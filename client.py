"""
RTSP audio conference client with a Tkinter GUI.

Joins a call hosted by server.py using RTSP over TCP
(OPTIONS, DESCRIBE, SETUP, PLAY), then streams microphone audio to the
server and plays back the mixed audio the server sends in return over
RTP/UDP.

Quality / reliability features:
    * Jitter buffer with a real playout delay plus RTP sequence reordering and
      packet-loss concealment, for smooth playback over the internet.
    * Automatic gain control + noise gate so quiet mics are normalized instead
      of just boosted, and background hiss is suppressed.
    * UDP heartbeat keepalive so the server can prune dead clients.
    * Server-sent roster of participants; you can mute only yourself.

Run the GUI:
    python client.py

Run the headless CLI client:
    python gui_client.py --cli <server-ip>
"""

import socket
import struct
import threading
import time
import sys
import argparse
import numpy as np
import sounddevice as sd


RTP_VERSION = 2
PAYLOAD_TYPE = 97

# Control message types (must match server.py).
MSG_HEARTBEAT = 0x01
MSG_MUTE = 0x02
MSG_ROSTER = 0x03


def build_rtp_packet(sequence_number, timestamp, ssrc, audio_bytes):
    """Wraps raw audio bytes in a 12 byte RTP header."""
    byte1 = (RTP_VERSION << 6) | (0 << 5) | (0 << 4)
    byte2 = PAYLOAD_TYPE
    header = struct.pack(
        "!BBHII",
        byte1,
        byte2,
        sequence_number & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )
    return header + audio_bytes


def parse_rtp_packet(packet_bytes):
    """Splits an RTP packet back into its header fields and audio bytes."""
    header_bytes = packet_bytes[:12]
    audio_bytes = packet_bytes[12:]
    _, _, sequence_number, timestamp, ssrc = struct.unpack("!BBHII", header_bytes)
    return ssrc, sequence_number, timestamp, audio_bytes


def build_control_message(msg_type, payload=b""):
    return struct.pack("!BB", msg_type, len(payload)) + payload


def parse_control_message(data):
    if len(data) < 2:
        return None, b""
    msg_type, length = struct.unpack("!BB", data[:2])
    return msg_type, data[2:2 + length]



# Default server address. Left empty so the user must explicitly supply the
# server's IP (LAN IP on the same network, or the public IP + port forwarding
# when connecting across the internet). A hardcoded public IP here is a common
# cause of "can't connect" because it is wrong for LAN use and goes stale.
SERVER_IP = "127.0.0.1"

RTSP_PORT = 5540
SERVER_RTP_PORT = 5541
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 320  # 20ms of audio per RTP packet
FRAME_INTERVAL = SAMPLES_PER_FRAME / SAMPLE_RATE

MIC_GAIN = 4.0
PLAYBACK_GAIN = 2.5

# Jitter buffer: how many 20ms frames we hold before playing. Larger = more
# tolerant of jitter/loss, but adds latency.
JITTER_FRAMES = 4
# If we miss this many consecutive frames, we still hold the last good frame.
MAX_CONCEAL = 10

HEARTBEAT_INTERVAL = 2.0


class AudioProcessor:
    """Microphone-side AGC + noise gate so quiet mics are normalized and
    background noise is silenced. Pure NumPy, no external dependencies."""

    def __init__(self, target_peak=0.35, attack=0.02, release=0.002,
                 noise_floor=0.01, gain_cap=12.0):
        self.target_peak = target_peak
        self.attack = attack
        self.release = release
        self.noise_floor = noise_floor
        self.gain_cap = gain_cap
        self.gain = 1.0

    def process(self, samples_float32):
        # samples_float32: numpy array in [-1, 1] (mono).
        peak = float(np.max(np.abs(samples_float32))) + 1e-6

        # Desired gain to reach the target peak, smoothly tracked.
        desired = min(self.gain_cap, self.target_peak / peak)
        if desired > self.gain:
            # Attack: raise gain slowly to avoid pumping on transients.
            coeff = self.attack
        else:
            coeff = self.release
        self.gain += (desired - self.gain) * coeff

        out = samples_float32 * self.gain
        # Noise gate: if the signal is below the floor, fade it out.
        if peak < self.noise_floor:
            out *= max(0.0, (peak / self.noise_floor) ** 2)
        return out


class JitterBuffer:
    """Reorders RTP frames by sequence number and conceals losses.

    A real playout buffer: incoming frames are held until JITTER_FRAMES of
    audio have accumulated, then released in strict sequence order. This
    absorbs network jitter. Missing frames are concealed by linearly fading
    the last good frame toward silence (avoids the "stuck tone" of a pure
    hold), and once the conceal budget is exceeded we emit silence."""

    def __init__(self, size=SAMPLES_PER_FRAME, delay_frames=JITTER_FRAMES):
        self.size = size
        self.delay = max(1, delay_frames)
        # Buffered frames keyed by sequence number, so we can release in order.
        self.buffered = {}
        self.expected_seq = None      # next sequence number the player wants
        self.started = False          # becomes True once we have enough to play
        self.last_good = np.zeros(size, dtype=np.int16)
        self.conceal_count = 0

    def push(self, sequence_number, frame):
        self.buffered[sequence_number] = frame
        # Bound memory: drop the oldest outstanding sequence if we fall behind.
        if len(self.buffered) > 64:
            oldest = min(self.buffered)
            self.buffered.pop(oldest, None)

    def pop(self):
        """Returns the next frame in sequence order, concealing if missing."""
        # Not enough buffered yet: keep filling the playout window.
        if not self.started:
            if len(self.buffered) < self.delay:
                return self._conceal(playing=False)
            # Prime: start at the lowest sequence we have.
            self.expected_seq = min(self.buffered)
            self.started = True

        seq = self.expected_seq
        if seq in self.buffered:
            frame = self.buffered.pop(seq)
            self.expected_seq = (seq + 1) & 0xFFFF
            self.last_good = frame
            self.conceal_count = 0
            return frame

        # The expected frame never arrived: conceal the gap and skip ahead.
        self.expected_seq = (seq + 1) & 0xFFFF
        return self._conceal(playing=True)

    def _conceal(self, playing):
        if not playing:
            # Still filling the initial window; stay silent until we start.
            return np.zeros(self.size, dtype=np.int16)
        if self.conceal_count < MAX_CONCEAL and np.any(self.last_good):
            # Fade the last good frame toward silence (linear decay) to avoid
            # a hard repeated tone on loss.
            self.conceal_count += 1
            fade = max(0.0, 1.0 - self.conceal_count / MAX_CONCEAL)
            faded = (self.last_good * fade).astype(np.int16)
            self.last_good = faded
            return faded
        return np.zeros(self.size, dtype=np.int16)


class Client:
    def __init__(self, server_ip, display_name="", device=None):
        self.server_ip = server_ip
        self.display_name = display_name[:32]
        self.device = device  # sounddevice device id/name or None for default

        self.next_cseq = 1
        self.session_ssrc = None
        self.server_rtp_port = SERVER_RTP_PORT

        self.rtsp_socket = None

        self.audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.audio_socket.bind(("0.0.0.0", 0))
        self.local_rtp_port = self.audio_socket.getsockname()[1]

        self.jitter = JitterBuffer(SAMPLES_PER_FRAME)
        self.processor = AudioProcessor()

        self.mic_muted = False
        self.speaker_muted = False

        self.mic_gain = MIC_GAIN
        self.playback_gain = PLAYBACK_GAIN

        self.peak_lock = threading.Lock()
        self.mic_peak = 0.0
        self.speaker_peak = 0.0

        self.roster_lock = threading.Lock()
        self.roster = []  # list of (ssrc, name, muted) from the server

        self.outgoing_sequence_number = 0
        self.outgoing_timestamp = 0
        self.call_active = False

    # ---- RTSP handshake -------------------------------------------------

    def send_rtsp_request(self, method, extra_headers=None):
        request_lines = [f"{method} rtsp://{self.server_ip}/conference RTSP/1.0", f"CSeq: {self.next_cseq}"]
        if extra_headers:
            request_lines.extend(extra_headers)
        request_text = "\r\n".join(request_lines) + "\r\n\r\n"

        self.rtsp_socket.sendall(request_text.encode())
        self.next_cseq += 1

        response_file = self.rtsp_socket.makefile("r")
        status_line = response_file.readline().strip()

        headers = {}
        while True:
            header_line = response_file.readline().strip()
            if not header_line:
                break
            name, _, value = header_line.partition(":")
            headers[name.strip()] = value.strip()
        return status_line, headers

    def join_call(self):
        self.rtsp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtsp_socket.settimeout(30)
        self.rtsp_socket.connect((self.server_ip, RTSP_PORT))

        self.send_rtsp_request("OPTIONS")
        self.send_rtsp_request("DESCRIBE")

        extra = [f"Transport: RTP/UDP;client_port={self.local_rtp_port}"]
        if self.display_name:
            extra.append(f"X-Display-Name: {self.display_name}")
        status_line, headers = self.send_rtsp_request("SETUP", extra)
        if "200" not in status_line:
            raise RuntimeError(f"SETUP failed: {status_line}")

        for part in headers.get("Transport", "").split(";"):
            if part.startswith("server_port="):
                self.server_rtp_port = int(part.split("=")[1])
            elif part.startswith("ssrc="):
                self.session_ssrc = int(part.split("=")[1])

        self.send_rtsp_request("PLAY")
        print(f"Joined the call as '{self.display_name or 'anon'}' "
              f"-> {self.server_ip}:{self.server_rtp_port}")

    def leave_call(self):
        try:
            self.send_rtsp_request("TEARDOWN")
        except OSError:
            pass
        self.call_active = False

    # ---- Audio callbacks ------------------------------------------------

    def on_microphone_audio(self, input_samples, frame_count, time_info, status):
        if status:
            pass
        samples = input_samples[:, 0].astype(np.float32)

        peak = float(np.max(np.abs(samples)))
        with self.peak_lock:
            self.mic_peak = peak

        if self.mic_muted:
            audio_bytes = np.zeros_like(samples, dtype=np.int16).tobytes()
        else:
            # AGC + noise gate, then apply mic gain and convert to int16.
            processed = self.processor.process(samples)
            audio_bytes = (processed * 32767 * self.mic_gain).clip(-32767, 32767).astype(np.int16).tobytes()

        packet = build_rtp_packet(
            self.outgoing_sequence_number, self.outgoing_timestamp, self.session_ssrc, audio_bytes
        )
        self.outgoing_sequence_number = (self.outgoing_sequence_number + 1) & 0xFFFF
        self.outgoing_timestamp += SAMPLES_PER_FRAME

        try:
            self.audio_socket.sendto(packet, (self.server_ip, self.server_rtp_port))
        except OSError:
            pass

    def on_speaker_output(self, output_samples, frame_count, time_info, status):
        if self.speaker_muted:
            output_samples[:, 0] = 0.0
            return

        try:
            frame = self.jitter.pop()
        except Exception:
            frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.int16)

        with self.peak_lock:
            self.speaker_peak = float(np.max(np.abs(frame))) / 32768.0

        output_samples[:, 0] = frame.astype(np.float32) / 32768.0 * self.playback_gain

    def receive_thread(self):
        self.audio_socket.settimeout(1.0)
        while self.call_active:
            try:
                packet_bytes, _ = self.audio_socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(packet_bytes) < 2:
                continue

            first_byte = packet_bytes[0]
            is_rtp = (first_byte & 0xC0) == 0x80
            if not is_rtp:
                self._handle_control(packet_bytes)
                continue

            if len(packet_bytes) < 12:
                continue

            ssrc, seq, _, audio_bytes = parse_rtp_packet(packet_bytes)
            if ssrc != 0:
                # Not the mixed server output; ignore (defensive).
                continue
            frame = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(frame) != SAMPLES_PER_FRAME:
                continue
            self.jitter.push(seq, frame)

    def _handle_control(self, data):
        msg_type, payload = parse_control_message(data)
        if msg_type == MSG_ROSTER:
            self._update_roster(payload)

    def _update_roster(self, payload):
        if len(payload) < 2:
            return
        count = struct.unpack("!H", payload[:2])[0]
        pos = 2
        roster = []
        for _ in range(count):
            if pos + 6 > len(payload):
                break
            ssrc, muted, name_len = struct.unpack("!IBB", payload[pos:pos + 6])
            pos += 6
            name = payload[pos:pos + name_len].decode("utf-8", "replace")
            pos += name_len
            roster.append((ssrc, name, bool(muted)))
        with self.roster_lock:
            self.roster = roster

    def get_roster(self):
        with self.roster_lock:
            return list(self.roster)

    def send_mute_state(self, muted):
        if self.session_ssrc is None:
            return
        payload = struct.pack("!IB", self.session_ssrc, 1 if muted else 0)
        try:
            self.audio_socket.sendto(build_control_message(MSG_MUTE, payload),
                                     (self.server_ip, self.server_rtp_port))
        except OSError:
            pass

    def heartbeat_loop(self):
        while self.call_active:
            if self.session_ssrc is not None:
                payload = struct.pack("!I", self.session_ssrc)
                try:
                    self.audio_socket.sendto(build_control_message(MSG_HEARTBEAT, payload),
                                             (self.server_ip, self.server_rtp_port))
                except OSError:
                    pass
            time.sleep(HEARTBEAT_INTERVAL)

    def start(self):
        self.join_call()
        self.call_active = True

        threading.Thread(target=self.receive_thread, daemon=True).start()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

        self.input_stream = sd.InputStream(
            device=self.device, samplerate=SAMPLE_RATE, channels=1,
            blocksize=SAMPLES_PER_FRAME, callback=self.on_microphone_audio
        )
        self.output_stream = sd.OutputStream(
            device=self.device, samplerate=SAMPLE_RATE, channels=1,
            blocksize=SAMPLES_PER_FRAME, callback=self.on_speaker_output
        )
        self.input_stream.start()
        self.output_stream.start()
        print("On the call.")

    def stop(self):
        self.call_active = False
        try:
            self.input_stream.stop()
            self.input_stream.close()
            self.output_stream.stop()
            self.output_stream.close()
        except Exception:
            pass
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        self.leave_call()
        print("Left the call.")

    def get_peaks(self):
        with self.peak_lock:
            return self.mic_peak, self.speaker_peak


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox


class LevelMeter(tk.Canvas):
    def __init__(self, master, label, height=18, **kwargs):
        super().__init__(master, height=height, **kwargs)
        self.height = height
        self.label = label
        self.level = 0.0
        self._bar = self.create_rectangle(0, 0, 0, height, fill="#3ad07a", outline="")
        self._text = self.create_text(6, height / 2, anchor="w", text=label, fill="#ffffff")

    def set_level(self, value):
        self.level = max(0.0, min(1.0, value))
        width = self.winfo_width()
        bar_end = int(width * self.level)
        self.coords(self._bar, 0, 0, bar_end, self.height)
        color = "#e0584f" if self.level > 0.95 else ("#e0c14f" if self.level > 0.75 else "#3ad07a")
        self.itemconfig(self._bar, fill=color)


class ConferenceGUI:
    def __init__(self, root, default_ip):
        self.root = root
        self.client = None
        self.connected = False

        root.title("RTSP Audio Conference")
        root.resizable(True, True)
        root.configure(bg="#1f2430")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6)
        style.configure("TLabel", background="#1f2430", foreground="#e6e6e6")
        style.configure("TEntry", padding=4)
        style.configure("Treeview", background="#2a3140", foreground="#e6e6e6",
                        fieldbackground="#2a3140")
        style.configure("Treeview.Heading", background="#222838", foreground="#e6e6e6")

        pad = {"padx": 10, "pady": 6}

        # --- Connection row ---
        conn_frame = ttk.Frame(root)
        conn_frame.pack(fill="x", **pad)

        ttk.Label(conn_frame, text="Server IP:").pack(side="left")
        self.ip_var = tk.StringVar(value=default_ip)
        self.ip_entry = ttk.Entry(conn_frame, textvariable=self.ip_var, width=20)
        self.ip_entry.pack(side="left", padx=(6, 6))

        ttk.Label(conn_frame, text="Name:").pack(side="left")
        self.name_var = tk.StringVar(value="")
        self.name_entry = ttk.Entry(conn_frame, textvariable=self.name_var, width=14)
        self.name_entry.pack(side="left", padx=(0, 10))

        self.join_button = ttk.Button(conn_frame, text="Join Call", command=self.toggle_call)
        self.join_button.pack(side="left")

        # --- Audio device row ---
        dev_frame = ttk.Frame(root)
        dev_frame.pack(fill="x", **pad)
        ttk.Label(dev_frame, text="Audio device:").pack(side="left")
        self.devices = self._list_devices()
        self.device_var = tk.StringVar(value="Default")
        self.device_combo = ttk.Combobox(
            dev_frame, textvariable=self.device_var,
            values=list(self.devices.keys()), state="readonly", width=40
        )
        self.device_combo.pack(side="left", padx=(6, 0))

        # --- Status ---
        self.status_var = tk.StringVar(value="Not connected")
        self.status_label = ttk.Label(root, textvariable=self.status_var, foreground="#9aa4b2")
        self.status_label.pack(anchor="w", **pad)

        # --- Meters ---
        meter_frame = ttk.Frame(root)
        meter_frame.pack(fill="x", **pad)
        self.mic_meter = LevelMeter(meter_frame, "Mic", bg="#2a3140", width=420)
        self.mic_meter.pack(fill="x", pady=2)
        self.speaker_meter = LevelMeter(meter_frame, "Speaker", bg="#2a3140", width=420)
        self.speaker_meter.pack(fill="x", pady=2)

        # --- Controls ---
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(fill="x", **pad)
        self.mic_mute_var = tk.BooleanVar(value=False)
        self.mic_mute_button = ttk.Checkbutton(
            ctrl_frame, text="Mute Mic", variable=self.mic_mute_var,
            command=self.apply_mic_mute, state="disabled"
        )
        self.mic_mute_button.pack(side="left", padx=(0, 10))
        self.speaker_mute_var = tk.BooleanVar(value=False)
        self.speaker_mute_button = ttk.Checkbutton(
            ctrl_frame, text="Mute Speaker", variable=self.speaker_mute_var,
            command=self.apply_speaker_mute, state="disabled"
        )
        self.speaker_mute_button.pack(side="left")

        # --- Volume sliders ---
        vol_frame = ttk.Frame(root)
        vol_frame.pack(fill="x", **pad)
        ttk.Label(vol_frame, text="Mic Volume").pack(anchor="w")
        self.mic_gain_var = tk.DoubleVar(value=MIC_GAIN)
        self.mic_gain_slider = ttk.Scale(
            vol_frame, from_=1.0, to=10.0, variable=self.mic_gain_var,
            orient="horizontal", command=self.apply_mic_gain, state="disabled"
        )
        self.mic_gain_slider.pack(fill="x")
        ttk.Label(vol_frame, text="Playback Volume").pack(anchor="w", pady=(6, 0))
        self.play_gain_var = tk.DoubleVar(value=PLAYBACK_GAIN)
        self.play_gain_slider = ttk.Scale(
            vol_frame, from_=1.0, to=10.0, variable=self.play_gain_var,
            orient="horizontal", command=self.apply_play_gain, state="disabled"
        )
        self.play_gain_slider.pack(fill="x")

        # --- Roster (participants) ---
        ros_frame = ttk.LabelFrame(root, text="Participants")
        ros_frame.pack(fill="both", expand=True, **pad)
        self.roster_tree = ttk.Treeview(
            ros_frame, columns=("name", "muted"), show="headings", height=5
        )
        self.roster_tree.heading("name", text="Name")
        self.roster_tree.heading("muted", text="Self-muted")
        self.roster_tree.column("name", width=200)
        self.roster_tree.column("muted", width=60)
        self.roster_tree.pack(fill="both", expand=True, side="left")
        self.roster_tree.bind("<Double-1>", self.on_roster_double_click)

        self._poll_meters()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _list_devices(self):
        devices = {"Default": None}
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    label = f"{i}: {dev['name']}"
                    devices[label] = i
        except Exception:
            pass
        return devices

    def _device_id(self):
        sel = self.device_var.get()
        return self.devices.get(sel, None)

    def toggle_call(self):
        if not self.connected:
            self.join_call()
        else:
            self.leave_call()

    def join_call(self):
        ip = self.ip_var.get().strip()
        if not ip:
            messagebox.showerror("Error", "Enter a server IP address.")
            return
        name = self.name_var.get().strip()

        self.join_button.config(state="disabled")
        self.status_var.set(f"Connecting to {ip}...")
        self.root.update_idletasks()

        def worker():
            try:
                self.client = Client(ip, display_name=name, device=self._device_id())
                self.client.start()
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._join_failed(str(exc)))
                return
            self.root.after(0, self._join_succeeded)

        threading.Thread(target=worker, daemon=True).start()

    def _join_failed(self, reason):
        self.status_var.set("Connection failed: " + reason)
        self.join_button.config(state="normal")

    def _join_succeeded(self):
        self.connected = True
        self.status_var.set(f"Connected to {self.client.server_ip}")
        self.join_button.config(text="Leave Call")
        self.join_button.config(state="normal")

        self.mic_gain_var.set(self.client.mic_gain)
        self.play_gain_var.set(self.client.playback_gain)
        self.mic_mute_var.set(self.client.mic_muted)
        self.speaker_mute_var.set(self.client.speaker_muted)

        for w in (self.mic_mute_button, self.speaker_mute_button,
                  self.mic_gain_slider, self.play_gain_slider):
            w.config(state="normal")

    def leave_call(self):
        if self.client is not None:
            try:
                self.client.stop()
            except Exception:
                pass
            self.client = None
        self.connected = False
        self.status_var.set("Not connected")
        self.join_button.config(text="Join Call")
        for w in (self.mic_mute_button, self.speaker_mute_button,
                  self.mic_gain_slider, self.play_gain_slider):
            w.config(state="disabled")
        for row in self.roster_tree.get_children():
            self.roster_tree.delete(row)

    def apply_mic_mute(self):
        if self.client:
            self.client.mic_muted = self.mic_mute_var.get()
            self.client.send_mute_state(self.client.mic_muted)

    def apply_speaker_mute(self):
        if self.client:
            self.client.speaker_muted = self.speaker_mute_var.get()

    def apply_mic_gain(self, _=None):
        if self.client:
            self.client.mic_gain = self.mic_gain_var.get()

    def apply_play_gain(self, _=None):
        if self.client:
            self.client.playback_gain = self.play_gain_var.get()

    def on_roster_double_click(self, _):
        # Only the local participant's mute is controllable from here; the
        # server enforces it. Other rows show their own server-reported state
        # and are read-only.
        sel = self.roster_tree.selection()
        if not sel:
            return
        item = sel[0]
        name = self.roster_tree.item(item, "values")[0]
        if name != (self.client.display_name if self.client else ""):
            return
        new_state = not self.mic_mute_var.get()
        self.mic_mute_var.set(new_state)
        self.apply_mic_mute()

    def _refresh_roster(self):
        if not self.client:
            return
        roster = self.client.get_roster()
        existing = {self.roster_tree.item(r, "values")[0]: r
                    for r in self.roster_tree.get_children()}
        seen = set()
        for ssrc, name, muted in roster:
            seen.add(name)
            if name in existing:
                self.roster_tree.set(existing[name], "muted", "yes" if muted else "no")
            else:
                self.roster_tree.insert("", "end", values=(name, "yes" if muted else "no"))
        # Remove participants no longer present.
        for name, row in existing.items():
            if name not in seen:
                self.roster_tree.delete(row)

    def _poll_meters(self):
        if self.client is not None and self.connected:
            mic, speaker = self.client.get_peaks()
            self.mic_meter.set_level(mic)
            self.speaker_meter.set_level(speaker)
            self._refresh_roster()
        else:
            self.mic_meter.set_level(0.0)
            self.speaker_meter.set_level(0.0)
        self.root.after(120, self._poll_meters)

    def on_close(self):
        if self.connected:
            self.leave_call()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_gui(default_ip):
    root = tk.Tk()
    ConferenceGUI(root, default_ip)
    root.mainloop()


def run_cli(server_ip, display_name="", device=None):
    client = Client(server_ip, display_name=display_name, device=device)
    client.start()
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        client.stop()


def main():
    parser = argparse.ArgumentParser(description="RTSP audio conference client")
    parser.add_argument("ip", nargs="?", default=SERVER_IP, help="server IP address")
    parser.add_argument("--server", dest="server_opt", default=None, help="server IP (alt)")
    parser.add_argument("--name", default="", help="display name")
    parser.add_argument("--device", default=None, help="audio input device id")
    parser.add_argument("--cli", action="store_true", help="run headless CLI client")
    args = parser.parse_args()

    server_ip = args.server_opt or args.ip

    if not server_ip:
        parser.error("a server IP is required (positional IP or --server)")

    if args.cli:
        run_cli(server_ip, display_name=args.name, device=args.device)
    else:
        run_gui(server_ip)


if __name__ == "__main__":
    main()
