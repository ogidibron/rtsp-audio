"""
RTSP audio conference server

RTSP over TCP is used for session setup and control:
    OPTIONS   -> ask what commands the server supports
    DESCRIBE  -> ask the server to describe the audio stream
    SETUP     -> agree on which UDP ports to use for audio
    PLAY      -> start sending/receiving audio
    TEARDOWN  -> leave the call

RTP over UDP is used to actually move the audio once the call has started.

The server sits in the middle of the call. Every client sends its
microphone audio to the server. The server mixes together everyone
else's audio for each client (leaving that client's own voice out)
and sends the mix back to them. That is what makes it a group call
instead of just one person streaming to another.

Features:
    * Heartbeat keepalive over UDP so dead/NAT-timed-out clients are pruned.
    * Per-client mute: a client's audio can be excluded from the mix.
    * Roster: the server tracks participants and pushes the list.
    * Public address discovery aid for internet use (see --stun).
"""

import socket
import struct
import threading
import time
import argparse
from collections import deque
import numpy as np


RTP_VERSION = 2

# RTP payload type for the audio (97 = dynamic; maps to L16 in SDP).
PAYLOAD_TYPE = 97

# Control message types sent over the same UDP socket as RTP audio.
MSG_HEARTBEAT = 0x01   # client -> server: "I'm still here"
MSG_MUTE = 0x02        # client -> server: toggle this client's mute
MSG_ROSTER = 0x03      # server -> client: periodic participant list


def build_rtp_packet(sequence_number, timestamp, ssrc, audio_bytes):
    """Wraps raw audio bytes in a 12 byte RTP header."""
    byte1 = (RTP_VERSION << 6) | (0 << 5) | (0 << 4)  # version, padding, extension
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
    """A tiny control envelope: 1 byte type + 1 byte length + payload."""
    return struct.pack("!BB", msg_type, len(payload)) + payload


def parse_control_message(data):
    if len(data) < 2:
        return None, b""
    msg_type, length = struct.unpack("!BB", data[:2])
    return msg_type, data[2:2 + length]


RTSP_PORT = 5540           # TCP port clients connect to for call setup
SERVER_RTP_PORT = 5541     # UDP port the server listens on for audio + control

SAMPLE_RATE = 16000        # audio samples per second
SAMPLES_PER_FRAME = 320    # 20ms of audio per RTP packet
FRAME_INTERVAL = SAMPLES_PER_FRAME / SAMPLE_RATE  # seconds between audio frames

MIX_GAIN = 1.5             # boost the mixed output so it is clearly audible

HEARTBEAT_TIMEOUT = 8.0    # drop a client if no heartbeat for this long (seconds)
HEARTBEAT_INTERVAL = 2.0   # how often clients are expected to send heartbeats
ROSTER_INTERVAL = 3.0      # how often the server pushes the participant list


# All connected clients, keyed by their ssrc.
# Protected by clients_lock since multiple threads read and write it.
connected_clients = {}
clients_lock = threading.Lock()

next_ssrc = 1000  # simple counter used to hand out unique client ids


class ConnectedClient:
    """Everything the server needs to remember about one participant."""

    def __init__(self, ssrc, ip_address, rtp_port):
        self.ssrc = ssrc
        self.ip_address = ip_address        # client's IP address
        self.rtp_port = rtp_port            # UDP port on the client that receives the mix

        self.audio_frames = deque(maxlen=4)
        self.last_sequence = None           # last RTP sequence seen (for loss detection)
        self.lost_packets = 0
        self.received_packets = 0

        self.outgoing_sequence_number = 0   # sequence number for packets we send to this client
        self.outgoing_timestamp = 0         # timestamp for packets we send to this client

        self.muted = False                  # when True, this client's mic is excluded from mixes
        self.display_name = f"Client {ssrc}"

        self.return_address = None          # NAT-translated (ip, port) we actually received from
        self.last_heartbeat = time.time()   # for dead-client pruning


def get_roster():
    """Returns a sorted list of (ssrc, display_name, muted) for all clients."""
    with clients_lock:
        return sorted(
            ((c.ssrc, c.display_name, c.muted) for c in connected_clients.values()),
            key=lambda entry: entry[0],
        )


def handle_control_connection(connection, client_address):
    """Handles the RTSP style TCP connection for one client (OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN)."""
    global next_ssrc

    client_ssrc = None
    connection_file = connection.makefile("r")  # lets us read the request line by line

    try:
        while True:
            request_line = connection_file.readline()
            if not request_line:
                break  # client closed the connection
            request_line = request_line.strip()
            if not request_line:
                continue

            headers = {}
            while True:
                header_line = connection_file.readline().strip()
                if not header_line:
                    break
                name, _, value = header_line.partition(":")
                headers[name.strip()] = value.strip()

            method = request_line.split(" ")[0]
            cseq = headers.get("CSeq", "0")

            if method == "OPTIONS":
                connection.sendall(
                    f"RTSP/1.0 200 OK\r\n"
                    f"CSeq: {cseq}\r\n"
                    f"Public: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n\r\n".encode()
                )

            elif method == "DESCRIBE":
                description = (
                    f"m=audio {SERVER_RTP_PORT} RTP/AVP {PAYLOAD_TYPE}\r\n"
                    f"a=rtpmap:{PAYLOAD_TYPE} L16/{SAMPLE_RATE}/1\r\n"
                )
                connection.sendall(
                    f"RTSP/1.0 200 OK\r\n"
                    f"CSeq: {cseq}\r\n"
                    f"Content-Type: application/sdp\r\n"
                    f"Content-Length: {len(description)}\r\n\r\n"
                    f"{description}".encode()
                )

            elif method == "SETUP":
                transport_header = headers.get("Transport", "")
                client_rtp_port = None
                for part in transport_header.split(";"):
                    if part.startswith("client_port="):
                        client_rtp_port = int(part.split("=")[1])

                if client_rtp_port is None:
                    connection.sendall(f"RTSP/1.0 400 Bad Request\r\nCSeq: {cseq}\r\n\r\n".encode())
                    continue

                client_ssrc = next_ssrc
                next_ssrc += 1

                name = headers.get("X-Display-Name", "").strip() or f"Client {client_ssrc}"
                new_client = ConnectedClient(client_ssrc, client_address[0], client_rtp_port)
                new_client.display_name = name[:32]

                with clients_lock:
                    connected_clients[client_ssrc] = new_client

                connection.sendall(
                    f"RTSP/1.0 200 OK\r\n"
                    f"CSeq: {cseq}\r\n"
                    f"Transport: RTP/UDP;server_port={SERVER_RTP_PORT};ssrc={client_ssrc}\r\n"
                    f"Session: {client_ssrc}\r\n\r\n".encode()
                )
                print(f"[joined] {name} ({client_address[0]}) is now client {client_ssrc}")

            elif method == "PLAY":
                connection.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: {client_ssrc}\r\n\r\n".encode())
                print(f"[playing] client {client_ssrc} is now sending and receiving audio")

            elif method == "TEARDOWN":
                connection.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n\r\n".encode())
                if client_ssrc is not None:
                    with clients_lock:
                        connected_clients.pop(client_ssrc, None)
                print(f"[left] client {client_ssrc} left the call")
                break

            else:
                connection.sendall(f"RTSP/1.0 501 Not Implemented\r\nCSeq: {cseq}\r\n\r\n".encode())

    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        if client_ssrc is not None:
            with clients_lock:
                connected_clients.pop(client_ssrc, None)
        connection.close()


def receive_audio_from_clients(audio_socket):
    """Runs forever. Listens for RTP audio and control messages from all
    clients, stores each client's latest audio frame, and tracks liveness.

    The same socket is shared with mix_and_send_audio so the mixed audio we
    send back leaves from SERVER_RTP_PORT. That is what lets the reply
    traverse a remote client's NAT/firewall (symmetric RTP)."""
    print(f"[audio in] listening for microphone audio on UDP port {SERVER_RTP_PORT}")

    while True:
        try:
            packet_bytes, addr = audio_socket.recvfrom(2048)
        except OSError:
            continue

        if len(packet_bytes) < 2:
            continue

        # Distinguish control messages (first byte is a known MSG_* type) from
        # RTP packets (first byte starts with RTP version 2 -> 0x80..0xBF).
        first_byte = packet_bytes[0]
        is_rtp = (first_byte & 0xC0) == 0x80
        if not is_rtp:
            msg_type, payload = parse_control_message(packet_bytes)
            handle_control_message(msg_type, payload, addr, audio_socket)
            continue

        if len(packet_bytes) < 12:
            continue

        ssrc, seq, _, audio_bytes = parse_rtp_packet(packet_bytes)

        with clients_lock:
            sender = connected_clients.get(ssrc)
            if sender is not None:
                sender.audio_frames.append(audio_bytes)
                sender.return_address = addr
                sender.last_heartbeat = time.time()
                sender.received_packets += 1
                if sender.last_sequence is not None:
                    # Count wraps as lost if we skipped ahead.
                    expected = (sender.last_sequence + 1) & 0xFFFF
                    if seq != expected:
                        gap = (seq - expected) & 0xFFFF
                        sender.lost_packets += gap
                sender.last_sequence = seq


def handle_control_message(msg_type, payload, addr, audio_socket):
    if msg_type == MSG_HEARTBEAT:
        # Payload is the client's ssrc (4 bytes) so we can locate it even
        # before audio arrives.
        if len(payload) >= 4:
            ssrc = struct.unpack("!I", payload[:4])[0]
            with clients_lock:
                sender = connected_clients.get(ssrc)
                if sender is not None:
                    sender.return_address = addr
                    sender.last_heartbeat = time.time()
    elif msg_type == MSG_MUTE:
        # Payload: 4-byte ssrc + 1-byte 0/1 mute flag.
        if len(payload) >= 5:
            ssrc = struct.unpack("!I", payload[:4])[0]
            muted = bool(payload[4])
            with clients_lock:
                sender = connected_clients.get(ssrc)
                if sender is not None:
                    sender.muted = muted
                    print(f"[mute] client {ssrc} muted={muted}")


def prune_dead_clients():
    """Periodically drops clients that stopped sending heartbeats."""
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        now = time.time()
        with clients_lock:
            dead = [
                ssrc for ssrc, c in connected_clients.items()
                if now - c.last_heartbeat > HEARTBEAT_TIMEOUT
            ]
            for ssrc in dead:
                c = connected_clients.pop(ssrc)
                print(f"[timeout] client {ssrc} ({c.display_name}) dropped (no heartbeat)")


def broadcast_roster(audio_socket):
    """Periodically pushes the participant list to every client."""
    while True:
        time.sleep(ROSTER_INTERVAL)
        roster = get_roster()
        # Encode roster: count, then for each: ssrc(u32), muted(u8), name(bytes).
        body = struct.pack("!H", len(roster))
        for ssrc, name, muted in roster:
            name_bytes = name.encode("utf-8", "replace")[:32]
            body += struct.pack("!IBB", ssrc, muted, len(name_bytes)) + name_bytes

        msg = build_control_message(MSG_ROSTER, body)
        with clients_lock:
            # Only deliver to clients whose NAT-translated address we've
            # actually seen. A client's private UDP port is unreachable from
            # here, so skip any client we haven't heard from yet.
            targets = [
                c.return_address
                for c in connected_clients.values()
                if c.return_address is not None
            ]
        for target in targets:
            try:
                audio_socket.sendto(msg, target)
            except OSError:
                pass


def mix_and_send_audio(audio_socket):
    """Runs forever. Every 20ms, builds a personalized mix for each client
    (everyone else's audio, minus their own and minus muted talkers) and
    sends it to them. Replies are sent from SERVER_RTP_PORT (symmetric RTP)."""
    silent_frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.int32)
    next_loop_time = time.time()

    while True:
        loop_start_time = time.time()

        with clients_lock:
            clients_snapshot = list(connected_clients.values())

            frames_by_ssrc = {}
            for client in clients_snapshot:
                audio = None
                if client.audio_frames:
                    audio = client.audio_frames[-1]
                    client.audio_frames.clear()
                if audio is not None and len(audio) >= SAMPLES_PER_FRAME * 2:
                    frames_by_ssrc[client.ssrc] = np.frombuffer(
                        audio, dtype=np.int16
                    ).astype(np.int32)
                else:
                    frames_by_ssrc[client.ssrc] = silent_frame

            for client in clients_snapshot:
                other_frames = [
                    frame for ssrc, frame in frames_by_ssrc.items()
                    if ssrc != client.ssrc and not connected_clients[ssrc].muted
                ]

                if other_frames:
                    mixed_frame = sum(other_frames)
                    # Normalize by the number of active talkers so a single
                    # voice isn't quiet and many voices don't clip.
                    mixed_frame = mixed_frame / len(other_frames)
                    mixed_frame = (mixed_frame * MIX_GAIN).clip(-32768, 32767).astype(np.int16)
                else:
                    mixed_frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.int16)

                packet = build_rtp_packet(
                    client.outgoing_sequence_number,
                    client.outgoing_timestamp,
                    0,  # ssrc 0 means "this is the mixed server output"
                    mixed_frame.tobytes(),
                )
                client.outgoing_sequence_number += 1
                client.outgoing_timestamp += SAMPLES_PER_FRAME

                # Only send once we've learned the client's NAT-translated
                # address from an inbound packet. The client's private UDP port
                # (client.rtp_port) is unreachable across NAT, so we must not
                # fall back to it.
                target = client.return_address
                if target is None:
                    continue
                try:
                    audio_socket.sendto(packet, target)
                except OSError:
                    pass  # client's socket may have closed, ignore and continue

        next_loop_time += FRAME_INTERVAL
        time_spent = time.time() - loop_start_time
        sleep_time = next_loop_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_loop_time = time.time()


def discover_public_address(stun_host="stun.l.google.com", stun_port=19302, local_socket=None):
    """Best-effort public IP:port discovery via a STUN binding request.

    IMPORTANT: pass the *audio* socket (the one bound to SERVER_RTP_PORT) so the
    discovered mapping actually corresponds to the port clients will send to.
    Using a throwaway socket would reveal a different (useless) mapping.

    Returns (ip, port) or None.

    Note: STUN only works for cone NATs. Symmetric NATs assign a different
    mapping per destination, so the STUN-discovered port may not match the
    mapping used for the conference server. In that case port forwarding or a
    relay is still required.
    """
    try:
        # Reuse the audio socket when provided so the binding reflects port 5541.
        sock = local_socket if local_socket is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3.0)
        # STUN binding request: type 0x0001, length 0, magic cookie, 12-byte txid.
        magic = b"\x21\x12\xa4\x42"
        txid = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c"
        req = struct.pack("!HH", 0x0001, 0x0000) + magic + txid
        sock.sendto(req, (stun_host, stun_port))
        data, _ = sock.recvfrom(1024)
        # XOR-MAPPED-ADDRESS attribute type 0x0020.
        idx = 20
        while idx + 4 <= len(data):
            attr_type, attr_len = struct.unpack("!HH", data[idx:idx + 4])
            idx += 4
            if attr_type == 0x0020:
                family = data[idx + 1]
                # The port and address are XORed with the magic cookie (RFC 5389).
                xport = struct.unpack("!H", data[idx + 2:idx + 4])[0]
                port = xport ^ 0x2112
                if family == 0x01:  # IPv4
                    xaddr = data[idx + 4:idx + 8]
                    ip = ".".join(str(b ^ m) for b, m in zip(xaddr, magic))
                    return ip, port
            idx += attr_len
    except Exception as exc:
        print(f"[stun] discovery failed: {exc}")
    return None


def run_rtsp_server():
    """Accepts incoming RTSP control connections, one per client."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", RTSP_PORT))
    server_socket.listen()
    print(f"[rtsp] conference server listening on TCP port {RTSP_PORT}")

    while True:
        connection, client_address = server_socket.accept()
        threading.Thread(
            target=handle_control_connection, args=(connection, client_address), daemon=True
        ).start()


def main():
    parser = argparse.ArgumentParser(description="RTSP audio conference server")
    parser.add_argument("--stun", action="store_true", help="attempt STUN public address discovery")
    args = parser.parse_args()

    audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio_socket.bind(("0.0.0.0", SERVER_RTP_PORT))

    if args.stun:
        public = discover_public_address(local_socket=audio_socket)
        if public:
            print(f"[stun] public address appears to be {public[0]}:{public[1]}")
            print(f"[stun] clients on other networks should connect to that IP/port")

    threading.Thread(target=receive_audio_from_clients, args=(audio_socket,), daemon=True).start()
    threading.Thread(target=mix_and_send_audio, args=(audio_socket,), daemon=True).start()
    threading.Thread(target=prune_dead_clients, daemon=True).start()
    threading.Thread(target=broadcast_roster, args=(audio_socket,), daemon=True).start()
    run_rtsp_server()


if __name__ == "__main__":
    main()
