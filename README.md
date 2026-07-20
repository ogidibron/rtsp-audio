# RTSP Audio Conference

A small group voice-call app: a relay server mixes everyone's microphone
audio and streams the mix back to each participant over RTP/UDP, with RTSP
over TCP for call setup.

```
server.py       - conference server (mixes audio, relays to clients)
client.py       - client with Tkinter GUI + headless CLI mode
rtp.py          - shared RTP/control protocol helpers
```

## Requirements

```
pip install numpy sounddevice
```

`tkinter` ships with standard Python on Windows/macOS. On Linux install the
system package (e.g. `sudo apt install python3-tk`).

## Run

Start the server:

```
python server.py
```

Start a client (GUI):

```
python client.py 127.0.0.1
python client.py --server 127.0.0.1 --name Alice
```

To select a specific audio device, pass its numeric ID (run the client once
without `--device` to see the list printed in the GUI dropdown, or use
`sounddevice.query_devices()` in Python):

Headless CLI client (no GUI), useful for testing:

```
python client.py --cli 127.0.0.1
```

## Connecting over the internet (no LAN)

The code is IP-based, so the server just needs to be reachable. Options,
simplest first:

1. **Tailscale / ZeroTier (recommended)** - install on server + clients, use
   the assigned `100.x.x.x` IP. No port forwarding, works through any NAT.
2. **Port forwarding** - forward TCP `5540` and UDP `5541` to the server host,
   then connect to your public IP. Run `python server.py --stun` to *print* your
   discovered public address (informational only; the server still binds
   `0.0.0.0` and you must give clients the reachable IP/port).
3. **Cloud VPS** - run `server.py` on a VPS and open those two ports.

The client sends a UDP heartbeat and the server prunes clients that stop
sending it, so dropped connections are cleaned up. The server's replies use
symmetric RTP (sent back from the port you sent to) for NAT traversal. Note:
this works through many but not all NAT types (e.g. symmetric NATs may fail);
a VPN/tunnel is the most reliable option.

## Features

- **Jitter buffer + packet-loss concealment** - incoming RTP frames are held
  in a playout buffer (a few frames of delay) and released in strict sequence
  order, absorbing network jitter. Missing frames are concealed by fading the
  last good frame toward silence.
- **AGC + noise gate** - the mic is automatically normalized to a target level
  and background noise below the floor is suppressed (no external deps).
- **Per-user mute** - the server excludes a muted client's audio from everyone
  else's mix. Toggle "Mute Mic" in the GUI (sends a `MSG_MUTE` control packet).
  The roster's "Self-muted" column shows each participant's own mute state.
- **Participant roster** - the server periodically pushes the participant list;
  the GUI shows it in the "Participants" panel. (You can only mute yourself;
  other rows are read-only.)
- **Live level meters** - Mic and Speaker bars in the GUI.
- **Volume controls** - separate Mic and Playback gain sliders.
- **Audio device selection** - pick an input/output device from the dropdown.
- **Heartbeat keepalive** - the server prunes dead clients so the call stays
  healthy. If your connection drops, re-join from the GUI.

## Protocol notes

- RTSP-like TCP handshake: `OPTIONS`, `DESCRIBE`, `SETUP`, `PLAY`, `TEARDOWN`.
  `SETUP` accepts an `X-Display-Name` header for the participant name.
- RTP audio: 16-bit PCM (L16), 16 kHz, mono, 320 samples (20 ms) per packet,
  payload type 97.
- A separate control channel shares the RTP/UDP socket: heartbeat (`0x01`),
  mute (`0x02`), and roster (`0x03`) messages. They are distinguished from RTP
  by the first byte (control messages are not `0x80..0xBF`).

## Tuning

Constants at the top of each file:

- `server.py`: `MIX_GAIN`, `HEARTBEAT_TIMEOUT`, `HEARTBEAT_INTERVAL`,
  `ROSTER_INTERVAL`.
- `client.py`: `MIC_GAIN`, `PLAYBACK_GAIN`, `JITTER_FRAMES`, `MAX_CONCEAL`,
  `HEARTBEAT_INTERVAL`. `AudioProcessor` params control AGC target/gate.
