"""
Shared RTP packet and control-message helpers for the RTSP audio conference.

Both client.py and server.py import from here so the wire format stays in
one place.
"""

import struct


RTP_VERSION = 2
PAYLOAD_TYPE = 97

# Control message types (must match both ends).
MSG_HEARTBEAT = 0x01
MSG_MUTE = 0x02
MSG_ROSTER = 0x03


def build_rtp_packet(sequence_number, timestamp, ssrc, audio_bytes):
    """Wraps raw audio bytes in a 12-byte RTP header."""
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
    """A tiny control envelope: 1 byte type + 1 byte length + payload."""
    return struct.pack("!BB", msg_type, len(payload)) + payload


def parse_control_message(data):
    if len(data) < 2:
        return None, b""
    msg_type, length = struct.unpack("!BB", data[:2])
    return msg_type, data[2:2 + length]
