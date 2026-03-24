"""RTP codec abstraction — encode/decode between s16le PCM and wire format.

Supported codecs: PCMU (G.711 µ-law), PCMA (G.711 A-law).
Prepared for G.722 (wideband, 16 kHz).

Usage::

    from speech_pipeline.rtp_codec import PCMU, PCMA, codec_for_pt

    codec = PCMU
    wire = codec.encode(s16le_pcm)   # s16le → wire bytes
    pcm = codec.decode(wire)         # wire bytes → s16le
"""
from __future__ import annotations

import audioop


class RTPCodec:
    """Base class for RTP codecs."""

    __slots__ = ("name", "payload_type", "clock_rate", "sample_rate",
                 "frame_samples", "frame_bytes", "silence_byte")

    def __init__(self, name: str, payload_type: int, clock_rate: int,
                 sample_rate: int, frame_samples: int, frame_bytes: int,
                 silence_byte: int) -> None:
        self.name = name
        self.payload_type = payload_type
        self.clock_rate = clock_rate      # RTP timestamp clock
        self.sample_rate = sample_rate    # actual PCM sample rate
        self.frame_samples = frame_samples  # PCM samples per 20ms
        self.frame_bytes = frame_bytes    # wire bytes per 20ms frame
        self.silence_byte = silence_byte  # wire silence fill byte

    def encode(self, s16le: bytes) -> bytes:
        """Encode s16le PCM → wire format."""
        raise NotImplementedError

    def decode(self, wire: bytes) -> bytes:
        """Decode wire format → s16le PCM."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}(PT={self.payload_type}, {self.sample_rate}Hz)"


class _PCMU(RTPCodec):
    def __init__(self) -> None:
        # 8kHz, 160 samples/frame, 160 wire bytes/frame
        super().__init__("PCMU", 0, 8000, 8000, 160, 160, 0xFF)

    def encode(self, s16le: bytes) -> bytes:
        return audioop.lin2ulaw(s16le, 2)

    def decode(self, wire: bytes) -> bytes:
        return audioop.ulaw2lin(wire, 2)


class _PCMA(RTPCodec):
    def __init__(self) -> None:
        # 8kHz, 160 samples/frame, 160 wire bytes/frame
        super().__init__("PCMA", 8, 8000, 8000, 160, 160, 0xD5)

    def encode(self, s16le: bytes) -> bytes:
        return audioop.lin2alaw(s16le, 2)

    def decode(self, wire: bytes) -> bytes:
        return audioop.alaw2lin(wire, 2)


# Module-level singletons
PCMU = _PCMU()
PCMA = _PCMA()

CODECS_BY_PT = {0: PCMU, 8: PCMA}
CODECS_BY_NAME = {"PCMU": PCMU, "PCMA": PCMA}

# SDP offer preference order
CODEC_PREFERENCE = [PCMU, PCMA]


def codec_for_pt(pt: int) -> RTPCodec | None:
    """Look up codec by RTP payload type."""
    return CODECS_BY_PT.get(pt)


def negotiate_codec(remote_pts: list[int]) -> RTPCodec:
    """Pick best codec from remote's offered payload types.

    Returns our preferred codec that the remote also supports.
    Falls back to PCMU if nothing matches.
    """
    for codec in CODEC_PREFERENCE:
        if codec.payload_type in remote_pts:
            return codec
    return PCMU
