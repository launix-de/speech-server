"""RTP codec abstraction — encode/decode between s16le PCM and wire format.

Supported codecs: Opus, G.722, PCMU (G.711 µ-law), PCMA (G.711 A-law).

Usage::

    from speech_pipeline.rtp_codec import PCMU, PCMA, G722, codec_for_pt

    codec = PCMU
    wire = codec.encode(s16le_pcm)   # s16le → wire bytes
    pcm = codec.decode(wire)         # wire bytes → s16le
"""
from __future__ import annotations

import audioop
import io
import queue
import threading

import av


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

    def close(self) -> None:
        """Release codec resources held by a live RTP session."""

    def new_session_codec(self) -> RTPCodec:
        """Return a codec instance safe to bind to one RTP session."""
        return self

    @property
    def timestamp_step(self) -> int:
        return int(self.clock_rate * self.frame_samples / self.sample_rate)

    @property
    def sdp_rtpmap(self) -> str:
        return f"a=rtpmap:{self.payload_type} {self.name}/{self.clock_rate}\r\n"

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


class _G722(RTPCodec):
    """G.722 wideband codec (16 kHz PCM, 8 kHz RTP clock).

    Synchronous encode/decode via PyAV CodecContext — same pattern as Opus.
    """
    __slots__ = RTPCodec.__slots__ + (
        "_encoder",
        "_decoder",
        "_encode_lock",
        "_decode_lock",
    )

    def __init__(self, runtime: bool = False) -> None:
        # RTP payload type 9 keeps an 8kHz RTP clock although the decoded
        # PCM is 16kHz wideband audio (RFC 3551 compatibility quirk).
        super().__init__("G722", 9, 8000, 16000, 320, 160, 0x00)
        self._encoder = None
        self._decoder = None
        self._encode_lock = threading.Lock()
        self._decode_lock = threading.Lock()
        if runtime:
            self._init_runtime()

    def new_session_codec(self) -> RTPCodec:
        return _G722(runtime=True)

    def _init_runtime(self) -> None:
        if self._encoder is not None:
            return

        self._encoder = av.codec.CodecContext.create("g722", "w")
        self._encoder.sample_rate = self.sample_rate
        self._encoder.layout = "mono"
        self._encoder.format = "s16"

        self._decoder = av.codec.CodecContext.create("g722", "r")
        self._decoder.sample_rate = self.sample_rate
        self._decoder.layout = "mono"
        self._decoder.format = "s16"

    def encode(self, s16le: bytes) -> bytes:
        self._init_runtime()
        if not s16le:
            return b""
        frame = av.AudioFrame(
            format="s16",
            layout="mono",
            samples=len(s16le) // 2,
        )
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(s16le)
        with self._encode_lock:
            packets = self._encoder.encode(frame)
        return b"".join(bytes(packet) for packet in packets)

    def decode(self, wire: bytes) -> bytes:
        self._init_runtime()
        if not wire:
            return b""
        packet = av.Packet(wire)
        with self._decode_lock:
            frames = self._decoder.decode(packet)
        if frames:
            return b"".join(f.to_ndarray().tobytes() for f in frames)
        return b"\x00" * (self.frame_samples * 2)

    def close(self) -> None:
        self._encoder = None
        self._decoder = None


class _Opus(RTPCodec):
    """Opus wideband codec (48 kHz, RFC 7587).

    At 48 kHz Opus matches the ConferenceMixer's native sample rate,
    eliminating resampling entirely.  Variable bitrate encoding via PyAV.
    """
    __slots__ = RTPCodec.__slots__ + (
        "_encoder",
        "_decoder",
        "_encode_lock",
        "_decode_lock",
    )

    def __init__(self, runtime: bool = False) -> None:
        # PT 111 is conventional for Opus in SIP (dynamic range).
        # RTP clock is always 48000 (RFC 7587), mono but SDP says /2.
        super().__init__("opus", 111, 48000, 48000, 960, 0, 0x00)
        self._encoder = None
        self._decoder = None
        self._encode_lock = threading.Lock()
        self._decode_lock = threading.Lock()
        if runtime:
            self._init_runtime()

    @property
    def sdp_rtpmap(self) -> str:
        # RFC 7587: Opus MUST use "/2" channel count even for mono
        return (f"a=rtpmap:{self.payload_type} opus/48000/2\r\n"
                f"a=fmtp:{self.payload_type} useinbandfec=1;stereo=0;sprop-stereo=0\r\n")

    def new_session_codec(self) -> RTPCodec:
        return _Opus(runtime=True)

    def _init_runtime(self) -> None:
        if self._encoder is not None:
            return
        self._encoder = av.codec.CodecContext.create("libopus", "w")
        self._encoder.sample_rate = 48000
        self._encoder.layout = "mono"
        self._encoder.format = "s16"
        self._encoder.bit_rate = 32000

        self._decoder = av.codec.CodecContext.create("libopus", "r")
        self._decoder.sample_rate = 48000
        self._decoder.layout = "mono"
        self._decoder.format = "s16"

    def encode(self, s16le: bytes) -> bytes:
        self._init_runtime()
        if not s16le:
            return b""
        frame = av.AudioFrame(format="s16", layout="mono",
                              samples=len(s16le) // 2)
        frame.sample_rate = 48000
        frame.planes[0].update(s16le)
        with self._encode_lock:
            packets = self._encoder.encode(frame)
        return b"".join(bytes(p) for p in packets)

    def decode(self, wire: bytes) -> bytes:
        self._init_runtime()
        if not wire:
            return b""
        packet = av.Packet(wire)
        with self._decode_lock:
            frames = self._decoder.decode(packet)
        if frames:
            # Use to_ndarray() — planes[0] may contain stride padding
            return b"".join(f.to_ndarray().tobytes() for f in frames)
        return b"\x00" * (self.frame_samples * 2)

    def close(self) -> None:
        self._encoder = None
        self._decoder = None


# Module-level singletons
Opus = _Opus()
G722 = _G722()
PCMU = _PCMU()
PCMA = _PCMA()

CODECS_BY_PT = {111: Opus, 9: G722, 0: PCMU, 8: PCMA}
CODECS_BY_NAME = {"opus": Opus, "G722": G722, "PCMU": PCMU, "PCMA": PCMA}

# SDP offer preference order
CODEC_PREFERENCE = [Opus, G722, PCMU, PCMA]


def codec_for_pt(pt: int) -> RTPCodec | None:
    """Look up codec by RTP payload type."""
    codec = CODECS_BY_PT.get(pt)
    return codec.new_session_codec() if codec is not None else None


def negotiate_payload_type(remote_pts: list[int]) -> int:
    """Pick best supported payload type from a remote SDP offer."""
    for codec in CODEC_PREFERENCE:
        if codec.payload_type in remote_pts:
            return codec.payload_type
    return PCMU.payload_type


def negotiate_codec(remote_pts: list[int]) -> RTPCodec:
    """Pick best codec from remote's offered payload types.

    Returns our preferred codec that the remote also supports.
    Falls back to PCMU if nothing matches.
    """
    return codec_for_pt(negotiate_payload_type(remote_pts)) or PCMU
