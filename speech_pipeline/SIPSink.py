"""Sink stage: writes audio into a SIP call.

Input: s16le mono @ 48 kHz (same as mixer — no SampleRateConverter needed).
Downsamples 48k→8k internally and encodes to wire codec.
Same principle as CodecSocketSink which encodes 48kHz internally.
"""
from __future__ import annotations

import audioop
import logging

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-sink")


class SIPSink(Stage):
    """Terminal sink: receives 48 kHz s16le from mixer, downsamples to
    8 kHz, encodes to SIP codec, writes to call.

    No SampleRateConverter in the pipeline — conversion is internal,
    keeping the data path identical to the websocket codec path.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.input_format = AudioFormat(48000, "s16le")

    def run(self) -> None:
        if not self.upstream:
            return

        if not self.session.connected.is_set():
            _LOGGER.info("SIPSink: waiting for call to connect...")
            self.session.connected.wait(timeout=30)

        if self.session.hungup.is_set():
            _LOGGER.warning("SIPSink: call already hung up")
            return

        call = self.session.call

        from speech_pipeline.RTPSession import RTPSession
        if isinstance(call, RTPSession):
            self._run_rtp_session(call)
        else:
            self._run_pyvoip(call)

    def _run_rtp_session(self, rtp_session) -> None:
        """RTPSession: downsample 48k→8k, encode via codec, send."""
        _LOGGER.info("SIPSink: streaming via RTPSession %s to %s:%d",
                     rtp_session.codec, rtp_session.remote_host,
                     rtp_session.remote_port)
        state = None
        try:
            for pcm48 in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                pcm8, state = audioop.ratecv(
                    pcm48, 2, 1, 48000, 8000, state)
                rtp_session.write_s16le(pcm8)
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")

    def _run_pyvoip(self, call) -> None:
        """pyVoIP: downsample 48k→8k, encode to codec, write via ring buffer."""
        if not call.RTPClients:
            _LOGGER.warning("SIPSink: no RTP clients")
            return

        rtp = call.RTPClients[0]

        try:
            import pyVoIP
            pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0
        except ImportError:
            pass

        # Detect codec
        try:
            from pyVoIP.RTP import PayloadType
            if rtp.preference == PayloadType.PCMA:
                encode = lambda pcm: audioop.lin2alaw(pcm, 2)
                silence = 0xd5
                _LOGGER.info("SIPSink: pyVoIP A-law to %s:%d", rtp.outIP, rtp.outPort)
            else:
                encode = lambda pcm: audioop.lin2ulaw(pcm, 2)
                silence = 0xff
                _LOGGER.info("SIPSink: pyVoIP µ-law to %s:%d", rtp.outIP, rtp.outPort)
        except Exception:
            encode = lambda pcm: audioop.lin2ulaw(pcm, 2)
            silence = 0xff

        # Ring buffer + encode pass-through
        from .SIPSink import _RingBuffer
        rb = _RingBuffer(silence_byte=silence)
        rtp.pmout = rb
        rtp.encode_packet = lambda data: data

        state = None
        try:
            for pcm48 in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                pcm8, state = audioop.ratecv(
                    pcm48, 2, 1, 48000, 8000, state)
                call.write_audio(encode(pcm8))
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")


class _RingBuffer:
    """Ring buffer for pyVoIP trans() thread."""

    def __init__(self, silence_byte: int = 0xff):
        cap = 4096
        self._buf = bytearray(bytes([silence_byte]) * cap)
        self._cap = cap
        self._silence = silence_byte
        self._write_pos = 0
        self._read_pos = 0
        import threading
        self._lock = threading.Lock()
        self.rebuilding = False

    def write(self, offset: int, data: bytes) -> None:
        with self._lock:
            n = len(data)
            start = self._write_pos % self._cap
            if start + n <= self._cap:
                self._buf[start:start + n] = data
            else:
                first = self._cap - start
                self._buf[start:] = data[:first]
                self._buf[:n - first] = data[first:]
            self._write_pos += n
            if self._write_pos - self._read_pos > self._cap:
                self._read_pos = self._write_pos - self._cap

    def read(self, length: int = 160) -> bytes:
        with self._lock:
            avail = self._write_pos - self._read_pos
            if avail < length:
                return bytes([self._silence]) * length
            consume = length
            if avail > self._cap * 3 // 4:
                consume = length + 2
            elif avail > self._cap // 2:
                consume = length + 1
            if consume > avail:
                consume = length
            start = self._read_pos % self._cap
            self._read_pos += consume
            if start + consume <= self._cap:
                return bytes(self._buf[start:start + length])
            else:
                part1 = self._buf[start:]
                part2 = self._buf[:consume - len(part1)]
                return bytes((part1 + part2)[:length])
