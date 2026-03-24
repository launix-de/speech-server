from __future__ import annotations

import audioop
import logging
import threading

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-sink")

_FRAME = 160  # 160 bytes = 20ms @ 8kHz


class _RingBuffer:
    """Ring buffer for pyVoIP trans() thread.
    Only used for pyVoIP legs — RTPSession has its own buffering.
    """

    def __init__(self, silence_byte: int = 0xff):
        cap = 4096
        self._buf = bytearray(bytes([silence_byte]) * cap)
        self._cap = cap
        self._silence = silence_byte
        self._write_pos = 0
        self._read_pos = 0
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

    def read(self, length: int = _FRAME) -> bytes:
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


class SIPSink(Stage):
    """Sink stage: writes audio into a SIP call.

    Works with both pyVoIP calls and RTPSession.
    Input: s16le mono @ 8000 Hz (pipe() auto-resamples from mixer rate).

    For RTPSession: delegates encoding to the RTPSession's codec.
    For pyVoIP: detects codec via rtp.preference, encodes to codec.

    Terminal sink — drives the pipeline.
    """

    def __init__(self, session) -> None:
        super().__init__()
        self.session = session
        self.input_format = AudioFormat(8000, "s16le")

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

        # Detect call type: RTPSession (has codec) or pyVoIP (has RTPClients)
        from speech_pipeline.RTPSession import RTPSession
        if isinstance(call, RTPSession):
            self._run_rtp_session(call)
        else:
            self._run_pyvoip(call)

    def _run_rtp_session(self, rtp_session) -> None:
        """Direct RTP: pass s16le to RTPSession, codec handles encoding."""
        _LOGGER.info("SIPSink: streaming via RTPSession %s to %s:%d",
                     rtp_session.codec, rtp_session.remote_host,
                     rtp_session.remote_port)
        try:
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                rtp_session.write_s16le(pcm)
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")

    def _run_pyvoip(self, call) -> None:
        """pyVoIP: encode to codec, write to ring buffer, trans() sends."""
        if not call.RTPClients:
            _LOGGER.warning("SIPSink: no RTP clients")
            return

        rtp = call.RTPClients[0]

        # Fix pyVoIP trans() timing
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
        rb = _RingBuffer(silence_byte=silence)
        rtp.pmout = rb
        rtp.encode_packet = lambda data: data

        try:
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                call.write_audio(encode(pcm))
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")
