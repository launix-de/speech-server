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

    def __init__(self, session, leg=None) -> None:
        super().__init__()
        self.session = session
        self.leg = leg  # optional — enables BYE/CANCEL via leg.hangup()
        from speech_pipeline.RTPSession import RTPSession
        call = session.call if hasattr(session, "call") else None
        if isinstance(call, RTPSession):
            self.input_format = AudioFormat(call.codec.sample_rate, "s16le")
        else:
            self.input_format = AudioFormat(8000, "s16le")

    def _on_close(self) -> None:
        """Orderly teardown: send SIP BYE/CANCEL (idempotent)."""
        if self.session.hungup.is_set():
            return
        if self.leg is not None:
            try:
                self.leg.hangup()
                return
            except Exception as e:
                _LOGGER.debug("SIPSink: leg.hangup raised: %s", e)
        try:
            self.session.hangup()
        except Exception as e:
            _LOGGER.debug("SIPSink: session.hangup raised: %s", e)

    def _session_writer_lock(self):
        lock = getattr(self.session, "_sink_writer_lock", None)
        if lock is None:
            lock = threading.Lock()
            self.session._sink_writer_lock = lock
        return lock

    def _claim_session_writer(self) -> None:
        with self._session_writer_lock():
            current = getattr(self.session, "_active_sink_writer", None)
            if current and current is not self:
                current.cancel()
            self.session._active_sink_writer = self

    def _owns_session_writer(self) -> bool:
        return getattr(self.session, "_active_sink_writer", None) is self

    def _release_session_writer(self) -> None:
        with self._session_writer_lock():
            if getattr(self.session, "_active_sink_writer", None) is self:
                self.session._active_sink_writer = None

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
        self._claim_session_writer()
        natural_eof = False

        from speech_pipeline.RTPSession import RTPSession
        try:
            if isinstance(call, RTPSession):
                natural_eof = self._run_rtp_session(call)
            else:
                natural_eof = self._run_pyvoip(call)
        finally:
            self._release_session_writer()
            # Natural upstream EOF → orderly teardown via Stage.close().
            # External cancels and early returns skip this; the pipeline
            # owner is responsible for teardown in those cases.
            # The extra ``not self.cancelled`` guard covers the race where
            # cancel propagates via upstream's None-sentinel (the for-loop
            # exits "naturally" but cancel already flipped the flag).
            if natural_eof and not self.cancelled:
                self.close()

    def _run_rtp_session(self, rtp_session) -> bool:
        """Direct RTP: pass s16le to RTPSession. Returns True on natural EOF."""
        _LOGGER.info("SIPSink: streaming via RTPSession %s to %s:%d",
                     rtp_session.codec, rtp_session.remote_host,
                     rtp_session.remote_port)
        try:
            for pcm in self.upstream.stream_pcm24k():
                if (
                    self.cancelled
                    or self.session.hungup.is_set()
                    or not self._owns_session_writer()
                ):
                    return False
                rtp_session.write_s16le(pcm)
            return True
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
            return False
        finally:
            _LOGGER.info("SIPSink: stream ended")

    def _run_pyvoip(self, call) -> bool:
        """pyVoIP: encode + ring-buffer. Returns True on natural EOF."""
        if not call.RTPClients:
            _LOGGER.warning("SIPSink: no RTP clients")
            return False

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
                if (
                    self.cancelled
                    or self.session.hungup.is_set()
                    or not self._owns_session_writer()
                ):
                    return False
                call.write_audio(encode(pcm))
            return True
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
            return False
        finally:
            _LOGGER.info("SIPSink: stream ended")
