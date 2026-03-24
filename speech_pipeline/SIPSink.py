from __future__ import annotations

import audioop
import logging
import threading

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-sink")

_FRAME = 160  # 160 samples u8 = 20ms @ 8kHz


class _RingBuffer:
    """Ring buffer for continuous audio streaming.

    Writer (SIPSink) appends u8 audio from the mixer.
    Reader (pyVoIP trans()) reads 160 bytes every ~20ms.
    """

    def __init__(self, capacity: int = 16000):
        self._buf = bytearray(b"\x80" * capacity)
        self._cap = capacity
        self._write_pos = 0
        self._read_pos = 0
        self._lock = threading.Lock()
        self.rebuilding = False

    def write(self, offset: int, data: bytes) -> None:
        """Append to ring buffer. Offset is ignored (FIFO)."""
        with self._lock:
            # Protect against writer lapping reader
            if self._write_pos + len(data) - self._read_pos > self._cap:
                self._read_pos = self._write_pos + len(data) - self._cap
            for b in data:
                self._buf[self._write_pos % self._cap] = b
                self._write_pos += 1

    def read(self, length: int = _FRAME) -> bytes:
        """Non-blocking read."""
        self._read_id = id(self)  # Tag for identification
        with self._lock:
            avail = self._write_pos - self._read_pos
            if avail < length:
                if self._write_pos >= length:
                    self._read_pos = self._write_pos - length
                else:
                    return b"\x80" * length

            start = self._read_pos % self._cap
            self._read_pos += length

            if start + length <= self._cap:
                return bytes(self._buf[start:start + length])
            else:
                part1 = bytes(self._buf[start:])
                part2 = bytes(self._buf[:length - len(part1)])
                return part1 + part2


class SIPSink(Stage):
    """Sink stage: writes audio into a SIP call.

    Input: s16le mono @ 8000 Hz (pipe() auto-resamples from mixer rate).
    Converts s16le → u8 and feeds into a ring buffer.
    pyVoIP's trans() reads at its own pace (fixed by TRANSMIT_DELAY_REDUCTION=1).

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
        if not call.RTPClients:
            _LOGGER.warning("SIPSink: no RTP clients")
            return

        rtp = call.RTPClients[0]
        _LOGGER.info("SIPSink: streaming to %s:%d", rtp.outIP, rtp.outPort)

        # Fix pyVoIP trans() timing bug
        import pyVoIP
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0

        _LOGGER.info("SIPSink: codec=%s rate=%s",
                     rtp.preference, getattr(rtp.preference, 'rate', '?'))

        # Replace pyVoIP's broken BytesIO buffer with ring buffer
        rtp.pmout = _RingBuffer()

        try:
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                # s16le → u8, OHNE Ring-Buffer (direkt an pyVoIP's original pmout)
                signed8 = audioop.lin2lin(pcm, 2, 1)
                call.write_audio(audioop.bias(signed8, 1, 128))
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")
