from __future__ import annotations

import audioop
import logging
import threading

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("sip-sink")

_FRAME = 160  # 160 bytes µ-law = 20ms @ 8kHz


class _RingBuffer:
    """Ring buffer storing pre-encoded µ-law audio.

    Writer never blocks. Reader gets silence on underrun.
    Overrun: reader skips to latest data.
    """

    def __init__(self, capacity: int = 4096, silence_byte: int = 0xff):
        self._silence = silence_byte
        self._buf = bytearray(bytes([silence_byte]) * capacity)
        self._cap = capacity
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

            # Gentle catch-up when buffer > half full
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
                result = bytes(part1 + part2)
                return result[:length]


class SIPSink(Stage):
    """Sink stage: writes audio into a SIP call.

    Input: s16le mono @ 8000 Hz (pipe() auto-resamples from mixer rate).
    Encodes s16le → µ-law directly (full 16-bit precision) into a ring
    buffer. pyVoIP's trans() reads pre-encoded µ-law — encode_packet
    is overridden to pass-through.

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
        _LOGGER.info("SIPSink: codec=%s rate=%s",
                     rtp.preference, getattr(rtp.preference, 'rate', '?'))

        import pyVoIP
        pyVoIP.TRANSMIT_DELAY_REDUCTION = 1.0

        # Detect codec and set encoder
        from pyVoIP.RTP import PayloadType
        if rtp.preference == PayloadType.PCMA:
            self._encode = lambda pcm: audioop.lin2alaw(pcm, 2)
            silence = b"\xd5"  # alaw silence
            _LOGGER.info("SIPSink: encoding s16le → A-law")
        else:
            self._encode = lambda pcm: audioop.lin2ulaw(pcm, 2)
            silence = b"\xff"  # µ-law silence
            _LOGGER.info("SIPSink: encoding s16le → µ-law")

        # Ring buffer with pre-encoded audio
        rb = _RingBuffer(silence_byte=silence[0])
        rtp.pmout = rb

        # trans() reads pre-encoded data — skip pyVoIP's encode
        rtp.encode_packet = lambda data: data

        try:
            for pcm in self.upstream.stream_pcm24k():
                if self.cancelled or self.session.hungup.is_set():
                    break
                # s16le → codec directly (full 16-bit precision, no u8 bottleneck)
                call.write_audio(self._encode(pcm))
        except Exception as e:
            if not self.cancelled:
                _LOGGER.warning("SIPSink write error: %s", e)
        finally:
            _LOGGER.info("SIPSink: stream ended")
