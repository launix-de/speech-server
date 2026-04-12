"""Sink stage that requests orderly teardown of its upstream chain.

DSL use::

    sip:LEG -> hangup           # reject ringing call (CANCEL)
    play:X -> sip:LEG -> hangup # play, then hang up
    sip:LEG -> record:Y -> hangup

On ``run()`` the sink calls ``self.upstream.close()`` — the regular
``Stage.close()`` machinery propagates the signal through the chain and
each stage's ``_on_close()`` hook performs its resource release
(SIPSource/SIPSink send SIP BYE/CANCEL, FileRecorder flushes, etc.).
"""
from __future__ import annotations

import logging

from .base import AudioFormat, Stage

_LOGGER = logging.getLogger("hangup-sink")


class HangupSink(Stage):
    def __init__(self) -> None:
        super().__init__()
        self.input_format = AudioFormat(8000, "s16le")

    def run(self) -> None:
        _LOGGER.info("HangupSink: requesting upstream close")
        if self.upstream is not None:
            self.upstream.close()
