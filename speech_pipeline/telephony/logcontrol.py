from __future__ import annotations

import logging
from typing import Iterable


_FILTER_NAME = "_telephony_verbosity_filter"

_CALL_LOGGERS = (
    "telephony.call",
    "telephony.leg",
    "telephony.webclient",
    "telephony.pipe-executor",
    "telephony.sip-stack",
    "conference-mixer",
    "conference-leg",
    "sip-source",
    "sip-sink",
    "codec-source",
    "codec-sink",
    "rtp-session",
)

_WEBHOOK_LOGGERS = (
    "telephony.shared",
    "telephony.dispatcher",
    "webhook-sink",
)

_INIT_KEYWORDS = (
    "SIP stack started",
    "Built-in SIP stack enabled",
    "Pipeline control API enabled",
    "Telephony API enabled",
    "WebClient phone UI enabled",
    "Sending startup callback",
    "Startup callback response",
    "PBX registered",
    "trunk registered",
    "Account registered",
    "Subscriber ",
    "Pinged CRM heartbeat",
    "REGISTER:",
    "Trunk ",
)


def _matches_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


class TelephonyVerbosityFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = max(0, min(4, int(level)))

    def filter(self, record: logging.LogRecord) -> bool:
        if self.level >= 4:
            return True

        name = record.name
        msg = record.getMessage()

        telephony_related = (
            name.startswith("telephony.")
            or name in _CALL_LOGGERS
            or name in _WEBHOOK_LOGGERS
        )
        if not telephony_related:
            return True

        if self.level <= 0:
            return False

        if self.level == 1:
            if name == "piper-multi-server":
                return any(keyword in msg for keyword in _INIT_KEYWORDS)
            if name in ("telephony.auth", "telephony.pbx", "telephony.subscriber"):
                return True
            if name == "telephony.sip-stack":
                return any(keyword in msg for keyword in _INIT_KEYWORDS)
            return False

        if self.level == 2:
            if _matches_any(name, _CALL_LOGGERS):
                return record.levelno >= logging.INFO
            if name in ("telephony.auth", "telephony.pbx", "telephony.subscriber"):
                return record.levelno >= logging.INFO
            return False

        # level == 3
        if _matches_any(name, _CALL_LOGGERS) or _matches_any(name, _WEBHOOK_LOGGERS):
            return record.levelno >= logging.INFO
        if name in ("telephony.auth", "telephony.pbx", "telephony.subscriber"):
            return record.levelno >= logging.INFO
        return False


def configure(level: int, debug: bool = False) -> int:
    root = logging.getLogger()
    normalized = max(0, min(4, int(level)))

    if debug or normalized >= 4:
        root.setLevel(logging.DEBUG)

    for handler in root.handlers:
        filters = getattr(handler, "filters", [])
        for existing in list(filters):
            if getattr(existing, "_telephony_filter_marker", False):
                handler.removeFilter(existing)
        filt = TelephonyVerbosityFilter(normalized)
        filt._telephony_filter_marker = True  # type: ignore[attr-defined]
        handler.addFilter(filt)

    return normalized
