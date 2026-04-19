import logging

from speech_pipeline.telephony.logcontrol import TelephonyVerbosityFilter


def _record(name: str, level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_level_0_suppresses_telephony_but_keeps_other_logs():
    filt = TelephonyVerbosityFilter(0)
    assert not filt.filter(_record("telephony.sip-stack", logging.INFO, "REGISTER: foo registered"))
    assert filt.filter(_record("piper-multi-server", logging.INFO, "HTTP server started"))


def test_level_1_keeps_register_and_provision_logs_only():
    filt = TelephonyVerbosityFilter(1)
    assert filt.filter(_record("telephony.sip-stack", logging.INFO, "REGISTER: alice@crm.example.test registered"))
    assert filt.filter(_record("telephony.auth", logging.INFO, "Account registered: ExampleAccount"))
    assert not filt.filter(_record("telephony.leg", logging.INFO, "Leg created: foo"))
    assert not filt.filter(_record("telephony.shared", logging.INFO, "Webhook public → 200"))


def test_level_2_keeps_call_logs_but_not_webhook_logs():
    filt = TelephonyVerbosityFilter(2)
    assert filt.filter(_record("telephony.leg", logging.INFO, "Leg created: foo"))
    assert filt.filter(_record("telephony.sip-stack", logging.INFO, "Registered SIP device alice@crm.example.test dials 017..."))
    assert not filt.filter(_record("telephony.shared", logging.INFO, "Webhook public → 200"))
    assert not filt.filter(_record("webhook-sink", logging.INFO, "WebhookSink: streaming to https://example.com"))


def test_level_3_keeps_webhook_logs():
    filt = TelephonyVerbosityFilter(3)
    assert filt.filter(_record("telephony.shared", logging.INFO, "Webhook public → 200"))
    assert filt.filter(_record("webhook-sink", logging.INFO, "WebhookSink: streaming to https://example.com"))


def test_level_4_keeps_debug_logs():
    filt = TelephonyVerbosityFilter(4)
    assert filt.filter(_record("telephony.sip-stack", logging.DEBUG, "Source registration lookup keys: ['alice@crm.example.test']"))
