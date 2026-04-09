"""Tests for PyVoIPCallSession — the adapter wrapping pyVoIP calls
so that SIPSource/SIPSink can treat them like RTPCallSession."""
import queue
import threading
from unittest.mock import MagicMock

from speech_pipeline.telephony.leg import PyVoIPCallSession


def _make_session():
    mock_call = MagicMock()
    mock_call.read_audio.side_effect = Exception("stopped")
    return PyVoIPCallSession(mock_call), mock_call


class TestPyVoIPCallSessionInterface:
    """PyVoIPCallSession must expose connected, hungup, rx_queue, call —
    the same interface SIPSource/SIPSink rely on."""

    def test_connected_event_set_at_creation(self):
        session, _ = _make_session()
        assert isinstance(session.connected, threading.Event)
        assert session.connected.is_set()

    def test_hungup_event_unset_at_creation(self):
        session, _ = _make_session()
        assert isinstance(session.hungup, threading.Event)
        assert not session.hungup.is_set()

    def test_rx_queue_exists(self):
        session, _ = _make_session()
        assert isinstance(session.rx_queue, queue.Queue)

    def test_call_property_returns_underlying(self):
        session, mock_call = _make_session()
        assert session.call is mock_call


class TestPyVoIPCallSessionHangup:
    """hangup() must stop the rx pump, signal hungup, and forward to pyVoIP."""

    def test_hangup_sets_hungup_event(self):
        session, _ = _make_session()
        session.hangup()
        assert session.hungup.is_set()

    def test_hangup_stops_rx_pump(self):
        session, _ = _make_session()
        assert session._rx_pump_running is True
        session.hangup()
        assert session._rx_pump_running is False

    def test_hangup_calls_underlying_call(self):
        session, mock_call = _make_session()
        session.hangup()
        mock_call.hangup.assert_called_once()

    def test_hangup_tolerates_underlying_exception(self):
        session, mock_call = _make_session()
        mock_call.hangup.side_effect = RuntimeError("already dead")
        session.hangup()  # must not raise
        assert session.hungup.is_set()
