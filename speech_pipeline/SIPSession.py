from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

_LOGGER = logging.getLogger("sip-session")


def _find_free_udp_port(start: int = 5070, end: int = 5199) -> int:
    """Find a free UDP port in the given range."""
    for port in range(start, end):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("0.0.0.0", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free UDP port found in {start}-{end}")

try:
    from pyVoIP.VoIP.VoIP import VoIPPhone, CallState
    from pyVoIP.VoIP.status import PhoneStatus
except ImportError:
    VoIPPhone = None  # type: ignore
    PhoneStatus = None  # type: ignore
    CallState = None  # type: ignore


def _patch_voip_phone(phone: "VoIPPhone") -> None:
    """Fix pyVoIP race condition: ACK never sent for outbound INVITE.

    pyVoIP bug: SIP.invite() returns after 100 TRYING and releases
    recvLock.  The 200 OK is then picked up by recv_loop, which calls
    _callback_RESP_OK.  But VoIPPhone.call() hasn't stored the call in
    self.calls yet, so the lookup fails and ACK is never sent.
    Asterisk retransmits 200 OK for ~32s, then drops the call.

    Fix: replace _callback_RESP_OK with a version that waits briefly
    for the call_id to appear in self.calls.

    Also fixes gen_ack: pyVoIP generates a new To tag instead of using
    the tag from the 200 OK response.
    """
    import types

    def _patched_callback_RESP_OK(self, request):
        call_id = request.headers["Call-ID"]
        cseq_method = request.headers.get("CSeq", {}).get("method", "")

        # Only wait for INVITE 200 OK (not REGISTER, OPTIONS, etc.)
        if cseq_method != "INVITE":
            return

        # Wait up to 3s for call to be registered (race condition fix)
        for _ in range(60):
            if call_id in self.calls:
                break
            time.sleep(0.05)

        if call_id not in self.calls:
            _LOGGER.warning("200 OK for unregistered call %s — ACK not sent", call_id)
            return

        self.calls[call_id].answered(request)
        _LOGGER.debug("Call %s answered, sending ACK", call_id)

        # Send ACK with corrected To tag
        ack = _build_ack(self.sip, request)
        self.sip.out.sendto(ack.encode("utf8"), (self.server, self.port))

    def _build_ack(sip, request):
        """Build ACK using To tag from the 200 OK (not a new random tag)."""
        import pyVoIP
        tag = sip.tagLibrary[request.headers["Call-ID"]]
        to_raw = request.headers["To"]["raw"]

        # Use the To tag from the 200 OK (pyVoIP's gen_ack wrongly generates a new tag)
        to_tag = request.headers["To"].get("tag", "")
        if to_tag:
            to_header = f"{to_raw};tag={to_tag}"
        else:
            to_header = to_raw

        via = sip._gen_response_via_header(request)

        ack = f"ACK {to_raw.strip('<').strip('>')} SIP/2.0\r\n"
        ack += via
        ack += "Max-Forwards: 70\r\n"
        ack += f"To: {to_header}\r\n"
        ack += f"From: {request.headers['From']['raw']};tag={tag}\r\n"
        ack += f"Call-ID: {request.headers['Call-ID']}\r\n"
        ack += f"CSeq: {request.headers['CSeq']['check']} ACK\r\n"
        ack += f"User-Agent: pyVoIP {pyVoIP.__version__}\r\n"
        ack += "Content-Length: 0\r\n\r\n"
        return ack

    phone._callback_RESP_OK = types.MethodType(_patched_callback_RESP_OK, phone)

    # Suppress pyVoIP's noisy "TODO: Add 500 Error on Receiving SIP Response"
    # for unhandled status codes (e.g. 183 Session Progress).
    import pyVoIP as _pyVoIP
    _orig_debug = _pyVoIP.debug if hasattr(_pyVoIP, "debug") else None

    def _quiet_debug(msg, *args, **kwargs):
        if "TODO: Add 500 Error" in str(msg):
            return
        if _orig_debug:
            _orig_debug(msg, *args, **kwargs)

    _pyVoIP.debug = _quiet_debug
    # Also patch in the SIP module which imports debug at module level
    try:
        from pyVoIP import SIP as _SIP
        _SIP.debug = _quiet_debug
    except Exception:
        pass


class SIPSession:
    """Manages a pyVoIP SIP call lifecycle.

    Registers as a SIP client, dials the target extension, and provides
    the active call object for SIPSource/SIPSink to read/write audio.

    Audio format: unsigned 8-bit PCM (u8), 8000 Hz, mono.
    This is pyVoIP's internal format after codec decoding.
    """

    def __init__(
        self,
        target: str,
        server: str = "127.0.0.1",
        port: int = 5060,
        username: str = "piper",
        password: str = "piper123",
        sample_rate: int = 8000,
        local_port: int = 0,
    ) -> None:
        if VoIPPhone is None:
            raise RuntimeError("pyVoIP is not installed — pip install pyVoIP")
        self.target = target
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.sample_rate = sample_rate
        self.local_port = local_port or _find_free_udp_port()

        self.connected = threading.Event()
        self.hungup = threading.Event()

        self._phone: Optional[VoIPPhone] = None
        self._call = None

    def start(self) -> None:
        """Register with the SIP server and dial the target."""
        def _incoming(call):
            call.deny()

        # Determine local IP by connecting a UDP socket to the SIP server
        import socket as _sock
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        try:
            _s.connect((self.server, self.port))
            _local_ip = _s.getsockname()[0]
        finally:
            _s.close()

        self._phone = VoIPPhone(
            server=self.server,
            port=self.port,
            username=self.username,
            password=self.password,
            callCallback=_incoming,
            sipPort=self.local_port,
            myIP=_local_ip,
        )

        # Fix pyVoIP ACK race condition before starting
        _patch_voip_phone(self._phone)

        self._phone.start()

        # Wait for registration
        deadline = time.time() + 15
        while self._phone.get_status() != PhoneStatus.REGISTERED:
            if time.time() > deadline:
                raise RuntimeError(
                    f"SIP registration timeout for {self.username}@{self.server}:{self.port}"
                )
            time.sleep(0.3)

        _LOGGER.info("SIP registered as %s@%s:%d", self.username, self.server, self.port)

        # Dial
        self._call = self._phone.call(self.target)
        _LOGGER.info("SIP dialing %s", self.target)

        # Wait for answer
        deadline = time.time() + 30
        while self._call.state != CallState.ANSWERED:
            if self._call.state == CallState.ENDED:
                self.hungup.set()
                raise RuntimeError(f"SIP call to {self.target} ended/rejected")
            if time.time() > deadline:
                try:
                    self._call.hangup()
                except Exception:
                    pass
                self.hungup.set()
                raise RuntimeError(f"SIP call to {self.target} answer timeout")
            time.sleep(0.1)

        self.connected.set()
        _LOGGER.info("SIP call answered — connected to %s", self.target)

        # Monitor call state in background
        t = threading.Thread(target=self._monitor, daemon=True, name="sip-monitor")
        t.start()

    def _monitor(self) -> None:
        """Watch for call hangup."""
        while not self.hungup.is_set():
            if self._call and self._call.state == CallState.ENDED:
                self.hungup.set()
                self.connected.set()  # unblock waiters
                _LOGGER.info("SIP call ended")
                return
            time.sleep(0.5)

    @property
    def call(self):
        return self._call

    def hangup(self) -> None:
        self.hungup.set()
        if self._call:
            try:
                self._call.hangup()
            except Exception as e:
                _LOGGER.debug("SIP hangup: %s", e)
            self._call = None
        if self._phone:
            try:
                self._phone.stop()
            except Exception as e:
                _LOGGER.debug("SIP phone stop: %s", e)
            self._phone = None
        # Give pyVoIP threads time to notice NSD=False and exit
        time.sleep(0.3)
