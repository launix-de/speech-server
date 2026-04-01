"""Minimal SIP stack — replaces pyVoIP for both inbound and outbound calls.

Handles:
- SIP Client/Trunk: REGISTER with external PBX, INVITE outbound, Digest Auth
- SIP Registrar: accept REGISTER from client devices, authenticate via CRM
- Call signaling: INVITE, ACK, BYE, 100/180/200/401/407 responses

RTP is NOT handled here — only SIP signaling.  RTP ports/addresses are
communicated via SDP; actual audio goes through ConferenceMixer/SIPSource/SIPSink.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import socket
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests

from . import subscriber as sub_mod

_LOGGER = logging.getLogger("telephony.sip-stack")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_sock: Optional[socket.socket] = None
_local_ip: str = "0.0.0.0"
_public_ip: str = ""  # learned from REGISTER response (received= param)
_sip_port: int = 5060
_recv_thread: Optional[threading.Thread] = None
_running = False

# Trunk registrations: pbx_id -> _Trunk
_trunks: Dict[str, "_Trunk"] = {}
_trunks_lock = threading.Lock()

# Active calls: call_id -> SIPCall
_calls: Dict[str, "SIPCall"] = {}
_calls_lock = threading.Lock()

# Client device registrations (registrar): sip_user -> registration_key -> _Registration
_registrations: Dict[str, Dict[str, "_Registration"]] = {}
_registrations_lock = threading.Lock()

# Pending transactions: (branch or call_id, method) -> callback
_transactions: Dict[str, Callable] = {}
_transactions_lock = threading.Lock()

# Nonce -> True for registrar challenges we issued
_nonces: Dict[str, float] = {}

# ---------------------------------------------------------------------------
# SIPCall handle
# ---------------------------------------------------------------------------


@dataclass
class SIPCall:
    """Handle for an active SIP call."""
    call_id: str
    state: str = "dialing"          # dialing, ringing, answered, ended
    state_event: threading.Event = field(default_factory=threading.Event)
    remote_rtp_host: str = ""
    remote_rtp_port: int = 0
    local_rtp_port: int = 0
    negotiated_pt: int = 0          # payload type from SDP answer (0=PCMU)
    # Internal bookkeeping
    _from_header: str = ""
    _to_header: str = ""
    _to_tag: str = ""
    _via_branch: str = ""
    _cseq: int = 1
    _remote_addr: Tuple[str, int] = ("127.0.0.1", 5060)
    _local_tag: str = ""
    _contact_uri: str = ""
    _route: str = ""
    _caller_id: str = ""
    _cancel_requested: bool = False

    def _set_state(self, new_state: str) -> None:
        self.state = new_state
        self.state_event.set()
        self.state_event = threading.Event()


def _finalize_call(call_obj: SIPCall) -> None:
    call_obj._set_state("ended")
    with _calls_lock:
        _calls.pop(call_obj.call_id, None)


# ---------------------------------------------------------------------------
# Registration record (registrar side)
# ---------------------------------------------------------------------------


@dataclass
class _Registration:
    sip_user: str           # full SIP username (user@baseurl-without-https)
    contact_uri: str        # where to reach the device
    expires: float          # absolute time when registration expires
    user_id: int = 0        # CRM user id
    subscriber_id: str = "" # CRM subscriber this device belongs to
    base_url: str = ""      # CRM base URL for callbacks/device dial
    source_addr: Tuple[str, int] = ("", 0)  # actual REGISTER source tuple


def _registration_key(contact_uri: str, addr: Tuple[str, int]) -> str:
    return f"{addr[0]}:{addr[1]}|{contact_uri}"


# ---------------------------------------------------------------------------
# Trunk record (client side — outbound to PBX)
# ---------------------------------------------------------------------------


@dataclass
class _Trunk:
    pbx_id: str
    server: str
    port: int
    username: str
    password: str
    registered: bool = False
    local_tag: str = ""
    call_id_reg: str = ""
    cseq: int = 0
    _refresh_timer: Optional[threading.Timer] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _gen_call_id() -> str:
    return f"{_rand_hex(16)}@{_local_ip}"


def _gen_branch() -> str:
    return f"z9hG4bK-{_rand_hex(12)}"


def _gen_tag() -> str:
    return _rand_hex(8)


def _rand_hex(n: int) -> str:
    return "".join(random.choices(string.hexdigits[:16], k=n))


def _get_local_ip(remote_host: str, remote_port: int) -> str:
    """Determine local IP by connecting a UDP socket to a remote host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((remote_host, remote_port))
        return s.getsockname()[0]
    finally:
        s.close()


def _find_free_rtp_port(start: int = 10000, end: int = 20000) -> int:
    """Find a free even UDP port for RTP."""
    for _ in range(500):
        port = random.randrange(start, end, 2)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("0.0.0.0", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free RTP port in {start}-{end}")


# ---------------------------------------------------------------------------
# SDP helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find a free UDP port for RTP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sdp_ip_for_remote_sdp(remote_sdp_ip: str) -> str:
    """Choose our SDP IP to match the remote's network scope.

    If the remote advertises a private IP, reply with our private IP
    (direct LAN path).  If public, reply with our public IP.
    """
    if _is_private_ip(remote_sdp_ip):
        return _local_ip
    return _public_ip or _local_ip


def _is_private_ip(ip: str) -> bool:
    import ipaddress
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def _build_sdp(ip: str, rtp_port: int, codec=None) -> str:
    from speech_pipeline.rtp_codec import CODEC_PREFERENCE

    session_id = str(random.randint(100000, 999999))
    codecs = [codec] if codec is not None else CODEC_PREFERENCE
    payloads = " ".join(str(c.payload_type) for c in codecs)
    rtpmap = "".join(c.sdp_rtpmap for c in codecs)
    return (
        "v=0\r\n"
        f"o=tts-piper {session_id} {session_id} IN IP4 {ip}\r\n"
        "s=tts-piper\r\n"
        f"c=IN IP4 {ip}\r\n"
        "t=0 0\r\n"
        f"m=audio {rtp_port} RTP/AVP {payloads} 101\r\n"
        f"{rtpmap}"
        "a=rtpmap:101 telephone-event/8000\r\n"
        "a=fmtp:101 0-16\r\n"
        "a=sendrecv\r\n"
    )


def _parse_sdp_offer(sdp: str) -> Tuple[str, int, list[int]]:
    """Extract remote RTP host, port and offered payload types from SDP."""
    host = "0.0.0.0"
    port = 0
    remote_pts = []
    for line in sdp.splitlines():
        line = line.strip()
        if line.startswith("c=IN IP4 "):
            host = line.split()[-1]
        m = re.match(r"m=audio\s+(\d+)\s+\S+\s+([\d\s]+)", line)
        if m:
            port = int(m.group(1))
            remote_pts = [int(pt_str) for pt_str in m.group(2).split()]
    return host, port, remote_pts


def _parse_sdp(sdp: str) -> Tuple[str, int, int]:
    """Extract remote RTP host, port, and preferred codec from SDP."""
    from speech_pipeline.rtp_codec import negotiate_payload_type

    host, port, remote_pts = _parse_sdp_offer(sdp)
    payload_type = negotiate_payload_type(remote_pts)
    return host, port, payload_type


# ---------------------------------------------------------------------------
# SIP message parsing
# ---------------------------------------------------------------------------


def _parse_sip(data: bytes) -> dict:
    """Parse a SIP message into a dict.

    Returns:
        {
            "type": "request" | "response",
            "method": str,          # for requests
            "uri": str,             # for requests
            "status": int,          # for responses
            "reason": str,          # for responses
            "headers": dict,        # header-name (lower) -> value (str)
            "body": str,
            "raw_first_line": str,
        }
    """
    text = data.decode("utf-8", errors="replace")
    parts = text.split("\r\n\r\n", 1)
    header_section = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    lines = header_section.split("\r\n")
    first = lines[0]

    msg: dict = {"headers": {}, "body": body, "raw_first_line": first,
                 "raw_headers": lines[1:]}

    if first.startswith("SIP/2.0"):
        # Response: SIP/2.0 200 OK
        msg["type"] = "response"
        tokens = first.split(None, 2)
        msg["status"] = int(tokens[1])
        msg["reason"] = tokens[2] if len(tokens) > 2 else ""
    else:
        # Request: INVITE sip:user@host SIP/2.0
        msg["type"] = "request"
        tokens = first.split(None, 2)
        if not tokens:
            return None
        msg["method"] = tokens[0]
        msg["uri"] = tokens[1] if len(tokens) > 1 else ""

    # Parse headers (case-insensitive keys)
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        name = name.strip().lower()
        value = value.strip()
        # Multiple Via, Record-Route etc. — keep first occurrence (topmost)
        if name not in msg["headers"]:
            msg["headers"][name] = value
        elif name in ("via", "record-route", "route"):
            msg["headers"][name] += ", " + value

    return msg


def _get_header(msg: dict, name: str) -> str:
    return msg["headers"].get(name.lower(), "")


def _extract_tag(header_val: str, which: str = "tag") -> str:
    """Extract ;tag=xxx from a From/To header."""
    m = re.search(r";tag=([^\s;,>]+)", header_val)
    return m.group(1) if m else ""


def _extract_uri(header_val: str) -> str:
    """Extract sip:user@host from <sip:user@host>;tag=xxx."""
    m = re.search(r"<([^>]+)>", header_val)
    return m.group(1) if m else header_val.split(";")[0].strip()


def _extract_user(uri: str) -> str:
    """Extract username from sip:user@host."""
    m = re.match(r"sip:([^@]+)@", uri)
    return m.group(1) if m else ""


def _extract_host(uri: str) -> str:
    """Extract host from sip:user@host or sip:user@host:port."""
    m = re.match(r"sip:[^@]+@([^:;>]+)", uri)
    return m.group(1) if m else ""


def _parse_www_authenticate(header: str) -> dict:
    """Parse WWW-Authenticate or Proxy-Authenticate header."""
    result = {}
    # Remove "Digest " prefix
    header = re.sub(r"^Digest\s+", "", header, flags=re.IGNORECASE)
    for m in re.finditer(r'(\w+)="([^"]*)"', header):
        result[m.group(1)] = m.group(2)
    for m in re.finditer(r'(\w+)=([^",\s]+)', header):
        if m.group(1) not in result:
            result[m.group(1)] = m.group(2)
    return result


def _compute_digest_response(
    username: str, password: str, realm: str, nonce: str,
    method: str, uri: str, algorithm: str = "MD5",
) -> str:
    """Compute Digest Auth response value."""
    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()
    return _compute_digest_response_ha1(ha1, nonce, method, uri)


def _compute_digest_response_ha1(
    ha1: str, nonce: str, method: str, uri: str,
) -> str:
    """Compute Digest Auth response from pre-computed HA1."""
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    return hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()


def _build_authorization(
    auth_header_name: str, username: str, password: str,
    challenge: dict, method: str, uri: str,
) -> str:
    """Build Authorization or Proxy-Authorization header value."""
    realm = challenge.get("realm", "")
    nonce = challenge.get("nonce", "")
    response = _compute_digest_response(username, password, realm, nonce,
                                        method, uri)
    return (
        f'Digest username="{username}", realm="{realm}", nonce="{nonce}", '
        f'uri="{uri}", response="{response}", algorithm=MD5'
    )


# ---------------------------------------------------------------------------
# SIP message building
# ---------------------------------------------------------------------------


def _build_request(
    method: str, uri: str, *,
    call_id: str, from_header: str, to_header: str,
    cseq: int, via_branch: str,
    contact: str = "", body: str = "", extra_headers: str = "",
    remote_addr: Tuple[str, int] = ("127.0.0.1", 5060),
) -> str:
    """Build a SIP request message."""
    via = (f"SIP/2.0/UDP {_local_ip}:{_sip_port}"
           f";branch={via_branch};rport")
    if not contact:
        # Use public hostname if available (NAT traversal)
        contact_ip = _public_ip or _local_ip
        contact = f"<sip:{contact_ip}:{_sip_port}>"

    content_type = ""
    if body:
        content_type = "Content-Type: application/sdp\r\n"

    msg = (
        f"{method} {uri} SIP/2.0\r\n"
        f"Via: {via}\r\n"
        f"Max-Forwards: 70\r\n"
        f"From: {from_header}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq} {method}\r\n"
        f"Contact: {contact}\r\n"
        f"User-Agent: tts-piper SIP/1.0\r\n"
        f"{extra_headers}"
        f"{content_type}"
        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
        f"\r\n"
        f"{body}"
    )
    return msg


def _build_response(
    status: int, reason: str, msg: dict, *,
    body: str = "", extra_headers: str = "",
    to_tag: str = "",
) -> str:
    """Build a SIP response based on a received request."""
    via = _get_header(msg, "via")
    from_h = _get_header(msg, "from")
    to_h = _get_header(msg, "to")
    call_id = _get_header(msg, "call-id")
    cseq = _get_header(msg, "cseq")

    if to_tag and ";tag=" not in to_h:
        to_h += f";tag={to_tag}"

    contact = f"<sip:{_local_ip}:{_sip_port}>"

    content_type = ""
    if body:
        content_type = "Content-Type: application/sdp\r\n"

    # Reconstruct all Via headers from raw headers
    via_lines = ""
    for raw_line in msg.get("raw_headers", []):
        if raw_line.lower().startswith("via:") or raw_line.lower().startswith("v:"):
            via_lines += raw_line + "\r\n"

    if not via_lines:
        via_lines = f"Via: {via}\r\n"

    resp = (
        f"SIP/2.0 {status} {reason}\r\n"
        f"{via_lines}"
        f"From: {from_h}\r\n"
        f"To: {to_h}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq}\r\n"
        f"Contact: {contact}\r\n"
        f"User-Agent: tts-piper SIP/1.0\r\n"
        f"{extra_headers}"
        f"{content_type}"
        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
        f"\r\n"
        f"{body}"
    )
    return resp


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def _send(data: str, addr: Tuple[str, int]) -> None:
    """Send a SIP message via the shared socket."""
    if _sock is None:
        _LOGGER.error("SIP stack not initialized — cannot send")
        return
    first_line = data.split('\r\n', 1)[0]
    _LOGGER.info("SIP TX → %s: %s", addr, first_line)
    _LOGGER.debug("SIP TX raw to %s:\n%s", addr, data)
    try:
        _sock.sendto(data.encode("utf-8"), addr)
    except Exception as e:
        _LOGGER.error("Failed to send SIP to %s: %s", addr, e)


# ---------------------------------------------------------------------------
# Trunk registration (outbound to PBX)
# ---------------------------------------------------------------------------


def _send_register(trunk: _Trunk, auth_header: str = "") -> None:
    """Send REGISTER to the PBX."""
    trunk.cseq += 1
    trunk.call_id_reg = trunk.call_id_reg or _gen_call_id()
    trunk.local_tag = trunk.local_tag or _gen_tag()
    branch = _gen_branch()

    uri = f"sip:{trunk.server}"
    from_h = (f"<sip:{trunk.username}@{trunk.server}>"
              f";tag={trunk.local_tag}")
    to_h = f"<sip:{trunk.username}@{trunk.server}>"

    extra = ""
    if auth_header:
        extra = f"Authorization: {auth_header}\r\n"
    extra += "Expires: 300\r\n"

    msg = _build_request(
        "REGISTER", uri,
        call_id=trunk.call_id_reg, from_header=from_h, to_header=to_h,
        cseq=trunk.cseq, via_branch=branch,
        extra_headers=extra,
        remote_addr=(trunk.server, trunk.port),
    )
    _send(msg, (trunk.server, trunk.port))

    # Store transaction to handle response
    key = f"reg:{trunk.pbx_id}"
    with _transactions_lock:
        _transactions[key] = lambda resp: _handle_register_response(trunk, resp)


def _handle_register_response(trunk: _Trunk, msg: dict) -> None:
    status = msg["status"]

    if status == 200:
        trunk.registered = True
        # Learn public IP from Via received= parameter (NAT traversal)
        global _public_ip
        via = _get_header(msg, "via")
        _LOGGER.info("Trunk %s: REGISTER 200 Via: %s", trunk.pbx_id, via)
        m = re.search(r'received=([^;>\s]+)', via) if via else None
        if m and not _public_ip:
            _public_ip = m.group(1)
            _LOGGER.info("Learned public IP: %s (from Via received=)", _public_ip)
            # Re-register immediately with public IP in Contact
            threading.Thread(target=_send_register, args=(trunk,), daemon=True).start()
        _LOGGER.info("Trunk %s: registered with %s:%d as %s",
                     trunk.pbx_id, trunk.server, trunk.port, trunk.username)
        # Schedule re-registration
        expires = 300
        exp_header = _get_header(msg, "expires")
        if exp_header:
            try:
                expires = int(exp_header)
            except ValueError:
                pass
        trunk._refresh_timer = threading.Timer(
            max(expires - 30, 60), _send_register, args=(trunk,))
        trunk._refresh_timer.daemon = True
        trunk._refresh_timer.start()

    elif status in (401, 407):
        # Authentication required
        challenge_header = (_get_header(msg, "www-authenticate")
                            or _get_header(msg, "proxy-authenticate"))
        if not challenge_header:
            _LOGGER.error("Trunk %s: auth required but no challenge header",
                          trunk.pbx_id)
            return
        challenge = _parse_www_authenticate(challenge_header)
        uri = f"sip:{trunk.server}"
        auth_name = ("Authorization" if status == 401
                     else "Proxy-Authorization")
        auth_val = _build_authorization(
            auth_name, trunk.username, trunk.password,
            challenge, "REGISTER", uri)
        _send_register(trunk, auth_val)

    else:
        _LOGGER.warning("Trunk %s: REGISTER got %d %s",
                        trunk.pbx_id, status, msg.get("reason", ""))


# ---------------------------------------------------------------------------
# Outbound INVITE (via trunk)
# ---------------------------------------------------------------------------


def _send_invite(
    trunk: _Trunk, target: str, call_obj: SIPCall,
    auth_header: str = "", caller_id: str = "",
) -> None:
    """Send INVITE through a trunk to dial a target."""
    # Determine target URI
    if "@" in target:
        to_uri = f"sip:{target}"
    else:
        # Phone number — route via PBX
        to_uri = f"sip:{target}@{trunk.server}"

    call_obj._cseq += 1
    branch = _gen_branch()
    call_obj._via_branch = branch
    call_obj._remote_addr = (trunk.server, trunk.port)

    from_user = caller_id or trunk.username
    from_h = (f"<sip:{from_user}@{trunk.server}>"
              f";tag={call_obj._local_tag}")
    to_h = f"<{to_uri}>"
    call_obj._from_header = from_h
    call_obj._to_header = to_h

    sdp = _build_sdp(_local_ip, call_obj.local_rtp_port)

    extra = ""
    if auth_header:
        extra = f"Proxy-Authorization: {auth_header}\r\n"

    msg = _build_request(
        "INVITE", to_uri,
        call_id=call_obj.call_id, from_header=from_h, to_header=to_h,
        cseq=call_obj._cseq, via_branch=branch,
        body=sdp, extra_headers=extra,
        remote_addr=(trunk.server, trunk.port),
    )
    _send(msg, (trunk.server, trunk.port))


# ---------------------------------------------------------------------------
# Receiving — main dispatcher
# ---------------------------------------------------------------------------


def _recv_loop() -> None:
    """Main receive loop — reads SIP messages from the socket."""
    while _running:
        try:
            data, addr = _sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            if _running:
                _LOGGER.error("Socket error in recv loop", exc_info=True)
            break

        first_line = data.split(b'\r\n', 1)[0].decode('utf-8', errors='replace')
        _LOGGER.info("SIP RX ← %s: %s", addr, first_line)
        _LOGGER.debug("SIP RX raw ← %s:\n%s", addr,
                      data.decode('utf-8', errors='replace'))

        try:
            msg = _parse_sip(data)
        except Exception:
            _LOGGER.debug("Failed to parse SIP message from %s", addr,
                          exc_info=True)
            continue
        if msg is None:
            continue

        try:
            if msg["type"] == "response":
                _handle_response(msg, addr)
            else:
                _handle_request(msg, addr)
        except Exception:
            _LOGGER.error("Error handling SIP from %s", addr, exc_info=True)


def _handle_response(msg: dict, addr: Tuple[str, int]) -> None:
    """Handle an incoming SIP response."""
    status = msg["status"]
    call_id = _get_header(msg, "call-id")
    cseq_header = _get_header(msg, "cseq")  # "1 INVITE" etc.
    cseq_parts = cseq_header.split()
    cseq_method = cseq_parts[1] if len(cseq_parts) > 1 else ""

    # Check for trunk REGISTER responses
    if cseq_method == "REGISTER":
        # Find which trunk this belongs to
        with _trunks_lock:
            for trunk in _trunks.values():
                if trunk.call_id_reg == call_id:
                    _handle_register_response(trunk, msg)
                    return
        return

    # Look up active call
    with _calls_lock:
        call_obj = _calls.get(call_id)

    if not call_obj:
        _LOGGER.debug("Response for unknown call %s: %d", call_id, status)
        return

    if cseq_method == "INVITE":
        _handle_invite_response(call_obj, msg, addr)
    elif cseq_method == "BYE":
        if status == 200:
            _LOGGER.info("BYE acknowledged for call %s", call_id)


def _handle_invite_response(
    call_obj: SIPCall, msg: dict, addr: Tuple[str, int],
) -> None:
    """Handle response to our outbound INVITE."""
    status = msg["status"]
    call_id = call_obj.call_id

    if status == 100:
        _LOGGER.debug("Call %s: 100 Trying", call_id)

    elif status == 180 or status == 183:
        _LOGGER.info("Call %s: %d %s", call_id, status, msg.get("reason", ""))
        if call_obj.state == "dialing" and not call_obj._cancel_requested:
            call_obj._set_state("ringing")
        # Store To tag for later ACK/BYE
        to_tag = _extract_tag(_get_header(msg, "to"))
        if to_tag:
            call_obj._to_tag = to_tag

    elif status == 200:
        _LOGGER.info("Call %s: FULL 200 OK:\n%s", call_id,
                     "\n".join(f"  {k}: {v}" for k, v in msg.get("headers", {}).items()))
        to_tag = _extract_tag(_get_header(msg, "to"))
        if to_tag:
            call_obj._to_tag = to_tag

        # Extract RTP info + negotiated codec from SDP
        if msg["body"]:
            _LOGGER.info("Call %s: SDP body:\n%s", call_id, msg["body"][:300])
            host, port, pt = _parse_sdp(msg["body"])
            call_obj.remote_rtp_host = host
            call_obj.remote_rtp_port = port
            call_obj.negotiated_pt = pt

        # Store contact for future requests
        contact = _get_header(msg, "contact")
        _LOGGER.info("Call %s: 200 OK Contact header: %s", call_id, contact)
        if contact:
            call_obj._contact_uri = _extract_uri(contact)
            _LOGGER.info("Call %s: extracted contact_uri: %s", call_id, call_obj._contact_uri)

        # Store Record-Route for in-dialog routing
        rr = _get_header(msg, "record-route")
        if rr:
            call_obj._route = rr

        if call_obj._cancel_requested:
            _LOGGER.info("Call %s: 200 OK after CANCEL requested — sending ACK+BYE", call_id)
            _send_ack(call_obj, msg)
            _send_bye(call_obj)
            _finalize_call(call_obj)
            return

        _LOGGER.info("Call %s: answered (RTP %s:%d)",
                      call_id, call_obj.remote_rtp_host,
                      call_obj.remote_rtp_port)
        call_obj._set_state("answered")

        # Send ACK
        _send_ack(call_obj, msg)

    elif status in (401, 407):
        # Auth retry for INVITE
        challenge_header = (_get_header(msg, "www-authenticate")
                            if status == 401
                            else _get_header(msg, "proxy-authenticate"))
        if not challenge_header:
            _LOGGER.error("Call %s: auth required but no challenge", call_id)
            call_obj._set_state("ended")
            return

        # Find the trunk
        trunk = _find_trunk_for_call(call_obj)
        if not trunk:
            _LOGGER.error("Call %s: no trunk for auth retry", call_id)
            call_obj._set_state("ended")
            return

        # Send ACK for the challenge response (required by RFC 3261)
        _send_ack(call_obj, msg)

        challenge = _parse_www_authenticate(challenge_header)
        to_uri = _extract_uri(call_obj._to_header)
        auth_val = _build_authorization(
            "Proxy-Authorization" if status == 407 else "Authorization",
            trunk.username, trunk.password,
            challenge, "INVITE", to_uri)
        _send_invite(trunk, to_uri.replace("sip:", ""), call_obj, auth_val,
                     caller_id=call_obj._caller_id)

    elif status >= 400:
        _LOGGER.warning("Call %s: INVITE failed %d %s",
                        call_id, status, msg.get("reason", ""))
        # Send ACK for error responses
        _send_ack(call_obj, msg)
        _finalize_call(call_obj)


def _send_bye(call_obj: SIPCall) -> None:
    call_obj._cseq += 1
    branch = _gen_branch()

    to_h = call_obj._to_header
    if call_obj._to_tag and ";tag=" not in to_h:
        to_h += f";tag={call_obj._to_tag}"

    target_uri = call_obj._contact_uri or _extract_uri(to_h)
    if not target_uri.startswith("sip:"):
        target_uri = f"sip:{target_uri}"

    msg = _build_request(
        "BYE", target_uri,
        call_id=call_obj.call_id,
        from_header=call_obj._from_header,
        to_header=to_h,
        cseq=call_obj._cseq, via_branch=branch,
        remote_addr=call_obj._remote_addr,
    )
    _send(msg, call_obj._remote_addr)
    _LOGGER.info("Call %s: BYE sent", call_obj.call_id)


def _send_ack(call_obj: SIPCall, response_msg: dict) -> None:
    """Send ACK for a received response (200 OK or error)."""
    to_h = _get_header(response_msg, "to")
    to_uri = _extract_uri(to_h)
    if not to_uri.startswith("sip:"):
        to_uri = f"sip:{to_uri}"

    # ACK must use the same branch for non-2xx, new branch for 2xx
    status = response_msg.get("status", 0)
    branch = _gen_branch() if status == 200 else call_obj._via_branch

    cseq = call_obj._cseq

    # RFC 3261: ACK Request-URI = Contact URI from 200 OK (for 2xx)
    ack_uri = call_obj._contact_uri if call_obj._contact_uri and status == 200 else to_uri
    if not ack_uri.startswith("sip:"):
        ack_uri = f"sip:{ack_uri}"

    msg = _build_request(
        "ACK", ack_uri,
        call_id=call_obj.call_id,
        from_header=call_obj._from_header,
        to_header=to_h,
        cseq=cseq, via_branch=branch,
        remote_addr=call_obj._remote_addr,
    )
    _LOGGER.info("ACK for %s → %s: %s", call_obj.call_id, call_obj._remote_addr,
                 msg[:200].replace('\r\n', ' | '))
    _send(msg, call_obj._remote_addr)


def _find_trunk_for_call(call_obj: SIPCall) -> Optional[_Trunk]:
    """Find the trunk that owns a call based on remote address."""
    with _trunks_lock:
        for trunk in _trunks.values():
            if (trunk.server, trunk.port) == call_obj._remote_addr:
                return trunk
            # Also check by resolved IP
            try:
                resolved = socket.gethostbyname(trunk.server)
                if (resolved, trunk.port) == call_obj._remote_addr:
                    return trunk
            except socket.gaierror:
                pass
    return None


# ---------------------------------------------------------------------------
# Handling inbound requests
# ---------------------------------------------------------------------------


def _handle_request(msg: dict, addr: Tuple[str, int]) -> None:
    method = msg["method"]

    if method == "REGISTER":
        _handle_inbound_register(msg, addr)
    elif method == "INVITE":
        _handle_inbound_invite(msg, addr)
    elif method == "ACK":
        _LOGGER.debug("ACK received from %s", addr)
    elif method == "BYE":
        _handle_inbound_bye(msg, addr)
    elif method == "OPTIONS":
        # Reply 200 OK to keep-alive
        resp = _build_response(200, "OK", msg, to_tag=_gen_tag())
        _send(resp, addr)
    elif method == "CANCEL":
        _handle_inbound_cancel(msg, addr)
    else:
        _LOGGER.debug("Unhandled SIP method %s from %s", method, addr)


# ---------------------------------------------------------------------------
# Registrar — accept REGISTER from client devices
# ---------------------------------------------------------------------------


def _resolve_sip_identity(to_uri: str) -> Tuple[str, str, Optional[dict]]:
    """Resolve SIP identity from a To URI.

    Supports two formats:
      New: sip:milan.hradecky@crm.launix.de  (plain user, SIP domain)
      Old: sip:user%40launix.de/crm@srv...   (encoded CRM info in username)
      Old: sip:user:launix.de~crm@srv...     (: separator, ~ for /)

    Returns (username, realm, subscriber) or (username, realm, None) on failure.
    """
    from urllib.parse import unquote
    raw_user = _extract_user(to_uri)
    decoded_user = unquote(raw_user)

    # Old format: encoded CRM info in username (contains @ or : after decode)
    if "@" in decoded_user or ":" in decoded_user:
        crm_user, base_url = _split_sip_user(decoded_user)
        realm = _realm_from_sip_user(decoded_user)
        subscriber = _find_subscriber_by_base_url(base_url) if base_url else None
        return (crm_user, realm, subscriber)

    # New format: plain username, SIP domain identifies subscriber
    sip_domain = _extract_host(to_uri)
    subscriber = sub_mod.find_by_sip_domain(sip_domain)
    # Fallback: domain is raw base_url path (e.g. launix.de/crm)
    if not subscriber and "/" in sip_domain:
        subscriber = _find_subscriber_by_base_url(f"https://{sip_domain}")
    return (decoded_user, sip_domain, subscriber)


def _handle_inbound_register(msg: dict, addr: Tuple[str, int]) -> None:
    """Process REGISTER from a SIP client device."""
    to_h = _get_header(msg, "to")
    to_uri = _extract_uri(to_h)
    username, realm, _ = _resolve_sip_identity(to_uri)
    sip_domain = _extract_host(to_uri)
    # Canonical SIP AOR for registration storage — always normalize
    raw_aor = f"{username}@{sip_domain}" if sip_domain else username
    sip_user = _normalize_sip_user(raw_aor)
    contact = _get_header(msg, "contact")

    # Check Authorization header
    auth_h = _get_header(msg, "authorization")
    if not auth_h:
        # Challenge with 401
        nonce = _rand_hex(32)
        _nonces[nonce] = time.time()
        challenge = (
            f'Digest realm="{realm}", '
            f'nonce="{nonce}", algorithm=MD5'
        )
        extra = f"WWW-Authenticate: {challenge}\r\n"
        resp = _build_response(401, "Unauthorized", msg,
                               extra_headers=extra, to_tag=_gen_tag())
        _send(resp, addr)
        return

    # Verify credentials via CRM callback
    auth_params = _parse_www_authenticate(auth_h)
    nonce = auth_params.get("nonce", "")

    # Check nonce validity
    if nonce not in _nonces:
        _LOGGER.warning("REGISTER from %s: unknown nonce", addr)
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    # Resolve subscriber
    username, realm_resolved, subscriber = _resolve_sip_identity(to_uri)
    if not username:
        _LOGGER.warning("REGISTER: no username in %s", to_uri)
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    if not subscriber:
        _LOGGER.warning("REGISTER: no subscriber for %s", to_uri)
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    # Call CRM to get HA1
    raw_sip_user = auth_params.get("username", sip_user)
    auth_realm = auth_params.get("realm", realm_resolved)
    ha1_info = _crm_login(subscriber, username, auth_realm, raw_sip_user)
    if not ha1_info:
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    # Verify digest using HA1
    ha1 = ha1_info["ha1"]
    expected = _compute_digest_response_ha1(
        ha1, nonce, "REGISTER",
        auth_params.get("uri", f"sip:{_local_ip}:{_sip_port}"))
    actual = auth_params.get("response", "")

    if expected != actual:
        _LOGGER.warning("REGISTER: digest mismatch for %s", sip_user)
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    # Keep nonce valid until it expires (SIP clients reuse it on re-REGISTER)

    # Parse expiry
    exp_header = _get_header(msg, "expires")
    expires = int(exp_header or "3600")
    if expires < 30:
        expires = 30

    # Build contact URI from the contact header or from addr
    if contact:
        contact_uri = _extract_uri(contact)
    else:
        contact_uri = f"sip:{sip_user}@{addr[0]}:{addr[1]}"

    # Store/update registration
    reg_key = _registration_key(contact_uri, addr)
    with _registrations_lock:
        regs = _registrations.setdefault(sip_user, {})
        regs[reg_key] = _Registration(
            sip_user=sip_user,
            contact_uri=contact_uri,
            expires=time.time() + expires,
            user_id=ha1_info.get("user_id", 0),
            subscriber_id=subscriber.get("id", ""),
            base_url=subscriber.get("base_url", ""),
            source_addr=addr,
        )

    _LOGGER.info("REGISTER: %s registered (contact=%s, expires=%d)",
                 sip_user, contact_uri, expires)

    extra = f"Expires: {expires}\r\n"
    resp = _build_response(200, "OK", msg, extra_headers=extra,
                           to_tag=_gen_tag())
    _send(resp, addr)


def _split_sip_user_sep(sip_user: str) -> Tuple[str, str, str]:
    """Split SIP username at the CRM separator (@ or :).

    Supports two formats:
      user@example.com/app   (percent-encoded %40 is decoded first)
      user:example.com/app   (for SIP clients that reject @ in usernames)
      user:example.com~app   (~ as / alternative for clients that reject /)

    @ takes precedence so that %40-encoded users keep working.
    Returns (username, rest, separator) or ("", "", "") if no separator found.
    """
    from urllib.parse import unquote
    sip_user = unquote(sip_user)
    for sep in ("@", ":"):
        if sep in sip_user:
            username, rest = sip_user.split(sep, 1)
            rest = rest.replace("~", "/")
            return (username, rest, sep)
    return ("", "", "")


def _realm_from_sip_user(sip_user: str) -> str:
    """Extract realm from SIP username.

    SIP user format: username@baseurl  or  username:baseurl
    Realm = the baseurl part.
    """
    _, rest, _ = _split_sip_user_sep(sip_user)
    return rest if rest else sip_user


def _split_sip_user(sip_user: str) -> Tuple[str, str]:
    """Split SIP username into (crm_username, base_url).

    Input:  user@example.com/app  (or user%40example.com/app or user:example.com~app)
    Output: ("user", "https://example.com/app")
    """
    username, rest, _ = _split_sip_user_sep(sip_user)
    if not username or not rest:
        return ("", "")
    base_url = f"https://{rest}"
    return (username, base_url)


def _find_subscriber_by_base_url(base_url: str) -> Optional[dict]:
    """Find a subscriber whose base_url matches (ignoring trailing slash)."""
    base_url = base_url.rstrip("/")
    for sub in sub_mod.list_all():
        if sub.get("base_url", "").rstrip("/") == base_url:
            return sub
    return None


def _crm_login(
    subscriber: dict, username: str, realm: str, full_sip_user: str,
) -> Optional[dict]:
    """Call CRM login endpoint to get HA1 for digest verification.

    GET {base_url}/Telephone/SpeechServer/login?username={user}&realm={realm}&sip_user={full_sip_user}
    Authorization: Bearer {bearer_token}
    Response: {"ha1": "md5hash", "user_id": 123}
    """
    base_url = subscriber.get("base_url", "").rstrip("/")
    token = subscriber.get("bearer_token", "")
    if not base_url or not token:
        _LOGGER.warning("CRM login: subscriber missing base_url or token")
        return None

    url = f"{base_url}/Telephone/SpeechServer/login"
    params = {
        "username": username,
        "realm": realm,
        "sip_user": full_sip_user,
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "ha1" in data:
                return data
            _LOGGER.warning("CRM login: response missing ha1: %s", data)
        else:
            _LOGGER.warning("CRM login: HTTP %d from %s", resp.status_code, url)
    except Exception:
        _LOGGER.error("CRM login failed for %s", url, exc_info=True)

    return None


# ---------------------------------------------------------------------------
# Inbound INVITE — route to registered device
# ---------------------------------------------------------------------------


def _is_from_trunk(addr: Tuple[str, int]) -> Optional[str]:
    """Check if an INVITE comes from a registered trunk. Returns pbx_id or None."""
    with _trunks_lock:
        for pbx_id, trunk in _trunks.items():
            # Match by server IP (resolve hostname)
            try:
                trunk_ip = socket.gethostbyname(trunk.server)
            except Exception:
                trunk_ip = trunk.server
            if addr[0] == trunk_ip:
                return pbx_id
    return None


def _handle_inbound_invite(msg: dict, addr: Tuple[str, int]) -> None:
    """Handle inbound INVITE — either from trunk (incoming call) or to a registered device."""
    # Check if this INVITE comes from a trunk PBX
    pbx_id = _is_from_trunk(addr)
    if pbx_id:
        _handle_trunk_invite(msg, addr, pbx_id)
        return

    # Otherwise: either route to a registered device, or treat a registered
    # SIP client dialing an external number as a CRM-orchestrated outbound call.
    to_h = _get_header(msg, "to")
    to_uri = _extract_uri(to_h)
    target_user = _extract_user(to_uri)

    # Send 100 Trying immediately
    resp = _build_response(100, "Trying", msg)
    _send(resp, addr)

    reg = _get_registration(target_user)
    if reg:
        _LOGGER.info("Routing INVITE for %s to %s", target_user, reg.contact_uri)
        _proxy_invite(msg, addr, reg)
        return

    source_user, source_reg = _find_source_registration(msg, addr)
    if source_reg:
        _handle_registered_client_invite(msg, addr, source_user, source_reg, target_user)
        return

    _LOGGER.warning("INVITE for unregistered user %s", target_user)
    resp = _build_response(404, "Not Found", msg, to_tag=_gen_tag())
    _send(resp, addr)


def _handle_trunk_invite(msg: dict, addr: Tuple[str, int], pbx_id: str) -> None:
    """Handle incoming call from a trunk PBX with Early Media.

    Sends 183 Session Progress (not 200 OK) so RTP starts immediately
    for hold music.  The CRM decides when to answer via the API.
    """
    from_h = _get_header(msg, "from")
    to_h = _get_header(msg, "to")
    caller = _extract_user(_extract_uri(from_h))
    callee = _extract_user(_extract_uri(to_h))
    call_id = _get_header(msg, "call-id")

    _LOGGER.info("Trunk INVITE on %s: %s → %s", pbx_id, caller, callee)

    # SIP trunks retransmit the same INVITE until they get a final answer.
    # Re-send our existing provisional/final response, but do not spawn a
    # second inbound leg or a second CRM incoming webhook.
    existing = _trunk_dialogs.get(call_id)
    if existing:
        code = 200 if existing.get("answered") else 183
        phrase = "OK" if code == 200 else "Session Progress"
        resp = _build_response(
            code,
            phrase,
            msg,
            to_tag=existing["to_tag"],
            body=existing["sdp"],
        )
        _send(resp, addr)
        _LOGGER.info("Trunk INVITE %s: resent %d %s", call_id, code, phrase)
        return

    # Parse remote's codec preference from INVITE SDP
    _, _, remote_pt = _parse_sdp(msg.get("body", ""))
    from speech_pipeline.rtp_codec import codec_for_pt, PCMU
    negotiated_codec = codec_for_pt(remote_pt) or PCMU

    to_tag = _gen_tag()
    rtp_port = _find_free_port()
    sdp = _build_sdp(_local_ip, rtp_port, codec=negotiated_codec)

    # Send 183 Session Progress with SDP — starts Early Media RTP
    resp = _build_response(183, "Session Progress", msg, to_tag=to_tag, body=sdp)
    _send(resp, addr)
    _LOGGER.info(
        "Trunk INVITE: sent 183 Session Progress (Early Media RTP :%d, codec=%s)",
        rtp_port,
        negotiated_codec,
    )

    # Store the dialog info so we can send 200 OK later via answer_trunk_leg()
    _trunk_dialogs[call_id] = {
        "msg": msg,
        "addr": addr,
        "to_tag": to_tag,
        "rtp_port": rtp_port,
        "sdp": sdp,
        "pbx_id": pbx_id,
        "answered": False,
        "negotiated_pt": negotiated_codec.payload_type,
    }

    # Delegate to sip_listener
    from . import sip_listener
    threading.Thread(
        target=sip_listener._handle_trunk_call,
        args=(pbx_id, caller, callee, msg, addr, rtp_port),
        daemon=True,
    ).start()


# Trunk dialog storage for deferred 200 OK
_trunk_dialogs: Dict[str, dict] = {}


def answer_trunk_leg(sip_call_id: str) -> bool:
    """Send 200 OK for a trunk leg (deferred answer after Early Media).

    Called by the API when the CRM decides to answer the call.
    """
    dialog = _trunk_dialogs.get(sip_call_id)
    if not dialog or dialog["answered"]:
        return False

    resp = _build_response(200, "OK", dialog["msg"],
                            to_tag=dialog["to_tag"], body=dialog["sdp"])
    _send(resp, dialog["addr"])
    dialog["answered"] = True
    _LOGGER.info("Trunk leg %s: sent 200 OK (answered)", sip_call_id)
    return True


def hangup_trunk_leg(sip_call_id: str) -> bool:
    """End a deferred inbound dialog created via Early Media/183.

    Used for both real trunk inbound legs and registered-client device_dial
    legs, where the active SIP dialog is tracked in ``_trunk_dialogs`` rather
    than as a standalone ``SIPCall`` object.
    """
    dialog = _trunk_dialogs.get(sip_call_id)
    if not dialog:
        return False

    msg = dialog["msg"]
    addr = dialog["addr"]
    local_to_tag = dialog.get("to_tag", "")
    answered = bool(dialog.get("answered"))

    local_from = _get_header(msg, "to")
    if local_to_tag and ";tag=" not in local_from:
        local_from += f";tag={local_to_tag}"
    remote_to = _get_header(msg, "from")

    contact_h = _get_header(msg, "contact")
    target_uri = _extract_uri(contact_h) if contact_h else _extract_uri(remote_to)
    if not target_uri.startswith("sip:"):
        target_uri = f"sip:{target_uri}"

    cseq_h = _get_header(msg, "cseq")
    try:
        cseq_num = int((cseq_h.split()[0] if cseq_h else "1")) + 1
    except Exception:
        cseq_num = 2

    if answered:
        req = _build_request(
            "BYE",
            target_uri,
            call_id=sip_call_id,
            from_header=local_from,
            to_header=remote_to,
            cseq=cseq_num,
            via_branch=_gen_branch(),
            remote_addr=addr,
        )
        _send(req, addr)
        _LOGGER.info("Trunk leg %s: BYE sent", sip_call_id)
    else:
        req = _build_request(
            "CANCEL",
            _extract_uri(_get_header(msg, "to")),
            call_id=sip_call_id,
            from_header=_get_header(msg, "from"),
            to_header=_get_header(msg, "to"),
            cseq=1,
            via_branch=_extract_branch(_get_header(msg, "via")) or _gen_branch(),
            remote_addr=addr,
        )
        _send(req, addr)
        _LOGGER.info("Trunk leg %s: CANCEL sent", sip_call_id)

    session = dialog.get("session")
    if session is not None and hasattr(session, "hungup"):
        try:
            session.hungup.set()
        except Exception:
            pass
    rtp = dialog.get("rtp_session")
    if rtp is not None:
        try:
            rtp.stop()
        except Exception:
            pass
    _trunk_dialogs.pop(sip_call_id, None)
    return True


def _get_registrations(sip_user: str) -> List[_Registration]:
    with _registrations_lock:
        normalized = _normalize_sip_user(sip_user)
        regs = _registrations.get(sip_user, {}) or _registrations.get(normalized, {})
        if not regs:
            return []
        now = time.time()
        expired = [key for key, reg in regs.items() if reg.expires < now]
        for key in expired:
            regs.pop(key, None)
        if not regs:
            _registrations.pop(sip_user, None)
            return []
        return list(regs.values())


def _get_registration(sip_user: str) -> Optional[_Registration]:
    regs = _get_registrations(sip_user)
    return regs[0] if regs else None


def _find_source_registration(msg: dict, addr: Tuple[str, int]) -> Tuple[str, Optional[_Registration]]:
    """Resolve which registered client sent this INVITE."""
    from_h = _get_header(msg, "from")
    from_uri = _extract_uri(from_h)
    from_user = _extract_user(from_uri)
    regs = _get_registrations(from_user)
    if regs:
        for reg in regs:
            if reg.source_addr == addr:
                return from_user, reg
        return from_user, regs[0]

    now = time.time()
    stale: List[Tuple[str, str]] = []
    with _registrations_lock:
        for sip_user, reg_map in list(_registrations.items()):
            for reg_key, candidate in list(reg_map.items()):
                if candidate.expires < now:
                    stale.append((sip_user, reg_key))
                    continue
                if candidate.source_addr == addr:
                    for stale_user, stale_key in stale:
                        reg_bucket = _registrations.get(stale_user)
                        if reg_bucket:
                            reg_bucket.pop(stale_key, None)
                            if not reg_bucket:
                                _registrations.pop(stale_user, None)
                    return sip_user, candidate
                m = re.match(r"sip:([^@]+@)?([^:;>]+)(?::(\d+))?", candidate.contact_uri or "")
                if not m:
                    continue
                host = m.group(2)
                port = int(m.group(3)) if m.group(3) else 5060
                if host == addr[0] and port == addr[1]:
                    for stale_user, stale_key in stale:
                        reg_bucket = _registrations.get(stale_user)
                        if reg_bucket:
                            reg_bucket.pop(stale_key, None)
                            if not reg_bucket:
                                _registrations.pop(stale_user, None)
                    return sip_user, candidate
        for stale_user, stale_key in stale:
            reg_bucket = _registrations.get(stale_user)
            if reg_bucket:
                reg_bucket.pop(stale_key, None)
                if not reg_bucket:
                    _registrations.pop(stale_user, None)
    return "", None


def _handle_registered_client_invite(
    msg: dict,
    addr: Tuple[str, int],
    source_user: str,
    source_reg: _Registration,
    target_user: str,
) -> None:
    """Forward a registered SIP client's external dial attempt to the CRM."""
    from . import auth as auth_mod, dispatcher, leg as leg_mod, sip_listener
    from speech_pipeline.RTPSession import RTPSession, RTPCallSession
    from speech_pipeline.rtp_codec import PCMU, codec_for_pt, negotiate_payload_type

    call_id = _get_header(msg, "call-id")
    subscriber = None
    if source_reg.subscriber_id:
        subscriber = sub_mod.get(source_reg.subscriber_id)
    if not subscriber and source_reg.base_url:
        subscriber = _find_subscriber_by_base_url(source_reg.base_url)
    if not subscriber:
        _, base_url = _split_sip_user(source_user)
        if base_url:
            subscriber = _find_subscriber_by_base_url(base_url)
    if not subscriber and "@" in source_user:
        domain = source_user.split("@", 1)[1]
        subscriber = sub_mod.find_by_sip_domain(domain)
    if not subscriber:
        _LOGGER.warning("Registered INVITE from %s: no subscriber found", source_user)
        resp = _build_response(403, "Forbidden", msg, to_tag=_gen_tag())
        _send(resp, addr)
        return

    existing = _trunk_dialogs.get(call_id)
    if existing:
        code = 200 if existing.get("answered") else 183
        phrase = "OK" if code == 200 else "Session Progress"
        resp = _build_response(
            code,
            phrase,
            msg,
            to_tag=existing["to_tag"],
            body=existing["sdp"],
        )
        _send(resp, addr)
        _LOGGER.info("Registered INVITE %s: resent %d %s", call_id, code, phrase)
        return

    _LOGGER.info("Registered SIP device %s dials %s", source_user, target_user)

    remote_host, remote_port, remote_pts = _parse_sdp_offer(msg.get("body", ""))
    remote_pt = negotiate_payload_type(remote_pts)
    negotiated_codec = codec_for_pt(remote_pt) or PCMU
    to_tag = _gen_tag()
    rtp_port = _find_free_port()
    sdp = _build_sdp(_sdp_ip_for_remote_sdp(remote_host), rtp_port, codec=negotiated_codec)

    resp = _build_response(183, "Session Progress", msg, to_tag=to_tag, body=sdp)
    _send(resp, addr)

    _trunk_dialogs[call_id] = {
        "msg": msg,
        "addr": addr,
        "to_tag": to_tag,
        "rtp_port": rtp_port,
        "sdp": sdp,
        "pbx_id": "",
        "answered": False,
        "negotiated_pt": negotiated_codec.payload_type if negotiated_codec else 0,
        "source_user": source_user,
    }

    rtp = RTPSession(rtp_port, remote_host, remote_port, codec=negotiated_codec)
    rtp.start()
    session = RTPCallSession(rtp)

    acct = auth_mod.get_account(subscriber["account_id"]) if subscriber["account_id"] != "__admin__" else None
    pbx_id = acct.get("pbx", "") if acct else ""
    leg = leg_mod.create_leg(
        direction="inbound",
        number=source_user,
        pbx_id=pbx_id,
        subscriber_id=subscriber["id"],
        voip_call=rtp,
    )
    leg.status = "ringing"
    leg._sip_msg = msg
    leg._sip_addr = addr
    leg._rtp_port = rtp_port
    leg._rtp_session = rtp
    leg._sip_session = session
    leg._sip_call_id = call_id

    _trunk_dialogs[call_id]["session"] = session
    _trunk_dialogs[call_id]["rtp_session"] = rtp
    _trunk_dialogs[call_id]["leg_id"] = leg.leg_id

    dispatcher.fire_subscriber_event(subscriber, "device_dial", {
        "callId": leg.leg_id,
        "caller": source_user,
        "callee": target_user,
        "number": target_user,
        "leg_id": leg.leg_id,
        "sip_user": source_user,
        "user_id": source_reg.user_id,
    })

    threading.Thread(
        target=sip_listener._wait_for_bridge,
        args=(leg, None),
        daemon=True,
        name=f"dial-bridge-{leg.leg_id}",
    ).start()


def _proxy_invite(
    original_msg: dict, caller_addr: Tuple[str, int], reg: _Registration,
) -> None:
    """Forward an INVITE to a registered device's contact URI."""
    contact_uri = reg.contact_uri

    # Parse the contact URI to get host:port
    m = re.match(r"sip:([^@]+@)?([^:;>]+)(?::(\d+))?", contact_uri)
    if not m:
        _LOGGER.error("Cannot parse contact URI: %s", contact_uri)
        resp = _build_response(500, "Server Error", original_msg,
                               to_tag=_gen_tag())
        _send(resp, caller_addr)
        return

    device_host = m.group(2)
    device_port = int(m.group(3)) if m.group(3) else 5060
    device_addr = (device_host, device_port)

    # Build new INVITE to the device
    call_id = _get_header(original_msg, "call-id")
    from_h = _get_header(original_msg, "from")
    to_h = _get_header(original_msg, "to")
    cseq = _get_header(original_msg, "cseq")
    cseq_num = int(cseq.split()[0]) if cseq else 1
    branch = _gen_branch()

    body = original_msg.get("body", "")

    extra = f"Record-Route: <sip:{_local_ip}:{_sip_port};lr>\r\n"

    fwd = _build_request(
        "INVITE", contact_uri,
        call_id=call_id, from_header=from_h, to_header=to_h,
        cseq=cseq_num, via_branch=branch,
        body=body, extra_headers=extra,
        remote_addr=device_addr,
    )
    _send(fwd, device_addr)

    # Track this proxied call so we can relay responses back
    key = f"proxy:{call_id}"
    with _transactions_lock:
        _transactions[key] = lambda resp_msg, _ca=caller_addr, _om=original_msg: (
            _relay_proxy_response(resp_msg, _ca, _om)
        )


def _relay_proxy_response(
    resp_msg: dict, caller_addr: Tuple[str, int], original_msg: dict,
) -> None:
    """Relay a response from the callee device back to the original caller."""
    # Rebuild the response with the original Via headers
    status = resp_msg["status"]
    reason = resp_msg.get("reason", "")
    body = resp_msg.get("body", "")

    resp = _build_response(status, reason, original_msg, body=body,
                           to_tag=_extract_tag(_get_header(resp_msg, "to")))
    _send(resp, caller_addr)


# ---------------------------------------------------------------------------
# BYE handling
# ---------------------------------------------------------------------------


def _handle_inbound_bye(msg: dict, addr: Tuple[str, int]) -> None:
    """Handle BYE from remote party."""
    call_id = _get_header(msg, "call-id")

    trunk_dialog = _trunk_dialogs.pop(call_id, None)

    with _calls_lock:
        call_obj = _calls.get(call_id)

    if call_obj:
        _LOGGER.info("Call %s: received BYE", call_id)
        call_obj._set_state("ended")
        with _calls_lock:
            _calls.pop(call_id, None)

    if trunk_dialog:
        _LOGGER.info("Trunk dialog %s: received BYE", call_id)
        session = trunk_dialog.get("session")
        if session is not None and hasattr(session, "hungup"):
            try:
                session.hungup.set()
            except Exception:
                pass
        rtp = trunk_dialog.get("rtp_session")
        if rtp is not None:
            try:
                rtp.stop()
            except Exception:
                pass

    # Always reply 200 OK to BYE
    resp = _build_response(200, "OK", msg, to_tag=_gen_tag())
    _send(resp, addr)


def _handle_inbound_cancel(msg: dict, addr: Tuple[str, int]) -> None:
    """Handle CANCEL from remote party."""
    call_id = _get_header(msg, "call-id")

    trunk_dialog = _trunk_dialogs.pop(call_id, None)

    with _calls_lock:
        call_obj = _calls.get(call_id)

    if call_obj:
        _LOGGER.info("Call %s: received CANCEL", call_id)
        call_obj._set_state("ended")
        with _calls_lock:
            _calls.pop(call_id, None)

    if trunk_dialog:
        _LOGGER.info("Trunk dialog %s: received CANCEL", call_id)
        session = trunk_dialog.get("session")
        if session is not None and hasattr(session, "hungup"):
            try:
                session.hungup.set()
            except Exception:
                pass
        rtp = trunk_dialog.get("rtp_session")
        if rtp is not None:
            try:
                rtp.stop()
            except Exception:
                pass

    # Reply 200 OK to CANCEL
    resp = _build_response(200, "OK", msg, to_tag=_gen_tag())
    _send(resp, addr)


# ---------------------------------------------------------------------------
# Periodic cleanup
# ---------------------------------------------------------------------------


def _cleanup_loop() -> None:
    """Periodically clean up expired registrations and stale nonces."""
    while _running:
        time.sleep(60)
        now = time.time()

        # Expire registrations
        with _registrations_lock:
            for sip_user, reg_map in list(_registrations.items()):
                expired = [key for key, reg in reg_map.items() if reg.expires < now]
                for key in expired:
                    _LOGGER.info("Registration expired: %s (%s)", sip_user, key)
                    reg_map.pop(key, None)
                if not reg_map:
                    _registrations.pop(sip_user, None)

        # Expire nonces older than 5 minutes
        stale = [n for n, t in _nonces.items() if now - t > 300]
        for n in stale:
            del _nonces[n]


# ===========================================================================
# Public API
# ===========================================================================


def init(sip_port: int) -> None:
    """Start the SIP stack on the given UDP port."""
    global _sock, _local_ip, _sip_port, _recv_thread, _running

    if _running:
        _LOGGER.warning("SIP stack already running")
        return

    _sip_port = sip_port

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _sock.bind(("0.0.0.0", sip_port))
    _sock.settimeout(1.0)

    # Determine local IP (best guess via default route)
    try:
        _local_ip = _get_local_ip("8.8.8.8", 53)
    except Exception:
        _local_ip = "127.0.0.1"

    _running = True

    _recv_thread = threading.Thread(target=_recv_loop, daemon=True,
                                    name="sip-recv")
    _recv_thread.start()

    cleanup = threading.Thread(target=_cleanup_loop, daemon=True,
                               name="sip-cleanup")
    cleanup.start()

    _LOGGER.info("SIP stack started on :%d (local IP %s)", sip_port, _local_ip)

    import atexit
    atexit.register(shutdown)


def shutdown() -> None:
    """Graceful shutdown: notify all registered clients and trunks."""
    global _running
    if not _running:
        return
    _LOGGER.info("SIP stack shutting down...")

    # Send BYE to all active calls
    with _calls_lock:
        for call_obj in list(_calls.values()):
            try:
                hangup(call_obj)
            except Exception:
                pass

    # Unregister from all trunks
    with _trunks_lock:
        for pbx_id in list(_trunks.keys()):
            try:
                unregister_trunk(pbx_id)
            except Exception:
                pass

    _running = False
    _LOGGER.info("SIP stack stopped")


def register_trunk(
    pbx_id: str, server: str, port: int, username: str, password: str,
) -> None:
    """Register as a SIP client on an external PBX (for outbound calls)."""
    _LOGGER.warning(
        "Trunk %s: provisioning register to %s:%d as %s (password_len=%d)",
        pbx_id,
        server,
        port,
        username,
        len(password or ""),
    )
    trunk = _Trunk(
        pbx_id=pbx_id,
        server=server,
        port=port,
        username=username,
        password=password,
    )
    with _trunks_lock:
        old = _trunks.get(pbx_id)
        if old and old._refresh_timer:
            old._refresh_timer.cancel()
        _trunks[pbx_id] = trunk

    _send_register(trunk)


def unregister_trunk(pbx_id: str) -> None:
    """Unregister from an external PBX."""
    with _trunks_lock:
        trunk = _trunks.pop(pbx_id, None)

    if not trunk:
        return

    if trunk._refresh_timer:
        trunk._refresh_timer.cancel()

    # Send REGISTER with Expires: 0
    trunk.cseq += 1
    branch = _gen_branch()
    uri = f"sip:{trunk.server}"
    from_h = (f"<sip:{trunk.username}@{trunk.server}>"
              f";tag={trunk.local_tag}")
    to_h = f"<sip:{trunk.username}@{trunk.server}>"

    msg = _build_request(
        "REGISTER", uri,
        call_id=trunk.call_id_reg, from_header=from_h, to_header=to_h,
        cseq=trunk.cseq, via_branch=branch,
        extra_headers="Expires: 0\r\n",
        remote_addr=(trunk.server, trunk.port),
    )
    _send(msg, (trunk.server, trunk.port))
    _LOGGER.info("Trunk %s: unregistered from %s", pbx_id, trunk.server)


def call(pbx_id: str, target: str, caller_id: str = "") -> SIPCall:
    """Originate an outbound call via a trunk.

    Args:
        pbx_id: Which PBX trunk to use.
        target: Phone number (+491747712705) or SIP URI (user@domain).
        caller_id: Optional caller ID (phone number) for From header.

    Returns:
        SIPCall handle.  Monitor ``state`` and ``state_event`` for progress.
    """
    with _trunks_lock:
        trunk = _trunks.get(pbx_id)
    if not trunk:
        raise ValueError(f"No trunk registered for PBX {pbx_id}")
    if not trunk.registered:
        raise RuntimeError(f"Trunk {pbx_id} not yet registered with PBX")

    rtp_port = _find_free_rtp_port()
    call_id = _gen_call_id()
    call_obj = SIPCall(
        call_id=call_id,
        local_rtp_port=rtp_port,
        _local_tag=_gen_tag(),
        _remote_addr=(trunk.server, trunk.port),
    )

    with _calls_lock:
        _calls[call_id] = call_obj

    call_obj._caller_id = caller_id
    _send_invite(trunk, target, call_obj, caller_id=caller_id)
    _LOGGER.info("Call %s: INVITE sent to %s via trunk %s (caller_id=%s)",
                 call_id, target, pbx_id, caller_id or trunk.username)
    return call_obj


def _call_device_to_registration(sip_user: str, reg: dict) -> SIPCall:
    """Originate a call directly to a single registered SIP contact."""
    contact_uri = reg["contact_uri"]
    # Parse contact URI: sip:user@host:port
    m = re.match(r"sip:([^@]+)@([^:;>]+)(?::(\d+))?", contact_uri)
    if not m:
        raise ValueError(f"Invalid contact URI: {contact_uri}")
    device_host = m.group(2)
    device_port = int(m.group(3)) if m.group(3) else 5060

    rtp_port = _find_free_rtp_port()
    call_id = _gen_call_id()
    call_obj = SIPCall(
        call_id=call_id,
        local_rtp_port=rtp_port,
        _local_tag=_gen_tag(),
        _remote_addr=(device_host, device_port),
    )

    with _calls_lock:
        _calls[call_id] = call_obj

    # Build and send INVITE directly to device
    sdp = _build_sdp(_sdp_ip_for_remote_sdp(device_host), rtp_port)
    branch = _gen_branch()
    call_obj._cseq = 1
    call_obj._via_branch = branch  # needed for CANCEL (must match INVITE branch)

    from_uri = f"sip:conference@{_local_ip}"
    to_uri = contact_uri
    call_obj._from_header = f"<{from_uri}>;tag={call_obj._local_tag}"
    call_obj._to_header = f"<{to_uri}>"

    invite = (
        f"INVITE {to_uri} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {_local_ip}:{_sip_port};branch={branch};rport\r\n"
        f"Max-Forwards: 70\r\n"
        f"From: <{from_uri}>;tag={call_obj._local_tag}\r\n"
        f"To: <{to_uri}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 INVITE\r\n"
        f"Contact: <sip:{_local_ip}:{_sip_port}>\r\n"
        f"Content-Type: application/sdp\r\n"
        f"User-Agent: tts-piper-sip\r\n"
        f"Content-Length: {len(sdp)}\r\n"
        f"\r\n{sdp}"
    )

    _send(invite, (device_host, device_port))
    _LOGGER.info("Call %s: INVITE sent directly to device %s (%s:%d)",
                 call_id, sip_user, device_host, device_port)
    return call_obj


def call_device(sip_user: str, reg: dict) -> SIPCall:
    """Backward-compatible single-contact device dial."""
    return _call_device_to_registration(sip_user, reg)


def call_registered_user(sip_user: str) -> SIPCall:
    """Originate to all active registrations of a local SIP identity.

    The first device that answers wins; all other ringing branches are
    cancelled immediately.
    """
    regs = get_registrations(sip_user)
    if not regs:
        raise RuntimeError(f"SIP device {sip_user} not registered")
    if len(regs) == 1:
        return _call_device_to_registration(sip_user, regs[0])

    master = SIPCall(
        call_id=_gen_call_id(),
        local_rtp_port=0,
        _local_tag=_gen_tag(),
    )
    master._fanout_children = []  # type: ignore[attr-defined]
    master._fanout_winner = None  # type: ignore[attr-defined]
    master._fanout_lock = threading.Lock()  # type: ignore[attr-defined]

    def _copy_winner_state(child: SIPCall) -> None:
        master.local_rtp_port = child.local_rtp_port
        master.remote_rtp_host = child.remote_rtp_host
        master.remote_rtp_port = child.remote_rtp_port
        master.negotiated_pt = child.negotiated_pt
        master._from_header = child._from_header
        master._to_header = child._to_header
        master._to_tag = child._to_tag
        master._via_branch = child._via_branch
        master._cseq = child._cseq
        master._remote_addr = child._remote_addr
        master._local_tag = child._local_tag
        master._contact_uri = child._contact_uri
        master._route = child._route
        master._caller_id = child._caller_id

    def _watch_child(child: SIPCall) -> None:
        while child.state not in ("answered", "ended"):
            child.state_event.wait(timeout=1.0)
            child.state_event.clear()
            if master.state == "ended":
                return
            if child.state == "ringing" and master.state == "dialing":
                master._set_state("ringing")

        if child.state == "answered":
            with master._fanout_lock:  # type: ignore[attr-defined]
                if master.state == "ended" or master._fanout_winner is not None:  # type: ignore[attr-defined]
                    winner = False
                else:
                    master._fanout_winner = child  # type: ignore[attr-defined]
                    _copy_winner_state(child)
                    master._set_state("answered")
                    winner = True
            if winner:
                for other in list(master._fanout_children):  # type: ignore[attr-defined]
                    if other is child or other.state == "ended":
                        continue
                    try:
                        hangup(other)
                    except Exception:
                        _LOGGER.debug("Failed to cancel losing branch for %s", sip_user, exc_info=True)
            return

        with master._fanout_lock:  # type: ignore[attr-defined]
            if master._fanout_winner is None and all(c.state == "ended" for c in master._fanout_children):  # type: ignore[attr-defined]
                master._set_state("ended")

    for reg in regs:
        child = _call_device_to_registration(sip_user, reg)
        master._fanout_children.append(child)  # type: ignore[attr-defined]
        threading.Thread(
            target=_watch_child,
            args=(child,),
            daemon=True,
            name=f"fork-{sip_user}-{child.call_id[:8]}",
        ).start()

    _LOGGER.info("Call %s: fanout INVITE sent to %d contacts for %s",
                 master.call_id, len(regs), sip_user)
    return master


def _normalize_sip_user(sip_user: str) -> str:
    """Normalize old-format SIP user to new user@sip_domain format.

    milan.hradecky@launix.de/crm -> milan.hradecky@crm.launix.de
    milan.hradecky@crm.launix.de -> milan.hradecky@crm.launix.de (unchanged)
    """
    from urllib.parse import unquote
    decoded = unquote(sip_user)
    username, base_url = _split_sip_user(decoded)
    if username and base_url:
        sip_domain = sub_mod.base_url_to_sip_domain(base_url)
        return f"{username}@{sip_domain}"
    return decoded


def get_registrations(sip_user: str) -> List[dict]:
    """Look up all active registrations for a SIP username."""
    with _registrations_lock:
        from urllib.parse import unquote
        needle = _normalize_sip_user(sip_user)
        result: List[dict] = []
        for key, reg_map in list(_registrations.items()):
            if _normalize_sip_user(key) != needle:
                continue
            expired = [reg_key for reg_key, reg in reg_map.items() if reg.expires < time.time()]
            for reg_key in expired:
                reg_map.pop(reg_key, None)
            if not reg_map:
                _registrations.pop(key, None)
                continue
            for reg in reg_map.values():
                result.append({
                    "sip_user": reg.sip_user,
                    "contact_uri": reg.contact_uri,
                    "expires": reg.expires,
                    "user_id": reg.user_id,
                    "subscriber_id": reg.subscriber_id,
                    "base_url": reg.base_url,
                    "source_addr": reg.source_addr,
                })
        return result


def get_registration(sip_user: str) -> Optional[dict]:
    """Backward-compatible lookup of the first active registration."""
    regs = get_registrations(sip_user)
    return regs[0] if regs else None


def is_local_sip_user(sip_user: str) -> bool:
    """Return True if the SIP identity belongs to a known CRM subscriber."""
    if _get_registrations(sip_user):
        return True
    # Old format: encoded base_url in username
    _, base_url = _split_sip_user(sip_user)
    if base_url and _find_subscriber_by_base_url(base_url):
        return True
    # New format: user@sip_domain
    if "@" in sip_user:
        domain = sip_user.split("@", 1)[1]
        if sub_mod.find_by_sip_domain(domain):
            return True
    return False


def hangup(call_obj: SIPCall) -> None:
    """Send BYE (established) or CANCEL (ringing) to end a call."""
    if call_obj.state == "ended":
        return

    children = getattr(call_obj, "_fanout_children", None)
    if children is not None:
        for child in list(children):
            if child.state == "ended":
                continue
            try:
                hangup(child)
            except Exception:
                _LOGGER.debug("Failed to hang up forked child call %s", child.call_id, exc_info=True)
        call_obj._set_state("ended")
        return

    if call_obj.state in ("dialing", "ringing"):
        # Not yet established — send CANCEL (RFC 3261 §9.1)
        if call_obj._cancel_requested:
            return
        branch = call_obj._via_branch
        msg = _build_request(
            "CANCEL", _extract_uri(call_obj._to_header),
            call_id=call_obj.call_id,
            from_header=call_obj._from_header,
            to_header=call_obj._to_header,
            cseq=1, via_branch=branch,
            remote_addr=call_obj._remote_addr,
        )
        _send(msg, call_obj._remote_addr)
        _LOGGER.info("Call %s: CANCEL sent", call_obj.call_id)
        call_obj._cancel_requested = True
        return
    else:
        # Established — send BYE
        _send_bye(call_obj)
        _finalize_call(call_obj)
