"""DTLS-SRTP support for RTP sessions.

Provides optional media encryption via DTLS key exchange + SRTP.
When the remote offers a DTLS fingerprint in SDP, a DTLS handshake
is performed on the RTP UDP socket and SRTP keys are derived.
If no fingerprint is offered, plain RTP is used (no encryption).

Adapted from aiortc's RTCDtlsTransport (MIT license).
"""
from __future__ import annotations

import binascii
import datetime
import logging
import os
import threading
from typing import Optional, Tuple

import pylibsrtp
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from OpenSSL import SSL

_LOGGER = logging.getLogger("dtls-srtp")

# SRTP protection profiles (ordered by preference)
_SRTP_PROFILES = []
for _profile_name, _libsrtp_id, _openssl_name, _key_len, _salt_len in [
    ("AEAD_AES_128_GCM", pylibsrtp.Policy.SRTP_PROFILE_AEAD_AES_128_GCM,
     b"SRTP_AEAD_AES_128_GCM", 16, 12),
    ("AES128_CM_SHA1_80", pylibsrtp.Policy.SRTP_PROFILE_AES128_CM_SHA1_80,
     b"SRTP_AES128_CM_SHA1_80", 16, 14),
]:
    try:
        pylibsrtp.Policy(srtp_profile=_libsrtp_id)
        _SRTP_PROFILES.append((_openssl_name, _libsrtp_id, _key_len, _salt_len))
    except pylibsrtp.Error:
        pass

# Module-level ephemeral certificate (generated once at import time)
_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
_now = datetime.datetime.now(tz=datetime.timezone.utc)
_name = x509.Name([x509.NameAttribute(
    x509.NameOID.COMMON_NAME,
    binascii.hexlify(os.urandom(16)).decode("ascii"),
)])
CERTIFICATE = (
    x509.CertificateBuilder()
    .subject_name(_name)
    .issuer_name(_name)
    .public_key(_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(_now - datetime.timedelta(days=1))
    .not_valid_after(_now + datetime.timedelta(days=365))
    .sign(_key, hashes.SHA256(), default_backend())
)
PRIVATE_KEY = _key

# Fingerprint for SDP
_hex = CERTIFICATE.fingerprint(hashes.SHA256()).hex().upper()
LOCAL_FINGERPRINT = ":".join(_hex[i:i+2] for i in range(0, len(_hex), 2))


def sdp_attributes() -> str:
    """Return SDP lines for DTLS-SRTP (fingerprint + setup)."""
    return (
        f"a=fingerprint:sha-256 {LOCAL_FINGERPRINT}\r\n"
        f"a=setup:actpass\r\n"
    )


def parse_sdp_fingerprint(sdp: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract fingerprint and setup role from SDP.

    Returns (fingerprint, setup) or (None, None) if not present.
    """
    fingerprint = None
    setup = None
    for line in sdp.splitlines():
        line = line.strip()
        if line.startswith("a=fingerprint:sha-256 "):
            fingerprint = line.split(" ", 2)[1] if " " in line[len("a=fingerprint:"):] else None
            # Actually parse more carefully
            parts = line.split(" ", 1)
            if len(parts) == 2:
                fingerprint = parts[1]  # "sha-256 AB:CD:..."
                # We only need the hex part
                fp_parts = line.split(" ", 2)
                if len(fp_parts) >= 3:
                    fingerprint = fp_parts[2]
        elif line.startswith("a=setup:"):
            setup = line[len("a=setup:"):]
    return fingerprint, setup


class DtlsSrtpSession:
    """Synchronous DTLS-SRTP session on an existing UDP socket.

    Usage::

        session = DtlsSrtpSession(udp_socket, role="server")
        session.do_handshake()  # blocks until complete
        # Now use protect/unprotect for SRTP
        srtp_packet = session.protect(rtp_packet)
        rtp_packet = session.unprotect(srtp_packet)
    """

    def __init__(self, sock, role: str = "server",
                 remote_addr: Tuple[str, int] = None) -> None:
        """
        :param sock: The UDP socket (already bound).
        :param role: "server" (we received INVITE) or "client" (we sent INVITE).
        :param remote_addr: Remote (host, port) for sendto.
        """
        self._sock = sock
        self._role = role
        self._remote_addr = remote_addr
        self._rx_srtp: Optional[pylibsrtp.Session] = None
        self._tx_srtp: Optional[pylibsrtp.Session] = None
        self.encrypted = False
        self._lock = threading.Lock()

        # Create SSL context
        openssl_profiles = b":".join(p[0] for p in _SRTP_PROFILES)
        ctx = SSL.Context(SSL.DTLS_METHOD)
        ctx.set_verify(
            SSL.VERIFY_PEER | SSL.VERIFY_FAIL_IF_NO_PEER_CERT,
            lambda *args: True
        )
        ctx.use_certificate(CERTIFICATE)
        ctx.use_privatekey(PRIVATE_KEY)
        ctx.set_tlsext_use_srtp(openssl_profiles)

        self._ssl = SSL.Connection(ctx)
        if role == "server":
            self._ssl.set_accept_state()
        else:
            self._ssl.set_connect_state()

        self._bio_in = SSL._lib.BIO_new(SSL._lib.BIO_s_mem())
        self._bio_out = SSL._lib.BIO_new(SSL._lib.BIO_s_mem())
        SSL._lib.SSL_set_bio(self._ssl._ssl, self._bio_in, self._bio_out)

    def do_handshake(self, timeout: float = 5.0) -> bool:
        """Perform DTLS handshake. Blocks until complete or timeout.

        Returns True if handshake succeeded, False otherwise.
        """
        import time
        deadline = time.monotonic() + timeout
        self._sock.settimeout(0.5)

        try:
            while not self.encrypted and time.monotonic() < deadline:
                try:
                    self._ssl.do_handshake()
                    self.encrypted = True
                except SSL.WantReadError:
                    # Send any pending outbound DTLS data
                    self._flush_bio_out()
                    # Read incoming DTLS data from socket
                    if not self._read_dtls_from_socket():
                        continue
                except SSL.Error as exc:
                    _LOGGER.warning("DTLS handshake failed: %s", exc)
                    return False
        except Exception as exc:
            _LOGGER.warning("DTLS handshake error: %s", exc)
            return False

        if not self.encrypted:
            _LOGGER.warning("DTLS handshake timed out")
            return False

        # Send final handshake messages
        self._flush_bio_out()

        # Extract SRTP keying material
        self._setup_srtp()
        _LOGGER.info("DTLS-SRTP handshake complete (role=%s)", self._role)
        return True

    def _flush_bio_out(self) -> None:
        """Send any pending DTLS data from the BIO to the UDP socket."""
        buf = bytearray(4096)
        while True:
            n = SSL._lib.BIO_read(self._bio_out, SSL._ffi.from_buffer(buf), len(buf))
            if n <= 0:
                break
            if self._remote_addr:
                try:
                    self._sock.sendto(bytes(buf[:n]), self._remote_addr)
                except Exception:
                    pass

    def _read_dtls_from_socket(self) -> bool:
        """Read a DTLS packet from the socket and feed to BIO."""
        try:
            data, addr = self._sock.recvfrom(4096)
        except (TimeoutError, OSError):
            return False
        if data and is_dtls(data):
            SSL._lib.BIO_write(self._bio_in, data, len(data))
            if self._remote_addr is None:
                self._remote_addr = addr
            return True
        return False

    def feed_dtls(self, data: bytes) -> None:
        """Feed a DTLS packet received by the RTP recv loop."""
        SSL._lib.BIO_write(self._bio_in, data, len(data))
        try:
            self._ssl.do_handshake()
            if not self.encrypted:
                self.encrypted = True
                self._flush_bio_out()
                self._setup_srtp()
                _LOGGER.info("DTLS-SRTP handshake complete (via feed)")
        except SSL.WantReadError:
            self._flush_bio_out()
        except SSL.Error:
            pass

    def _setup_srtp(self) -> None:
        """Extract SRTP keys from DTLS session and create SRTP sessions."""
        profile_name = self._ssl.get_selected_srtp_profile()
        for openssl_name, libsrtp_id, key_len, salt_len in _SRTP_PROFILES:
            if openssl_name == profile_name:
                break
        else:
            _LOGGER.warning("No matching SRTP profile for %s", profile_name)
            return

        material = self._ssl.export_keying_material(
            b"EXTRACTOR-dtls_srtp",
            2 * (key_len + salt_len),
        )

        # Key layout: client_key | server_key | client_salt | server_salt
        if self._role == "server":
            tx_key = (material[key_len:2*key_len]
                      + material[2*key_len+salt_len:2*key_len+2*salt_len])
            rx_key = (material[:key_len]
                      + material[2*key_len:2*key_len+salt_len])
        else:
            tx_key = (material[:key_len]
                      + material[2*key_len:2*key_len+salt_len])
            rx_key = (material[key_len:2*key_len]
                      + material[2*key_len+salt_len:2*key_len+2*salt_len])

        rx_policy = pylibsrtp.Policy(
            key=rx_key,
            ssrc_type=pylibsrtp.Policy.SSRC_ANY_INBOUND,
            srtp_profile=libsrtp_id,
        )
        rx_policy.allow_repeat_tx = True
        rx_policy.window_size = 1024
        self._rx_srtp = pylibsrtp.Session(rx_policy)

        tx_policy = pylibsrtp.Policy(
            key=tx_key,
            ssrc_type=pylibsrtp.Policy.SSRC_ANY_OUTBOUND,
            srtp_profile=libsrtp_id,
        )
        tx_policy.allow_repeat_tx = True
        tx_policy.window_size = 1024
        self._tx_srtp = pylibsrtp.Session(tx_policy)

    def protect(self, rtp_packet: bytes) -> bytes:
        """Encrypt an RTP packet → SRTP."""
        if self._tx_srtp is None:
            return rtp_packet  # no encryption
        return self._tx_srtp.protect(rtp_packet)

    def unprotect(self, srtp_packet: bytes) -> Optional[bytes]:
        """Decrypt an SRTP packet → RTP."""
        if self._rx_srtp is None:
            return srtp_packet  # no encryption
        try:
            return self._rx_srtp.unprotect(srtp_packet)
        except pylibsrtp.Error:
            return None  # decryption failed, drop packet


def is_dtls(data: bytes) -> bool:
    """Check if a UDP packet is a DTLS record (vs RTP/SRTP)."""
    if not data:
        return False
    # DTLS content types are 20-63, RTP/SRTP starts at 128+
    return 20 <= data[0] <= 63
