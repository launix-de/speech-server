"""URL safety checks — prevents SSRF via webhook, play, and other URL-accepting elements."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def is_safe_url(url: str) -> bool:
    """Return True if the URL points to a public (non-internal) host.

    Blocks: private IPs, loopback, link-local, metadata endpoints.
    Allows: public IPs and hostnames that resolve to public IPs.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Resolve hostname to IP(s) and check each
    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
        # Block AWS/GCP/Azure metadata endpoints
        if ip_str.startswith("169.254."):
            return False

    return True


def require_safe_url(url: str) -> None:
    """Raise ValueError if the URL targets an internal/private network."""
    if not is_safe_url(url):
        raise ValueError(
            f"URL '{url}' targets a private or internal network — blocked for security"
        )
