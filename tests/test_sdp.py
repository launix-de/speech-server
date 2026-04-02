"""SDP build, parse, and codec negotiation tests."""
import pytest


def test_build_sdp_single_codec():
    from speech_pipeline.telephony.sip_stack import _build_sdp
    from speech_pipeline.rtp_codec import PCMU
    sdp = _build_sdp("192.168.1.1", 10000, codec=PCMU, include_dtls=False)

    assert "c=IN IP4 192.168.1.1" in sdp
    assert "m=audio 10000" in sdp
    assert "PCMU/8000" in sdp
    assert "opus" not in sdp.lower()  # only PCMU


def test_build_sdp_all_codecs():
    from speech_pipeline.telephony.sip_stack import _build_sdp
    sdp = _build_sdp("10.0.0.1", 20000, include_dtls=False)

    assert "opus/48000" in sdp
    assert "G722/8000" in sdp
    assert "PCMU/8000" in sdp
    assert "PCMA/8000" in sdp
    assert "telephone-event" in sdp


def test_build_sdp_dtls_fingerprint():
    from speech_pipeline.telephony.sip_stack import _build_sdp
    sdp = _build_sdp("10.0.0.1", 20000, include_dtls=True)

    assert "a=fingerprint:sha-256" in sdp
    assert "a=setup:actpass" in sdp


def test_parse_sdp_offer():
    from speech_pipeline.telephony.sip_stack import _parse_sdp_offer

    sdp = (
        "v=0\r\n"
        "o=- 123 456 IN IP4 10.0.0.2\r\n"
        "c=IN IP4 10.0.0.2\r\n"
        "m=audio 30000 RTP/AVP 9 0 8 101\r\n"
        "a=rtpmap:9 G722/8000\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
    )
    host, port, pts = _parse_sdp_offer(sdp)
    assert host == "10.0.0.2"
    assert port == 30000
    assert 9 in pts   # G722
    assert 0 in pts   # PCMU
    assert 8 in pts   # PCMA
    assert 101 in pts  # telephone-event


def test_parse_sdp_negotiate():
    from speech_pipeline.telephony.sip_stack import _parse_sdp

    # Remote offers G722 + PCMU
    sdp = (
        "c=IN IP4 10.0.0.2\r\n"
        "m=audio 30000 RTP/AVP 9 0\r\n"
    )
    host, port, pt = _parse_sdp(sdp)
    assert host == "10.0.0.2"
    assert port == 30000
    assert pt == 9  # G722 preferred over PCMU


def test_parse_sdp_dtls():
    from speech_pipeline.telephony.sip_stack import _parse_sdp_dtls

    sdp = (
        "a=fingerprint:sha-256 AB:CD:EF:01:23\r\n"
        "a=setup:active\r\n"
    )
    fp, setup = _parse_sdp_dtls(sdp)
    assert fp == "AB:CD:EF:01:23"
    assert setup == "active"

    # No DTLS
    fp2, setup2 = _parse_sdp_dtls("c=IN IP4 10.0.0.1\r\n")
    assert fp2 == ""
    assert setup2 == ""


def test_parse_sdp_opus_negotiation():
    from speech_pipeline.telephony.sip_stack import _parse_sdp

    sdp = (
        "c=IN IP4 10.0.0.2\r\n"
        "m=audio 40000 RTP/AVP 111 9 0 8 101\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
    )
    host, port, pt = _parse_sdp(sdp)
    assert pt == 111  # Opus preferred


def test_subscriber_sip_domain():
    from speech_pipeline.telephony.subscriber import base_url_to_sip_domain

    assert base_url_to_sip_domain("https://launix.de/crm") == "crm.launix.de"
    assert base_url_to_sip_domain("https://launix.de/fop/crm-neu") == "crm-neu.fop.launix.de"
    assert base_url_to_sip_domain("https://crm-neu.launix.de") == "crm-neu.launix.de"
    assert base_url_to_sip_domain("https://example.com") == "example.com"
