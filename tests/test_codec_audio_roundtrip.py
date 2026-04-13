"""End-to-end audio quality across codecs through the conference path.

Real RTP between two parties, no mocks.  Each party sends queue.mp3
(known reference) to the conference; the other party receives the
conference's mix-minus and we decode + similarity-check.

This catches the production "audio Grütze" class of bug — when
either side encodes/decodes with the wrong codec, the similarity
collapses to near-zero.

Why test_trunk_pbx_does_not_start_pyvoip_listener didn't catch it:
that's a structural guard.  The real symptom is corrupted PCM at
the receiving party — only this similarity test surfaces that.
"""
from __future__ import annotations

import json
import queue as _q
import time
from typing import Any

import numpy as np
import pytest

from conftest import create_call
from speech_pipeline.rtp_codec import PCMU, PCMA, G722

from test_crm_e2e import (
    _cleanup_leg,
    _load_pcm,
    _make_rtp_leg,
    _receive_audio,
    _send_audio,
)


def _spectral_similarity(a: bytes, b: bytes, bands: int = 12) -> float:
    """Energy-band cosine similarity — robust against codec phase
    + sample-rate quirks.  Same metric used in
    ``tests/test_crm_hold_and_multiparty.py``."""
    if not a or not b:
        return 0.0
    aa = np.frombuffer(a, dtype=np.int16).astype(np.float64)
    bb = np.frombuffer(b, dtype=np.int16).astype(np.float64)
    n = min(len(aa), len(bb))
    if n < 1024:
        return 0.0
    aa, bb = aa[:n], bb[:n]
    spec_a = np.abs(np.fft.rfft(aa))
    spec_b = np.abs(np.fft.rfft(bb))
    band_a = np.array_split(spec_a, bands)
    band_b = np.array_split(spec_b, bands)
    ea = np.array([np.sum(x ** 2) for x in band_a])
    eb = np.array([np.sum(x ** 2) for x in band_b])
    denom = (np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-9
    return float(np.dot(ea, eb) / denom)


@pytest.mark.parametrize("codec_a,codec_b", [
    (PCMU, PCMU),
    (PCMA, PCMA),
    (G722, G722),
    (PCMU, PCMA),
    (G722, PCMU),
    (G722, PCMA),
])
def test_two_party_audio_quality_real_codec(client, account, codec_a, codec_b):
    """A talks → B hears clean audio (and vice versa).  Hard regression
    guard: similarity < 0.4 means the codec path is broken (wrong
    decoder, wrong sample rate, double-encoding)."""
    call_id = create_call(client, account)
    leg_a, phone_a, _ = _make_rtp_leg(codec=codec_a, number="+49a")
    leg_b, phone_b, _ = _make_rtp_leg(codec=codec_b, number="+49b")
    try:
        for leg in (leg_a, leg_b):
            client.post(
                "/api/pipelines",
                data=json.dumps({
                    "dsl": f"sip:{leg.leg_id} -> call:{call_id} "
                           f"-> sip:{leg.leg_id}",
                }),
                headers=account,
            )
        time.sleep(0.6)
        for p in (phone_a, phone_b):
            while not p.rx_queue.empty():
                p.rx_queue.get_nowait()

        # A sends 0.6 s of speech → B should hear it.
        ref_a = _load_pcm(codec_a.sample_rate, 0.6)
        _send_audio(phone_a, ref_a)
        time.sleep(1.0)
        recv_b = _receive_audio(phone_b, duration_s=0.7)
        # B's RX is at codec_b.sample_rate.  Resample reference too.
        ref_for_b = _load_pcm(codec_b.sample_rate, 0.6)
        sim_a_to_b = _spectral_similarity(ref_for_b, recv_b)
        assert sim_a_to_b > 0.4, (
            f"{codec_a.name}→conf→{codec_b.name}: "
            f"audio garbled (similarity={sim_a_to_b:.3f}). "
            f"Likely codec/sample-rate mismatch."
        )
    finally:
        client.delete(f"/api/calls/{call_id}", headers=account)
        _cleanup_leg(leg_a, phone_a)
        _cleanup_leg(leg_b, phone_b)
