"""Shared pytest fixtures for speech-pipeline tests.

All tests use ephemeral ports (OS-assigned) to avoid conflicts
with the production server running on standard ports.
"""
import json
import socket
import struct
import pytest
from flask import Flask


# ---------------------------------------------------------------------------
# Audio helpers (used by codec / mixer / RTP tests)
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    """Get an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def generate_sine_pcm(freq_hz: float = 440.0, duration_s: float = 1.0,
                      sample_rate: int = 8000, amplitude: int = 8000) -> bytes:
    """Generate a sine wave as s16le PCM bytes."""
    import math
    samples = int(sample_rate * duration_s)
    data = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        data.append(struct.pack("<h", max(-32768, min(32767, value))))
    return b"".join(data)


def audio_similarity(reference: bytes, recorded: bytes) -> tuple:
    """Compare two s16le PCM buffers via normalized cross-correlation.

    Returns (similarity, delay_samples).
    similarity: 0.0-1.0 (1.0 = identical)
    delay_samples: positive = recorded is delayed
    """
    import numpy as np
    ref = np.frombuffer(reference, dtype=np.int16).astype(np.float64)
    rec = np.frombuffer(recorded, dtype=np.int16).astype(np.float64)

    if len(ref) == 0 or len(rec) == 0:
        return 0.0, 0

    correlation = np.correlate(rec, ref, mode="full")
    peak_idx = int(np.argmax(np.abs(correlation)))
    delay_samples = peak_idx - len(ref) + 1

    norm = float(np.sqrt(np.sum(ref ** 2) * np.sum(rec ** 2)))
    similarity = float(abs(correlation[peak_idx]) / norm) if norm > 0 else 0.0

    return similarity, delay_samples


# ---------------------------------------------------------------------------
# Telephony API test constants
# ---------------------------------------------------------------------------

ADMIN_TOKEN = "test-admin-token-12345"
ACCOUNT_TOKEN = "test-account-token-67890"
ACCOUNT_ID = "test-account"
SUBSCRIBER_ID = "test-subscriber"

ACCOUNT2_TOKEN = "test-account2-token-99999"
ACCOUNT2_ID = "test-account2"
SUBSCRIBER2_ID = "test-subscriber2"


# ---------------------------------------------------------------------------
# Telephony API fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a minimal Flask app with telephony API for testing."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    from speech_pipeline.telephony.auth import init as auth_init
    auth_init(ADMIN_TOKEN)

    from speech_pipeline.telephony.api import api as telephony_bp
    app.register_blueprint(telephony_bp)

    from speech_pipeline.pipeline_api import api as pipeline_bp
    app.register_blueprint(pipeline_bp)

    from speech_pipeline.metrics_api import api as metrics_bp
    app.register_blueprint(metrics_bp)

    import speech_pipeline.telephony._shared as _shared
    _shared.flask_app = app

    yield app

    # Cleanup
    from speech_pipeline.telephony import auth, subscriber, call_state, leg, pbx
    for cid in list(call_state._calls.keys()):
        try:
            call_state._calls[cid].mixer.cancel()
            call_state.delete_call(cid)
        except Exception:
            pass
    for lid in list(leg._legs.keys()):
        try:
            leg.delete_leg(lid)
        except Exception:
            pass
    auth._accounts.clear()
    auth._nonces.clear()
    subscriber._subscribers.clear()
    subscriber._did_map.clear()
    subscriber._sip_domain_map.clear()
    pbx._pbx_list.clear()
    from speech_pipeline import live_pipeline
    for pid in list(live_pipeline._pipelines.keys()):
        try:
            live_pipeline._pipelines[pid].cancel()
        except Exception:
            pass
    live_pipeline._pipelines.clear()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def admin(client):
    """Returns headers for admin auth."""
    return {"Authorization": f"Bearer {ADMIN_TOKEN}",
            "Content-Type": "application/json"}


@pytest.fixture
def account(client, admin):
    """Set up PBX + account + subscriber, return account headers."""
    headers = {"Authorization": f"Bearer {ACCOUNT_TOKEN}",
               "Content-Type": "application/json"}
    # PBX
    client.put("/api/pbx/TestPBX",
               data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
               headers=admin)
    # Account
    client.put(f"/api/accounts/{ACCOUNT_ID}",
               data=json.dumps({"token": ACCOUNT_TOKEN, "pbx": "TestPBX"}),
               headers=admin)
    # Subscriber
    client.put(f"/api/subscribe/{SUBSCRIBER_ID}",
               data=json.dumps({
                   "base_url": "https://example.com/crm",
                   "bearer_token": "sub-token-xyz",
               }),
               headers=headers)
    return headers


@pytest.fixture
def account2(client, admin):
    """Set up a SECOND account with its own PBX + subscriber."""
    headers = {"Authorization": f"Bearer {ACCOUNT2_TOKEN}",
               "Content-Type": "application/json"}
    client.put("/api/pbx/TestPBX2",
               data=json.dumps({"sip_proxy": "", "sip_user": "", "sip_password": ""}),
               headers=admin)
    client.put(f"/api/accounts/{ACCOUNT2_ID}",
               data=json.dumps({"token": ACCOUNT2_TOKEN, "pbx": "TestPBX2"}),
               headers=admin)
    client.put(f"/api/subscribe/{SUBSCRIBER2_ID}",
               data=json.dumps({
                   "base_url": "https://other.example.com/crm",
                   "bearer_token": "sub2-token",
               }),
               headers=headers)
    return headers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_call(client, headers, subscriber_id=SUBSCRIBER_ID):
    """Create a call via the API and return its call_id."""
    resp = client.post("/api/calls",
                       data=json.dumps({"subscriber_id": subscriber_id}),
                       headers=headers)
    assert resp.status_code == 201, f"create_call failed: {resp.data}"
    return resp.get_json()["call_id"]
