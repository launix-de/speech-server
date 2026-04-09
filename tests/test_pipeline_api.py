"""REST API tests for the pipeline blueprint (/api/pipelines).

Covers: auth, CRUD, stage inspection, stage hot-update, stage delete,
stage replace, text input, and error cases.
"""
from __future__ import annotations

import json
import pytest
from flask import Flask

ADMIN_TOKEN = "test-pipeline-admin-token"


@pytest.fixture
def app():
    """Flask app with only the pipeline blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    from speech_pipeline.pipeline_api import api as pipeline_bp, init
    init(ADMIN_TOKEN)
    app.register_blueprint(pipeline_bp)

    yield app

    # Cleanup: remove all live pipelines
    from speech_pipeline import live_pipeline
    for pid in list(live_pipeline._pipelines.keys()):
        p = live_pipeline._pipelines.get(pid)
        if p:
            try:
                p.cancel()
            except Exception:
                pass
    live_pipeline._pipelines.clear()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def admin():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}",
            "Content-Type": "application/json"}


def _register_pipeline(stages=None, dsl="test"):
    """Register a LivePipeline directly in the registry (no DSL build)."""
    from speech_pipeline.live_pipeline import LivePipeline, register
    from speech_pipeline.base import AudioFormat, Stage

    p = LivePipeline(dsl=dsl)
    p.state = "running"

    if stages is None:
        # Default: one gain stage
        from speech_pipeline.GainStage import GainStage
        g = GainStage(48000, 1.0)
        p.add_stage(g, "gain", {"gain": 1.0})
    else:
        for stage, typ, config in stages:
            p.add_stage(stage, typ, config)

    register(p)
    return p


# ===========================================================================
# Auth
# ===========================================================================

class TestPipelineAuth:
    def test_no_auth_rejected(self, client):
        assert client.get("/api/pipelines").status_code == 401

    def test_wrong_token_rejected(self, client):
        h = {"Authorization": "Bearer wrong", "Content-Type": "application/json"}
        assert client.get("/api/pipelines", headers=h).status_code == 403

    def test_missing_bearer_prefix(self, client):
        h = {"Authorization": ADMIN_TOKEN, "Content-Type": "application/json"}
        assert client.get("/api/pipelines", headers=h).status_code == 401

    def test_admin_can_access(self, client, admin):
        assert client.get("/api/pipelines", headers=admin).status_code == 200


# ===========================================================================
# Pipeline CRUD
# ===========================================================================

class TestPipelineCRUD:
    def test_list_empty(self, client, admin):
        resp = client.get("/api/pipelines", headers=admin)
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_list_with_pipeline(self, client, admin):
        p = _register_pipeline()
        resp = client.get("/api/pipelines", headers=admin)
        assert resp.status_code == 200
        ids = [x["id"] for x in resp.get_json()]
        assert p.id in ids

    def test_get_pipeline(self, client, admin):
        p = _register_pipeline()
        resp = client.get(f"/api/pipelines/{p.id}", headers=admin)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] == p.id
        assert data["state"] == "running"
        assert data["dsl"] == "test"

    def test_get_pipeline_detail_has_stages(self, client, admin):
        p = _register_pipeline()
        resp = client.get(f"/api/pipelines/{p.id}", headers=admin)
        data = resp.get_json()
        assert isinstance(data["stages"], list)
        assert len(data["stages"]) == 1
        assert data["stages"][0]["type"] == "gain"

    def test_get_nonexistent_pipeline(self, client, admin):
        resp = client.get("/api/pipelines/nosuch", headers=admin)
        assert resp.status_code == 404

    def test_delete_pipeline(self, client, admin):
        p = _register_pipeline()
        resp = client.delete(f"/api/pipelines/{p.id}", headers=admin)
        assert resp.status_code == 204
        resp = client.get(f"/api/pipelines/{p.id}", headers=admin)
        assert resp.status_code == 404

    def test_delete_nonexistent_pipeline(self, client, admin):
        resp = client.delete("/api/pipelines/nosuch", headers=admin)
        assert resp.status_code == 404


class TestPipelineCreate:
    def test_create_missing_dsl(self, client, admin):
        resp = client.post("/api/pipelines",
                           data=json.dumps({}),
                           headers=admin)
        assert resp.status_code == 400

    def test_create_empty_dsl(self, client, admin):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": ""}),
                           headers=admin)
        assert resp.status_code == 400

    def test_create_invalid_dsl(self, client, admin):
        resp = client.post("/api/pipelines",
                           data=json.dumps({"dsl": "bogus_unknown_element"}),
                           headers=admin)
        assert resp.status_code == 400


# ===========================================================================
# Stage inspection
# ===========================================================================

class TestStageInspection:
    def test_list_stages(self, client, admin):
        p = _register_pipeline()
        resp = client.get(f"/api/pipelines/{p.id}/stages", headers=admin)
        assert resp.status_code == 200
        stages = resp.get_json()
        assert len(stages) == 1
        assert stages[0]["type"] == "gain"

    def test_list_stages_nonexistent_pipeline(self, client, admin):
        resp = client.get("/api/pipelines/nosuch/stages", headers=admin)
        assert resp.status_code == 404

    def test_get_stage(self, client, admin):
        from speech_pipeline.GainStage import GainStage
        g = GainStage(48000, 1.5)
        p = _register_pipeline(stages=[(g, "gain", {"gain": 1.5})])
        resp = client.get(f"/api/pipelines/{p.id}/stages/{g.id}",
                          headers=admin)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "gain"
        assert data["config"]["gain"] == 1.5

    def test_get_stage_includes_formats(self, client, admin):
        from speech_pipeline.GainStage import GainStage
        g = GainStage(48000, 1.0)
        p = _register_pipeline(stages=[(g, "gain", {})])
        resp = client.get(f"/api/pipelines/{p.id}/stages/{g.id}",
                          headers=admin)
        data = resp.get_json()
        # GainStage has input_format and output_format
        assert "input_format" in data or "output_format" in data

    def test_get_nonexistent_stage(self, client, admin):
        p = _register_pipeline()
        resp = client.get(f"/api/pipelines/{p.id}/stages/nosuch",
                          headers=admin)
        assert resp.status_code == 404


# ===========================================================================
# Stage hot-update (PATCH)
# ===========================================================================

class TestStageHotUpdate:
    def test_patch_gain(self, client, admin):
        from speech_pipeline.GainStage import GainStage
        g = GainStage(48000, 1.0)
        p = _register_pipeline(stages=[(g, "gain", {"gain": 1.0})])
        resp = client.patch(f"/api/pipelines/{p.id}/stages/{g.id}",
                            data=json.dumps({"gain": 2.5}),
                            headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["updated"]["gain"] == 2.5

    def test_patch_delay(self, client, admin):
        from speech_pipeline.DelayLine import DelayLine
        d = DelayLine(48000, 100.0)
        p = _register_pipeline(stages=[(d, "delay", {"delay_ms": 100.0})])
        resp = client.patch(f"/api/pipelines/{p.id}/stages/{d.id}",
                            data=json.dumps({"delay_ms": 50.0}),
                            headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["updated"]["delay_ms"] == 50.0

    def test_patch_unsupported_stage(self, client, admin):
        from speech_pipeline.SampleRateConverter import SampleRateConverter
        s = SampleRateConverter(8000, 48000)
        p = _register_pipeline(stages=[(s, "resample", {})])
        resp = client.patch(f"/api/pipelines/{p.id}/stages/{s.id}",
                            data=json.dumps({"gain": 2.0}),
                            headers=admin)
        assert resp.status_code == 422

    def test_patch_nonexistent_pipeline(self, client, admin):
        resp = client.patch("/api/pipelines/nosuch/stages/x",
                            data=json.dumps({"gain": 1.0}),
                            headers=admin)
        assert resp.status_code == 404

    def test_patch_nonexistent_stage(self, client, admin):
        p = _register_pipeline()
        resp = client.patch(f"/api/pipelines/{p.id}/stages/nosuch",
                            data=json.dumps({"gain": 1.0}),
                            headers=admin)
        assert resp.status_code == 404


# ===========================================================================
# Stage delete
# ===========================================================================

class TestStageDelete:
    def test_delete_processor_stage(self, client, admin):
        """Deleting a processor reconnects upstream and downstream."""
        from speech_pipeline.GainStage import GainStage
        from speech_pipeline.SampleRateConverter import SampleRateConverter
        from speech_pipeline.base import Stage

        src = Stage()
        gain = GainStage(48000, 1.0)
        sink = Stage()
        src.pipe(gain)
        gain.pipe(sink)

        p = _register_pipeline(stages=[
            (src, "source", {}),
            (gain, "gain", {}),
            (sink, "sink", {}),
        ])
        p.edges = [(src.id, gain.id), (gain.id, sink.id)]

        resp = client.delete(f"/api/pipelines/{p.id}/stages/{gain.id}",
                             headers=admin)
        assert resp.status_code == 204

        # gain should be removed, src→sink directly connected
        assert gain.id not in p.stages
        assert sink.upstream is src
        assert src.downstream is sink

    def test_delete_source_stage_rejected(self, client, admin):
        """Source stages (no upstream) cannot be deleted."""
        from speech_pipeline.base import Stage
        src = Stage()
        p = _register_pipeline(stages=[(src, "source", {})])
        resp = client.delete(f"/api/pipelines/{p.id}/stages/{src.id}",
                             headers=admin)
        assert resp.status_code == 422

    def test_delete_nonexistent_stage(self, client, admin):
        p = _register_pipeline()
        resp = client.delete(f"/api/pipelines/{p.id}/stages/nosuch",
                             headers=admin)
        assert resp.status_code == 404


# ===========================================================================
# Text input
# ===========================================================================

class TestPipelineInput:
    def test_input_nonexistent_pipeline(self, client, admin):
        resp = client.post("/api/pipelines/nosuch/input",
                           data=json.dumps({"text": "hello"}),
                           headers=admin)
        assert resp.status_code == 404

    def test_input_no_text_input_stage(self, client, admin):
        p = _register_pipeline()
        resp = client.post(f"/api/pipelines/{p.id}/input",
                           data=json.dumps({"text": "hello"}),
                           headers=admin)
        assert resp.status_code == 422

    def test_input_text(self, client, admin):
        import queue as q
        p = _register_pipeline()
        p._text_input_queue = q.Queue()
        resp = client.post(f"/api/pipelines/{p.id}/input",
                           data=json.dumps({"text": "Hallo Welt"}),
                           headers=admin)
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert p._text_input_queue.get_nowait() == "Hallo Welt"

    def test_input_eof(self, client, admin):
        import queue as q
        p = _register_pipeline()
        p._text_input_queue = q.Queue()
        resp = client.post(f"/api/pipelines/{p.id}/input",
                           data=json.dumps({"eof": True}),
                           headers=admin)
        assert resp.status_code == 200
        assert p._text_input_queue.get_nowait() is None

    def test_input_text_and_eof(self, client, admin):
        import queue as q
        p = _register_pipeline()
        p._text_input_queue = q.Queue()
        resp = client.post(f"/api/pipelines/{p.id}/input",
                           data=json.dumps({"text": "Bye", "eof": True}),
                           headers=admin)
        assert resp.status_code == 200
        assert p._text_input_queue.get_nowait() == "Bye"
        assert p._text_input_queue.get_nowait() is None


# ===========================================================================
# Pipeline metadata
# ===========================================================================

class TestPipelineMetadata:
    def test_pipeline_has_created_at(self, client, admin):
        p = _register_pipeline()
        resp = client.get(f"/api/pipelines/{p.id}", headers=admin)
        data = resp.get_json()
        assert "created_at" in data
        assert isinstance(data["created_at"], float)

    def test_pipeline_list_shows_stage_count(self, client, admin):
        from speech_pipeline.GainStage import GainStage
        from speech_pipeline.SampleRateConverter import SampleRateConverter
        g = GainStage(48000, 1.0)
        s = SampleRateConverter(8000, 48000)
        p = _register_pipeline(stages=[
            (g, "gain", {}),
            (s, "resample", {}),
        ])
        resp = client.get("/api/pipelines", headers=admin)
        for entry in resp.get_json():
            if entry["id"] == p.id:
                assert entry["stages"] == 2
                break

    def test_pipeline_detail_has_edges(self, client, admin):
        from speech_pipeline.GainStage import GainStage
        from speech_pipeline.base import Stage
        src = Stage()
        g = GainStage(48000, 1.0)
        p = _register_pipeline(stages=[
            (src, "source", {}),
            (g, "gain", {}),
        ])
        p.add_edge(src.id, g.id)
        resp = client.get(f"/api/pipelines/{p.id}", headers=admin)
        data = resp.get_json()
        assert len(data["edges"]) == 1
        assert data["edges"][0]["from"] == src.id
        assert data["edges"][0]["to"] == g.id


# ===========================================================================
# Multiple pipelines
# ===========================================================================

class TestMultiplePipelines:
    def test_list_multiple(self, client, admin):
        p1 = _register_pipeline(dsl="pipeline1")
        p2 = _register_pipeline(dsl="pipeline2")
        resp = client.get("/api/pipelines", headers=admin)
        ids = [x["id"] for x in resp.get_json()]
        assert p1.id in ids
        assert p2.id in ids

    def test_delete_one_keeps_other(self, client, admin):
        p1 = _register_pipeline(dsl="keep")
        p2 = _register_pipeline(dsl="delete")
        client.delete(f"/api/pipelines/{p2.id}", headers=admin)
        resp = client.get(f"/api/pipelines/{p1.id}", headers=admin)
        assert resp.status_code == 200
