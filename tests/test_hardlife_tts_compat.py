"""Caller-defined HTTP pipelines: authenticated preparation, secret playback."""
import threading

import pytest
from flask import Flask

from speech_pipeline.base import AudioFormat, Stage
from speech_pipeline import http_audio, live_pipeline
from speech_pipeline.pipeline_api import api, init

TOKEN = 'http-test-admin'


@pytest.fixture
def prepared(monkeypatch):
    sources = []

    class Source(Stage):
        output_format = AudioFormat(24000, 's16le')

        def estimate_frames_24k(self):
            return 240

        def stream_pcm24k(self):
            yield b'\x01\x00' * 240

    def create(executor, typ, elem_id, params):
        source = Source()
        sources.append((typ, params, source))
        return source

    monkeypatch.setattr('speech_pipeline.telephony.pipe_executor.CallPipeExecutor._create_stage', create)
    init(TOKEN)
    app = Flask(__name__)
    app.register_blueprint(api)
    app.config['TESTING'] = True
    yield app.test_client(), sources
    for pipeline in live_pipeline.list_all():
        pipeline.cancel()
        live_pipeline.unregister(pipeline.id)


def prepare(client, timeout=1, **extra):
    return client.post('/api/pipelines', headers={'Authorization': 'Bearer ' + TOKEN},
                       json={'dsl': 'tts{"text":"Hallo"}', 'http': {'timeout': timeout}, **extra})


def test_preparation_requires_credentials(prepared):
    client, sources = prepared
    assert client.post('/api/pipelines', json={'dsl': 'tts{"text":"Hallo"}', 'http': {}}).status_code == 401
    assert not sources


def test_secret_url_is_single_use_and_cleans_up(prepared):
    client, sources = prepared
    result = prepare(client)
    assert result.status_code == 201
    data = result.get_json()
    assert live_pipeline.get(data['id']) is not None
    assert client.get('/api/audio/guessed').status_code == 404
    response = client.get(data['audio_url'])
    assert response.status_code == 200
    assert response.data.startswith(b'RIFF')
    response.close()
    assert sources[0][2].closed
    assert live_pipeline.get(data['id']) is None
    assert client.get(data['audio_url']).status_code == 404


def test_unclaimed_endpoint_expires_and_releases_source(prepared):
    client, sources = prepared
    result = prepare(client, timeout=.03).get_json()
    done = threading.Event()
    sources[0][2].add_close_callback(done.set)
    assert done.wait(1)
    assert client.get(result['audio_url']).status_code == 404
    assert live_pipeline.get(result['id']) is None


def test_disconnect_closes_prepared_chain(prepared):
    client, sources = prepared
    result = prepare(client).get_json()
    response = client.get(result['audio_url'], buffered=False)
    next(iter(response.response))
    response.close()
    assert sources[0][2].closed
    assert live_pipeline.get(result['id']) is None


@pytest.mark.parametrize('timeout', [0, -1, 'nan', 'inf', 'bad', 601])
def test_invalid_timeout_does_not_build_stages(prepared, timeout):
    client, sources = prepared
    assert prepare(client, timeout).status_code == 400
    assert not sources


def test_dsl_is_owned_by_caller(prepared):
    client, sources = prepared
    result = prepare(client, dsl='tts{"text":"Hallo"} -> vc{"url":"carli.wav"}')
    assert result.status_code == 201
    assert [typ for typ, _, _ in sources] == ['tts', 'vc']
    assert sources[1][1]['url'] == 'carli.wav'
    pipeline = live_pipeline.get(result.get_json()['id'])
    assert sources[1][2].upstream is sources[0][2]
    pipeline.cancel()
    assert all(source.closed for _, _, source in sources)


def test_build_failure_cleans_partial_chain(prepared, monkeypatch):
    client, sources = prepared
    original = __import__('speech_pipeline.telephony.pipe_executor', fromlist=['CallPipeExecutor']).CallPipeExecutor._create_stage

    def failing(executor, typ, elem_id, params):
        if typ == 'vc':
            raise ValueError('missing reference')
        return original(executor, typ, elem_id, params)

    monkeypatch.setattr('speech_pipeline.telephony.pipe_executor.CallPipeExecutor._create_stage', failing)
    assert prepare(client, dsl='tts{"text":"Hallo"} -> vc{"url":"missing.wav"}').status_code == 400
    assert sources[0][2].closed
    assert not live_pipeline.list_all()


def test_head_does_not_consume_capability(prepared):
    client, _ = prepared
    data = prepare(client).get_json()
    assert client.head(data['audio_url']).status_code == 405
    response = client.get(data['audio_url'])
    assert response.status_code == 200
    response.close()


def test_accounts_cannot_inspect_or_delete_other_pipelines(prepared):
    from speech_pipeline.telephony import auth
    client, _ = prepared
    data = prepare(client).get_json()
    auth.put_account('http-other', {'token': 'other-account-token'})
    headers = {'Authorization': 'Bearer other-account-token'}
    try:
        assert client.get('/api/pipelines', headers=headers).get_json() == []
        for method in [client.get, client.delete]:
            assert method('/api/pipelines/' + data['id'], headers=headers).status_code == 404
        own = client.post('/api/pipelines', headers=headers,
                          json={'dsl': 'tts{"text":"Hello"}', 'http': {}}).get_json()
        visible = client.get('/api/pipelines', headers=headers).get_json()
        assert [pipeline['id'] for pipeline in visible] == [own['id']]
    finally:
        auth.delete_account('http-other')


def test_direct_render_reuses_sink_and_cleans_registry(prepared):
    client, sources = prepared
    response = client.post('/api/pipelines', headers={'Authorization': 'Bearer '+TOKEN},
                           json={'dsl': 'tts{"text":"Hello"}', 'render': True})
    assert response.status_code == 200
    assert len(response.data) == 44 + 480
    assert response.data.startswith(b'RIFF')
    response.close()
    assert sources[0][2].closed
    assert not live_pipeline.list_all()


def test_empty_pbx_allowlist_disallows_phone_access(prepared):
    from speech_pipeline.telephony import auth
    auth.put_account('tts-only-test', {'token': 'tts-only-token', 'pbx': []})
    try:
        assert not auth.check_pbx_access('tts-only-test', 'phone-system')
    finally:
        auth.delete_account('tts-only-test')
