"""HTTP audio deadlines release owned chains without stopping shared audio."""
import threading

import pytest

from speech_pipeline.base import AudioFormat, Stage
from speech_pipeline.ResponseWriter import ResponseWriter
from speech_pipeline.RawResponseWriter import RawResponseWriter
from speech_pipeline.FileFetcher import FileFetcher
from speech_pipeline.ConferenceMixer import ConferenceMixer
from speech_pipeline.ConferenceEndpoint import ConferenceSource, _Coupling
from speech_pipeline.AudioMixer import AudioMixer


class Source(Stage):
    output_format = AudioFormat(8000, "s16le")

    def __init__(self, chunks=None):
        super().__init__()
        self.chunks = chunks if chunks is not None else [b'\x01\x00' * 8]
        self.cleaned = 0
        self.exhausted = False

    def stream_pcm24k(self):
        try:
            for chunk in self.chunks:
                if self.cancelled:
                    break
                yield chunk
        finally:
            self.exhausted = True

    def _on_close(self):
        self.cleaned += 1


def wait_closed(stage):
    done = threading.Event()
    stage.add_close_callback(done.set)
    assert done.wait(2), "HTTP sink did not expire"


def test_unclaimed_response_expires_entire_exclusive_chain():
    source = Source()
    middle = source.pipe(Stage())
    sink = middle.pipe(ResponseWriter(timeout=0.03))
    wait_closed(source)
    assert sink.cancelled and sink.closed
    assert middle.cancelled and middle.closed
    assert source.cancelled and source.closed
    assert source.cleaned == 1
    assert source.reference_count == middle.reference_count == 0
    assert sink._timer is None


def test_expiring_one_branch_preserves_other_consumers():
    source = Source()
    first = source.pipe(ResponseWriter(timeout=0.03, est_frames_24k=8))
    second = source.pipe(ResponseWriter(est_frames_24k=8))
    wait_closed(first)
    assert source.reference_count == 1
    assert not source.closed and not source.cancelled
    assert not second.closed and not second.cancelled
    assert b''.join(second.stream_pcm24k()).startswith(b'RIFF')
    assert source.exhausted and source.closed
    assert source.cleaned == 1


def test_retained_owner_protects_shared_source_until_release():
    source = Source()
    reference = source.retain()
    sink = source.pipe(ResponseWriter(timeout=0.03))
    wait_closed(sink)
    assert not source.closed and not source.cancelled
    reference.release()
    reference.release()
    assert source.closed and source.reference_count == 0
    assert source.cleaned == 1


def test_completion_stops_deadline_and_releases_source():
    source = Source()
    sink = source.pipe(ResponseWriter(est_frames_24k=8, timeout=10))
    assert len(b''.join(sink.stream())) == 44 + 16
    assert source.closed and sink.closed
    assert not source.cancelled and not sink.cancelled
    assert sink._timer is None


@pytest.mark.parametrize("chunks", [0, 1])
def test_disconnect_at_header_or_pcm_closes_pipeline(chunks):
    source = Source([b'\x01\x00' * 8] * 4)
    sink = source.pipe(ResponseWriter(est_frames_24k=32, timeout=10))
    stream = sink.stream_pcm24k()
    assert next(stream).startswith(b'RIFF')
    if chunks:
        next(stream)
    stream.close()
    assert sink.cancelled and sink.closed and source.closed
    assert source.cleaned == 1
    assert sink._timer is None


def test_upstream_exception_closes_resources():
    class Broken(Source):
        def stream_pcm24k(self):
            raise RuntimeError("synthesis failed")
            yield b''

    source = Broken()
    sink = source.pipe(ResponseWriter(timeout=10))
    with pytest.raises(RuntimeError, match="synthesis failed"):
        list(sink.stream())
    assert source.closed and sink.closed and sink.cancelled
    assert sink._timer is None


def test_rewiring_releases_old_processor_and_preserves_source():
    source, middle, terminal = Source(), Stage(), Stage()
    source.pipe(middle).pipe(terminal)
    source.pipe(terminal)
    assert middle.closed and middle.reference_count == 0
    assert not source.closed and source.reference_count == 1
    terminal.close()
    assert source.closed


def test_reference_release_is_thread_safe_and_idempotent():
    source = Source()
    references = [source.retain() for _ in range(8)]
    threads = [threading.Thread(target=lambda r=r: (r.release(), r.release())) for r in references]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive()
    assert source.cleaned == 1 and source.reference_count == 0


def test_raw_file_response_uses_same_deadline_and_cleanup(tmp_path):
    path = tmp_path / 'audio.mp3'
    path.write_bytes(b'abcde')
    source = FileFetcher(str(path))
    sink = source.pipe(RawResponseWriter(timeout=10))
    assert b''.join(sink.stream()) == b'abcde'
    assert sink.closed and source.closed and source._handle.closed
    assert sink._timer is None


def test_shared_mixer_survives_one_http_deadline():
    mixer = AudioMixer('shared')
    first = mixer.pipe(ResponseWriter(timeout=0.03))
    second = mixer.pipe(ResponseWriter())
    wait_closed(first)
    assert not mixer.cancelled and not mixer.closed
    second.cancel()
    assert mixer.cancelled and mixer.closed


def test_conference_output_timeout_keeps_other_participant_and_owner():
    mixer = ConferenceMixer('shared')
    owner = mixer.retain()
    first = ConferenceSource(_Coupling(mixer))
    second = ConferenceSource(_Coupling(mixer))
    first.coupling.activate()
    second.coupling.activate()
    first_queue, second_queue = first.coupling.out_queue, second.coupling.out_queue
    sink = first.pipe(ResponseWriter(timeout=0.03))
    wait_closed(first)
    assert sink.closed
    assert not mixer.cancelled and not mixer.closed
    assert all(entry.queue is not first_queue for entry in mixer._sinks)
    assert any(entry.queue is second_queue for entry in mixer._sinks)
    assert not second.closed
    second.close()
    assert not mixer.closed  # call/container still owns it
    owner.release()
    assert mixer.closed and mixer.cancelled


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_deadline_must_be_finite_positive(timeout):
    with pytest.raises(ValueError, match='timeout'):
        ResponseWriter(timeout=timeout)


def test_deadline_releases_suspended_generator_resources():
    source = Source([b'\x01\x00' * 8] * 4)
    sink = source.pipe(ResponseWriter(timeout=0.05))
    stream = sink.stream()
    next(stream)  # WAV header
    next(stream)  # suspend the upstream generator with resources open
    wait_closed(sink)
    assert source.exhausted, "deadline did not unwind suspended upstream"
    assert sink._stream is None
    stream.close()


def test_conference_stream_releases_only_its_output_on_timeout():
    mixer = ConferenceMixer('two-http-outputs')
    first = mixer.pipe(ResponseWriter(timeout=0.1))
    second = mixer.pipe(ResponseWriter())
    stream = first.stream()
    next(stream)
    # Inject a frame directly into the mixer's test output queue. No actual
    # audio clock, SIP socket, or inference model is needed for this contract.
    original = mixer.add_output

    def output(*args, **kwargs):
        q = original(*args, **kwargs)
        q.put(b'\x01\x00' * 8)
        return q

    mixer.add_output = output
    next(stream)
    assert len(mixer._sinks) == 1
    wait_closed(first)
    assert not mixer.closed and not mixer.cancelled
    assert mixer._sinks == []  # no stale queue retained for the retired reader
    assert mixer.reference_count == 1
    stream.close()
    second.cancel()
    assert mixer.closed


def test_buffered_wav_has_exact_length_even_when_source_closes_on_eof():
    import io
    import wave
    class Finite(Stage):
        output_format = AudioFormat(24000, 's16le')
        def stream_pcm24k(self):
            try:
                yield b'\x01\x00' * 240
            finally:
                self.close()
    source = Finite()
    sink = source.pipe(ResponseWriter(timeout=1, buffered=True, estimated_length=False))
    data = b''.join(sink.stream_pcm24k())
    assert len(data) == 44 + 480
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getnframes() == 240
        assert wav.getframerate() == 24000
    assert source.closed and sink.closed
    assert sink._buffer.closed


def test_failed_voice_conversion_releases_files_and_does_not_silently_pass_through(tmp_path, monkeypatch):
    import tempfile
    from speech_pipeline.VCConverter import VCConverter
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    target = tmp_path / 'target.wav'
    target.write_bytes(b'target')
    def broken(source, reference, output):
        raise RuntimeError('conversion failed')
    pipeline = Source().pipe(VCConverter(target, vc_convert=broken)).pipe(ResponseWriter(buffered=True))
    with pytest.raises(RuntimeError, match='conversion failed'):
        list(pipeline.stream_pcm24k())
    assert pipeline.closed
    assert sorted(p.name for p in tmp_path.iterdir()) == ['target.wav']
