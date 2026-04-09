"""Tests for FileFetcher._classify and its callers."""
import inspect

from speech_pipeline.FileFetcher import FileFetcher


class TestClassify:

    def test_http_url(self):
        kind, value = FileFetcher._classify("https://example.com/audio.wav")
        assert kind == "http"
        assert value == "https://example.com/audio.wav"

    def test_http_url_plain(self):
        kind, value = FileFetcher._classify("http://localhost:5000/file.mp3")
        assert kind == "http"

    def test_local_file(self):
        kind, value = FileFetcher._classify("/tmp/audio.wav")
        assert kind == "file"

    def test_no_public_classify(self):
        """_classify is private; there must be no public 'classify'."""
        assert not hasattr(FileFetcher, "classify"), \
            "FileFetcher should not have public classify"


class TestCallerConsistency:

    def test_vcconverter_calls_private_classify(self):
        from speech_pipeline.VCConverter import VCConverter
        source = inspect.getsource(VCConverter)
        assert "FileFetcher.classify(" not in source
        assert "FileFetcher._classify(" in source

    def test_pitchadjuster_calls_private_classify(self):
        from speech_pipeline.PitchAdjuster import PitchAdjuster
        source = inspect.getsource(PitchAdjuster)
        assert "FileFetcher.classify(" not in source
        assert "FileFetcher._classify(" in source
