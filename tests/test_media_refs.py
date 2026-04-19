from __future__ import annotations

from pathlib import Path

import pytest

import speech_pipeline.media_refs as media_refs
from speech_pipeline.media_refs import resolve_media_ref


def test_resolve_media_ref_accepts_public_https_url(monkeypatch: pytest.MonkeyPatch):
    url = "https://cdn.example.com/audio/hold.mp3"
    monkeypatch.setattr(media_refs, "require_safe_url", lambda value: None)
    assert resolve_media_ref(url, None) == url


def test_resolve_media_ref_rejects_absolute_path(tmp_path: Path):
    with pytest.raises(ValueError, match="absolute filesystem paths"):
        resolve_media_ref("/etc/passwd", str(tmp_path))


def test_resolve_media_ref_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match=r"\.\."):
        resolve_media_ref("../secret.wav", str(tmp_path))


def test_resolve_media_ref_requires_media_folder_for_filenames():
    with pytest.raises(ValueError, match="media folder"):
        resolve_media_ref("voice.wav", None)


def test_resolve_media_ref_resolves_relative_filename_under_media_folder(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    target = media / "voice.wav"
    target.write_bytes(b"stub")

    assert resolve_media_ref("voice.wav", str(media)) == str(target.resolve())
