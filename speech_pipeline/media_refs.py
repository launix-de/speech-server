from __future__ import annotations

from pathlib import Path, PurePosixPath

from .url_safety import require_safe_url


def resolve_media_ref(ref: str, media_folder: str | None) -> str:
    """Resolve a media reference to either a safe URL or a file in media_folder.

    Allowed:
    - absolute http(s) URLs to public hosts
    - relative filenames / subpaths below ``media_folder``

    Rejected:
    - absolute filesystem paths
    - ``..`` traversal
    - filenames when no media folder is configured
    """
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("media reference is required")

    if ref.startswith(("http://", "https://")):
        require_safe_url(ref)
        return ref

    if not media_folder:
        raise ValueError(
            "media reference must be an absolute http(s) URL or a filename "
            "under the configured media folder"
        )

    pure = PurePosixPath(ref)
    if pure.is_absolute():
        raise ValueError("absolute filesystem paths are not allowed")
    if any(part in ("..", ".", "") for part in pure.parts):
        raise ValueError("relative media paths must not contain '.' or '..'")

    base = Path(media_folder).resolve()
    resolved = (base / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as e:
        raise ValueError("media path escapes the configured media folder") from e
    if not resolved.exists():
        raise ValueError(f"media file not found: {ref}")
    return str(resolved)
