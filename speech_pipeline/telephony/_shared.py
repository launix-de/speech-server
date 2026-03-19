"""Shared references set during app initialization.

Avoids circular imports — modules read from here instead of
importing Flask app context.
"""
from __future__ import annotations

from typing import Any, Optional

# Set by piper_multi_server.py during create_app()
tts_registry: Optional[Any] = None  # TTSRegistry
flask_app: Optional[Any] = None
