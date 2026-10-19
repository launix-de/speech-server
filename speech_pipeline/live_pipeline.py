"""LivePipeline: runtime model for inspectable and mutable pipelines."""
from __future__ import annotations

import queue
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import Stage

_LOGGER = logging.getLogger("live-pipeline")

# ---- Global pipeline registry ----

_pipelines: Dict[str, LivePipeline] = {}
_lock = threading.Lock()


def register(pipeline: LivePipeline) -> None:
    with _lock:
        _pipelines[pipeline.id] = pipeline


def unregister(pipeline_id: str) -> Optional[LivePipeline]:
    with _lock:
        return _pipelines.pop(pipeline_id, None)


def get(pipeline_id: str) -> Optional[LivePipeline]:
    with _lock:
        return _pipelines.get(pipeline_id)


def list_all() -> List[LivePipeline]:
    with _lock:
        return list(_pipelines.values())


# ---- LivePipeline ----

class LivePipeline:
    """Runtime representation of a running pipeline with named stages."""

    def __init__(self, dsl: str = "") -> None:
        self.id: str = uuid4().hex[:12]
        self.dsl: str = dsl
        self.account_id: Optional[str] = None
        self.stages: Dict[str, Stage] = {}       # stage.id -> Stage
        self.stage_types: Dict[str, str] = {}     # stage.id -> type string
        self.stage_configs: Dict[str, dict] = {}  # stage.id -> config dict
        self.edges: List[Tuple[str, str]] = []    # (from_id, to_id)
        self.run = None                           # PipelineRun reference
        self.created_at: float = time.time()
        self.state: str = "created"               # created | running | stopped

    def add_stage(self, stage: Stage, typ: str, config: Optional[dict] = None) -> str:
        self.stages[stage.id] = stage
        self.stage_types[stage.id] = typ
        self.stage_configs[stage.id] = config or {}
        return stage.id

    def add_edge(self, from_id: str, to_id: str) -> None:
        self.edges.append((from_id, to_id))

    def get_stage(self, stage_id: str) -> Optional[Stage]:
        return self.stages.get(stage_id)

    def cancel(self) -> None:
        self.state = "stopped"
        sink = getattr(self, "_http_sink", None)
        if sink is not None:
            sink.cancel()
        if self.run:
            self.run.cancel()

    def to_dict(self, detail: bool = False) -> dict:
        d: dict = {
            "id": self.id,
            "dsl": self.dsl,
            "state": self.state,
            "created_at": self.created_at,
            "stages": len(self.stages),
        }
        if detail:
            d["stages"] = []
            for sid, stage in self.stages.items():
                entry: dict = {
                    "id": sid,
                    "type": self.stage_types.get(sid, "unknown"),
                    "config": self.stage_configs.get(sid, {}),
                }
                if stage.output_format:
                    entry["output_format"] = {
                        "sample_rate": stage.output_format.sample_rate,
                        "encoding": stage.output_format.encoding,
                    }
                if stage.input_format:
                    entry["input_format"] = {
                        "sample_rate": stage.input_format.sample_rate,
                        "encoding": stage.input_format.encoding,
                    }
                d["stages"].append(entry)
            d["edges"] = [{"from": f, "to": t} for f, t in self.edges]
        return d


# ---- CellRunner: queue-boundary wrapper for hot-swappable stages ----

class CellRunner:
    """Wraps a stage with input/output queues so it can be replaced at runtime.

    The wrapped stage runs in its own thread, pulling from input_q and
    pushing to output_q.  To swap: stop the old cell, start a new one
    with the same queues.

    Usage as a Stage-compatible object:
    - upstream writes to cell.input_q
    - downstream reads from cell.output_q via cell.as_source()
    """

    def __init__(self, stage: Stage, input_q: queue.Queue, output_q: queue.Queue) -> None:
        self.stage = stage
        self.input_q = input_q
        self.output_q = output_q
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    def start(self) -> None:
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopped.set()
        self.stage.cancel()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        from .QueueSource import QueueSource

        src = QueueSource(self.input_q)
        src.pipe(self.stage)
        try:
            for chunk in self.stage.stream_pcm24k():
                if self._stopped.is_set():
                    break
                try:
                    self.output_q.put(chunk, timeout=2.0)
                except queue.Full:
                    pass
        except Exception as e:
            _LOGGER.warning("CellRunner error: %s", e)
        finally:
            try:
                self.output_q.put(None, timeout=1.0)
            except Exception:
                pass

    def swap(self, new_stage: Stage) -> None:
        """Replace the running stage with a new one. Brief gap at boundary."""
        self.stop()
        self.stage = new_stage
        self.start()
