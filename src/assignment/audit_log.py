"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Remember request details and start time until its output is recorded."""
        key = request_id or user_id
        self._open[key] = {
            "user_id": user_id,
            "request_id": request_id,
            "input": text,
            "started_at": utc_now_iso(),
            # Monotonic time is used for elapsed duration so clock adjustments
            # cannot produce negative or inflated latency values.
            "started_monotonic": time.monotonic(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append the completed request and its guardrail decision to the log."""
        key = request_id or user_id
        opened = self._open.pop(key, None)
        latency_ms = None
        if opened is not None:
            latency_ms = round(
                max(0.0, time.monotonic() - opened["started_monotonic"]) * 1000,
                3,
            )

        self.logs.append(
            {
                "user_id": user_id,
                "request_id": request_id,
                "input": opened.get("input") if opened else None,
                "output": text,
                "started_at": opened.get("started_at") if opened else None,
                "completed_at": utc_now_iso(),
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
