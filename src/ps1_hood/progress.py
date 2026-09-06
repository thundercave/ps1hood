"""Live run status for the studio GUI (no capture browser in the UI)."""

from __future__ import annotations

from typing import Any

STAGE_WEIGHTS = {
    "discover": 8,
    "capture": 42,
    "crop": 6,
    "satellite": 5,
    "bag": 8,
    "align": 22,
    "interpolate": 5,
    "reconstruct": 4,
}


class RunStatus:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "running": False,
            "stage": "",
            "queued": 0,
            "captured": 0,
            "skipped": 0,
            "aligned": 0,
            "bag_buildings": 0,
            "current": "",
            "percent": 0.0,
            "log": [],
            "error": None,
            "live_version": 0,
        }

    def update(self, stage: str, message: str = "", **fields: Any) -> None:
        if stage:
            self.data["stage"] = stage
        if message:
            self.data["log"].append({"stage": stage, "message": message})
            if len(self.data["log"]) > 400:
                self.data["log"] = self.data["log"][-300:]
        for key, value in fields.items():
            self.data[key] = value
        self.data["live_version"] = int(self.data["live_version"]) + 1
        self._recompute_percent()

    def _recompute_percent(self) -> None:
        stage = str(self.data.get("stage") or "")
        ordered = list(STAGE_WEIGHTS)
        done = 0.0
        total = float(sum(STAGE_WEIGHTS.values()) or 1)
        if stage not in STAGE_WEIGHTS:
            if self.data.get("running"):
                return
            self.data["percent"] = 100.0 if not self.data.get("error") else float(self.data["percent"])
            return
        for name in ordered:
            if name == stage:
                break
            done += STAGE_WEIGHTS[name]
        if stage == "capture":
            queued = max(1, int(self.data.get("queued") or 1))
            captured = int(self.data.get("captured") or 0) + int(self.data.get("skipped") or 0)
            frac = min(1.0, captured / queued)
            done += STAGE_WEIGHTS["capture"] * frac
        elif stage == "align":
            queued = max(1, int(self.data.get("queued") or 1))
            aligned = int(self.data.get("aligned") or 0)
            done += STAGE_WEIGHTS["align"] * min(1.0, aligned / queued)
        else:
            done += STAGE_WEIGHTS[stage] * 0.35
        self.data["percent"] = round(min(99.0, 100.0 * done / total), 1)

    def finish(self, error: str | None = None) -> None:
        self.data["running"] = False
        self.data["error"] = error
        if error:
            self.data["stage"] = "error"
        else:
            self.data["percent"] = 100.0
            self.data["stage"] = "done"
        self.data["live_version"] = int(self.data["live_version"]) + 1


def emit(progress: Any, stage: str, message: str, **fields: Any) -> None:
    import logging

    logging.getLogger("ps1hood").info("%s: %s", stage, message)
    if progress is None:
        return
    if hasattr(progress, "update"):
        progress.update(stage, message, **fields)
    else:
        progress(stage, message)
