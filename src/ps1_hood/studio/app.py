"""Tiny Flask studio."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from ps1_hood.config import ProjectSpec, Settings
from ps1_hood.geo import BBox
from ps1_hood.pipeline import run_all
from ps1_hood.progress import RunStatus
from ps1_hood.project import create_project, default_runs_root, open_project

STATIC = Path(__file__).resolve().parent / "static"

_jobs: dict[str, RunStatus] = {}


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/viewer")
    def viewer():
        return send_from_directory(STATIC, "viewer.html")

    @app.get("/api/runs")
    def api_runs():
        root = default_runs_root()
        items = []
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (child / "project.yaml").is_file():
                    spec = open_project(child.name).load_spec()
                    items.append(
                        {
                            "name": spec.name,
                            "bbox": spec.bbox.as_dict(),
                            "source": spec.source,
                            "has_cloud": (child / "recon" / "cloud.ply").is_file(),
                            "has_scene": (child / "recon" / "scene.json").is_file(),
                            "has_live": (child / "live.json").is_file(),
                            "has_bag": (child / "bag" / "buildings.json").is_file(),
                        }
                    )
        return jsonify(items)

    @app.post("/api/runs")
    def api_create():
        body = request.get_json(force=True)
        settings = Settings.from_env()
        spec = ProjectSpec(
            name=str(body["name"]).strip().replace(" ", "-"),
            bbox=BBox.from_dict(body["bbox"]),
            source=body.get("source") or settings.source,
            spacing_m=float(body.get("spacing_m") or 8),
            interp_steps=int(body.get("interp_steps") or 8),
        )
        project = create_project(spec)
        return jsonify({"ok": True, "name": spec.name, "root": str(project.root)})

    @app.post("/api/runs/<name>/run")
    def api_run(name: str):
        status = _jobs.get(name)
        if status is not None and status.data.get("running"):
            return jsonify({"ok": False, "error": "already running"}), 409
        settings = Settings.from_env()
        project = open_project(name)
        status = RunStatus()
        status.data["running"] = True
        _jobs[name] = status

        def work() -> None:
            try:
                run_all(project, settings, progress=status)
                status.finish()
            except Exception as exc:  # noqa: BLE001
                status.finish(str(exc))

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"ok": True})

    @app.get("/api/runs/<name>/status")
    def api_status(name: str):
        job = _jobs.get(name)
        if job is not None:
            return jsonify(job.data)
        # CLI/align writes live.json without a studio job — still report it
        # so the 3D pane updates without clicking "run pipeline".
        payload = {
            "running": False,
            "log": [],
            "percent": 0,
            "queued": 0,
            "captured": 0,
            "skipped": 0,
            "bag_buildings": 0,
            "aligned": 0,
            "stage": "idle",
            "live_version": 0,
        }
        try:
            project = open_project(name)
            path = project.live_path
            if path.is_file():
                live = json.loads(path.read_text(encoding="utf-8"))
                payload["bag_buildings"] = len(live.get("buildings") or [])
                payload["aligned"] = len(live.get("cameras") or [])
                payload["stage"] = "done"
                payload["percent"] = 100
                payload["live_version"] = int(path.stat().st_mtime)
        except Exception:
            pass
        return jsonify(payload)

    @app.get("/api/runs/<name>/live")
    def api_live(name: str):
        path = open_project(name).live_path
        if not path.is_file():
            return jsonify({"buildings": [], "cameras": []})
        resp = send_file(path)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/runs/<name>/scene")
    def api_scene(name: str):
        path = open_project(name).recon_dir / "scene.json"
        if not path.is_file():
            return jsonify({"error": "no scene yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/cloud.ply")
    def api_cloud(name: str):
        path = open_project(name).recon_dir / "cloud.ply"
        if not path.is_file():
            return jsonify({"error": "no cloud yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/satellite.jpg")
    def api_sat(name: str):
        path = open_project(name).satellite_dir / "ortho.jpg"
        if not path.is_file():
            return jsonify({"error": "no satellite yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/file")
    def api_file(name: str):
        rel = request.args.get("path", "")
        root = open_project(name).root.resolve()
        path = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        if root not in path.parents and path != root:
            # also allow absolute paths that already live under the run
            if not str(path).startswith(str(root)):
                return jsonify({"error": "path outside run"}), 400
        if not path.is_file():
            return jsonify({"error": "missing"}), 404
        return send_file(path)

    return app


def serve(host: str, port: int) -> None:
    create_app().run(host=host, port=port, debug=False, threaded=True)
