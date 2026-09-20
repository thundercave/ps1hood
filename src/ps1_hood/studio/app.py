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
                            "has_cloud_zclean": (child / "recon" / "cloud_zclean.ply").is_file(),
                            "has_cloud_offtile": (child / "recon" / "cloud_offtile.ply").is_file(),
                            "has_compare": (child / "recon" / "compare" / "summary.json").is_file(),
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

    @app.get("/api/runs/<name>/cloud_zclean.ply")
    def api_cloud_zclean(name: str):
        path = open_project(name).recon_dir / "cloud_zclean.ply"
        if not path.is_file():
            return jsonify({"error": "no cloud_zclean yet — run: ps1hood cloud-zclean <run>"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/cloud_offtile.ply")
    def api_cloud_offtile(name: str):
        path = open_project(name).recon_dir / "cloud_offtile.ply"
        if not path.is_file():
            return jsonify({"error": "no cloud_offtile yet — run: ps1hood cloud-offtile <run>"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/facades.obj")
    def api_facades_obj(name: str):
        path = open_project(name).recon_dir / "facades.obj"
        if not path.is_file():
            return jsonify({"error": "no facades yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/facades.mtl")
    def api_facades_mtl(name: str):
        path = open_project(name).recon_dir / "facades.mtl"
        if not path.is_file():
            return jsonify({"error": "no facades mtl yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/roofs.obj")
    def api_roofs_obj(name: str):
        path = open_project(name).recon_dir / "roofs.obj"
        if not path.is_file():
            return jsonify({"error": "no roofs yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/roofs.mtl")
    def api_roofs_mtl(name: str):
        path = open_project(name).recon_dir / "roofs.mtl"
        if not path.is_file():
            return jsonify({"error": "no roofs mtl yet"}), 404
        return send_file(path)



    @app.get("/api/runs/<name>/street.obj")
    def api_street_obj(name: str):
        path = open_project(name).recon_dir / "street.obj"
        if not path.is_file():
            return jsonify({"error": "no street yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/street.mtl")
    def api_street_mtl(name: str):
        path = open_project(name).recon_dir / "street.mtl"
        if not path.is_file():
            return jsonify({"error": "no street mtl yet"}), 404
        return send_file(path)


    @app.get("/api/runs/<name>/compare/summary.json")
    def api_compare_summary(name: str):
        path = open_project(name).recon_dir / "compare" / "summary.json"
        if not path.is_file():
            return jsonify({"error": "no compare yet — run: ps1hood compare <run>"}), 404
        return send_file(path, mimetype="application/json")

    @app.get("/api/runs/<name>/compare/<path:filename>")
    def api_compare_file(name: str, filename: str):
        folder = open_project(name).recon_dir / "compare"
        path = (folder / filename).resolve()
        if not str(path).startswith(str(folder.resolve())) or not path.is_file():
            return jsonify({"error": "missing"}), 404
        return send_file(path)


    @app.get("/api/runs/<name>/textures/<path:filename>")
    def api_facade_texture(name: str, filename: str):
        root = open_project(name).recon_dir / "textures"
        # Prevent path escape while allowing nested texture names.
        target = (root / filename).resolve()
        if root.resolve() not in target.parents and target != root.resolve():
            return jsonify({"error": "path outside textures"}), 400
        if not target.is_file():
            return jsonify({"error": "missing"}), 404
        return send_file(target)

    @app.get("/api/runs/<name>/satellite.jpg")
    def api_sat(name: str):
        path = open_project(name).satellite_dir / "ortho.jpg"
        if not path.is_file():
            return jsonify({"error": "no satellite yet"}), 404
        return send_file(path)

    @app.get("/api/runs/<name>/planes.json")
    def api_planes_json(name: str):
        path = open_project(name).recon_dir / "planes.json"
        if not path.is_file():
            return jsonify({"error": "no planes yet"}), 404
        resp = send_file(path, mimetype="application/json")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/runs/<name>/sculpt/cameras")
    def api_sculpt_cameras(name: str):
        """Known posed cams for sculpt bake dropdown (no free-pose)."""
        from ps1_hood.reconstruct.keyframes import load_keyframes

        project = open_project(name)
        frames = load_keyframes(project)
        items = []
        for i, fr in enumerate(frames):
            items.append(
                {
                    "index": i,
                    "pano_id": fr.get("pano_id"),
                    "e": fr.get("e"),
                    "n": fr.get("n"),
                    "u": fr.get("u"),
                    "heading": fr.get("heading"),
                    "path": fr.get("path"),
                }
            )
        return jsonify({"cameras": items, "count": len(items)})

    @app.post("/api/runs/<name>/sculpt")
    def api_sculpt(name: str):
        """Bak + write one façade edit; optional re-bake from known pano."""
        from ps1_hood.reconstruct.keyframes import load_keyframes
        from ps1_hood.reconstruct.sculpt import apply_sculpt

        body = request.get_json(force=True) or {}
        plane_id = body.get("plane_id") or body.get("id")
        if not plane_id:
            return jsonify({"ok": False, "error": "plane_id required"}), 400
        project = open_project(name)
        frames = load_keyframes(project)
        for fr in frames:
            if "path" not in fr and fr.get("shot_path"):
                fr["path"] = fr["shot_path"]
        try:
            result = apply_sculpt(
                project.recon_dir,
                str(plane_id),
                delta_d=body.get("delta_d"),
                delta_t=body.get("delta_t"),
                width_m=body.get("width_m"),
                height_m=body.get("height_m"),
                corners=body.get("corners") or body.get("quad"),
                n=body.get("n"),
                d=body.get("d"),
                bake=bool(body.get("bake", True)),
                bake_cam=body.get("bake_cam"),
                frames=frames,
                margin_px=float(body.get("margin_px") or 120.0),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(result)

    @app.post("/api/runs/<name>/sculpt/undo")
    def api_sculpt_undo(name: str):
        from ps1_hood.reconstruct.sculpt import undo_sculpt

        body = request.get_json(force=True, silent=True) or {}
        try:
            result = undo_sculpt(
                open_project(name).recon_dir,
                stamp=body.get("stamp"),
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(result)


    @app.post("/api/runs/<name>/sat-offset/pairs")
    def api_sat_offset_pairs(name: str):
        """Fit SE(2) from ≥3 yellow↔red corner picks — preview only, no apply."""
        from ps1_hood.align.sat_offset import SatOffsetError, fit_pairs_se2, persist_t_pick

        body = request.get_json(force=True) or {}
        pairs = body.get("pairs")
        if not isinstance(pairs, list):
            return jsonify({"ok": False, "error": "pairs must be a list"}), 400
        project = open_project(name)
        try:
            payload = fit_pairs_se2(
                pairs,
                max_rms_m=float(body.get("max_rms_m") or 1.5),
                min_pairs=int(body.get("min_pairs") or 3),
                max_yaw_deg=float(body.get("max_yaw_deg") or 15.0),
                max_translation_m=float(body.get("max_translation_m") or 10.0),
            )
            paths = persist_t_pick(project, payload)
        except SatOffsetError as exc:
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        return jsonify(
            {
                "ok": True,
                "applied": False,
                "tx_m": payload["tx_m"],
                "ty_m": payload["ty_m"],
                "yaw_deg": payload["yaw_deg"],
                "s": payload["s"],
                "pivot_e": payload["pivot_e"],
                "pivot_n": payload["pivot_n"],
                "rms_m": payload["rms_m"],
                "n_pairs": payload["n_pairs"],
                "source": payload["source"],
                "preview": payload.get("preview") or {},
                "pairs": payload.get("pairs") or [],
                "paths": paths,
                "T": {
                    "tx_m": payload["tx_m"],
                    "ty_m": payload["ty_m"],
                    "yaw_deg": payload["yaw_deg"],
                    "s": payload["s"],
                    "pivot_e": payload["pivot_e"],
                    "pivot_n": payload["pivot_n"],
                    "source": payload["source"],
                    "rms_m": payload["rms_m"],
                    "n_pairs": payload["n_pairs"],
                },
            }
        )

    @app.post("/api/runs/<name>/sat-offset/cam-road-pairs")
    def api_sat_offset_cam_road_pairs(name: str):
        """Fit SE(2) from cam↔road-center picks or a drawn centerline polyline.

        Preview only — writes align/T_pick_cam_road.json. No apply.
        Body: ``{pairs:[...]}`` and/or ``{polyline:[{e,n},...]}``.
        """
        from ps1_hood.align.sat_offset import (
            SatOffsetError,
            fit_cam_road_from_pairs,
            fit_cam_road_from_polyline,
            persist_t_pick_cam_road,
        )

        body = request.get_json(force=True) or {}
        project = open_project(name)
        try:
            if body.get("polyline"):
                payload = fit_cam_road_from_polyline(
                    project,
                    body["polyline"],
                    search_r_m=float(body.get("search_r_m") or 15.0),
                    max_rms_m=float(body.get("max_rms_m") or 2.0),
                    min_pairs=int(body.get("min_pairs") or 4),
                    max_yaw_deg=float(body.get("max_yaw_deg") or 10.0),
                    max_translation_m=float(body.get("max_translation_m") or 12.0),
                    max_mad_m=float(body.get("max_mad_m") or 1.5),
                    translation_only=bool(body.get("translation_only", True)),
                )
            elif isinstance(body.get("pairs"), list):
                payload = fit_cam_road_from_pairs(
                    body["pairs"],
                    max_rms_m=float(body.get("max_rms_m") or 2.0),
                    min_pairs=int(body.get("min_pairs") or 4),
                    max_yaw_deg=float(body.get("max_yaw_deg") or 10.0),
                    max_translation_m=float(body.get("max_translation_m") or 12.0),
                    max_mad_m=float(body.get("max_mad_m") or 1.5),
                    translation_only=bool(body.get("translation_only", False)),
                )
            else:
                return jsonify({"ok": False, "error": "need pairs:[...] or polyline:[{e,n},...]"}), 400
            paths = persist_t_pick_cam_road(project, payload)
        except SatOffsetError as exc:
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        return jsonify(
            {
                "ok": True,
                "applied": False,
                "tx_m": payload["tx_m"],
                "ty_m": payload["ty_m"],
                "yaw_deg": payload["yaw_deg"],
                "rms_m": payload["rms_m"],
                "mad_m": payload.get("mad_m"),
                "median_abs_m": payload.get("median_abs_m"),
                "n_pairs": payload["n_pairs"],
                "source": payload["source"],
                "preview": payload.get("preview") or {},
                "paths": paths,
                "T": {
                    "tx_m": payload["tx_m"],
                    "ty_m": payload["ty_m"],
                    "yaw_deg": payload["yaw_deg"],
                    "s": payload.get("s", 1.0),
                    "pivot_e": payload.get("pivot_e"),
                    "pivot_n": payload.get("pivot_n"),
                    "source": payload["source"],
                    "rms_m": payload["rms_m"],
                    "n_pairs": payload["n_pairs"],
                },
                "note": "preview only — apply via POST …/sat-offset/apply with from=T_pick_cam_road + confirm:true",
            }
        )

    @app.post("/api/runs/<name>/sat-offset/apply")
    def api_sat_offset_apply(name: str):
        """Explicit confirm: bak then apply T_pick (or given --from) via sat-offset apply.

        Never auto-applies Chamfer T_force — body.confirm must be true.
        """
        from ps1_hood.align.sat_offset import SatOffsetError, apply_forced_se2, load_t_force

        body = request.get_json(force=True, silent=True) or {}
        if not body.get("confirm"):
            return jsonify(
                {
                    "ok": False,
                    "error": "confirm:true required — preview first via POST …/sat-offset/pairs",
                    "applied": False,
                }
            ), 400
        project = open_project(name)
        which = str(body.get("from") or body.get("which") or "T_pick").strip()
        if which.endswith(".json"):
            src = Path(which)
            if not src.is_absolute():
                src = project.root / src if (project.root / src).is_file() else (project.align_dir / Path(which).name)
        elif which in ("T_pick", "pick", "studio"):
            src = project.align_dir / "T_pick.json"
        elif which in ("T_pick_cam_road", "cam_road", "cam-road", "camroad"):
            src = project.align_dir / "T_pick_cam_road.json"
        elif which in ("T_cam_road", "cams-road", "cams_road"):
            src = project.align_dir / "T_cam_road.json"
        elif which in ("T_force", "force", "chamfer"):
            # Allowed only with explicit confirm + which — still not auto.
            src = project.align_dir / "T_force.json"
        else:
            src = project.align_dir / f"{which}.json" if not which.endswith(".json") else project.align_dir / which
        try:
            T = load_t_force(src)
            meta = apply_forced_se2(
                project,
                T,
                targets=str(body.get("targets") or "cams,cloud,facades,planes"),
                skip=str(body.get("skip") or "roofs,street"),
                bak=bool(body.get("bak", True)),
            )
        except SatOffsetError as exc:
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 404
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc), "applied": False}), 400
        return jsonify({"ok": True, "applied": True, "from": str(src), **meta})


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
