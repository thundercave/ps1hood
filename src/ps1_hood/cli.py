"""Command line for ps1-hood."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ps1_hood import __version__
from ps1_hood.config import Settings
from ps1_hood.geo import BBox
from ps1_hood.pipeline import STAGES, run_all  # align-prior sat (default) or bag
from ps1_hood.project import create_project, default_runs_root, open_project
from ps1_hood.config import ProjectSpec

log = logging.getLogger("ps1hood")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Rebuild a street-scale 3D neighborhood from Street View + satellite."""
    _setup_logging()


@main.command("init")
@click.argument("name")
@click.option("--south", type=float, required=True)
@click.option("--west", type=float, required=True)
@click.option("--north", type=float, required=True)
@click.option("--east", type=float, required=True)
@click.option(
    "--source",
    type=click.Choice(["google_web", "google_js", "google_static", "mapillary"]),
    default=None,
)
@click.option(
    "--spacing",
    type=float,
    default=None,
    help="metres between Street View probes (default 8; smoke preset 25)",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="interpolated frames between neighbouring SVs (default 8; smoke preset 3)",
)
@click.option(
    "--heading-step",
    type=int,
    default=None,
    help="degrees between 360° screengrabs (default 45; smoke preset 90)",
)
@click.option(
    "--max-panos",
    type=int,
    default=None,
    help="hard-cap panoramas after discover (evenly subsampled by index)",
)
@click.option(
    "--preset",
    type=click.Choice(["smoke"]),
    default=None,
    help="smoke: spacing=25, heading_step=90, steps=3, max_panos=8, extra_pitches=[0]",
)
def init_cmd(
    name: str,
    south: float,
    west: float,
    north: float,
    east: float,
    source: str | None,
    spacing: float | None,
    steps: int | None,
    heading_step: int | None,
    max_panos: int | None,
    preset: str | None,
) -> None:
    """Create a run from a geographic bounding box."""
    settings = Settings.from_env()
    # Defaults; --preset smoke fills cheap demo budget (CLI flags still win).
    spacing_m = 8.0 if spacing is None else spacing
    interp_steps = 8 if steps is None else steps
    h_step = 45 if heading_step is None else heading_step
    pitches = [-30, 0, 18]
    cap = max_panos
    if preset == "smoke":
        if spacing is None:
            spacing_m = 25.0
        if steps is None:
            interp_steps = 3
        if heading_step is None:
            h_step = 90
        if max_panos is None:
            cap = 8
        pitches = [0]
    spec = ProjectSpec(
        name=name,
        bbox=BBox(south=south, west=west, north=north, east=east),
        source=source or settings.source,
        spacing_m=spacing_m,
        interp_steps=interp_steps,
        heading_step=h_step,
        extra_pitches=pitches,
        max_panos=cap,
    )
    project = create_project(spec)
    click.echo(f"created {project.root}")
    click.echo(f"  bbox {spec.bbox.width_m():.0f} × {spec.bbox.height_m():.0f} m")
    click.echo(f"  source {spec.source}")
    if spec.max_panos is not None:
        click.echo(f"  max_panos {spec.max_panos}")
    if preset:
        click.echo(f"  preset {preset}")
    click.echo("next:  ps1hood run " + name)


@main.command("bag-download")
def bag_download_cmd() -> None:
    """Show / resume the national 3DBAG GeoPackage zip (~18 GB, queried by bbox)."""
    from ps1_hood.capture.bag import BAG_ZIP_BYTES, BAG_ZIP_URL, bag_zip_path, zip_status

    path = bag_zip_path()
    st = zip_status(path)
    click.echo(f"{st['path']}")
    click.echo(f"  {st['have']}/{st['expected']} bytes  ({st['percent']}%)")
    if st["complete"]:
        click.echo("complete — bbox queries will not unpack the whole country")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"downloading {BAG_ZIP_URL}")
    import subprocess

    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "8",
            "--retry-all-errors",
            "-C",
            "-",
            "-o",
            str(path),
            BAG_ZIP_URL,
        ],
        check=False,
    )
    st = zip_status(path)
    click.echo(f"now {st['percent']}%  complete={st['complete']}")
    if st["have"] and st["have"] != BAG_ZIP_BYTES:
        click.echo(f"expected {BAG_ZIP_BYTES} bytes")


@main.command("run")
@click.argument("name")
@click.option("--from-stage", "from_stage", default="discover", type=click.Choice(STAGES))
@click.option(
    "--align-prior",
    "align_prior",
    type=click.Choice(["sat", "bag"]),
    default=None,
    help="Absolute XY prior for align: sat (Ortho, default) or bag (legacy debug).",
)
@click.option("--sat-edge-weight", "sat_edge_weight", type=float, default=None)
@click.option(
    "--cloud-clip-sat/--no-cloud-clip-sat",
    "cloud_clip_sat",
    default=False,
    show_default=True,
    help="Opt-in floater tool: clip MA cloud to Ortho ENU ± margin. Prefer unclipped product cloud.",
)
@click.option("--sat-cloud-margin-m", "sat_cloud_margin_m", type=float, default=None)
def run_cmd(
    name: str,
    from_stage: str,
    align_prior: str | None,
    sat_edge_weight: float | None,
    cloud_clip_sat: bool,
    sat_cloud_margin_m: float | None,
) -> None:
    """Run the pipeline (or resume from a stage)."""
    settings = Settings.from_env()
    project = open_project(name)
    if align_prior is not None:
        spec = project.load_spec()
        spec.align_prior = align_prior
        project.save_spec(spec)
    click.echo(f"run {project.root}  from {from_stage}")
    try:
        run_all(
            project,
            settings,
            from_stage=from_stage,
            align_prior=align_prior,
            sat_edge_weight=sat_edge_weight,
            cloud_clip_sat=cloud_clip_sat,
            sat_cloud_margin_m=sat_cloud_margin_m,
        )
    except Exception as exc:
        click.echo(f"failed: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo("done.  ps1hood studio   → open the 3D viewer")


@main.command("discover")
@click.argument("name")
def discover_cmd(name: str) -> None:
    from ps1_hood.pipeline import stage_discover

    stage_discover(open_project(name), Settings.from_env())


@main.command("capture")
@click.argument("name")
def capture_cmd(name: str) -> None:
    from ps1_hood.pipeline import stage_capture

    stage_capture(open_project(name), Settings.from_env())


@main.command("crop")
@click.argument("name")
def crop_cmd(name: str) -> None:
    from ps1_hood.pipeline import stage_crop

    stage_crop(open_project(name), Settings.from_env())


@main.command("satellite")
@click.argument("name")
def satellite_cmd(name: str) -> None:
    from ps1_hood.pipeline import stage_satellite

    stage_satellite(open_project(name))


@main.command("align")
@click.argument("name")
@click.option(
    "--align-prior",
    "align_prior",
    type=click.Choice(["sat", "bag"]),
    default="sat",
    show_default=True,
    help="Absolute XY prior: sat = Ortho edge+NCC + SE(2) (no BAG snap); bag = legacy.",
)
@click.option(
    "--sat-edge-weight",
    "sat_edge_weight",
    type=float,
    default=0.65,
    show_default=True,
    help="Weight for Ortho Canny edge term in fused sat score (w_ncc = 1 - w_edge).",
)
@click.option(
    "--cloud-clip-sat/--no-cloud-clip-sat",
    "cloud_clip_sat",
    default=False,
    show_default=True,
    help="Opt-in floater tool: after sat seat, clip MA cloud.ply to Ortho ENU ± margin. Prefer unclipped product cloud.",
)
@click.option(
    "--sat-cloud-margin-m",
    "sat_cloud_margin_m",
    type=float,
    default=2.0,
    show_default=True,
    help="ENU margin (m) outside Ortho bbox when --cloud-clip-sat.",
)
@click.option(
    "--cloud-zclean/--no-cloud-zclean",
    "cloud_zclean",
    default=False,
    show_default=True,
    help="Opt-in soft Z gate after sat seat: write recon/cloud_zclean.ply (does not replace product cloud).",
)
@click.option(
    "--zclean-margin-m",
    "zclean_margin_m",
    type=float,
    default=1.5,
    show_default=True,
    help="Drop pts inside sat roof AABB with z > shell_z + margin (m).",
)
def align_cmd(
    name: str,
    align_prior: str,
    sat_edge_weight: float,
    cloud_clip_sat: bool,
    sat_cloud_margin_m: float,
    cloud_zclean: bool,
    zclean_margin_m: float,
) -> None:
    from ps1_hood.pipeline import stage_align

    project = open_project(name)
    spec = project.load_spec()
    if getattr(spec, "align_prior", None) != align_prior:
        spec.align_prior = align_prior
        project.save_spec(spec)
    stage_align(
        project,
        align_prior=align_prior,
        sat_edge_weight=sat_edge_weight,
        cloud_clip_sat=cloud_clip_sat,
        sat_cloud_margin_m=sat_cloud_margin_m,
        cloud_zclean=cloud_zclean,
        zclean_margin_m=zclean_margin_m,
    )


@main.command("interpolate")
@click.argument("name")
def interpolate_cmd(name: str) -> None:
    """FILM/flow midframes between panos; poses always from ``lerp_pose`` ENU.

    Appearance only — never invents extrinsics. Asserts full FILM-rate ENU +
    PINHOLE size before densify/recon consume midframes.
    """
    from ps1_hood.pipeline import stage_interpolate

    stage_interpolate(open_project(name))


@main.command("reconstruct")
@click.argument("name")
@click.option(
    "--backend",
    type=click.Choice(["flow", "colmap", "colmap_posed", "sift", "mast3r", "export"]),
    default=None,
)
@click.option(
    "--matcher",
    type=click.Choice(["sift", "mast3r"]),
    default=None,
    help="posed COLMAP matcher: sift (default) or mast3r (GPU, ENU-locked). "
    "Implied mast3r when --backend mast3r.",
)
def reconstruct_cmd(name: str, backend: str | None, matcher: str | None) -> None:
    from ps1_hood.pipeline import stage_reconstruct

    project = open_project(name)
    if backend or matcher:
        spec = project.load_spec()
        if backend:
            spec.recon_backend = backend
            if backend == "mast3r" and matcher is None:
                spec.recon_matcher = "mast3r"
        if matcher:
            spec.recon_matcher = matcher
        project.save_spec(spec)
    stage_reconstruct(project)


@main.command("densify")
@click.argument("name")
@click.option(
    "--backend",
    type=click.Choice(["openmvs", "mapanything"]),
    default="openmvs",
    help="optional densify backend (openmvs=AGPL binary; mapanything=CUDA Meta)",
)
@click.option("--resolution-level", default=2, show_default=True, type=int)
@click.option("--number-views", default=4, show_default=True, type=int)
@click.option(
    "--images",
    "images_path",
    type=click.Path(path_type=Path, exists=False),
    default=None,
    help="override COLMAP images dir (default recon/colmap/images)",
)
@click.option(
    "--sparse",
    "sparse_path",
    type=click.Path(path_type=Path, exists=False),
    default=None,
    help="override posed sparse model (default recon/colmap/sparse_posed)",
)
@click.option(
    "--stride",
    default=2,
    show_default=True,
    type=int,
    help="MapAnything: subsample every Nth frame (COLMAP demo / bundle)",
)
@click.option(
    "--max-views",
    default=48,
    show_default=True,
    type=int,
    help="MapAnything Path B: cap views after stride",
)
@click.option(
    "--apache/--research",
    default=True,
    help="MapAnything: apache weights (default) vs CC-BY-NC research checkpoint",
)
@click.option(
    "--export-only",
    is_flag=True,
    default=False,
    help="MapAnything: write pose-locked bundle only (no CUDA infer)",
)
@click.option(
    "--prefer-interp-bundle/--prefer-colmap",
    "prefer_interp_bundle",
    default=True,
    help="MapAnything PR-B: prefer pose-locked FILM/lerp midframe bundle "
    "(default) over Path A COLMAP sparse (no mids). Auto-skips Path A when "
    "midframes are present even if --prefer-colmap.",
)
@click.option(
    "--backup/--no-backup",
    "backup_cloud",
    default=True,
    help="MapAnything: backup recon/cloud.ply (+ cloud_mapanything.ply) before "
    "replace. Never touches façades/roofs. Default on.",
)
def densify_cmd(
    name: str,
    backend: str,
    resolution_level: int,
    number_views: int,
    images_path: Path | None,
    sparse_path: Path | None,
    stride: int,
    max_views: int,
    apache: bool,
    export_only: bool,
    prefer_interp_bundle: bool,
    backup_cloud: bool,
) -> None:
    """Optional densify: OpenMVS (CPU/AGPL) or MapAnything (CUDA, ENU-locked).

    OpenMVS: InterfaceCOLMAP + DensifyPointCloud on PATH → openmvs/scene_dense.ply.
    MapAnything (PR-B): FILM/lerp midframes + locked ENU cam2world + K;
    never ignore_pose_inputs. Prefer --apache. Without CUDA use --export-only
    then scripts/run_mapanything_bundle.py. Soft sat clip stays opt-in on
    align/run (--cloud-clip-sat); densify default is unclipped product cloud.
    """
    project = open_project(name)
    if backend == "openmvs":
        from ps1_hood.reconstruct.openmvs import INSTALL_HINT, run_openmvs_densify, which_openmvs

        if which_openmvs() is None:
            click.echo(INSTALL_HINT, err=True)
            raise SystemExit(1)
        try:
            meta = run_openmvs_densify(
                project.root,
                images_path=Path(images_path) if images_path else None,
                sparse_path=Path(sparse_path) if sparse_path else None,
                resolution_level=resolution_level,
                number_views=number_views,
            )
        except Exception as exc:
            click.echo(f"densify failed: {exc}", err=True)
            raise SystemExit(1) from exc
        click.echo(
            f"densify ok  {meta['path']}  points={meta['points']}  backend={meta['backend']}"
        )
        return

    if backend == "mapanything":
        from ps1_hood.reconstruct import mapanything as ma

        try:
            frames = ma.resolve_mapanything_frames(
                project, stride=1, max_views=None, prefer_interp=True
            )
            bundle_meta = ma.export_mapanything_bundle(
                frames,
                project.root / "mapanything" / "bundle",
                stride=stride,
                max_views=max_views,
                run_name=name,
            )
        except Exception as exc:
            click.echo(f"mapanything export failed: {exc}", err=True)
            raise SystemExit(1) from exc
        n_mids = ma.count_interpolated_frames(frames)
        click.echo(
            f"mapanything bundle  {bundle_meta['path']}  "
            f"views={bundle_meta['n_views']}  mids≈{n_mids}  "
            f"pose_lock=True  ignore_pose_inputs=False"
        )
        if export_only:
            click.echo(
                "export-only: on CUDA host run "
                "python scripts/run_mapanything_bundle.py "
                f"{bundle_meta['path']} --apache"
            )
            return
        try:
            meta = ma.run_mapanything_densify(
                project.root,
                project,
                stride=stride,
                max_views=max_views,
                apache=apache,
                prefer_colmap=not prefer_interp_bundle,
                force_interp_bundle=prefer_interp_bundle,
                backup=backup_cloud,
            )
        except Exception as exc:
            click.echo(f"densify failed: {exc}", err=True)
            click.echo(ma.INSTALL_HINT, err=True)
            raise SystemExit(1) from exc
        bak = meta.get("backups") or {}
        bak_note = f"  backups={len(bak)}" if bak else ""
        click.echo(
            f"densify ok  {meta.get('path') or meta.get('cloud_mapanything')}  "
            f"points={meta.get('points')}  backend=mapanything  "
            f"pose_lock={meta.get('pose_lock')}  path={meta.get('path_kind')}  "
            f"mids={meta.get('n_midframes', 0)}"
            f"{bak_note}  façades/roofs untouched"
        )
        return

    raise SystemExit(f"unsupported densify backend: {backend}")


@main.group("export")
def export_group() -> None:
    """Export run artefacts for external tools."""


@export_group.command("mapanything-bundle")
@click.argument("name")
@click.option("--stride", default=2, show_default=True, type=int)
@click.option("--max-views", default=48, show_default=True, type=int)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path, exists=False),
    default=None,
    help="override output dir (default runs/<name>/mapanything/bundle)",
)
def export_mapanything_bundle_cmd(
    name: str,
    stride: int,
    max_views: int,
    out_dir: Path | None,
) -> None:
    """Export images + ENU cam2world + K for MapAnything (no GPU required).

    Writes manifest.json with pose_lock=True. Feed to
    scripts/run_mapanything_bundle.py on a CUDA host. Never use
    --ignore_pose_inputs when inferring.
    """
    from ps1_hood.reconstruct import mapanything as ma

    project = open_project(name)
    try:
        frames = ma.resolve_mapanything_frames(
            project, stride=1, max_views=None, prefer_interp=True
        )
        dest = Path(out_dir) if out_dir else project.root / "mapanything" / "bundle"
        meta = ma.export_mapanything_bundle(
            frames,
            dest,
            stride=stride,
            max_views=max_views,
            run_name=name,
        )
    except Exception as exc:
        click.echo(f"export failed: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"exported {meta['path']}  views={meta['n_views']}  "
        f"format={meta['format']}  pose_lock=True"
    )


@main.command("facades")
@click.argument("name")
@click.option(
    "--source",
    type=click.Choice(["mapanything", "recon", "auto"]),
    default="mapanything",
    show_default=True,
    help="compat: MA peel PLY only (sets --ma-source). Does not set A's xyz.",
)
@click.option(
    "--a-source",
    type=click.Choice(["flow", "product", "mapanything", "recon"]),
    default="flow",
    show_default=True,
    help="A-arm xyz/seeds: recon/cloud_flow.ply|cloud.ply, product planes.json, or MA (debug)",
)
@click.option(
    "--ma-source",
    type=click.Choice(["mapanything", "recon", "auto"]),
    default=None,
    help="MA peel PLY (default: --source / mapanything). Independent of --a-source.",
)
@click.option(
    "--control-out",
    is_flag=True,
    default=False,
    help="Write facades.control.* / planes.control.json; never clobber *.candidate or product",
)
@click.option(
    "--planarize/--no-planarize",
    default=True,
    show_default=True,
    help="Path α Open3D/numpy segment_plane peel (auto-on for dense MA PLY in extract_facades)",
)
@click.option("--zncc-accept", default=0.35, show_default=True, type=float)
@click.option("--voxel", "voxel_m", default=0.08, show_default=True, type=float)
@click.option("--plane-dist", "plane_dist_m", default=0.08, show_default=True, type=float)
@click.option("--max-planes", default=16, show_default=True, type=int)
@click.option(
    "--keep-previous-on-fail/--no-keep-previous-on-fail",
    default=True,
    show_default=True,
    help="Preserve product on 0 accepts (*.failed) or when new result is not strictly better (*.candidate)",
)
@click.option(
    "--fallback-heading/--no-fallback-heading",
    default=True,
    show_default=True,
    help="If Path α ZNCC keeps 0 after hybrid, retry Milestone A heading×distance search",
)
@click.option(
    "--hybrid-heading/--no-hybrid-heading",
    default=True,
    show_default=True,
    help="Enable Path α hybrid A+MA (default dual-arm full A search + MA peels)",
)
@click.option(
    "--hybrid-a-full-search/--no-hybrid-a-full-search",
    default=True,
    show_default=True,
    help="Hybrid A arm = full search_photo_consistent_planes (on); off = inject seeds into score_planar_hyps",
)
@click.option(
    "--split-trigger",
    "split_trigger_width_m",
    default=8.0,
    show_default=True,
    type=float,
    help="Split peels wider than this (m) into façade windows before ZNCC",
)
@click.option(
    "--split-window",
    "split_window_m",
    default=8.0,
    show_default=True,
    type=float,
    help="Façade window length along peel right-axis (m)",
)
@click.option(
    "--split-overlap",
    "split_overlap_m",
    default=2.0,
    show_default=True,
    type=float,
    help="Overlap between consecutive split windows (m)",
)
@click.option(
    "--min-inliers",
    default=400,
    show_default=True,
    type=int,
    help="Peel: minimum inliers for a RANSAC plane (Path α)",
)
@click.option(
    "--residual-stop",
    default=1500,
    show_default=True,
    type=int,
    help="Peel: stop when remaining cloud points fall below this",
)
@click.option(
    "--vertical-dot",
    default=0.15,
    show_default=True,
    type=float,
    help="Peel: max |n·up| to accept as vertical façade",
)
@click.option(
    "--peel-max-planes",
    default=24,
    show_default=True,
    type=int,
    help="Peel budget (distinct from keep --max-planes)",
)
@click.option(
    "--nms-xy",
    "nms_xy_m",
    default=6.0,
    show_default=True,
    type=float,
    help="NMS XY center radius (m); split siblings use --nms-xy-split",
)
@click.option(
    "--nms-xy-split",
    "nms_xy_split_m",
    default=4.0,
    show_default=True,
    type=float,
    help="NMS XY radius (m) when either hyp has split_parent",
)
@click.option(
    "--union-strategy",
    type=click.Choice(["nms", "a_priority"]),
    default="a_priority",
    show_default=True,
    help="Hybrid union: keep A first then non-dup MA (a_priority), or ZNCC-sorted NMS (nms)",
)
@click.option(
    "--ps1-rectify/--no-ps1-rectify",
    default=True,
    show_default=True,
    help="Manhattan-rectify accepted façade quads (PS1) before warp/write",
)
@click.option(
    "--ps1-tex-size",
    default=128,
    show_default=True,
    type=int,
    help="Nearest resize short side (128² or 128×256) + RGB555; 0 disables",
)
@click.option(
    "--sat-roofs/--no-sat-roofs",
    default=False,
    show_default=True,
    help="Also build sat-locked roof/yard shells → recon/roofs.obj (PR-A)",
)
@click.option(
    "--gap-fill/--no-gap-fill",
    default=False,
    show_default=True,
    help="PR-C: manhattan+corner seeds from sat roofs + MA peels for side walls; "
    "a_priority; quality-keep vs product (no peel spam / no BAG)",
)
@click.option(
    "--worst-cams",
    default=None,
    type=str,
    help="Comma-separated compare cam ids (pano_hNNN) for targeted gap-fill seeds",
)
@click.option(
    "--worst-from-compare",
    default=None,
    type=int,
    help="Auto-load top-N worst cam ids from recon/compare/summary.json",
)
@click.option(
    "--max-gap-adds",
    default=3,
    show_default=True,
    type=int,
    help="Cap new gap-fill planes after stricter multi-view + sat AABB gates",
)
@click.option(
    "--sat-aabb-gate",
    "sat_aabb_gate_m",
    default=2.0,
    show_default=True,
    type=float,
    help="Gap-add center must be within this many metres of a sat roof boundary "
    "(0 disables; soft-pass when no roofs)",
)
@click.option(
    "--ma-peel-cap",
    default=3,
    show_default=True,
    type=int,
    help="Max MA peels kept for gap-fill scoring (rethink: stop peel spam)",
)
@click.option(
    "--gap-min-views",
    default=2,
    show_default=True,
    type=int,
    help="Gap adds need ZNCC≥0.35 on at least this many scoring pair/cams",
)
@click.option(
    "--gap-seeds",
    default="sat-edge",
    show_default=True,
    type=click.Choice(["legacy", "sat-edge", "both"]),
    help="Gap seeds: sat-edge (uncovered roof AABB + facing cams), "
    "legacy (manhattan/corner/worst-cam), or both",
)
@click.option(
    "--sat-edge-reanchor/--no-sat-edge-reanchor",
    default=True,
    show_default=True,
    help="After ZNCC, re-anchor sat_edge center onto seed edge segment "
    "(reject refine_drift if |Δd| exceeds --sat-edge-max-drift)",
)
@click.option(
    "--sat-edge-max-drift",
    "sat_edge_max_drift_m",
    default=3.0,
    show_default=True,
    type=float,
    help="Max |d_refined − d_anchor| (m) when re-anchoring sat_edge",
)
@click.option(
    "--sat-aabb-gate-sat-edge",
    "sat_aabb_gate_sat_edge_m",
    default=None,
    type=float,
    help="Optional softer AABB gate (m) for sat_edge only (e.g. 5.0). "
    "Does NOT widen gate for MA peels / manhattan. Default: use --sat-aabb-gate",
)
@click.option(
    "--gap-nms-xy",
    "gap_nms_xy_m",
    default=3.0,
    show_default=True,
    type=float,
    help="a_priority dup XY (m) for sat_edge/ma_gap_sat_edge vs product_lock only. "
    "Other MA sources keep --nms-xy (default 6).",
)
@click.option(
    "--gap-nms-d-tol",
    "gap_nms_d_tol_m",
    default=1.0,
    show_default=True,
    type=float,
    help="a_priority |Δd| tol (m) for sat_edge vs product_lock only "
    "(global NMS keeps 2.5).",
)
@click.option(
    "--gap-nms-no-opposite/--gap-nms-opposite",
    "gap_nms_no_opposite",
    default=True,
    show_default=True,
    help="Drop |d+d_k| opposite-normal clause for sat_edge vs product_lock "
    "(default on). Other sources keep opposite-d.",
)
def facades_cmd(
    name: str,
    source: str,
    a_source: str,
    ma_source: str | None,
    control_out: bool,
    planarize: bool,
    zncc_accept: float,
    voxel_m: float,
    plane_dist_m: float,
    max_planes: int,
    keep_previous_on_fail: bool,
    fallback_heading: bool,
    hybrid_heading: bool,
    hybrid_a_full_search: bool,
    split_trigger_width_m: float,
    split_window_m: float,
    split_overlap_m: float,
    min_inliers: int,
    residual_stop: int,
    vertical_dot: float,
    peel_max_planes: int,
    nms_xy_m: float,
    nms_xy_split_m: float,
    union_strategy: str,
    ps1_rectify: bool,
    ps1_tex_size: int,
    sat_roofs: bool,
    gap_fill: bool,
    worst_cams: str | None,
    worst_from_compare: int | None,
    max_gap_adds: int,
    sat_aabb_gate_m: float,
    ma_peel_cap: int,
    gap_min_views: int,
    gap_seeds: str,
    sat_edge_reanchor: bool,
    sat_edge_max_drift_m: float,
    sat_aabb_gate_sat_edge_m: float | None,
    gap_nms_xy_m: float,
    gap_nms_d_tol_m: float,
    gap_nms_no_opposite: bool,
) -> None:
    """Path α: planarize dense ENU cloud → ZNCC-gated façades.obj + planes.json.

    Prefer MapAnything product PLY for plane seeds; still photo-ZNCC gate.
    No OSM/BAG hero. Residual organic omitted (no Poisson in α1).

    ``--gap-fill`` (PR-C): keep product A core; add ZNCC-gated manhattan /
    sat-corner seeds + road-rejected MA peels for side/return walls; promote
    only via quality-keep (never demote product).

    ``--worst-cams`` / ``--worst-from-compare``: seed planes toward those cams
    (dist 8–20 m × yaw 0/±45/±90); filter manhattan/corner to visible in them;
    lock product, NMS-add new, max_keep 24, bake new textures only.

    Gap-add hygiene: ``--sat-aabb-gate`` / ``--gap-min-views`` / ``--max-gap-adds``
    / ``--ma-peel-cap`` tighten multi-view + sat footprint for *new* planes only
    (product_lock untouched). ``--gap-seeds sat-edge`` (default) invents
    uncovered roof-edge hyps; zero facing cams → ``recon/gap_needs.json``.

    ``--sat-edge-reanchor`` (default on): after ZNCC, snap sat_edge center back
    onto its seed edge; reject ``refine_drift`` if |Δd| > ``--sat-edge-max-drift``.
    AABB distance is to that ``edge_id`` segment. Optional
    ``--sat-aabb-gate-sat-edge`` softens the gate for sat_edge only (not MA peels).

    ``--gap-nms-xy`` / ``--gap-nms-d-tol`` / ``--gap-nms-no-opposite``: edge-aware
    a_priority dup for ``ma_gap_sat_edge`` vs product_lock only (cover check,
    tighter XY/Δd, no opposite-d). Does **not** loosen global NMS for other
    sources.
    """
    from ps1_hood.geo import LocalFrame
    from ps1_hood.reconstruct.facades import extract_facades
    from ps1_hood.reconstruct.keyframes import load_keyframes
    from ps1_hood.reconstruct.planarize import resolve_a_ply, resolve_ma_ply

    project = open_project(name)

    ma_src = ma_source or source
    ctx = click.get_current_context(silent=True)
    source_explicit = False
    if ctx is not None:
        src_info = ctx.get_parameter_source("source")
        source_explicit = src_info == click.core.ParameterSource.COMMANDLINE
    if source_explicit and ma_source is None:
        log.warning(
            "facades: --source=%s sets MA peel PLY only; A xyz uses --a-source=%s",
            source,
            a_source,
        )
    try:
        ply = resolve_ma_ply(project.root, ma_src)
    except FileNotFoundError as exc:
        if planarize:
            click.echo(f"facades: {exc}", err=True)
            raise SystemExit(1) from exc
        ply = None
        log.warning("facades: no MA peel PLY (%s) — A-only / control", exc)

    a_ply = resolve_a_ply(project.root, a_source)
    if a_source == "product":
        a_ply = None
    elif a_ply is None and a_source in {"flow", "recon"}:
        log.warning(
            "facades: no flow/recon PLY for --a-source=%s — A will fall back "
            "(product lock if planes.json, else MA ply)",
            a_source,
        )

    frames = load_keyframes(project)
    for fr in frames:
        if "path" not in fr and fr.get("shot_path"):
            fr["path"] = fr["shot_path"]

    sat = None
    sat_meta = project.satellite_dir / "meta.json"
    if sat_meta.is_file():
        sat = project.read_json(sat_meta)
        if isinstance(sat, dict) and sat.get("path"):
            p = Path(sat["path"])
            if not p.is_file():
                cand = project.root / sat["path"]
                if cand.is_file():
                    sat["path"] = str(cand)

    try:
        frame = LocalFrame.from_bbox(project.load_spec().bbox)
    except Exception:
        frame = None

    try:
        meta = extract_facades(
            ply,
            project.recon_dir / "facades.obj",
            n_planes=max_planes,
            frames=frames,
            satellite=sat if isinstance(sat, dict) else None,
            local_frame=frame,
            zncc_accept=zncc_accept,
            planarize=planarize,
            voxel_m=voxel_m,
            plane_dist_m=plane_dist_m,
            keep_previous_on_fail=keep_previous_on_fail,
            fallback_heading=fallback_heading,
            hybrid_heading=hybrid_heading,
            hybrid_a_full_search=hybrid_a_full_search,
            split_trigger_width_m=split_trigger_width_m,
            split_window_m=split_window_m,
            split_overlap_m=split_overlap_m,
            min_inliers=min_inliers,
            residual_stop=residual_stop,
            vertical_dot=vertical_dot,
            peel_max_planes=peel_max_planes,
            nms_xy_m=nms_xy_m,
            nms_xy_split_m=nms_xy_split_m,
            union_strategy=union_strategy,
            a_ply_path=a_ply,
            a_source=a_source,
            control_out=control_out,
            ps1_rectify=ps1_rectify,
            ps1_tex_size=(None if int(ps1_tex_size) <= 0 else int(ps1_tex_size)),
            gap_fill=gap_fill,
            project_root=project.root,
            worst_cam_ids=(
                [c.strip() for c in str(worst_cams).split(",") if c.strip()]
                if worst_cams
                else None
            ),
            worst_from_compare=worst_from_compare,
            max_gap_adds=int(max_gap_adds),
            sat_aabb_gate_m=float(sat_aabb_gate_m),
            ma_peel_cap=int(ma_peel_cap),
            gap_min_views=int(gap_min_views),
            gap_seeds_mode=str(gap_seeds),
            sat_edge_reanchor=bool(sat_edge_reanchor),
            sat_edge_max_drift_m=float(sat_edge_max_drift_m),
            sat_aabb_gate_sat_edge_m=(
                None
                if sat_aabb_gate_sat_edge_m is None
                else float(sat_aabb_gate_sat_edge_m)
            ),
            gap_nms_xy_m=float(gap_nms_xy_m),
            gap_nms_d_tol_m=float(gap_nms_d_tol_m),
            gap_nms_no_opposite=bool(gap_nms_no_opposite),
        )
    except Exception as exc:
        click.echo(f"facades failed: {exc}", err=True)
        raise SystemExit(1) from exc

    # Merge into scene.json if present
    scene_path = project.recon_dir / "scene.json"
    if scene_path.is_file():
        import json

        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        cloud = payload.get("cloud") if isinstance(payload.get("cloud"), dict) else {}
        cloud = dict(cloud or {})
        cloud["facades"] = meta
        cloud["planar_source"] = meta.get("source")
        payload["cloud"] = cloud
        payload["facades"] = meta
        scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_planes_out = int(meta.get("planes") or 0)
    preserved = bool(meta.get("preserved_previous"))
    rejected_weaker = bool(meta.get("rejected_weaker"))
    if rejected_weaker and preserved:
        status = "preserved_better"
    elif n_planes_out > 0:
        status = "ok"
    elif preserved:
        status = "preserved"
    else:
        status = "FAIL"
    click.echo(
        f"facades {status}  planes={n_planes_out}  textured={meta.get('textured')}  "
        f"source={meta.get('source')}  mean_zncc={meta.get('mean_zncc')}  "
        f"preserved_previous={preserved}  "
        f"candidate_planes={meta.get('candidate_planes')}  "
        f"reason={meta.get('candidate_reason')}  "
        f"a_ply={meta.get('a_ply') or a_ply}  ma_ply={meta.get('ma_ply') or ply}  "
        f"a_kept={meta.get('a_kept')}  ma_added={meta.get('ma_added')}  "
        f"output_kind={meta.get('output_kind')}"
    )
    if sat_roofs:
        from ps1_hood.reconstruct.sat_roofs import SatRoofError, build_sat_roofs

        try:
            rmeta = build_sat_roofs(project.root)
            click.echo(
                f"sat-roofs ok  roofs={rmeta.get('n_roof')}  yards={rmeta.get('n_yard')}  "
                f"textured={rmeta.get('textured')}  mean_edge_m={rmeta.get('mean_edge_m')}  "
                f"gate_target_ok={rmeta.get('gate_target_ok')}  obj={rmeta.get('obj')}"
            )
            if scene_path.is_file():
                import json as _json

                payload = _json.loads(scene_path.read_text(encoding="utf-8"))
                payload["roofs"] = rmeta
                scene_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        except SatRoofError as exc:
            click.echo(f"sat-roofs failed: {exc}", err=True)
            raise SystemExit(1) from exc
    # Product intact on quality-keep; only fail-hard when nothing usable remains
    if n_planes_out <= 0 and not preserved:
        raise SystemExit(1)


@main.command("ps1-facades")
@click.argument("name")
@click.option(
    "--ps1-tex-size",
    default=128,
    show_default=True,
    type=int,
    help="Nearest resize short side (128² or 128×256) + RGB555",
)
@click.option(
    "--no-backup",
    is_flag=True,
    default=False,
    help="Do not write facades.obj.bak before rewriting quads",
)
def ps1_facades_cmd(name: str, ps1_tex_size: int, no_backup: bool) -> None:
    """Post-process existing product: Manhattan rectify + PS1 textures (no re-extract).

    Rectifies planes.json (or parses facades.obj), nearest-resizes façade JPGs,
    RGB555-quantizes, rewrites facades.obj quads. Quality-keep is not re-run.
    """
    from ps1_hood.reconstruct.ps1_facades import postprocess_run

    project = open_project(name)
    try:
        meta = postprocess_run(
            project.recon_dir,
            tex_size=int(ps1_tex_size),
            backup_obj=not no_backup,
        )
    except Exception as exc:
        click.echo(f"ps1-facades failed: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"ps1-facades ok  planes={meta.get('planes')}  "
        f"textures={meta.get('textures_rewritten')}  "
        f"tex_size={meta.get('ps1_tex_size')}  "
        f"obj={meta.get('obj')}  planes_json={meta.get('planes_json')}"
    )


@main.command("facades-retexture")
@click.argument("name")
@click.option(
    "--bare-only/--all",
    default=True,
    show_default=True,
    help="Only bake planes missing JPG or map_Kd (default); --all retries every plane",
)
@click.option(
    "--candidate",
    "stage_candidate",
    is_flag=True,
    default=False,
    help="Write facades.candidate.* + textures/candidate/; quality-keep promote",
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    help="Bake into live product with *.pre_retex backup",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List bare façade indices only; no writes",
)
@click.option(
    "--margin-px",
    default=120.0,
    show_default=True,
    type=float,
    help="Warp out-of-frame margin (softens historic 40px hard fail)",
)
@click.option(
    "--top-k-cams",
    default=5,
    show_default=True,
    type=int,
    help="Try top-K frontal×coverage cameras per bare plane",
)
@click.option(
    "--ps1-tex-size",
    default=128,
    show_default=True,
    type=int,
    help="Nearest resize short side after warp (0 = skip PS1 quantize)",
)
@click.option(
    "--no-promote",
    is_flag=True,
    default=False,
    help="With --candidate, leave candidate staged even if clause1 would promote",
)
def facades_retexture_cmd(
    name: str,
    bare_only: bool,
    stage_candidate: bool,
    in_place: bool,
    dry_run: bool,
    margin_px: float,
    top_k_cams: int,
    ps1_tex_size: int,
    no_promote: bool,
) -> None:
    """Re-texture bare façades only (no gap-fill / planarize / densify).

    Post-rectify full-quad warp often fails after ZNCC patch pass — this retries
    with multi-cam + looser margin. Prefer ``--dry-run`` then ``--bare-only
    --candidate`` (quality-keep promote when textured↑ planes unchanged).
    """
    from ps1_hood.reconstruct.facades import retexture_bare_planes
    from ps1_hood.reconstruct.keyframes import load_keyframes

    if sum(1 for f in (stage_candidate, in_place, dry_run) if f) > 1:
        click.echo(
            "facades-retexture: pass only one of --candidate / --in-place / --dry-run",
            err=True,
        )
        raise SystemExit(2)
    if not dry_run and not stage_candidate and not in_place:
        # Pack default path: candidate staging
        stage_candidate = True

    project = open_project(name)
    frames = load_keyframes(project)
    for fr in frames:
        if "path" not in fr and fr.get("shot_path"):
            fr["path"] = fr["shot_path"]

    tex_size = int(ps1_tex_size) if int(ps1_tex_size) > 0 else None
    try:
        meta = retexture_bare_planes(
            project.root,
            margin_px=float(margin_px),
            top_k_cams=int(top_k_cams),
            ps1_tex_size=tex_size,
            frames=frames,
            bare_only=bool(bare_only),
            candidate=bool(stage_candidate) and not dry_run and not in_place,
            in_place=bool(in_place) and not dry_run,
            dry_run=bool(dry_run),
            promote=not no_promote,
        )
    except Exception as exc:
        click.echo(f"facades-retexture failed: {exc}", err=True)
        raise SystemExit(1) from exc

    if dry_run:
        click.echo(
            f"facades-retexture dry-run  planes={meta.get('planes')}  "
            f"bare={meta.get('bare')}  ids={meta.get('bare_ids')}  "
            f"textured_before={meta.get('textured_before')}"
        )
        return

    qk = meta.get("quality_keep") or {}
    click.echo(
        f"facades-retexture ok  planes={meta.get('planes')}  "
        f"baked={meta.get('baked')}  failed={meta.get('failed')}  "
        f"textured={meta.get('textured_before')}→{meta.get('textured_after')}  "
        f"promoted={meta.get('promoted')}  "
        f"reason={meta.get('promote_reason') or qk.get('reason')}  "
        f"obj={meta.get('obj')}"
    )



@main.group("gap-needs")
def gap_needs_group() -> None:
    """Far-side SV for recon/gap_needs.json (prefer existing facing; fetch better probes)."""


@gap_needs_group.command("show")
@click.argument("name")
@click.option(
    "--need-id",
    "--id",
    "need_id",
    default=None,
    help="Single need id / edge_id (e.g. roof_099_e0); default: all",
)
def gap_needs_show_cmd(name: str, need_id: str | None) -> None:
    """List gap_needs entries with ENU→ll, probes, and existing nearly-facing cams."""
    from ps1_hood.reconstruct.gap_needs import (
        GapNeedsMissing,
        load_gap_needs,
        local_frame_for_run,
        select_need,
        summarize_need,
    )

    project = open_project(name)
    path = project.recon_dir / "gap_needs.json"
    try:
        payload = load_gap_needs(path)
    except GapNeedsMissing as exc:
        click.echo(f"gap-needs show: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"gap-needs show: {exc}", err=True)
        raise SystemExit(1) from exc

    try:
        needs = select_need(payload, need_id)
        frame = local_frame_for_run(project.root)
    except Exception as exc:
        click.echo(f"gap-needs show: {exc}", err=True)
        raise SystemExit(1) from exc

    cams_path = project.align_dir / "cameras.json"
    cameras = project.read_json(cams_path) if cams_path.is_file() else []
    click.echo(
        f"gap_needs {path}  run={payload.get('run')}  "
        f"product_planes={payload.get('product_planes')}  needs={len(payload.get('needs') or [])}"
    )
    for need in needs:
        s = summarize_need(need, frame, cameras)
        click.echo(
            f"  {s['id']}  ENU=({s['e']:.2f},{s['n']:.2f})  "
            f"heading={s['heading']:.1f}  ll=({s['lat']:.7f},{s['lon']:.7f})  "
            f"reason={s.get('reason')}"
        )
        click.echo(
            f"    prefer existing facing: {s['n_existing_facing']} pano(s) "
            "(fetch still probes for better)"
        )
        for ef in s["existing_facing"][:6]:
            click.echo(
                f"      {ef['pano_id']}  h={ef['heading']}  "
                f"dist={ef['dist_m']:.1f}m  score={ef['score']:.3f}"
            )
        for pr in s["probes"]:
            click.echo(
                f"    probe r={pr['radius_m']:.0f}m  "
                f"ENU=({pr['e']:.2f},{pr['n']:.2f})  "
                f"ll=({pr['lat']:.7f},{pr['lon']:.7f})"
            )


@gap_needs_group.command("fetch")
@click.argument("name")
@click.option(
    "--need-id",
    "--id",
    "need_id",
    default=None,
    help="Single need id / edge_id (e.g. roof_099_e0); default: all needs",
)
@click.option(
    "--radii",
    default="8,12,16,20",
    show_default=True,
    help="Comma-separated probe radii in metres along outward normal",
)
@click.option(
    "--max-panos",
    default=6,
    show_default=True,
    type=int,
    help="Cap on new panos appended per need",
)
@click.option(
    "--align/--no-align",
    default=True,
    show_default=True,
    help="After capture/crop, align with --align-prior sat (no free-pose)",
)
@click.option(
    "--film/--no-film",
    "do_film",
    default=True,
    show_default=True,
    help="If <2 facing frames after prefer+fetch, ENU-lerp FILM spur (PR-B)",
)
@click.option(
    "--capture/--no-capture",
    "do_capture",
    default=True,
    show_default=True,
    help="Capture/crop new panos (reuse existing capture path); off = discover-only",
)
def gap_needs_fetch_cmd(
    name: str,
    need_id: str | None,
    radii: str,
    max_panos: int,
    align: bool,
    do_film: bool,
    do_capture: bool,
) -> None:
    """Probe opposite the wall, prefer existing facing, fetch better SV, sat-align.

    Sacred: fixed poses only · --align-prior sat · lock 18 · no sat_edge peels.
    """
    from ps1_hood.reconstruct.gap_needs import (
        GapNeedsMissing,
        build_probe_seeds,
        capture_new_panos_only,
        crop_new_shots_only,
        existing_pano_ids,
        filter_facing_panos_enu,
        find_existing_facing,
        interpolate_facing_spur,
        load_gap_needs,
        local_frame_for_run,
        lookup_new_panos,
        merge_discover_panos,
        need_id_of,
        select_need,
        unique_facing_pano_count,
    )

    settings = Settings.from_env()
    project = open_project(name)
    path = project.recon_dir / "gap_needs.json"
    try:
        payload = load_gap_needs(path)
        needs = select_need(payload, need_id)
        frame = local_frame_for_run(project.root)
    except GapNeedsMissing as exc:
        click.echo(f"gap-needs fetch: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"gap-needs fetch: {exc}", err=True)
        raise SystemExit(1) from exc

    try:
        radii_m = tuple(float(x.strip()) for x in radii.split(",") if x.strip())
    except ValueError as exc:
        click.echo(f"gap-needs fetch: bad --radii {radii!r}", err=True)
        raise SystemExit(1) from exc
    if not radii_m:
        click.echo("gap-needs fetch: --radii empty", err=True)
        raise SystemExit(1)

    cams_path = project.align_dir / "cameras.json"
    cameras = project.read_json(cams_path) if cams_path.is_file() else []
    discover_path = project.discover_dir / "panos.json"
    discover = (
        project.read_json(discover_path)
        if discover_path.is_file()
        else {"source": "gap_needs", "panos": []}
    )
    have_ids = existing_pano_ids(discover)
    # also treat captured raw dirs as existing
    if project.raw_dir.is_dir():
        for d in project.raw_dir.iterdir():
            if d.is_dir() and not d.name.startswith("seed-"):
                have_ids.add(d.name)

    lookup_fn = None
    if settings.google_maps_api_key:
        from ps1_hood.capture.google_static import lookup_pano

        key = settings.google_maps_api_key

        def lookup_fn(lat: float, lon: float):  # noqa: F811
            return lookup_pano(lat, lon, key)

    all_new: list[dict] = []
    facing_after: dict[str, int] = {}

    for need in needs:
        nid = need_id_of(need)
        existing = find_existing_facing(cameras, need)
        n_exist = unique_facing_pano_count(existing)
        click.echo(
            f"need {nid}: prefer existing facing={n_exist} "
            f"(e.g. {[c.get('pano_id') for c in existing[:3]]}); "
            "fetch still probes for better"
        )
        seeds = build_probe_seeds(need, frame, radii_m=radii_m, need_id=nid)
        resolved = lookup_new_panos(
            seeds,
            existing_ids=have_ids,
            lookup_fn=lookup_fn,
            max_panos=int(max_panos),
        )
        # Prefer panos that can face; keep provisional seeds if lookup empty
        facing_new = filter_facing_panos_enu(resolved, need, frame)
        chosen = facing_new if facing_new else resolved
        for p in chosen:
            have_ids.add(str(p.get("pano_id")))
        all_new.extend(chosen)
        facing_after[nid] = n_exist  # updated after capture/align

    if not all_new:
        click.echo(
            "gap-needs fetch: no new panos to attach "
            "(existing facing may already cover — still prefer those)"
        )
    else:
        discover = merge_discover_panos(discover, all_new)
        project.write_json(discover_path, discover)
        click.echo(f"appended {len(all_new)} pano(s) → {discover_path}")

    new_shots: list = []
    if do_capture and all_new:
        try:
            new_shots = capture_new_panos_only(project, settings, all_new)
            click.echo(f"captured {len(new_shots)} frame(s) from new panos")
            if new_shots:
                cropped = crop_new_shots_only(project, settings, new_shots)
                click.echo(f"cropped {len(cropped)} frame(s)")
        except Exception as exc:
            click.echo(f"gap-needs fetch capture failed: {exc}", err=True)
            raise SystemExit(1) from exc
    elif not do_capture:
        click.echo("skip capture (--no-capture)")

    if align and (all_new or do_capture):
        from ps1_hood.pipeline import stage_align

        spec = project.load_spec()
        if getattr(spec, "align_prior", None) != "sat":
            spec.align_prior = "sat"
            project.save_spec(spec)
        click.echo("align --align-prior sat (seat new cams into same ENU; no free-pose)")
        try:
            stage_align(project, align_prior="sat")
        except Exception as exc:
            click.echo(f"gap-needs fetch align failed: {exc}", err=True)
            raise SystemExit(1) from exc
        # refresh cameras after align
        if cams_path.is_file():
            cameras = project.read_json(cams_path)

    # Recompute facing counts; optional FILM spur if still <2
    for need in needs:
        nid = need_id_of(need)
        existing = find_existing_facing(cameras, need)
        n_face = unique_facing_pano_count(existing)
        facing_after[nid] = n_face
        click.echo(f"need {nid}: facing panos after fetch/align = {n_face}")
        if do_film and n_face < 2:
            click.echo(
                f"need {nid}: <2 facing — ENU-lerp FILM spur "
                "(appearance only; reuse PR-B lerp_pose)"
            )
            try:
                spur = interpolate_facing_spur(project, existing if len(existing) >= 2 else existing)
                # If only 1 facing, try spur with nearest aligned poses to that cam
                if not spur and existing:
                    # pair with nearest other camera by XY
                    anchor = existing[0]
                    others = [
                        c
                        for c in cameras
                        if c.get("pano_id") != anchor.get("pano_id")
                    ]
                    others.sort(
                        key=lambda c: (
                            (float(c["e"]) - float(anchor["e"])) ** 2
                            + (float(c["n"]) - float(anchor["n"])) ** 2
                        )
                    )
                    pair = [anchor] + others[:1]
                    spur = interpolate_facing_spur(project, pair)
                click.echo(f"need {nid}: spur midframes={len(spur)}")
            except Exception as exc:
                click.echo(f"gap-needs FILM spur failed: {exc}", err=True)
                raise SystemExit(1) from exc
        elif n_face >= 2:
            click.echo(f"need {nid}: ≥2 facing — skip FILM densify")

    click.echo(
        "next: ps1hood facades "
        f"{name} --a-source product --no-gap-fill --no-planarize  "
        "# photo search; NOT sat_edge; lock 18"
    )



@main.command("cloud-zclean")
@click.argument("name")
@click.option(
    "--margin-m",
    "--margin",
    "margin_m",
    type=float,
    default=1.5,
    show_default=True,
    help="Drop pts inside sat roof AABB with z > shell_z + margin (m).",
)
@click.option(
    "--aabb-inset-m",
    "aabb_inset_m",
    type=float,
    default=0.5,
    show_default=True,
    help="Inset (m) applied to roof AABB footprints before the Z gate.",
)
@click.option(
    "--drop-sinks/--no-drop-sinks",
    "drop_sinks",
    default=False,
    show_default=True,
    help="Also drop pts inside roof AABB with z < ground_z - 1.0 m (off by default).",
)
@click.option(
    "--replace-product/--no-replace-product",
    "replace_product",
    default=False,
    show_default=True,
    help="Also rewrite cloud.ply from zclean (default: write cloud_zclean.ply sidecar only).",
)
@click.option(
    "--offtile/--no-offtile",
    "offtile",
    default=False,
    show_default=True,
    help="Also run soft support offtile gate → recon/cloud_offtile.ply (opt-in).",
)
@click.option(
    "--support-dilate-m",
    "support_dilate_m",
    type=float,
    default=10.0,
    show_default=True,
    help="Dilate roof∪yard∪street AABBs (m) for offtile support keep-mask.",
)
@click.option(
    "--far-m",
    "far_m",
    type=float,
    default=15.0,
    show_default=True,
    help="Drop pts farther than this (m) from dilated support (offtile).",
)
@click.option(
    "--z-out-m",
    "z_out_m",
    type=float,
    default=8.0,
    show_default=True,
    help="Drop off-support pts with z > local_ground + this (m) (offtile).",
)
def cloud_zclean_cmd(
    name: str,
    margin_m: float,
    aabb_inset_m: float,
    drop_sinks: bool,
    replace_product: bool,
    offtile: bool,
    support_dilate_m: float,
    far_m: float,
    z_out_m: float,
) -> None:
    """Opt-in soft Z gate vs sat roof shells — drop sky floaters above roofs.

    Inside each sat roof footprint AABB (inset), drop cloud pts with
    z > shell_z + margin. Yards/street outside AABBs unchanged. Writes
    recon/cloud_zclean.ply (+ bak); does NOT change default product to XY-clipped.
    Façades/roofs shells untouched. Mapillary garage fill is a separate non-goal.

    With --offtile: also write recon/cloud_offtile.ply via soft support gate
    (dilated roof∪yard∪street; not Ortho XY±2 m).
    """
    from ps1_hood.align.georef import offtile_recon_clouds, zclean_recon_clouds

    project = open_project(name)
    roofs = project.recon_dir / "roofs.json"
    if not roofs.is_file():
        click.echo(
            f"cloud-zclean: missing {roofs} — run: ps1hood roofs {name}",
            err=True,
        )
        raise SystemExit(1)
    if not (project.recon_dir / "cloud.ply").is_file():
        click.echo(f"cloud-zclean: missing recon/cloud.ply in {name}", err=True)
        raise SystemExit(1)

    stats = zclean_recon_clouds(
        project.recon_dir,
        margin_m=float(margin_m),
        aabb_inset_m=float(aabb_inset_m),
        drop_sinks=bool(drop_sinks),
        replace_product=bool(replace_product),
    )
    georef_path = project.align_dir / "georef.json"
    georef = project.read_json(georef_path) if georef_path.is_file() else {}
    georef["cloud_zclean"] = stats
    project.align_dir.mkdir(parents=True, exist_ok=True)
    project.write_json(georef_path, georef)

    click.echo(
        f"cloud-zclean ok  kept={stats.get('kept')}  dropped={stats.get('dropped')}  "
        f"margin_m={stats.get('margin_m')}  n_roofs={stats.get('n_roofs')}  "
        f"sidecar=recon/cloud_zclean.ply  replace_product={replace_product}"
    )

    if offtile:
        ot = offtile_recon_clouds(
            project.recon_dir,
            dilate_m=float(support_dilate_m),
            far_m=float(far_m),
            z_out_m=float(z_out_m),
            poses_path=project.align_dir / "poses.json",
            replace_product=False,
        )
        georef["cloud_offtile"] = ot
        project.write_json(georef_path, georef)
        click.echo(
            f"cloud-offtile ok  kept={ot.get('kept')}  dropped={ot.get('dropped')}  "
            f"dropped_far={ot.get('dropped_far')}  dropped_z={ot.get('dropped_z')}  "
            f"dilate_m={ot.get('dilate_m')}  far_m={ot.get('far_m')}  "
            f"z_out_m={ot.get('z_out_m')}  sidecar=recon/cloud_offtile.ply"
        )


@main.command("cloud-offtile")
@click.argument("name")
@click.option(
    "--support-dilate-m",
    "support_dilate_m",
    type=float,
    default=10.0,
    show_default=True,
    help="Dilate roof∪yard∪street AABBs (m) for the keep-mask (~8–12).",
)
@click.option(
    "--far-m",
    "far_m",
    type=float,
    default=15.0,
    show_default=True,
    help="Drop pts farther than this (m) from dilated support.",
)
@click.option(
    "--z-out-m",
    "z_out_m",
    type=float,
    default=8.0,
    show_default=True,
    help="Drop off-support pts with z > local_ground + this (m).",
)
@click.option(
    "--cam-corridor-m",
    "cam_corridor_m",
    type=float,
    default=6.0,
    show_default=True,
    help="Also keep ±this (m) around cam poses (0 disables).",
)
@click.option(
    "--replace-product/--no-replace-product",
    "replace_product",
    default=False,
    show_default=True,
    help="Also rewrite cloud.ply from offtile (default: sidecar only).",
)
def cloud_offtile_cmd(
    name: str,
    support_dilate_m: float,
    far_m: float,
    z_out_m: float,
    cam_corridor_m: float,
    replace_product: bool,
) -> None:
    """Opt-in soft support gate — drop off-tile / sky-halo floaters.

    Dilates sat roof∪yard∪street (~10 m). Outside support: drop if
    z ≫ local ground (+z-out) OR farther than far-m. Soft fringe kept.
    Writes recon/cloud_offtile.ply (+ bak/georef). NOT Ortho XY±2 m.
    Façades untouched. Does not densify.
    """
    from ps1_hood.align.georef import offtile_recon_clouds

    project = open_project(name)
    roofs = project.recon_dir / "roofs.json"
    if not roofs.is_file():
        click.echo(
            f"cloud-offtile: missing {roofs} — run: ps1hood roofs {name}",
            err=True,
        )
        raise SystemExit(1)
    if not (project.recon_dir / "cloud.ply").is_file():
        click.echo(f"cloud-offtile: missing recon/cloud.ply in {name}", err=True)
        raise SystemExit(1)

    stats = offtile_recon_clouds(
        project.recon_dir,
        dilate_m=float(support_dilate_m),
        far_m=float(far_m),
        z_out_m=float(z_out_m),
        cam_corridor_m=float(cam_corridor_m),
        poses_path=project.align_dir / "poses.json",
        replace_product=bool(replace_product),
    )
    georef_path = project.align_dir / "georef.json"
    georef = project.read_json(georef_path) if georef_path.is_file() else {}
    georef["cloud_offtile"] = stats
    project.align_dir.mkdir(parents=True, exist_ok=True)
    project.write_json(georef_path, georef)

    click.echo(
        f"cloud-offtile ok  kept={stats.get('kept')}  dropped={stats.get('dropped')}  "
        f"dropped_far={stats.get('dropped_far')}  dropped_z={stats.get('dropped_z')}  "
        f"dilate_m={stats.get('dilate_m')}  far_m={stats.get('far_m')}  "
        f"z_out_m={stats.get('z_out_m')}  local_ground_z={stats.get('local_ground_z')}  "
        f"sidecar=recon/cloud_offtile.ply  replace_product={replace_product}"
    )


@main.command("roofs")
@click.argument("name")
@click.option(
    "--min-area",
    "min_area_m2",
    default=8.0,
    show_default=True,
    type=float,
    help="Minimum footprint area (m²) after sat segmentation",
)
@click.option(
    "--edge-gate",
    "edge_gate_m",
    default=1.0,
    show_default=True,
    type=float,
    help="Target mean shell↔sat-edge distance (m); logs gate_target_ok",
)
def roofs_cmd(name: str, min_area_m2: float, edge_gate_m: float) -> None:
    """Sat-locked roof/yard shells from Ortho (PR-A). Writes recon/roofs.obj.

    XY from sat absolute ENU; Z from MA high-z in footprint or façade top (yards near ground; street/cam spill rejected).
    No BAG/OSM extruded shells. Fail-loud if ortho missing or zero footprints.
    """
    from ps1_hood.reconstruct.sat_roofs import SatRoofError, build_sat_roofs

    project = open_project(name)
    try:
        meta = build_sat_roofs(
            project.root,
            min_area_m2=float(min_area_m2),
            edge_gate_m=float(edge_gate_m),
        )
    except SatRoofError as exc:
        click.echo(f"roofs failed: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"roofs failed: {exc}", err=True)
        raise SystemExit(1) from exc

    scene_path = project.recon_dir / "scene.json"
    if scene_path.is_file():
        import json

        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        payload["roofs"] = meta
        scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(
        f"roofs ok  roofs={meta.get('n_roof')}  yards={meta.get('n_yard')}  "
        f"textured={meta.get('textured')}  mean_edge_m={meta.get('mean_edge_m')}  "
        f"p90_edge_m={meta.get('p90_edge_m')}  gate_target_ok={meta.get('gate_target_ok')}  "
        f"obj={meta.get('obj')}"
    )



@main.command("street")
@click.argument("name")
@click.option(
    "--min-area-m2",
    default=8.0,
    show_default=True,
    type=float,
    help="Minimum street tile area (m²)",
)
@click.option(
    "--building-dilate-m",
    default=1.5,
    show_default=True,
    type=float,
    help="Dilate roof∪yard punch (m) so walls get no ground slab",
)
@click.option(
    "--cam-corridor-m",
    default=4.0,
    show_default=True,
    type=float,
    help="Optional cam XY corridor dilate (m); 0 disables",
)
@click.option(
    "--camera-height-m",
    default=2.5,
    show_default=True,
    type=float,
    help="ground_z = median(cam_u) − this (m)",
)
def street_cmd(
    name: str,
    min_area_m2: float,
    building_dilate_m: float,
    cam_corridor_m: float,
    camera_height_m: float,
) -> None:
    """Sat-locked street/ground shells from Ortho. Writes recon/street.obj.

    street_mask − dilated roof∪yard → flat ENU tiles at ground_z (planes or
    cam_u−height), sat-crop textured. Never touches façades.* / roofs.* / planes.json.
    """
    from ps1_hood.reconstruct.sat_street import SatStreetError, build_sat_street

    project = open_project(name)
    try:
        meta = build_sat_street(
            project.root,
            min_area_m2=float(min_area_m2),
            building_dilate_m=float(building_dilate_m),
            cam_corridor_m=float(cam_corridor_m),
            camera_height_m=float(camera_height_m),
        )
    except SatStreetError as exc:
        click.echo(f"street failed: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"street failed: {exc}", err=True)
        raise SystemExit(1) from exc

    scene_path = project.recon_dir / "scene.json"
    if scene_path.is_file():
        import json

        payload = json.loads(scene_path.read_text(encoding="utf-8"))
        payload["street"] = meta
        scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    click.echo(
        f"street ok  tiles={meta.get('n_street')}  textured={meta.get('textured')}  "
        f"ground_z={meta.get('ground_z')}  z_source={meta.get('z_source')}  "
        f"obj={meta.get('obj')}"
    )



@main.command("compare")
@click.argument("name")
@click.option(
    "--max-cams",
    default=40,
    show_default=True,
    type=int,
    help="Max horizon cameras after (pano,heading≈45°) dedupe",
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output dir (default: <run>/recon/compare)",
)
@click.option(
    "--patch",
    default=64,
    show_default=True,
    type=int,
    help="Ortho patch size for ZNCC",
)
def compare_cmd(name: str, max_cams: int, out_dir: Path | None, patch: int) -> None:
    """Diagnose-only: reproject façades+roofs into SV; ZNCC + edge + sat Chamfer.

    Writes recon/compare/*_hNNN.jpg overlays (photo | mesh | absdiff) and
    summary.json ranked by worst ZNCC. Does not densify, free-pose, or wipe product.
    """
    from ps1_hood.reconstruct.compare import CompareError, run_compare

    project = open_project(name)
    try:
        summary = run_compare(
            project.root,
            max_cams=int(max_cams),
            out_dir=out_dir,
            patch=int(patch),
        )
    except CompareError as exc:
        click.echo(f"compare failed: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"compare failed: {exc}", err=True)
        raise SystemExit(1) from exc

    g = summary.get("global") or {}
    click.echo(
        f"compare ok  cams={summary.get('n_cams_scored')}/"
        f"{summary.get('n_cams_considered')}  "
        f"quads={summary.get('n_quads_product')}  "
        f"facade_zncc={g.get('facade_zncc_mean')}  "
        f"roof_zncc={g.get('roof_zncc_mean')}  "
        f"facade_edge={g.get('facade_edge_mean')}  "
        f"sat_edge_m={g.get('sat_edge_mean_m')}  "
        f"soft_warn={g.get('soft_zncc_warn')}  "
        f"out={summary.get('out_dir')}"
    )
    click.echo("worst cams (lowest façade zncc):")
    for w in summary.get("worst") or []:
        kinds = ",".join(w.get("kinds") or []) or "?"
        fz = w.get("facade_zncc_mean")
        fz_s = f"{fz:.3f}" if fz == fz else "nan"
        fe = w.get("facade_edge_mean", w.get("edge_mean"))
        fe_s = f"{fe:.3f}" if fe == fe else "nan"
        click.echo(
            f"  {w['id']}  facade_zncc={fz_s}  "
            f"facade_edge={fe_s}  kinds={kinds}  "
            f"quads={w['n_quads']}  overlay={w['overlay']}"
        )


@main.command("sculpt-apply")
@click.argument("name")
@click.option("--plane", "plane_id", required=True, help="facade_XX id")
@click.option("--delta-d", type=float, default=None, help="Nudge along normal (m), clamp ±3")
@click.option("--delta-t", type=float, default=None, help="Optional tangent slide (m), clamp ±2")
@click.option("--width-m", type=float, default=None)
@click.option("--height-m", type=float, default=None)
@click.option("--bake/--no-bake", default=True, show_default=True)
@click.option("--bake-cam", default=None, help="pano_id of known posed cam")
@click.option("--margin-px", type=float, default=120.0, show_default=True)
def sculpt_apply_cmd(
    name: str,
    plane_id: str,
    delta_d: float | None,
    delta_t: float | None,
    width_m: float | None,
    height_m: float | None,
    bake: bool,
    bake_cam: str | None,
    margin_px: float,
) -> None:
    """ENU façade sculpt: bak + write one plane (no free-pose / no wipe)."""
    from ps1_hood.reconstruct.keyframes import load_keyframes
    from ps1_hood.reconstruct.sculpt import apply_sculpt

    project = open_project(name)
    frames = load_keyframes(project)
    for fr in frames:
        if "path" not in fr and fr.get("shot_path"):
            fr["path"] = fr["shot_path"]
    try:
        meta = apply_sculpt(
            project.recon_dir,
            plane_id,
            delta_d=delta_d,
            delta_t=delta_t,
            width_m=width_m,
            height_m=height_m,
            bake=bake,
            bake_cam=bake_cam,
            frames=frames,
            margin_px=margin_px,
        )
    except Exception as exc:
        click.echo(f"sculpt-apply failed: {exc}", err=True)
        raise SystemExit(1) from exc
    pl = meta.get("plane") or {}
    click.echo(
        f"sculpt-apply ok  plane={meta.get('plane_id')}  "
        f"d={pl.get('d')}  w={pl.get('width_m')}  h={pl.get('height_m')}  "
        f"bake={meta.get('bake_ok')}  planes={meta.get('plane_count')}  "
        f"bak={meta.get('bak', {}).get('stamp')}"
    )
    if meta.get("bake_error"):
        click.echo(f"  bake note: {meta['bake_error']}")


@main.command("sculpt-undo")
@click.argument("name")
@click.option("--stamp", default=None, help="bak_sculpt stamp; default=latest")
def sculpt_undo_cmd(name: str, stamp: str | None) -> None:
    """Restore latest bak_sculpt_* (planes/obj/mtl/textures)."""
    from ps1_hood.reconstruct.sculpt import undo_sculpt

    project = open_project(name)
    try:
        meta = undo_sculpt(project.recon_dir, stamp=stamp)
    except Exception as exc:
        click.echo(f"sculpt-undo failed: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"sculpt-undo ok  stamp={meta.get('stamp')}  "
        f"restored={len(meta.get('restored') or [])}"
    )




@main.group("sat-offset")
def sat_offset_group() -> None:
    """Forced SE(2): Chamfer, cam→road, Studio picks, or apply (bak first)."""


@sat_offset_group.command("measure")
@click.argument("name")
@click.option(
    "--from",
    "from_mode",
    default="facades",
    type=click.Choice(["facades", "cams-road"], case_sensitive=False),
    show_default=True,
    help="facades=Chamfer façade↔Canny; cams-road=cam→street centerline (alias measure-cams)",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output json (default: align/T_force.json or align/T_cam_road.json)",
)
@click.option("--search-r-m", default=None, type=float, help="Search radius m (facades default 8; cams-road 15)")
@click.option("--min-len-m", default=3.0, show_default=True, type=float)
@click.option("--max-rms-m", default=None, type=float, help="RMS gate m (facades 1.5; cams-road 2.0)")
@click.option("--min-pairs", default=4, show_default=True, type=int)
@click.option("--max-yaw-deg", default=None, type=float)
@click.option("--max-translation-m", default=None, type=float)
@click.option(
    "--multistart/--no-multistart",
    default=False,
    show_default=True,
    help="Optional Chamfer multi-start polish (facades only)",
)
@click.option(
    "--overlay/--no-overlay",
    default=False,
    show_default=True,
    help="Write overlay PNG (cams-road only)",
)
def sat_offset_measure_cmd(
    name: str,
    from_mode: str,
    out_path: Path | None,
    search_r_m: float | None,
    min_len_m: float,
    max_rms_m: float | None,
    min_pairs: int,
    max_yaw_deg: float | None,
    max_translation_m: float | None,
    multistart: bool,
    overlay: bool,
) -> None:
    """Measure one rigid SE(2). Default: façade↔Ortho Canny → T_force.json.

    Use ``--from cams-road`` (or ``measure-cams``) for cam track → street
    centerline → T_cam_road.json. Does **not** apply.
    """
    from ps1_hood.align.sat_offset import (
        SatOffsetError,
        measure_cam_road_se2,
        measure_facade_sat_se2,
        persist_t_cam_road,
        write_t_force,
    )

    project = open_project(name)
    mode = (from_mode or "facades").lower()
    if mode == "cams-road":
        try:
            payload = measure_cam_road_se2(
                project,
                search_r_m=float(search_r_m) if search_r_m is not None else 15.0,
                max_rms_m=float(max_rms_m) if max_rms_m is not None else 2.0,
                min_pairs=int(min_pairs),
                max_yaw_deg=float(max_yaw_deg) if max_yaw_deg is not None else 10.0,
                max_translation_m=float(max_translation_m) if max_translation_m is not None else 12.0,
            )
            paths = persist_t_cam_road(
                project, payload, t_path=out_path, overlay=bool(overlay)
            )
        except SatOffsetError as exc:
            click.echo(f"sat-offset measure failed: {exc}", err=True)
            raise SystemExit(1) from exc
        t_norm = float(payload.get("t_norm_m", 0.0))
        click.echo(
            f"sat-offset measure ok  source=cam_street_centerline  "
            f"tx={payload['tx_m']:.3f}  ty={payload['ty_m']:.3f}  "
            f"yaw={payload['yaw_deg']:.3f}  rms={payload['rms_m']:.3f}  "
            f"n_cams={payload['n_cams']}  n_pairs={payload['n_pairs']}  "
            f"||t||={t_norm:.3f}  applied=false  out={paths['T_cam_road']}"
        )
        return

    dest = Path(out_path) if out_path else (project.align_dir / "T_force.json")
    if not dest.is_absolute():
        dest = project.root / dest
    try:
        payload = measure_facade_sat_se2(
            project,
            search_r_m=float(search_r_m) if search_r_m is not None else 8.0,
            min_len_m=float(min_len_m),
            max_rms_m=float(max_rms_m) if max_rms_m is not None else 1.5,
            min_pairs=int(min_pairs),
            max_yaw_deg=float(max_yaw_deg) if max_yaw_deg is not None else 15.0,
            max_translation_m=float(max_translation_m) if max_translation_m is not None else 10.0,
            multistart=bool(multistart),
        )
    except SatOffsetError as exc:
        click.echo(f"sat-offset measure failed: {exc}", err=True)
        raise SystemExit(1) from exc
    write_t_force(dest, payload)
    click.echo(
        f"sat-offset measure ok  tx={payload['tx_m']:.3f}  ty={payload['ty_m']:.3f}  "
        f"yaw={payload['yaw_deg']:.3f}  rms={payload['rms_m']:.3f}  "
        f"n_pairs={payload['n_pairs']}  out={dest}"
    )


@sat_offset_group.command("measure-cams")
@click.argument("name")
@click.option(
    "--out",
    "out_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output T_cam_road.json (default: <run>/align/T_cam_road.json)",
)
@click.option(
    "--search-r",
    "search_r_m",
    default=15.0,
    show_default=True,
    type=float,
    help="Max cam→centerline NN distance (m)",
)
@click.option("--min-nn-m", default=0.5, show_default=True, type=float)
@click.option("--max-rms-m", default=2.0, show_default=True, type=float)
@click.option("--min-pairs", default=4, show_default=True, type=int)
@click.option("--max-yaw-deg", default=10.0, show_default=True, type=float)
@click.option("--max-translation-m", default=12.0, show_default=True, type=float)
@click.option(
    "--yaw-zero-deg",
    default=2.0,
    show_default=True,
    type=float,
    help="If |yaw| below this, fit tx,ty only (yaw=0)",
)
@click.option(
    "--se2-mode",
    default="auto",
    type=click.Choice(["auto", "translation", "full"], case_sensitive=False),
    show_default=True,
    help="auto: yaw=0 if |yaw|≤yaw-zero-deg; translation: force yaw=0; full: always SE(2)",
)
@click.option(
    "--overlay/--no-overlay",
    default=False,
    show_default=True,
    help="Write align/T_cam_road_overlay.png (red cams → cyan mapped)",
)
def sat_offset_measure_cams_cmd(
    name: str,
    out_path: Path | None,
    search_r_m: float,
    min_nn_m: float,
    max_rms_m: float,
    min_pairs: int,
    max_yaw_deg: float,
    max_translation_m: float,
    yaw_zero_deg: float,
    se2_mode: str,
    overlay: bool,
) -> None:
    """Measure SE(2) from unique cam XY → Ortho street_mask centerline.

    Writes align/T_cam_road.json. Does **not** apply — use
    `sat-offset apply --from align/T_cam_road.json` after Studio preview.
    One rigid SE(2) only — no free-pose. Skips roofs/street on apply.
    """
    from ps1_hood.align.sat_offset import (
        SatOffsetError,
        measure_cam_road_se2,
        persist_t_cam_road,
    )

    project = open_project(name)
    try:
        mode = (se2_mode or "auto").lower()
        t_only = None if mode == "auto" else (mode == "translation")
        payload = measure_cam_road_se2(
            project,
            search_r_m=float(search_r_m),
            min_nn_m=float(min_nn_m),
            max_rms_m=float(max_rms_m),
            min_pairs=int(min_pairs),
            max_yaw_deg=float(max_yaw_deg),
            max_translation_m=float(max_translation_m),
            yaw_zero_deg=float(yaw_zero_deg),
            translation_only=t_only,
        )
        paths = persist_t_cam_road(
            project,
            payload,
            t_path=out_path,
            overlay=bool(overlay),
        )
    except SatOffsetError as exc:
        click.echo(f"sat-offset measure-cams failed: {exc}", err=True)
        raise SystemExit(1) from exc
    t_norm = float(payload.get("t_norm_m", 0.0))
    click.echo(
        f"sat-offset measure-cams ok  tx={payload['tx_m']:.3f}  ty={payload['ty_m']:.3f}  "
        f"yaw={payload['yaw_deg']:.3f}  rms={payload['rms_m']:.3f}  "
        f"n_cams={payload['n_cams']}  n_pairs={payload['n_pairs']}  "
        f"mean_nn_m={payload['mean_nn_m']:.3f}  ||t||={t_norm:.3f}  "
        f"translation_only={payload.get('translation_only')}  "
        f"applied=false  out={paths['T_cam_road']}"
        + (f"  overlay={paths['overlay']}" if paths.get("overlay") else "")
    )


@sat_offset_group.command("fit-pairs")
@click.argument("name")
@click.option(
    "--pairs",
    "pairs_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="JSON list of {yellow:{e,n}, red:{e,n}} (or {pairs:[...]})",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output T_pick.json (default: <run>/align/T_pick.json)",
)
@click.option("--max-rms-m", default=1.5, show_default=True, type=float)
@click.option("--min-pairs", default=3, show_default=True, type=int)
@click.option("--max-yaw-deg", default=15.0, show_default=True, type=float)
@click.option("--max-translation-m", default=10.0, show_default=True, type=float)
def sat_offset_fit_pairs_cmd(
    name: str,
    pairs_path: Path,
    out_path: Path | None,
    max_rms_m: float,
    min_pairs: int,
    max_yaw_deg: float,
    max_translation_m: float,
) -> None:
    """Fit SE(2) from ≥3 Studio yellow↔red corner pairs (preview only).

    Writes align/T_pick.json + T_pick_pairs.json. Does **not** apply —
    use `sat-offset apply --from align/T_pick.json` after visual confirm.
    """
    from ps1_hood.align.sat_offset import (
        SatOffsetError,
        fit_pairs_se2,
        load_pairs_json,
        persist_t_pick,
    )

    project = open_project(name)
    try:
        pairs = load_pairs_json(Path(pairs_path))
        payload = fit_pairs_se2(
            pairs,
            max_rms_m=float(max_rms_m),
            min_pairs=int(min_pairs),
            max_yaw_deg=float(max_yaw_deg),
            max_translation_m=float(max_translation_m),
        )
        paths = persist_t_pick(project, payload, t_path=out_path)
    except SatOffsetError as exc:
        click.echo(f"sat-offset fit-pairs failed: {exc}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"sat-offset fit-pairs ok  tx={payload['tx_m']:.3f}  ty={payload['ty_m']:.3f}  "
        f"yaw={payload['yaw_deg']:.3f}  rms={payload['rms_m']:.3f}  "
        f"n_pairs={payload['n_pairs']}  applied=false  "
        f"T_pick={paths['T_pick']}  pairs={paths['T_pick_pairs']}"
    )


@sat_offset_group.command("apply")
@click.argument("name")
@click.option(
    "--from",
    "from_path",
    default=None,
    type=click.Path(path_type=Path),
    help="T json (default: align/T_force.json; also T_pick.json / T_cam_road.json)",
)
@click.option(
    "--targets",
    default="cams,cloud,facades,planes",
    show_default=True,
    help="Comma list: cams,cloud,facades,planes[,roofs,street]",
)
@click.option(
    "--skip",
    default="roofs,street",
    show_default=True,
    help="Comma list to leave alone (sat-native roofs/street by default)",
)
@click.option("--bak/--no-bak", default=True, show_default=True)
def sat_offset_apply_cmd(
    name: str,
    from_path: Path | None,
    targets: str,
    skip: str,
    bak: bool,
) -> None:
    """Bak then apply one forced SE(2) to cams+cloud+façades/planes.

    Skips roofs/street by default. Undo: restore align.bak_force_* / recon/bak_force_*.
    """
    from ps1_hood.align.sat_offset import SatOffsetError, apply_forced_se2, load_t_force

    project = open_project(name)
    src = Path(from_path) if from_path else (project.align_dir / "T_force.json")
    if not src.is_absolute():
        src = project.root / src
    try:
        T = load_t_force(src)
        meta = apply_forced_se2(
            project,
            T,
            targets=targets,
            skip=skip,
            bak=bool(bak),
        )
    except SatOffsetError as exc:
        click.echo(f"sat-offset apply failed: {exc}", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"sat-offset apply failed: {exc}", err=True)
        raise SystemExit(1) from exc
    stats = meta.get("stats") or {}
    bak_meta = meta.get("bak") or {}
    click.echo(
        f"sat-offset apply ok  tx={meta['T']['tx_m']:.3f}  ty={meta['T']['ty_m']:.3f}  "
        f"yaw={meta['T']['yaw_deg']:.3f}  targets={','.join(meta['targets'])}  "
        f"skip={','.join(meta['skip'])}  "
        f"poses={stats.get('poses')}  cloud={stats.get('cloud.ply')}  "
        f"facades={stats.get('facades.obj')}  planes={stats.get('planes.json')}  "
        f"roofs={stats.get('roofs.obj')}  street={stats.get('street.obj')}  "
        f"bak={bak_meta.get('stamp')}"
    )


@main.command("studio")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8765, type=int)
def studio_cmd(host: str, port: int) -> None:
    """Leaflet bbox picker + Three.js point-cloud viewer."""
    from ps1_hood.studio.app import serve

    click.echo(f"studio  http://{host}:{port}")
    serve(host, port)


@main.command("setup-browser")
def setup_browser_cmd() -> None:
    """Download the bundled Chromium that screengrabs the 360 panos."""
    import subprocess

    click.echo("installing Playwright Chromium…")
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    click.echo("done. capture will drive this browser, not your desktop Chrome.")


@main.command("list")
def list_cmd() -> None:
    root = default_runs_root()
    if not root.is_dir():
        click.echo("no runs yet")
        return
    for child in sorted(root.iterdir()):
        if (child / "project.yaml").is_file():
            click.echo(child.name)


def main_argv(argv: list[str] | None = None) -> None:
    sys.argv = ["ps1hood", *(argv or sys.argv[1:])]
    main()
