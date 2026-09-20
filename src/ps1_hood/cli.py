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
def align_cmd(
    name: str,
    align_prior: str,
    sat_edge_weight: float,
    cloud_clip_sat: bool,
    sat_cloud_margin_m: float,
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
    )


@main.command("interpolate")
@click.argument("name")
def interpolate_cmd(name: str) -> None:
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
) -> None:
    """Optional densify: OpenMVS (CPU/AGPL) or MapAnything (CUDA, ENU-locked).

    OpenMVS: InterfaceCOLMAP + DensifyPointCloud on PATH → openmvs/scene_dense.ply.
    MapAnything: fixed ENU cam2world + K; never ignore_pose_inputs. Prefer
    --apache. Without CUDA use --export-only then scripts/run_mapanything_bundle.py.
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
        click.echo(
            f"mapanything bundle  {bundle_meta['path']}  "
            f"views={bundle_meta['n_views']}  pose_lock=True"
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
            )
        except Exception as exc:
            click.echo(f"densify failed: {exc}", err=True)
            click.echo(ma.INSTALL_HINT, err=True)
            raise SystemExit(1) from exc
        click.echo(
            f"densify ok  {meta.get('path') or meta.get('cloud_mapanything')}  "
            f"points={meta.get('points')}  backend=mapanything  "
            f"pose_lock={meta.get('pose_lock')}  path={meta.get('path_kind')}"
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
) -> None:
    """Path α: planarize dense ENU cloud → ZNCC-gated façades.obj + planes.json.

    Prefer MapAnything product PLY for plane seeds; still photo-ZNCC gate.
    No OSM/BAG hero. Residual organic omitted (no Poisson in α1).
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

    XY from sat absolute ENU; Z from MA cloud median in footprint or façade top.
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
