"""Command line for ps1-hood."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ps1_hood import __version__
from ps1_hood.config import Settings
from ps1_hood.geo import BBox
from ps1_hood.pipeline import STAGES, run_all  # includes bag + 3DBAG edge snap
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
def run_cmd(name: str, from_stage: str) -> None:
    """Run the pipeline (or resume from a stage)."""
    settings = Settings.from_env()
    project = open_project(name)
    click.echo(f"run {project.root}  from {from_stage}")
    try:
        run_all(project, settings, from_stage=from_stage)
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
def align_cmd(name: str) -> None:
    from ps1_hood.pipeline import stage_align

    stage_align(open_project(name))


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
