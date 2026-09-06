"""On-disk run folder layout."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ps1_hood.config import ProjectSpec

RUNS_DIRNAME = "runs"


class Project:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.spec_path = self.root / "project.yaml"

    @property
    def osm_dir(self) -> Path:
        return self.root / "osm"

    @property
    def satellite_dir(self) -> Path:
        return self.root / "satellite"

    @property
    def bag_dir(self) -> Path:
        return self.root / "bag"

    @property
    def live_path(self) -> Path:
        return self.root / "live.json"

    @property
    def discover_dir(self) -> Path:
        return self.root / "discover"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def cropped_dir(self) -> Path:
        return self.root / "cropped"

    @property
    def align_dir(self) -> Path:
        return self.root / "align"

    @property
    def interp_dir(self) -> Path:
        return self.root / "interp"

    @property
    def recon_dir(self) -> Path:
        return self.root / "recon"

    def ensure_dirs(self) -> None:
        for d in (
            self.osm_dir,
            self.satellite_dir,
            self.bag_dir,
            self.discover_dir,
            self.raw_dir,
            self.cropped_dir,
            self.align_dir,
            self.interp_dir,
            self.recon_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def load_spec(self) -> ProjectSpec:
        data = yaml.safe_load(self.spec_path.read_text(encoding="utf-8"))
        return ProjectSpec.from_dict(data)

    def save_spec(self, spec: ProjectSpec) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_dirs()
        self.spec_path.write_text(
            yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8"
        )

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_json(self, path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))


def default_runs_root(cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    return base / RUNS_DIRNAME


def create_project(spec: ProjectSpec, runs_root: Path | None = None) -> Project:
    root = (runs_root or default_runs_root()) / spec.name
    project = Project(root)
    project.save_spec(spec)
    return project


def open_project(name_or_path: str, runs_root: Path | None = None) -> Project:
    path = Path(name_or_path)
    if path.is_dir() and (path / "project.yaml").is_file():
        return Project(path)
    root = (runs_root or default_runs_root()) / name_or_path
    if not (root / "project.yaml").is_file():
        raise FileNotFoundError(f"no project at {root}")
    return Project(root)
