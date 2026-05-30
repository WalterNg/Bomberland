from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from config import (
    CURRENT_DIR,
    LEGACY_RUNS_DIR,
    LEGACY_WEIGHTS_DIR,
    load_agent_config,
    resolve_config_path,
)


RUN_DIR_PATTERN = re.compile(r"^(?P<date>\d{6})-(?P<time>\d{6})$")
CHECKPOINT_PATTERN = re.compile(r"^(?P<name>\d+|last)\.pth$")


def _timestamp() -> str:
    return datetime.now().strftime("%d%m%y-%H%M%S")


def _run_datetime(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.name, "%d%m%y-%H%M%S")
    except ValueError:
        return None


def _configured_paths(config_path: str | Path | None = None) -> tuple[Path, Path, Path, Path | None]:
    cfg, cfg_path = load_agent_config(config_path)
    base_dir = cfg_path.parent

    paths = cfg.get("paths", {})
    inference = cfg.get("inference", {})

    runs_dir = resolve_config_path(paths.get("runs_dir", "runs"), base_dir) or (CURRENT_DIR / "runs")
    weights_dir = resolve_config_path(paths.get("weights_dir", "weights"), base_dir) or (CURRENT_DIR / "weights")
    bootstrap_checkpoint = resolve_config_path(
        paths.get("bootstrap_checkpoint", "weights/v1_280526_154105/v1.pth"),
        base_dir,
    )
    configured_checkpoint = resolve_config_path(inference.get("checkpoint_path"), base_dir)
    return runs_dir, weights_dir, bootstrap_checkpoint or (weights_dir / "v1_280526_154105" / "v1.pth"), configured_checkpoint


def _existing_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)],
        key=lambda path: _run_datetime(path) or datetime.min,
    )


def _latest_checkpoint_in_root(root: Path) -> Path | None:
    run_dirs = _existing_run_dirs(root)
    if not run_dirs:
        return None

    for run_dir in reversed(run_dirs):
        last_path = last_checkpoint_path(run_dir)
        if last_path.exists():
            return last_path

        checkpoints = sorted(
            [
                path
                for path in run_dir.iterdir()
                if path.is_file() and CHECKPOINT_PATTERN.match(path.name) and path.name != "last.pth"
            ],
            key=lambda path: int(path.stem),
        )
        if checkpoints:
            return checkpoints[-1]

    return None


def create_run_dir(root: Path | None = None, config_path: str | Path | None = None) -> Path:
    if root is None:
        root, _, _, _ = _configured_paths(config_path)

    root.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = root / _timestamp()
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir


def ensure_run_dir(root: Path | None = None, config_path: str | Path | None = None) -> Path:
    if root is None:
        root, _, _, _ = _configured_paths(config_path)

    run_dirs = _existing_run_dirs(root)
    if run_dirs:
        return run_dirs[-1]
    return create_run_dir(root=root, config_path=config_path)


def checkpoint_path_for_episode(run_dir: Path, episode_number: int) -> Path:
    return run_dir / f"{int(episode_number)}.pth"


def last_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "last.pth"


def latest_checkpoint_path(
    root: Path | None = None,
    config_path: str | Path | None = None,
) -> Path | None:
    runs_dir, weights_dir, bootstrap_checkpoint, configured_checkpoint = _configured_paths(config_path)
    search_root = Path(root) if root is not None else runs_dir

    if configured_checkpoint is not None and configured_checkpoint.exists():
        return configured_checkpoint

    latest = _latest_checkpoint_in_root(search_root)
    if latest is not None:
        return latest

    if root is None and search_root != LEGACY_RUNS_DIR:
        legacy_latest = _latest_checkpoint_in_root(LEGACY_RUNS_DIR)
        if legacy_latest is not None:
            return legacy_latest

    if bootstrap_checkpoint.exists():
        return bootstrap_checkpoint

    legacy_bootstrap = LEGACY_WEIGHTS_DIR / "v1_280526_154105" / "v1.pth"
    if legacy_bootstrap.exists():
        return legacy_bootstrap

    legacy_weights_bootstrap = weights_dir / "v1_280526_154105" / "v1.pth"
    if legacy_weights_bootstrap.exists():
        return legacy_weights_bootstrap

    return None
