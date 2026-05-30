from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
CONFIG_FILENAME = "config.json"
CONFIG_PATH = CURRENT_DIR / CONFIG_FILENAME
LEGACY_WEIGHTS_DIR = ROOT_DIR / "weights"
LEGACY_RUNS_DIR = ROOT_DIR / "runs"
CHECKPOINT_ENV_VAR = "AGENT_CHECKPOINT_PATH"

DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "agent": {
        "team_id": "shielded_dueling_double_dqn",
    },
    "paths": {
        "runs_dir": "runs",
        "weights_dir": "weights",
        "model_filename": "shielded_dueling_double_dqn.pth",
        "bootstrap_checkpoint": "weights/v1_280526_154105/v1.pth",
    },
    "inference": {
        "checkpoint_path": "runs/280526-164057/last.pth",
        "device": "cpu",
    },
    "training": {
        "enemy_type": "league",
        "num_episodes": 200,
        "max_steps": 500,
        "seed": 86,
        "save_model": True,
        "pretrained_model": None,
        "opponent_paths": [],
        "batch_size": 128,
        "learning_starts": 10_000,
        "train_freq": 4,
        "target_update_interval": 4_000,
        "replay_capacity": 200_000,
        "n_step": 3,
        "gamma": 0.99,
        "learning_rate": 1e-4,
        "epsilon_start": 1.0,
        "epsilon_final": 0.05,
        "epsilon_decay_steps": 300_000,
        "beta_start": 0.4,
        "beta_final": 1.0,
        "beta_growth_steps": 300_000,
        "max_grad_norm": 10.0,
        "checkpoint_interval_episodes": 100,
    },
}

MODEL_FILENAME = DEFAULT_AGENT_CONFIG["paths"]["model_filename"]
MAX_PLAYERS = 4

BASELINE_FILES = {
    "simple": ROOT_DIR / "agent" / "simple_rule_agent.py",
    "smarter": ROOT_DIR / "agent" / "smarter_rule_agent.py",
    "tactical": ROOT_DIR / "agent" / "tactical_rule_agent.py",
    "genius": ROOT_DIR / "agent" / "genius_rule_agent.py",
    "box_farmer": ROOT_DIR / "agent" / "box_farmer_agent.py",
    "random": ROOT_DIR / "agent" / "random_agent.py",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_config_path(config_path: str | Path | None = None) -> Path:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if path.is_dir():
        return path / CONFIG_FILENAME
    return path


def load_agent_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """
    Load the Shielded Dueling Double DQN configuration from JSON.

    Returns a tuple of (merged_config, resolved_config_path).
    The resolved path is useful for resolving relative paths declared in JSON.
    """

    path = _normalize_config_path(config_path)
    merged = deepcopy(DEFAULT_AGENT_CONFIG)

    if not path.exists():
        return merged, path

    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    if not isinstance(loaded, dict):
        raise ValueError(f"Agent config must be a JSON object: {path}")

    return _deep_merge(merged, loaded), path


def resolve_config_path(raw_path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if raw_path is None:
        return None

    if isinstance(raw_path, str) and not raw_path.strip():
        return None

    candidate = Path(raw_path).expanduser()

    if candidate.is_absolute():
        return candidate

    base = Path(base_dir) if base_dir is not None else CURRENT_DIR
    return (base / candidate).resolve()


def training_config_from_json(config_path: str | Path | None = None) -> "TrainingConfig":
    cfg, _ = load_agent_config(config_path)
    training = cfg["training"]
    opponent_paths = tuple(training.get("opponent_paths") or ())
    pretrained_model = training.get("pretrained_model")

    return TrainingConfig(
        enemy_type=str(training["enemy_type"]),
        num_episodes=int(training["num_episodes"]),
        max_steps=int(training["max_steps"]),
        seed=int(training["seed"]),
        save_model=bool(training["save_model"]),
        pretrained_model=str(pretrained_model) if pretrained_model is not None else None,
        opponent_paths=opponent_paths,
        batch_size=int(training["batch_size"]),
        learning_starts=int(training["learning_starts"]),
        train_freq=int(training["train_freq"]),
        target_update_interval=int(training["target_update_interval"]),
        replay_capacity=int(training["replay_capacity"]),
        n_step=int(training["n_step"]),
        gamma=float(training["gamma"]),
        learning_rate=float(training["learning_rate"]),
        epsilon_start=float(training["epsilon_start"]),
        epsilon_final=float(training["epsilon_final"]),
        epsilon_decay_steps=int(training["epsilon_decay_steps"]),
        beta_start=float(training["beta_start"]),
        beta_final=float(training["beta_final"]),
        beta_growth_steps=int(training["beta_growth_steps"]),
        max_grad_norm=float(training["max_grad_norm"]),
        checkpoint_interval_episodes=int(training["checkpoint_interval_episodes"]),
    )

@dataclass(slots=True)
class TrainingConfig:
    enemy_type: str = "league"
    num_episodes: int = 200
    max_steps: int = 500
    seed: int = 86
    save_model: bool = True
    pretrained_model: str | None = None
    opponent_paths: tuple[str, ...] = ()
    batch_size: int = 128
    learning_starts: int = 10_000
    train_freq: int = 4
    target_update_interval: int = 4_000
    replay_capacity: int = 200_000
    n_step: int = 3
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 300_000
    beta_start: float = 0.4
    beta_final: float = 1.0
    beta_growth_steps: int = 300_000
    max_grad_norm: float = 10.0
    checkpoint_interval_episodes: int = 100
