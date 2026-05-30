from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

from config import BASELINE_FILES, MAX_PLAYERS, ROOT_DIR


def ensure_import_paths() -> None:
    for path in (ROOT_DIR, ROOT_DIR / "agent"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_agent_from_file(agent_path: Path, agent_id: int):
    module_name = f"baseline_{agent_path.stem}_{agent_id}_{abs(hash(str(agent_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith("Agent"):
                agent_cls = attr
                break
    if agent_cls is None:
        raise AttributeError(f"No Agent class found in {agent_path}")

    try:
        return agent_cls(agent_id)
    except TypeError:
        return agent_cls()


def build_builtin_agent(name: str, agent_id: int):
    path = BASELINE_FILES.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Unknown or missing baseline agent: {name}")
    ensure_import_paths()
    return load_agent_from_file(path, agent_id)


def load_opponent_pool(opponent_paths: tuple[str, ...]) -> list[tuple[str, str]]:
    pool: list[tuple[str, str]] = []
    for name in ("simple", "smarter", "tactical", "genius", "box_farmer", "random"):
        pool.append(("builtin", name))
    for path in opponent_paths:
        pool.append(("path", path))
    return pool


def spawn_opponents(enemy_type: str, opponent_paths: tuple[str, ...] = ()) -> list[object]:
    agents: list[object] = [None] * MAX_PLAYERS
    if enemy_type == "league":
        pool = load_opponent_pool(opponent_paths)
        for pid in range(1, MAX_PLAYERS):
            kind, value = random.choice(pool)
            if kind == "builtin":
                agents[pid] = build_builtin_agent(value, pid)
                continue

            path = Path(value)
            if path.is_dir():
                path = path / "agent.py"
            ensure_import_paths()
            agents[pid] = load_agent_from_file(path, pid)
        return agents

    for pid in range(1, MAX_PLAYERS):
        agents[pid] = build_builtin_agent(enemy_type, pid)
    return agents

