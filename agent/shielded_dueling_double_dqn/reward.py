from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from engine import Map


REWARD_DICT = {
    "survival": 0.005,
    "win": 2.0,
    "rank_1": 1.0,
    "rank_2": 0.3,
    "rank_3": -0.5,
    "agent_death": -1.0,
    "self_destruction": -2.0,
    "box_destroyed": 0.2,
    "item_radius": 0.25,
    "item_capacity": 0.2,
    "enemy_kill": 1.5,
    "escape_danger": 0.1,
    "enter_danger": -0.05,
    "bad_bomb": -0.05,
    "dead_end": -0.05,
    "stop": -0.01,
    "oscillation": -0.03,
}

_OSC_HISTORY: dict[int, tuple[tuple[int, int], tuple[int, int] | None]] = {}


def _as_player_array(players) -> np.ndarray:
    if isinstance(players, dict):
        if not players:
            return np.zeros((0, 5), dtype=np.int32)
        max_idx = max(int(pid) for pid in players.keys())
        arr = np.zeros((max_idx + 1, 5), dtype=np.int32)
        for pid, player in players.items():
            arr[int(pid)] = np.asarray(player, dtype=np.int32).reshape(-1)[:5]
        return arr

    arr = np.asarray(players)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.int32, copy=False)


def _as_bomb_array(bombs) -> np.ndarray:
    if bombs is None:
        return np.zeros((0, 4), dtype=np.int32)
    arr = np.asarray(bombs)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        padded = np.zeros((arr.shape[0], 4), dtype=np.int32)
        padded[:, : arr.shape[1]] = arr.astype(np.int32, copy=False)
        return padded
    return arr[:, :4].astype(np.int32, copy=False)


def _parse_bomb_row(bomb) -> tuple[int, int, int, int] | None:
    arr = np.asarray(bomb, dtype=np.int32).reshape(-1)
    if arr.size < 2:
        return None
    bx, by = int(arr[0]), int(arr[1])
    timer = int(arr[2]) if arr.size > 2 else 7
    owner_id = int(arr[3]) if arr.size > 3 else 0
    return bx, by, timer, owner_id


def _bomb_radius_from_obs(players, owner_id: int) -> int:
    arr = _as_player_array(players)
    if 0 <= owner_id < len(arr):
        return 1 + int(arr[owner_id][4])
    return 1


def _in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]


def _explosion_tiles_for_bomb(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for dist in range(1, radius + 1):
            tx, ty = bx + dx * dist, by + dy * dist
            if not _in_bounds(grid, tx, ty):
                break
            cell = int(grid[tx, ty])
            if cell == Map.WALL:
                break
            tiles.add((tx, ty))
            if cell == Map.BOX:
                break
    return tiles


def _bombs_in_blast(obs, x: int, y: int) -> tuple[bool, int | None]:
    bombs = _as_bomb_array(obs.get("bombs"))
    if bombs.size == 0:
        return False, None

    grid = np.asarray(obs["map"], dtype=np.int32)
    players = obs["players"]
    hit = False
    min_timer = None
    for bomb in bombs:
        parsed = _parse_bomb_row(bomb)
        if parsed is None:
            continue
        bx, by, timer, owner_id = parsed
        radius = _bomb_radius_from_obs(players, owner_id)
        blast = _explosion_tiles_for_bomb(grid, bx, by, radius)
        if (x, y) in blast:
            hit = True
            min_timer = timer if min_timer is None else min(min_timer, timer)
    return hit, min_timer


def _enemy_alive_count(players, agent_id: int) -> int:
    arr = _as_player_array(players)
    return sum(1 for pid, row in enumerate(arr) if pid != agent_id and int(row[2]) == 1)


def _box_count(grid: np.ndarray) -> int:
    return int(np.sum(grid == Map.BOX))


def _is_dead_end(grid: np.ndarray, x: int, y: int) -> bool:
    if not _in_bounds(grid, x, y):
        return True
    open_neighbors = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if _in_bounds(grid, nx, ny) and int(grid[nx, ny]) in (Map.GRASS, Map.ITEM_RADIUS, Map.ITEM_CAPACITY):
            open_neighbors += 1
    return open_neighbors <= 1


def _oscillation_penalty(agent_id: int, prev_pos: tuple[int, int], curr_pos: tuple[int, int], reset: bool = False) -> float:
    if reset:
        _OSC_HISTORY.pop(agent_id, None)
        return 0.0

    last = _OSC_HISTORY.get(agent_id)
    penalty = 0.0
    if last is not None:
        prev_prev, prev_last = last
        if prev_prev == curr_pos and prev_last == prev_pos:
            penalty = REWARD_DICT["oscillation"]
    _OSC_HISTORY[agent_id] = (prev_pos, curr_pos)
    return penalty


def final_rank_bonus(obs, agent_id: int, survival_steps: np.ndarray | None = None) -> float:
    players = _as_player_array(obs["players"])
    if len(players) == 0:
        return 0.0

    if survival_steps is None:
        survival_steps = np.zeros(len(players), dtype=np.float32)

    ranking = sorted(
        range(len(players)),
        key=lambda pid: (
            int(players[pid][2]),
            float(survival_steps[pid]),
            int(players[pid][3]),
            int(players[pid][0]),
            int(players[pid][1]),
        ),
        reverse=True,
    )
    rank = ranking.index(int(agent_id))
    bonuses = [REWARD_DICT["win"], REWARD_DICT["rank_1"], REWARD_DICT["rank_2"], REWARD_DICT["rank_3"]]
    return float(bonuses[rank]) if rank < len(bonuses) else float(bonuses[-1])


def compute_reward(prev_obs, curr_obs, agent_id: int) -> float:
    if prev_obs is None:
        _oscillation_penalty(agent_id, (0, 0), (0, 0), reset=True)
        return 0.0

    prev_players = _as_player_array(prev_obs["players"])
    curr_players = _as_player_array(curr_obs["players"])
    prev_grid = np.asarray(prev_obs["map"], dtype=np.int32)
    curr_grid = np.asarray(curr_obs["map"], dtype=np.int32)

    prev_alive = int(prev_players[agent_id][2])
    curr_alive = int(curr_players[agent_id][2])
    prev_pos = (int(prev_players[agent_id][0]), int(prev_players[agent_id][1]))
    curr_pos = (int(curr_players[agent_id][0]), int(curr_players[agent_id][1]))

    if prev_alive == 1 and curr_alive == 0:
        in_own_blast, _ = _bombs_in_blast(prev_obs, prev_pos[0], prev_pos[1])
        return float(REWARD_DICT["self_destruction"] if in_own_blast else REWARD_DICT["agent_death"])

    reward = float(REWARD_DICT["survival"] if curr_alive == 1 else 0.0)

    prev_enemy_alive = _enemy_alive_count(prev_players, agent_id)
    curr_enemy_alive = _enemy_alive_count(curr_players, agent_id)
    if curr_enemy_alive < prev_enemy_alive:
        reward += REWARD_DICT["enemy_kill"] * (prev_enemy_alive - curr_enemy_alive)
    if curr_enemy_alive == 0 and prev_enemy_alive > 0:
        reward += REWARD_DICT["win"]

    prev_box_count = _box_count(prev_grid)
    curr_box_count = _box_count(curr_grid)
    if curr_box_count < prev_box_count:
        reward += REWARD_DICT["box_destroyed"] * (prev_box_count - curr_box_count)

    prev_item = prev_grid[curr_pos[0], curr_pos[1]] if _in_bounds(prev_grid, curr_pos[0], curr_pos[1]) else Map.GRASS
    curr_radius = int(curr_players[agent_id][4])
    prev_radius = int(prev_players[agent_id][4])
    prev_bombs_left = int(prev_players[agent_id][3])
    curr_bombs_left = int(curr_players[agent_id][3])
    if prev_item == Map.ITEM_RADIUS and curr_radius > prev_radius:
        reward += REWARD_DICT["item_radius"]
    if prev_item == Map.ITEM_CAPACITY and curr_bombs_left > prev_bombs_left:
        reward += REWARD_DICT["item_capacity"]

    prev_in_blast, _ = _bombs_in_blast(prev_obs, prev_pos[0], prev_pos[1])
    curr_in_blast, _ = _bombs_in_blast(curr_obs, curr_pos[0], curr_pos[1])
    if prev_in_blast and not curr_in_blast:
        reward += REWARD_DICT["escape_danger"]
    elif not prev_in_blast and curr_in_blast and (prev_pos != curr_pos):
        reward += REWARD_DICT["enter_danger"]

    if prev_pos == curr_pos and prev_bombs_left == curr_bombs_left and curr_alive == 1:
        reward += REWARD_DICT["stop"]

    if curr_alive == 1 and _is_dead_end(curr_grid, curr_pos[0], curr_pos[1]) and curr_in_blast:
        reward += REWARD_DICT["dead_end"]

    if curr_bombs_left < prev_bombs_left:
        blast = _explosion_tiles_for_bomb(curr_grid, curr_pos[0], curr_pos[1], 1 + int(curr_players[agent_id][4]))
        bomb_value = any(int(curr_grid[x, y]) == Map.BOX for x, y in blast)
        if not bomb_value:
            bomb_value = any(
                int(row[2]) == 1 and (int(row[0]), int(row[1])) in blast
                for pid, row in enumerate(curr_players)
                if pid != agent_id
            )
        if not bomb_value:
            reward += REWARD_DICT["bad_bomb"]

    reward += _oscillation_penalty(agent_id, prev_pos, curr_pos)
    return float(reward)
