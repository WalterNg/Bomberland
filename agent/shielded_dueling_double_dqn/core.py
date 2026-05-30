from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

import numpy as np
import torch
import torch.nn as nn


class Action:
    STOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    PLACE_BOMB = 5


MOVE_ACTIONS = (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN)
ACTION_PRIORITY = {
    Action.LEFT: 0,
    Action.RIGHT: 1,
    Action.UP: 2,
    Action.DOWN: 3,
    Action.STOP: 4,
    Action.PLACE_BOMB: 5,
}

MAX_BOMB_TIMER = 7.0
MAX_BOMB_RADIUS = 5.0
MAX_BOMB_CAPACITY = 5.0
MAX_DISTANCE = 24.0
SPATIAL_CHANNELS = 15
VECTOR_FEATURES = 10
ACTION_DIM = 6


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


def _in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]


def _next_pos(pos: tuple[int, int], action: int) -> tuple[int, int]:
    x, y = pos
    if action == Action.LEFT:
        return x - 1, y
    if action == Action.RIGHT:
        return x + 1, y
    if action == Action.UP:
        return x, y - 1
    if action == Action.DOWN:
        return x, y + 1
    return x, y


def _passable(grid: np.ndarray, x: int, y: int) -> bool:
    return _in_bounds(grid, x, y) and int(grid[x, y]) in (0, 3, 4)


def _player_row(players: np.ndarray, agent_id: int) -> np.ndarray | None:
    if agent_id < 0 or agent_id >= len(players):
        return None
    return players[agent_id]


def _bomb_radius(players: np.ndarray, owner_id: int) -> int:
    if 0 <= owner_id < len(players):
        return max(1, int(players[owner_id][4]) + 1)
    return 1


def _line_clear(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    sx, sy = start
    ex, ey = end
    if sx == ex:
        step = 1 if ey > sy else -1
        for y in range(sy + step, ey, step):
            if int(grid[sx, y]) in (1, 2):
                return False
        return True
    if sy == ey:
        step = 1 if ex > sx else -1
        for x in range(sx + step, ex, step):
            if int(grid[x, sy]) in (1, 2):
                return False
        return True
    return False


def _blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    tiles = {(bx, by)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for dist in range(1, radius + 1):
            tx, ty = bx + dx * dist, by + dy * dist
            if not _in_bounds(grid, tx, ty):
                break
            cell = int(grid[tx, ty])
            if cell == 1:
                break
            tiles.add((tx, ty))
            if cell == 2:
                break
    return tiles


def _danger_masks(grid: np.ndarray, bombs: np.ndarray, players: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    danger_1 = np.zeros_like(grid, dtype=np.float32)
    danger_2 = np.zeros_like(grid, dtype=np.float32)
    danger_3 = np.zeros_like(grid, dtype=np.float32)
    danger_4plus = np.zeros_like(grid, dtype=np.float32)

    for bomb in bombs:
        bx, by, timer, owner_id = (int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3]))
        radius = _bomb_radius(players, owner_id)
        blast = _blast_tiles(grid, bx, by, radius)

        if timer <= 1:
            target = danger_1
        elif timer <= 2:
            target = danger_2
        elif timer <= 3:
            target = danger_3
        else:
            target = danger_4plus

        for tx, ty in blast:
            target[tx, ty] = 1.0
            if timer <= 1:
                danger_2[tx, ty] = 1.0
                danger_3[tx, ty] = 1.0
            elif timer <= 2:
                danger_3[tx, ty] = 1.0

    return danger_1, danger_2, danger_3, danger_4plus


@dataclass(slots=True)
class ObservationBundle:
    spatial: np.ndarray
    vector: np.ndarray
    grid: np.ndarray
    players: np.ndarray
    bombs: np.ndarray
    agent_id: int
    my_pos: tuple[int, int]
    alive: bool
    bombs_left: int
    bomb_radius: int
    step_index: int
    danger_1: np.ndarray
    danger_2: np.ndarray
    danger_3: np.ndarray
    danger_4plus: np.ndarray
    safe_mask: np.ndarray


def encode_observation(obs: dict, agent_id: int, step_index: int = 0, max_steps: int = 500) -> ObservationBundle:
    grid = np.asarray(obs["map"], dtype=np.int32)
    players = _as_player_array(obs["players"])
    bombs = _as_bomb_array(obs["bombs"])

    player = _player_row(players, int(agent_id))
    if player is None:
        player = np.array([0, 0, 0, 0, 0], dtype=np.int32)

    alive = bool(int(player[2]) == 1)
    my_x, my_y = int(player[0]), int(player[1])
    bombs_left = int(player[3])
    bomb_radius = max(1, int(player[4]) + 1)

    wall = (grid == 1).astype(np.float32)
    box = (grid == 2).astype(np.float32)
    grass = (grid == 0).astype(np.float32)
    item_radius = (grid == 3).astype(np.float32)
    item_capacity = (grid == 4).astype(np.float32)

    own_pos = np.zeros_like(grid, dtype=np.float32)
    if alive and _in_bounds(grid, my_x, my_y):
        own_pos[my_x, my_y] = 1.0

    enemy_pos = np.zeros_like(grid, dtype=np.float32)
    enemy_targets: list[tuple[int, int]] = []
    for pid, row in enumerate(players):
        if pid == int(agent_id) or int(row[2]) != 1:
            continue
        ex, ey = int(row[0]), int(row[1])
        if _in_bounds(grid, ex, ey):
            enemy_pos[ex, ey] = 1.0
            enemy_targets.append((ex, ey))

    bomb_present = np.zeros_like(grid, dtype=np.float32)
    bomb_timer = np.zeros_like(grid, dtype=np.float32)
    bomb_radius_mask = np.zeros_like(grid, dtype=np.float32)
    for bomb in bombs:
        bx, by, timer, owner_id = (int(bomb[0]), int(bomb[1]), int(bomb[2]), int(bomb[3]))
        if not _in_bounds(grid, bx, by):
            continue
        bomb_present[bx, by] = 1.0
        bomb_timer[bx, by] = max(bomb_timer[bx, by], float(timer) / MAX_BOMB_TIMER)
        bomb_radius_mask[bx, by] = max(bomb_radius_mask[bx, by], float(_bomb_radius(players, owner_id)) / MAX_BOMB_RADIUS)

    danger_1, danger_2, danger_3, danger_4plus = _danger_masks(grid, bombs, players)
    safe_mask = ((grass + item_radius + item_capacity) > 0).astype(np.float32)
    safe_mask *= (bomb_present == 0).astype(np.float32)
    safe_mask *= (danger_3 == 0).astype(np.float32)

    spatial = np.stack(
        [
            wall,
            box,
            grass,
            item_radius,
            item_capacity,
            own_pos,
            enemy_pos,
            bomb_present,
            bomb_timer,
            bomb_radius_mask,
            danger_1,
            danger_2,
            danger_3,
            danger_4plus,
            safe_mask,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    boxes_remaining = float(np.sum(box))
    bombs_on_map = float(len(bombs))

    def _nearest_manhattan(origin: tuple[int, int], targets: Iterable[tuple[int, int]]) -> float:
        ox, oy = origin
        best = None
        for tx, ty in targets:
            dist = abs(ox - int(tx)) + abs(oy - int(ty))
            best = dist if best is None else min(best, dist)
        if best is None:
            return 1.0
        return float(min(best, MAX_DISTANCE) / MAX_DISTANCE)

    nearest_enemy = _nearest_manhattan((my_x, my_y), enemy_targets)
    item_targets = [(int(x), int(y)) for x, y in zip(*np.where((grid == 3) | (grid == 4)))]
    nearest_item = _nearest_manhattan((my_x, my_y), item_targets)
    box_targets = [(int(x), int(y)) for x, y in zip(*np.where(grid == 2))]
    nearest_box = _nearest_manhattan((my_x, my_y), box_targets)

    safe_move_count = 0.0
    if alive and _in_bounds(grid, my_x, my_y):
        for action in MOVE_ACTIONS:
            nx, ny = _next_pos((my_x, my_y), action)
            if _passable(grid, nx, ny) and bomb_present[nx, ny] == 0 and danger_1[nx, ny] == 0:
                safe_move_count += 1.0

    vector = np.array(
        [
            min(max(float(bombs_left), 0.0), MAX_BOMB_CAPACITY) / MAX_BOMB_CAPACITY,
            float(bomb_radius) / MAX_BOMB_RADIUS,
            min(float(len(enemy_targets)), 3.0) / 3.0,
            nearest_enemy,
            nearest_item,
            nearest_box,
            min(float(step_index), float(max_steps)) / max(1.0, float(max_steps)),
            min(boxes_remaining, 100.0) / 100.0,
            min(bombs_on_map, 20.0) / 20.0,
            safe_move_count / 4.0,
        ],
        dtype=np.float32,
    )

    return ObservationBundle(
        spatial=spatial,
        vector=vector,
        grid=grid,
        players=players,
        bombs=bombs,
        agent_id=int(agent_id),
        my_pos=(my_x, my_y),
        alive=alive,
        bombs_left=bombs_left,
        bomb_radius=bomb_radius,
        step_index=int(step_index),
        danger_1=danger_1,
        danger_2=danger_2,
        danger_3=danger_3,
        danger_4plus=danger_4plus,
        safe_mask=safe_mask,
    )


class DuelingDQN(nn.Module):
    def __init__(self, spatial_channels: int = SPATIAL_CHANNELS, vector_dim: int = VECTOR_FEATURES, action_dim: int = ACTION_DIM):
        super().__init__()
        self.spatial_channels = spatial_channels
        self.vector_dim = vector_dim
        self.action_dim = action_dim

        self.map_encoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy_map = torch.zeros(1, spatial_channels, 13, 13)
            conv_out_dim = self.map_encoder(dummy_map).reshape(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.map_encoder(spatial).reshape(spatial.shape[0], -1)
        vector_feat = self.vector_encoder(vector)
        fused = torch.cat([spatial_feat, vector_feat], dim=1)
        fused = self.head(fused)
        value = self.value_stream(fused)
        advantage = self.advantage_stream(fused)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def load_flexible_state_dict(self, checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> tuple[int, int]:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return 0, 0

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        source_state = checkpoint.get("model_state_dict", checkpoint)
        target_state = self.state_dict()
        legacy_aliases = {
            "head.4.weight": "advantage_stream.2.weight",
            "head.4.bias": "advantage_stream.2.bias",
        }

        copied = 0
        partial = 0
        for source_key, source_tensor in source_state.items():
            target_key = source_key
            if target_key not in target_state and target_key in legacy_aliases:
                target_key = legacy_aliases[target_key]
            if target_key not in target_state:
                continue

            target_tensor = target_state[target_key]
            if target_tensor.shape == source_tensor.shape:
                target_state[target_key] = source_tensor.to(dtype=target_tensor.dtype).clone()
                copied += 1
                continue

            if target_tensor.ndim != source_tensor.ndim:
                continue

            patched = target_tensor.clone()
            slices = tuple(slice(0, min(dst, src)) for dst, src in zip(target_tensor.shape, source_tensor.shape))
            patched[slices] = source_tensor[slices].to(dtype=target_tensor.dtype)
            target_state[target_key] = patched
            partial += 1

        self.load_state_dict(target_state)
        return copied, partial


@dataclass(slots=True)
class RawTransition:
    spatial: np.ndarray
    vector: np.ndarray
    action: int
    reward: float
    next_spatial: np.ndarray
    next_vector: np.ndarray
    done: bool


@dataclass(slots=True)
class ReplayEntry:
    spatial: np.ndarray
    vector: np.ndarray
    action: int
    reward: float
    next_spatial: np.ndarray
    next_vector: np.ndarray
    done: bool
    discount: float


class NStepAccumulator:
    def __init__(self, n_step: int = 3, gamma: float = 0.99):
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queue: Deque[RawTransition] = deque()

    def push(self, transition: RawTransition) -> list[ReplayEntry]:
        self.queue.append(transition)
        if transition.done:
            return self.flush()
        return self._drain_ready()

    def _drain_ready(self) -> list[ReplayEntry]:
        entries: list[ReplayEntry] = []
        while len(self.queue) >= self.n_step:
            entries.append(self._build_entry(self.n_step))
            self.queue.popleft()
        return entries

    def flush(self) -> list[ReplayEntry]:
        entries: list[ReplayEntry] = []
        while self.queue:
            entries.append(self._build_entry(min(self.n_step, len(self.queue))))
            self.queue.popleft()
        return entries

    def _build_entry(self, horizon: int) -> ReplayEntry:
        reward = 0.0
        discount = 1.0
        next_spatial = self.queue[0].next_spatial
        next_vector = self.queue[0].next_vector
        done = False

        for idx in range(horizon):
            transition = self.queue[idx]
            reward += discount * float(transition.reward)
            next_spatial = transition.next_spatial
            next_vector = transition.next_vector
            done = bool(transition.done)
            if done:
                break
            discount *= self.gamma

        return ReplayEntry(
            spatial=self.queue[0].spatial,
            vector=self.queue[0].vector,
            action=int(self.queue[0].action),
            reward=float(reward),
            next_spatial=next_spatial,
            next_vector=next_vector,
            done=done,
            discount=float(discount),
        )


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, spatial_shape: tuple[int, ...], vector_dim: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.size = 0
        self.spatial_states = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.vector_states = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.next_spatial_states = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.next_vector_states = np.zeros((capacity, vector_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(self, entry: ReplayEntry) -> None:
        self.spatial_states[self.pos] = entry.spatial
        self.vector_states[self.pos] = entry.vector
        self.next_spatial_states[self.pos] = entry.next_spatial
        self.next_vector_states[self.pos] = entry.next_vector
        self.actions[self.pos] = int(entry.action)
        self.rewards[self.pos] = float(entry.reward)
        self.dones[self.pos] = 1.0 if entry.done else 0.0
        self.discounts[self.pos] = float(entry.discount)
        self.priorities[self.pos] = self.priorities[: self.size].max() if self.size > 0 else 1.0

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        priorities = self.priorities[: self.size] ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(self.size, size=batch_size, replace=self.size < batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return {
            "spatial": self.spatial_states[indices],
            "vector": self.vector_states[indices],
            "next_spatial": self.next_spatial_states[indices],
            "next_vector": self.next_vector_states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "discounts": self.discounts[indices],
            "weights": weights.astype(np.float32, copy=False),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6) -> None:
        self.priorities[indices] = np.abs(td_errors).astype(np.float32, copy=False) + float(epsilon)


class SafetyShield:
    def __init__(self, search_depth: int = 10):
        self.search_depth = int(search_depth)

    def safe_actions(self, bundle: ObservationBundle) -> list[int]:
        if not bundle.alive:
            return [Action.STOP]

        actions: list[int] = []
        grid = bundle.grid
        my_pos = bundle.my_pos
        bomb_tiles = {(int(b[0]), int(b[1])) for b in bundle.bombs}
        danger_now = bundle.danger_1 > 0
        danger_soon = bundle.danger_3 > 0

        if _in_bounds(grid, *my_pos) and not danger_now[my_pos] and not danger_soon[my_pos]:
            actions.append(Action.STOP)

        for action in MOVE_ACTIONS:
            nx, ny = _next_pos(my_pos, action)
            if not _passable(grid, nx, ny):
                continue
            if (nx, ny) in bomb_tiles:
                continue
            if danger_now[nx, ny]:
                continue
            if danger_soon[nx, ny] and not self._has_escape_path(grid, (nx, ny), bomb_tiles, danger_soon):
                continue
            if not self._has_escape_path(grid, (nx, ny), bomb_tiles, danger_soon):
                continue
            actions.append(action)

        if self._can_place_bomb(bundle):
            actions.append(Action.PLACE_BOMB)

        if not actions:
            return self.fallback_escape_actions(bundle)

        return actions

    def select_action(self, bundle: ObservationBundle, q_values: torch.Tensor) -> int:
        q = q_values.reshape(-1)
        safe_actions = self.safe_actions(bundle)
        if not safe_actions:
            return self.fallback_escape_actions(bundle)[0]

        best_action = safe_actions[0]
        best_score = float(q[best_action].item())
        best_priority = ACTION_PRIORITY[best_action]
        for action in safe_actions[1:]:
            score = float(q[action].item())
            priority = ACTION_PRIORITY[action]
            if score > best_score or (abs(score - best_score) <= 1e-6 and priority < best_priority):
                best_action = action
                best_score = score
                best_priority = priority
        return int(best_action)

    def fallback_escape_actions(self, bundle: ObservationBundle) -> list[int]:
        grid = bundle.grid
        my_pos = bundle.my_pos
        bomb_tiles = {(int(b[0]), int(b[1])) for b in bundle.bombs}
        danger = bundle.danger_1 > 0

        best_action = None
        best_score = -1_000_000
        for action in MOVE_ACTIONS + (Action.STOP,):
            if action == Action.STOP:
                nx, ny = my_pos
            else:
                nx, ny = _next_pos(my_pos, action)
                if not _passable(grid, nx, ny) or (nx, ny) in bomb_tiles:
                    continue
            score = 0
            if _in_bounds(grid, nx, ny) and not danger[nx, ny]:
                score += 10
            score += self._open_neighbors(grid, (nx, ny), bomb_tiles)
            score += self._distance_from_danger(grid, (nx, ny), bomb_tiles)
            if score > best_score or (score == best_score and ACTION_PRIORITY[action] < ACTION_PRIORITY.get(best_action, 999)):
                best_score = score
                best_action = action

        return [int(best_action)] if best_action is not None else [Action.STOP]

    def _can_place_bomb(self, bundle: ObservationBundle) -> bool:
        if not bundle.alive or bundle.bombs_left <= 0:
            return False

        grid = bundle.grid
        x, y = bundle.my_pos
        if not _in_bounds(grid, x, y):
            return False
        if bundle.bombs.size > 0 and np.any((bundle.bombs[:, 0] == x) & (bundle.bombs[:, 1] == y)):
            return False

        blast = _blast_tiles(grid, x, y, bundle.bomb_radius)
        boxes_hit = sum(1 for tx, ty in blast if int(grid[tx, ty]) == 2)
        enemy_hit = any(
            int(row[2]) == 1 and (int(row[0]), int(row[1])) in blast
            for pid, row in enumerate(bundle.players)
            if pid != bundle.agent_id
        )
        if boxes_hit <= 0 and not enemy_hit:
            return False

        blocked = {(int(b[0]), int(b[1])) for b in bundle.bombs}
        extra_danger = set(blast)
        return self._has_escape_path(grid, (x, y), blocked, bundle.danger_3 > 0, extra_danger=extra_danger)

    def _has_escape_path(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        blocked: set[tuple[int, int]],
        danger_mask: np.ndarray,
        extra_danger: set[tuple[int, int]] | None = None,
    ) -> bool:
        if extra_danger is None:
            extra_danger = set()

        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, depth = q.popleft()
            if depth > 0 and _in_bounds(grid, *pos):
                if not danger_mask[pos] and pos not in extra_danger:
                    return True
            if depth >= self.search_depth:
                continue
            for action in MOVE_ACTIONS:
                nx, ny = _next_pos(pos, action)
                next_pos = (nx, ny)
                if next_pos in seen:
                    continue
                if not _passable(grid, nx, ny) or next_pos in blocked or next_pos in extra_danger:
                    continue
                seen.add(next_pos)
                q.append((next_pos, depth + 1))
        return False

    def _open_neighbors(self, grid: np.ndarray, pos: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
        count = 0
        for action in MOVE_ACTIONS:
            nx, ny = _next_pos(pos, action)
            if _passable(grid, nx, ny) and (nx, ny) not in blocked:
                count += 1
        return count

    def _distance_from_danger(self, grid: np.ndarray, pos: tuple[int, int], blocked: set[tuple[int, int]]) -> int:
        score = 0
        for action in MOVE_ACTIONS:
            nx, ny = _next_pos(pos, action)
            if _passable(grid, nx, ny) and (nx, ny) not in blocked:
                score += 1
        return score

