from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from strategy import Strategy, StrategyDecision


ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_BOMB = 5

GRID_GRASS = 0
GRID_WALL = 1
GRID_BOX = 2
GRID_ITEM_RADIUS = 3
GRID_ITEM_CAPACITY = 4

BOMB_MAX_TIMER = 7
CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_PROFILE_NAME = "balanced"
PROFILE_NAME = "balanced"

DEFAULT_PARAMS = {
    "HORIZON": 8,
    "SAFE_MARGIN": 2,
    "BOMB_DECISION_MARGIN": 35.0,
    "SOLO_BOMB_DECISION_MARGIN": -15.0,
    "BOMB_TACTICAL_BONUS": 80.0,
    "SOLO_BOMB_TACTICAL_BONUS": 120.0,
    "ITEM_SCORE_MULTIPLIER": 1.0,
    "ITEM_PROGRESS_BONUS": 20.0,
    "ITEM_STEP_BONUS": 120.0,
    "STOP_AVOID_MARGIN": 8.0,
    "RECENT_BOMB_ESCAPE_BONUS": 20.0,
    "DEATH_PENALTY": -1_000_000.0,
    "NO_ESCAPE_PENALTY": -100_000.0,
    "DANGER_NEXT_STEP_PENALTY": -50_000.0,
    "DANGER_SOON_PENALTY": -5_000.0,
    "STOP_PENALTY": -10.0,
    "WASTE_BOMB_PENALTY": -80.0,
    "SAFE_ESCAPE_BONUS": 1_000.0,
    "OPEN_AREA_BONUS": 10.0,
    "SAFE_AREA_BONUS": 12.0,
    "ESCAPE_DISTANCE_PENALTY": 5.0,
    "BOX_VALUE": 60.0,
    "MULTI_BOX_BONUS": 30.0,
    "ENEMY_HIT_BONUS": 300.0,
    "ENEMY_TRAP_BONUS": 200.0,
    "ENEMY_NO_ESCAPE_BONUS": 500.0,
    "CAPACITY_ITEM_BONUS": 180.0,
    "CAPACITY_ITEM_STANDARD": 100.0,
    "RADIUS_ITEM_BONUS": 140.0,
    "RADIUS_ITEM_STANDARD": 90.0,
    "RADIUS_ITEM_LATE": 30.0,
}

INT_PARAMS = {"HORIZON", "SAFE_MARGIN"}


def _coerce_profile_params(raw_params: object) -> Dict[str, float]:
    params = dict(DEFAULT_PARAMS)
    if not isinstance(raw_params, dict):
        return params

    for key, default_value in DEFAULT_PARAMS.items():
        if key not in raw_params:
            continue
        value = raw_params[key]
        if key in INT_PARAMS:
            params[key] = int(value)
        else:
            params[key] = float(value)
    return params


def _load_profile_params(profile_name: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
    requested_profile = (profile_name or PROFILE_NAME or DEFAULT_PROFILE_NAME).strip().lower()

    try:
        raw_text = CONFIG_PATH.read_text(encoding="utf-8")
        raw_config = json.loads(raw_text)
    except Exception:
        return DEFAULT_PROFILE_NAME, dict(DEFAULT_PARAMS)

    profiles = raw_config.get("profiles", raw_config)
    if not isinstance(profiles, dict) or not profiles:
        return DEFAULT_PROFILE_NAME, dict(DEFAULT_PARAMS)

    default_profile = str(raw_config.get("default_profile", DEFAULT_PROFILE_NAME)).strip().lower()
    selected_profile = requested_profile if requested_profile in profiles else default_profile
    if selected_profile not in profiles:
        selected_profile = DEFAULT_PROFILE_NAME if DEFAULT_PROFILE_NAME in profiles else next(iter(profiles))

    return selected_profile, _coerce_profile_params(profiles.get(selected_profile))


@dataclass(frozen=True)
class BombState:
    x: int
    y: int
    timer: int
    owner_id: int
    radius: int


@dataclass
class ReachabilityResult:
    arrival_time_by_pos: Dict[Tuple[int, int], int]
    rest_safe_time_by_pos: Dict[Tuple[int, int], int]
    rest_safe_count: int
    best_rest_safe_distance: Optional[int]


@dataclass
class DangerTimeline:
    bomb_states: List[BombState]
    explosion_times: List[int]
    danger_by_t: List[set[Tuple[int, int]]]


class Agent:
    """Time-expanded tactical scoring agent for Bomberland."""

    MOVES = {
        ACTION_STOP: (0, 0),
        ACTION_LEFT: (-1, 0),
        ACTION_RIGHT: (1, 0),
        ACTION_UP: (0, -1),
        ACTION_DOWN: (0, 1),
    }
    SEARCH_ACTIONS = (ACTION_STOP, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN)
    team_id = "TimeExpandedTacticalScoringAgent"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.step_count = 0
        self._bomb_radius_cache: Dict[Tuple[int, int, int], int] = {}
        self._recent_bomb_pos: Optional[Tuple[int, int]] = None
        self._recent_bomb_turn: Optional[int] = None
        self.strategy = Strategy()
        self.collector_profile_name, self._collector_params = _load_profile_params("collector")
        self.balanced_profile_name, self._balanced_params = _load_profile_params("balanced")
        self.aggressive_profile_name, self._aggressive_params = _load_profile_params("aggressive")
        self.profile_name = self.balanced_profile_name
        self._apply_params(self._balanced_params)

    def _apply_params(self, params: Dict[str, float]) -> None:
        for key, value in params.items():
            setattr(self, key, value)

    def act(self, obs: dict) -> int:
        self.step_count += 1

        if not obs:
            return ACTION_STOP

        grid = np.asarray(obs.get("map"))
        players = self._as_player_array(obs.get("players"))
        if grid.size == 0 or grid.ndim != 2 or players is None or players.ndim != 2:
            return ACTION_STOP
        if self.agent_id < 0 or self.agent_id >= len(players):
            return ACTION_STOP
        if int(players[self.agent_id][2]) != 1:
            return ACTION_STOP

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        my_radius = max(1, 1 + int(bomb_bonus))

        bombs = self._parse_bombs(obs.get("bombs"), players)
        bomb_positions = {(bomb.x, bomb.y) for bomb in bombs}
        enemies = self._alive_enemy_positions(players)
        alive_count = 1 + len(enemies)
        box_count = int(np.count_nonzero(grid == GRID_BOX))
        blocked = set(bomb_positions) | set(enemies)
        blocked.discard(my_pos)

        item_targets = self._target_map_item(grid, bombs_left=int(bombs_left), bomb_radius=my_radius)
        nearest_item_distance = self._nearest_target_distance(my_pos, item_targets)
        strategy_decision = self.strategy.decide(
            alive_count=alive_count,
            box_count=box_count,
            nearest_item_distance=nearest_item_distance,
            item_targets_present=bool(item_targets),
        )
        self._sync_runtime_profile(strategy_decision)

        base_timeline = self._build_danger_timeline(
            grid=grid,
            players=players,
            bombs=bombs,
            horizon=self.HORIZON,
            extra_bomb=None,
        )

        if self._is_immediate_danger(my_pos, base_timeline.danger_by_t):
            return self._choose_escape_action(
                grid=grid,
                players=players,
                bombs=bombs,
                my_pos=my_pos,
                blocked=blocked,
                danger_by_t=base_timeline.danger_by_t,
            )

        candidates: List[Tuple[float, int]] = []

        for action in (ACTION_STOP, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN):
            if not self._is_move_legal(action, grid, my_pos, blocked):
                continue
            next_pos = self._next_pos(my_pos, action)
            score = self._score_move_action(
                action=action,
                grid=grid,
                players=players,
                my_pos=next_pos,
                original_pos=my_pos,
                bombs=bombs,
                blocked=blocked,
                danger_by_t=base_timeline.danger_by_t,
                bombs_left=int(bombs_left),
                my_radius=my_radius,
                alive_count=alive_count,
            )
            candidates.append((score, action))

        bomb_score = self._score_bomb_action(
            grid=grid,
            players=players,
            bombs=bombs,
            my_pos=my_pos,
            my_radius=my_radius,
            bombs_left=int(bombs_left),
            blocked=blocked,
            base_timeline=base_timeline,
            alive_count=alive_count,
        )
        solo_can_bomb_now = alive_count <= 2 and self.strategy.can_spam_bomb()
        if alive_count <= 2 and not solo_can_bomb_now:
            bomb_score = None
        if bomb_score is not None:
            candidates.append((bomb_score, ACTION_BOMB))

        if not candidates:
            return self._safe_fallback(grid=grid, my_pos=my_pos, blocked=blocked, danger_by_t=base_timeline.danger_by_t)

        candidates.sort(key=self._candidate_sort_key, reverse=True)
        _, best_action = candidates[0]
        best_non_bomb_action = max(
            (item for item in candidates if item[1] != ACTION_BOMB),
            key=self._candidate_sort_key,
        )[1]
        pause_solo_bombing = strategy_decision.pause_bomb_spam or self._should_pause_solo_bombing(
            grid=grid,
            my_pos=my_pos,
            best_non_bomb_action=best_non_bomb_action,
            bombs_left=int(bombs_left),
            my_radius=my_radius,
            alive_count=alive_count,
        )
        best_move_score = max(
            (score for score, action in candidates if action != ACTION_STOP),
            default=-inf,
        )
        if best_action == ACTION_STOP and best_move_score > -inf:
            if best_move_score + float(self.STOP_AVOID_MARGIN) >= candidates[0][0]:
                best_action = max(
                    (item for item in candidates if item[1] != ACTION_STOP),
                    key=self._candidate_sort_key,
                )[1]
        if bomb_score is not None and not pause_solo_bombing:
            if strategy_decision.force_bomb and solo_can_bomb_now and self._can_drop_solo_bomb(my_pos):
                self._remember_recent_bomb(my_pos)
                self.strategy.record_bomb_drop(solo=True)
                return ACTION_BOMB
            best_move_score = max((score for score, action in candidates if action != ACTION_BOMB), default=-inf)
            bomb_decision_margin = float(self.BOMB_DECISION_MARGIN)
            if alive_count <= 2:
                bomb_decision_margin = float(self.SOLO_BOMB_DECISION_MARGIN)
            if (bomb_score > 0 or alive_count <= 2) and bomb_score + bomb_decision_margin >= best_move_score:
                self._remember_recent_bomb(my_pos)
                self.strategy.record_bomb_drop(solo=alive_count <= 2)
                return ACTION_BOMB
        if best_action == ACTION_BOMB and not pause_solo_bombing:
            self._remember_recent_bomb(my_pos)
            self.strategy.record_bomb_drop(solo=alive_count <= 2)
        elif pause_solo_bombing and best_action == ACTION_BOMB:
            best_action = best_non_bomb_action
        elif alive_count <= 2 and best_action != ACTION_BOMB and self._can_drop_solo_bomb(my_pos):
            self.strategy.reset_solo_bomb_burst()
        return int(best_action)

    def _sync_runtime_profile(self, decision: StrategyDecision) -> None:
        target_profile_name = decision.profile_name
        if target_profile_name == self.profile_name and not decision.overrides:
            return

        self.profile_name = target_profile_name
        if target_profile_name == self.collector_profile_name:
            self._apply_params(self._collector_params)
        elif target_profile_name == self.aggressive_profile_name:
            self._apply_params(self._aggressive_params)
        else:
            self._apply_params(self._balanced_params)
        self._apply_params(decision.overrides)

    def _remember_recent_bomb(self, bomb_pos: Tuple[int, int]) -> None:
        self._recent_bomb_pos = bomb_pos
        self._recent_bomb_turn = self.step_count

    def _can_drop_solo_bomb(self, my_pos: Tuple[int, int]) -> bool:
        if self._recent_bomb_pos is None or self._recent_bomb_turn is None:
            return True
        recent_age = self.step_count - self._recent_bomb_turn
        if recent_age >= 2:
            return True
        return self._manhattan_distance(my_pos, self._recent_bomb_pos) >= 3

    def _should_pause_solo_bombing(
        self,
        grid: np.ndarray,
        my_pos: Tuple[int, int],
        best_non_bomb_action: int,
        bombs_left: int,
        my_radius: int,
        alive_count: int,
    ) -> bool:
        if alive_count > 2:
            return False
        item_targets = self._target_map_item(grid, bombs_left=bombs_left, bomb_radius=my_radius)
        if not item_targets:
            return False

        item_before = self._nearest_target_distance(my_pos, item_targets)
        if item_before is None or item_before > 4:
            return False
        if best_non_bomb_action == ACTION_STOP:
            return False

        next_pos = self._next_pos(my_pos, best_non_bomb_action)
        if not self._is_walkable(grid, next_pos[0], next_pos[1]):
            return False

        item_after = self._nearest_target_distance(next_pos, item_targets)
        if item_after is None:
            return False
        return item_after < item_before

    def _candidate_sort_key(self, item: Tuple[float, int]) -> Tuple[float, int, int]:
        score, action = item
        if action == ACTION_BOMB:
            action_priority = 2
        elif action == ACTION_STOP:
            action_priority = 0
        else:
            action_priority = 1
        return float(score), action_priority, -int(action)

    def _as_player_array(self, players: object) -> Optional[np.ndarray]:
        if players is None:
            return None
        if isinstance(players, dict):
            if not players:
                return None
            ordered = [players[key] for key in sorted(players)]
            arr = np.asarray(ordered)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        arr = np.asarray(players)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _parse_bombs(self, bombs: object, players: np.ndarray) -> List[BombState]:
        if bombs is None:
            self._bomb_radius_cache.clear()
            return []

        arr = np.asarray(bombs)
        if arr.size == 0:
            self._bomb_radius_cache.clear()
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        parsed: List[BombState] = []
        active_keys: set[Tuple[int, int, int]] = set()
        for row in arr:
            if len(row) < 2:
                continue
            x = int(row[0])
            y = int(row[1])
            timer = int(row[2]) if len(row) > 2 else BOMB_MAX_TIMER
            owner_id = int(row[3]) if len(row) > 3 else 0
            key = (x, y, owner_id)
            active_keys.add(key)
            radius = self._bomb_radius_cache.get(key)
            if radius is None:
                radius = self._bomb_radius_for_owner(players, owner_id)
                self._bomb_radius_cache[key] = radius
            parsed.append(BombState(x=x, y=y, timer=timer, owner_id=owner_id, radius=radius))

        for key in list(self._bomb_radius_cache):
            if key not in active_keys:
                del self._bomb_radius_cache[key]

        return parsed

    def _bomb_radius_for_owner(self, players: np.ndarray, owner_id: int) -> int:
        if 0 <= owner_id < len(players):
            return max(1, 1 + int(players[owner_id][4]))
        return 2

    def _alive_enemy_positions(self, players: np.ndarray) -> List[Tuple[int, int]]:
        enemies: List[Tuple[int, int]] = []
        for idx, player in enumerate(players):
            if idx == self.agent_id or int(player[2]) != 1:
                continue
            enemies.append((int(player[0]), int(player[1])))
        return enemies

    def _is_move_legal(self, action: int, grid: np.ndarray, my_pos: Tuple[int, int], blocked: set[Tuple[int, int]]) -> bool:
        if action == ACTION_STOP:
            return True
        next_pos = self._next_pos(my_pos, action)
        return self._is_walkable(grid, next_pos[0], next_pos[1]) and next_pos not in blocked

    def _next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid: np.ndarray, x: int, y: int) -> bool:
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _is_walkable(self, grid: np.ndarray, x: int, y: int) -> bool:
        return self._in_bounds(grid, x, y) and int(grid[x, y]) in {
            GRID_GRASS,
            GRID_ITEM_RADIUS,
            GRID_ITEM_CAPACITY,
        }

    def _blast_tiles(self, grid: np.ndarray, bx: int, by: int, radius: int) -> set[Tuple[int, int]]:
        tiles = {(bx, by)}
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for dist in range(1, radius + 1):
                x = bx + dx * dist
                y = by + dy * dist
                if not self._in_bounds(grid, x, y):
                    break
                cell = int(grid[x, y])
                if cell == GRID_WALL:
                    break
                tiles.add((x, y))
                if cell == GRID_BOX:
                    break
        return tiles

    def _build_danger_timeline(
        self,
        grid: np.ndarray,
        players: np.ndarray,
        bombs: Sequence[BombState],
        horizon: int,
        extra_bomb: Optional[BombState],
    ) -> DangerTimeline:
        bomb_states = list(bombs)
        if extra_bomb is not None:
            bomb_states.append(extra_bomb)

        explosion_times = [max(1, int(bomb.timer)) for bomb in bomb_states]

        changed = True
        while changed:
            changed = False
            for i, bomb_i in enumerate(bomb_states):
                blast_i = self._blast_tiles(grid, bomb_i.x, bomb_i.y, bomb_i.radius)
                time_i = explosion_times[i]
                for j, bomb_j in enumerate(bomb_states):
                    if i == j:
                        continue
                    if (bomb_j.x, bomb_j.y) not in blast_i:
                        continue
                    if explosion_times[j] > time_i:
                        explosion_times[j] = time_i
                        changed = True

        danger_by_t = [set() for _ in range(horizon + 1)]
        for bomb, time in zip(bomb_states, explosion_times):
            if time < 1 or time > horizon:
                continue
            danger_by_t[time].update(self._blast_tiles(grid, bomb.x, bomb.y, bomb.radius))

        return DangerTimeline(
            bomb_states=bomb_states,
            explosion_times=explosion_times,
            danger_by_t=danger_by_t,
        )

    def _is_immediate_danger(self, pos: Tuple[int, int], danger_by_t: Sequence[set[Tuple[int, int]]]) -> bool:
        return len(danger_by_t) > 1 and pos in danger_by_t[1]

    def _is_tile_dangerous_at(self, pos: Tuple[int, int], time: int, danger_by_t: Sequence[set[Tuple[int, int]]]) -> bool:
        if time < 1 or time >= len(danger_by_t):
            return False
        return pos in danger_by_t[time]

    def _is_rest_safe(self, pos: Tuple[int, int], time: int, danger_by_t: Sequence[set[Tuple[int, int]]]) -> bool:
        horizon = len(danger_by_t) - 1
        for offset in range(self.SAFE_MARGIN):
            check_time = time + offset
            if check_time > horizon:
                break
            if pos in danger_by_t[check_time]:
                return False
        return True

    def _min_danger_time(self, pos: Tuple[int, int], danger_by_t: Sequence[set[Tuple[int, int]]], start_time: int) -> Optional[int]:
        for time in range(max(1, start_time), len(danger_by_t)):
            if pos in danger_by_t[time]:
                return time
        return None

    def _danger_penalty(self, danger_time: Optional[int]) -> float:
        if danger_time is None:
            return 0.0
        if danger_time <= 1:
            return self.DEATH_PENALTY
        if danger_time == 2:
            return self.DANGER_NEXT_STEP_PENALTY
        if danger_time <= 4:
            return self.DANGER_SOON_PENALTY
        if danger_time <= 6:
            return -500.0
        return 0.0

    def _time_expanded_reachability(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        start_time: int,
        blocked: set[Tuple[int, int]],
        danger_by_t: Sequence[set[Tuple[int, int]]],
    ) -> ReachabilityResult:
        blocked_no_self = set(blocked)
        blocked_no_self.discard(start)

        arrival_time_by_pos: Dict[Tuple[int, int], int] = {}
        rest_safe_time_by_pos: Dict[Tuple[int, int], int] = {}
        visited: set[Tuple[int, int, int]] = set()
        queue = deque([(start[0], start[1], start_time)])
        visited.add((start[0], start[1], start_time))

        while queue:
            x, y, time = queue.popleft()
            pos = (x, y)

            prev_arrival = arrival_time_by_pos.get(pos)
            if prev_arrival is None or time < prev_arrival:
                arrival_time_by_pos[pos] = time

            if self._is_rest_safe(pos, time, danger_by_t):
                prev_rest = rest_safe_time_by_pos.get(pos)
                if prev_rest is None or time < prev_rest:
                    rest_safe_time_by_pos[pos] = time

            if time >= self.HORIZON:
                continue

            next_time = time + 1
            for action in self.SEARCH_ACTIONS:
                if action == ACTION_STOP:
                    nx, ny = x, y
                else:
                    nx, ny = self._next_pos(pos, action)
                    if not self._is_walkable(grid, nx, ny):
                        continue
                    if (nx, ny) in blocked_no_self:
                        continue
                if self._is_tile_dangerous_at((nx, ny), next_time, danger_by_t):
                    continue
                state = (nx, ny, next_time)
                if state in visited:
                    continue
                visited.add(state)
                queue.append(state)

        best_rest_safe_distance: Optional[int] = None
        for pos, arrival_time in rest_safe_time_by_pos.items():
            distance = arrival_time - start_time
            if best_rest_safe_distance is None or distance < best_rest_safe_distance:
                best_rest_safe_distance = distance

        return ReachabilityResult(
            arrival_time_by_pos=arrival_time_by_pos,
            rest_safe_time_by_pos=rest_safe_time_by_pos,
            rest_safe_count=len(rest_safe_time_by_pos),
            best_rest_safe_distance=best_rest_safe_distance,
        )

    def _target_map_item(self, grid: np.ndarray, bombs_left: int, bomb_radius: int) -> Dict[Tuple[int, int], float]:
        targets: Dict[Tuple[int, int], float] = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = int(grid[x, y])
                if cell == GRID_ITEM_CAPACITY:
                    value = self.CAPACITY_ITEM_BONUS if bombs_left <= 1 else self.CAPACITY_ITEM_STANDARD
                elif cell == GRID_ITEM_RADIUS:
                    if bomb_radius <= 2:
                        value = self.RADIUS_ITEM_BONUS
                    elif bomb_radius <= 4:
                        value = self.RADIUS_ITEM_STANDARD
                    else:
                        value = self.RADIUS_ITEM_LATE
                else:
                    continue
                targets[(x, y)] = max(targets.get((x, y), 0.0), value)
        return targets

    def _target_map_box_spots(self, grid: np.ndarray, bomb_radius: int, blocked: set[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        targets: Dict[Tuple[int, int], float] = {}
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if int(grid[x, y]) != GRID_BOX:
                    continue
                box_count = 0
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    tx, ty = x + dx, y + dy
                    if not self._is_walkable(grid, tx, ty):
                        continue
                    if (tx, ty) in blocked:
                        continue
                    box_count = max(box_count, self._count_boxes_in_blast(grid, (tx, ty), bomb_radius))
                    if box_count > 0:
                        value = box_count * self.BOX_VALUE
                        if box_count >= 2:
                            value += self.MULTI_BOX_BONUS
                        if box_count >= 3:
                            value += 60.0
                        targets[(tx, ty)] = max(targets.get((tx, ty), 0.0), value)
        return targets

    def _target_map_enemy_pressure(
        self,
        grid: np.ndarray,
        enemies: Sequence[Tuple[int, int]],
        bombs_left: int,
        bomb_radius: int,
        blocked: set[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        targets: Dict[Tuple[int, int], float] = {}
        base_value = 60.0
        if bombs_left >= 2 or bomb_radius >= 3:
            base_value += 40.0
        if bombs_left >= 3 or bomb_radius >= 4:
            base_value += 20.0

        for ex, ey in enemies:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tx, ty = ex + dx, ey + dy
                if not self._is_walkable(grid, tx, ty):
                    continue
                if (tx, ty) in blocked:
                    continue
                targets[(tx, ty)] = max(targets.get((tx, ty), 0.0), base_value)
        return targets

    def _best_reachable_target_score(
        self,
        reachability: ReachabilityResult,
        targets: Dict[Tuple[int, int], float],
        start_time: int,
    ) -> float:
        best = 0.0
        for pos, value in targets.items():
            target_time = reachability.rest_safe_time_by_pos.get(pos)
            if target_time is None:
                continue
            distance = target_time - start_time
            if distance < 0:
                continue
            score = value / float(1 + distance)
            if score > best:
                best = score
        return best

    def _escape_score(self, reachability: ReachabilityResult) -> float:
        if reachability.best_rest_safe_distance is None:
            return self.NO_ESCAPE_PENALTY
        return (
            self.SAFE_ESCAPE_BONUS
            + reachability.rest_safe_count * self.SAFE_AREA_BONUS
            - reachability.best_rest_safe_distance * self.ESCAPE_DISTANCE_PENALTY
        )

    def _open_neighbors(self, grid: np.ndarray, pos: Tuple[int, int], blocked: set[Tuple[int, int]]) -> int:
        count = 0
        for action in (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN):
            nx, ny = self._next_pos(pos, action)
            if self._is_walkable(grid, nx, ny) and (nx, ny) not in blocked:
                count += 1
        return count

    def _center_bonus(self, grid: np.ndarray, pos: Tuple[int, int]) -> float:
        center_x = (grid.shape[0] - 1) / 2.0
        center_y = (grid.shape[1] - 1) / 2.0
        dist = abs(pos[0] - center_x) + abs(pos[1] - center_y)
        return max(0.0, 8.0 - 0.75 * dist)

    def _nearest_target_distance(
        self,
        pos: Tuple[int, int],
        targets: Dict[Tuple[int, int], float],
    ) -> Optional[int]:
        if not targets:
            return None
        best: Optional[int] = None
        for target in targets:
            distance = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            if best is None or distance < best:
                best = distance
        return best

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _late_game_factor(self) -> float:
        if self.step_count >= 350:
            return 1.2
        if self.step_count >= 250:
            return 1.1
        return 1.0

    def _score_move_action(
        self,
        action: int,
        grid: np.ndarray,
        players: np.ndarray,
        my_pos: Tuple[int, int],
        original_pos: Tuple[int, int],
        bombs: Sequence[BombState],
        blocked: set[Tuple[int, int]],
        danger_by_t: Sequence[set[Tuple[int, int]]],
        bombs_left: int,
        my_radius: int,
        alive_count: int,
    ) -> float:
        reachability = self._time_expanded_reachability(
            grid=grid,
            start=my_pos,
            start_time=1,
            blocked=blocked,
            danger_by_t=danger_by_t,
        )

        danger_time = self._min_danger_time(my_pos, danger_by_t, start_time=1)
        danger_penalty = self._danger_penalty(danger_time)
        escape_score = self._escape_score(reachability)
        open_score = self._open_neighbors(grid, my_pos, blocked) * self.OPEN_AREA_BONUS
        center_score = self._center_bonus(grid, my_pos)

        item_targets = self._target_map_item(grid, bombs_left=bombs_left, bomb_radius=my_radius)
        box_targets = self._target_map_box_spots(grid, bomb_radius=my_radius, blocked=blocked)
        enemy_targets = self._target_map_enemy_pressure(
            grid=grid,
            enemies=self._alive_enemy_positions(players),
            bombs_left=bombs_left,
            bomb_radius=my_radius,
            blocked=blocked,
        )
        late_factor = self._late_game_factor()

        item_score = self._best_reachable_target_score(reachability, item_targets, start_time=1)
        item_score *= float(self.ITEM_SCORE_MULTIPLIER)
        box_score = self._best_reachable_target_score(reachability, box_targets, start_time=1)
        enemy_score = self._best_reachable_target_score(reachability, enemy_targets, start_time=1)
        item_before = self._nearest_target_distance(original_pos, item_targets)
        item_after = self._nearest_target_distance(my_pos, item_targets)
        item_progress_score = 0.0
        if item_before is not None and item_after is not None:
            if item_after < item_before:
                item_progress_score += float(self.ITEM_PROGRESS_BONUS) * float(item_before - item_after)
        if int(grid[my_pos[0], my_pos[1]]) in (GRID_ITEM_RADIUS, GRID_ITEM_CAPACITY):
            item_progress_score += float(self.ITEM_STEP_BONUS)

        recent_bomb_score = 0.0
        if self._recent_bomb_pos is not None and self._recent_bomb_turn is not None:
            recent_age = self.step_count - self._recent_bomb_turn
            if 0 <= recent_age <= 4 and alive_count <= 2:
                before_recent = self._manhattan_distance(original_pos, self._recent_bomb_pos)
                after_recent = self._manhattan_distance(my_pos, self._recent_bomb_pos)
                if after_recent > before_recent:
                    recent_bomb_score += float(self.RECENT_BOMB_ESCAPE_BONUS) * float(after_recent - before_recent)
                elif after_recent < before_recent:
                    recent_bomb_score -= float(self.RECENT_BOMB_ESCAPE_BONUS) * float(before_recent - after_recent)

        score = (
            escape_score
            + danger_penalty
            + open_score
            + center_score
            + recent_bomb_score
            + late_factor * (item_score + box_score + enemy_score + item_progress_score)
        )

        if action == ACTION_STOP:
            score += self.STOP_PENALTY

        if reachability.best_rest_safe_distance is None:
            score += self.NO_ESCAPE_PENALTY

        return float(score)

    def _count_boxes_in_blast(self, grid: np.ndarray, pos: Tuple[int, int], radius: int) -> int:
        return sum(1 for tile in self._blast_tiles(grid, pos[0], pos[1], radius) if int(grid[tile[0], tile[1]]) == GRID_BOX)

    def _score_bomb_action(
        self,
        grid: np.ndarray,
        players: np.ndarray,
        bombs: Sequence[BombState],
        my_pos: Tuple[int, int],
        my_radius: int,
        bombs_left: int,
        blocked: set[Tuple[int, int]],
        base_timeline: DangerTimeline,
        alive_count: int,
    ) -> Optional[float]:
        if bombs_left <= 0:
            return None
        if my_pos in {(bomb.x, bomb.y) for bomb in bombs}:
            return None

        hypothetical_bomb = BombState(
            x=my_pos[0],
            y=my_pos[1],
            timer=BOMB_MAX_TIMER,
            owner_id=self.agent_id,
            radius=my_radius,
        )

        bomb_timeline = self._build_danger_timeline(
            grid=grid,
            players=players,
            bombs=bombs,
            horizon=self.HORIZON,
            extra_bomb=hypothetical_bomb,
        )

        reachability = self._time_expanded_reachability(
            grid=grid,
            start=my_pos,
            start_time=1,
            blocked=blocked | {my_pos},
            danger_by_t=bomb_timeline.danger_by_t,
        )
        if reachability.best_rest_safe_distance is None:
            return None

        boxes_hit = self._count_boxes_in_blast(grid, my_pos, my_radius)
        box_value = boxes_hit * self.BOX_VALUE
        if boxes_hit >= 2:
            box_value += self.MULTI_BOX_BONUS
        if boxes_hit >= 3:
            box_value += 60.0

        base_explosion_map = {
            (bomb.x, bomb.y, bomb.owner_id): time
            for bomb, time in zip(base_timeline.bomb_states, base_timeline.explosion_times)
        }
        hypo_explosion_map = {
            (bomb.x, bomb.y, bomb.owner_id): time
            for bomb, time in zip(bomb_timeline.bomb_states, bomb_timeline.explosion_times)
        }

        enemy_positions = set(self._alive_enemy_positions(players))
        chain_value = 0.0
        for bomb in bomb_timeline.bomb_states:
            key = (bomb.x, bomb.y, bomb.owner_id)
            base_time = base_explosion_map.get(key)
            hypo_time = hypo_explosion_map.get(key)
            if base_time is None or hypo_time is None or hypo_time >= base_time:
                continue
            blast = self._blast_tiles(grid, bomb.x, bomb.y, bomb.radius)
            accelerated_boxes = sum(1 for tile in blast if int(grid[tile[0], tile[1]]) == GRID_BOX)
            accelerated_enemies = sum(1 for tile in blast if tile in enemy_positions)
            chain_value += (base_time - hypo_time) * 10.0
            chain_value += accelerated_boxes * 20.0
            chain_value += accelerated_enemies * self.ENEMY_HIT_BONUS

        enemy_trap_value = 0.0
        blast_tiles = self._blast_tiles(grid, my_pos[0], my_pos[1], my_radius)
        for enemy in enemy_positions:
            if enemy not in blast_tiles:
                continue
            enemy_reach = self._time_expanded_reachability(
                grid=grid,
                start=enemy,
                start_time=1,
                blocked=blocked | {enemy},
                danger_by_t=bomb_timeline.danger_by_t,
            )
            if enemy_reach.best_rest_safe_distance is None:
                enemy_trap_value += self.ENEMY_NO_ESCAPE_BONUS
            elif enemy_reach.rest_safe_count <= 2:
                enemy_trap_value += self.ENEMY_TRAP_BONUS
            else:
                enemy_trap_value += 100.0

        escape_cost = (
            reachability.best_rest_safe_distance * 10.0
            + max(0, 3 - reachability.rest_safe_count) * 120.0
        )
        self_risk = 0.0 if reachability.rest_safe_count > 0 else 5_000.0
        waste_penalty = self.WASTE_BOMB_PENALTY if boxes_hit == 0 and enemy_trap_value == 0 and chain_value == 0 else 0.0

        tactical_signal = box_value + enemy_trap_value + chain_value
        if tactical_signal > 0:
            tactical_signal += float(self.BOMB_TACTICAL_BONUS)
        if alive_count <= 2:
            tactical_signal += float(self.SOLO_BOMB_TACTICAL_BONUS)

        total_value = tactical_signal - escape_cost - self_risk - abs(waste_penalty)
        if alive_count <= 2:
            if total_value < -300.0 and self._recent_bomb_pos is not None:
                return None
            return float(self.SAFE_ESCAPE_BONUS + max(total_value, -300.0))
        elif total_value <= 0:
            return None
        return float(total_value + self.SAFE_ESCAPE_BONUS)

    def _choose_escape_action(
        self,
        grid: np.ndarray,
        players: np.ndarray,
        bombs: Sequence[BombState],
        my_pos: Tuple[int, int],
        blocked: set[Tuple[int, int]],
        danger_by_t: Sequence[set[Tuple[int, int]]],
    ) -> int:
        best_action = ACTION_STOP
        best_score = -inf

        for action in (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN):
            next_pos = self._next_pos(my_pos, action)
            if not self._is_walkable(grid, next_pos[0], next_pos[1]):
                continue
            if next_pos in blocked:
                continue
            if self._is_tile_dangerous_at(next_pos, 1, danger_by_t):
                continue

            reachability = self._time_expanded_reachability(
                grid=grid,
                start=next_pos,
                start_time=1,
                blocked=blocked,
                danger_by_t=danger_by_t,
            )
            if reachability.best_rest_safe_distance is None:
                score = -100_000.0
            else:
                score = (
                    self.SAFE_ESCAPE_BONUS
                    + reachability.rest_safe_count * 50.0
                    - reachability.best_rest_safe_distance * 15.0
                    + self._open_neighbors(grid, next_pos, blocked) * 15.0
                )
            if score > best_score:
                best_score = score
                best_action = action

        if best_score == -inf:
            return ACTION_STOP
        return int(best_action)

    def _safe_fallback(
        self,
        grid: np.ndarray,
        my_pos: Tuple[int, int],
        blocked: set[Tuple[int, int]],
        danger_by_t: Sequence[set[Tuple[int, int]]],
    ) -> int:
        for action in (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN, ACTION_STOP):
            if action == ACTION_STOP:
                return ACTION_STOP
            next_pos = self._next_pos(my_pos, action)
            if self._is_walkable(grid, next_pos[0], next_pos[1]) and next_pos not in blocked:
                if not self._is_tile_dangerous_at(next_pos, 1, danger_by_t):
                    return action
        return ACTION_STOP
