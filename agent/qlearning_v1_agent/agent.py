from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_BOMB = 5
N_ACTIONS = 6

MAP_GRASS = 0
MAP_WALL = 1
MAP_BOX = 2
MAP_ITEM_RADIUS = 3
MAP_ITEM_CAPACITY = 4

BOMB_BASE_TIMER = 7


@dataclass(frozen=True)
class BombState:
    x: int
    y: int
    timer: int
    owner_id: int


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _to_bomb_states(obs: dict) -> List[BombState]:
    bombs = obs["bombs"]
    arr = np.asarray(bombs)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    result: List[BombState] = []
    for row in arr:
        result.append(BombState(int(row[0]), int(row[1]), int(row[2]), int(row[3])))
    return result


def _bucket_distance(distance: Optional[int]) -> int:
    if distance is None:
        return 4
    if distance <= 1:
        return 0
    if distance <= 3:
        return 1
    if distance <= 6:
        return 2
    if distance <= 10:
        return 3
    return 4


def _direction_bucket(src: Tuple[int, int], dst: Optional[Tuple[int, int]]) -> int:
    if dst is None:
        return 0
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def _is_blocked(map_grid: np.ndarray, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= map_grid.shape[0] or y >= map_grid.shape[1]:
        return True
    return int(map_grid[x, y]) in (MAP_WALL, MAP_BOX)


def _adjacency_code(map_grid: np.ndarray, pos: Tuple[int, int], bombs: Iterable[BombState]) -> int:
    bomb_positions = {(b.x, b.y) for b in bombs}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    digits: List[int] = []
    for dx, dy in dirs:
        tx, ty = pos[0] + dx, pos[1] + dy
        if tx < 0 or ty < 0 or tx >= map_grid.shape[0] or ty >= map_grid.shape[1]:
            digits.append(0)
            continue
        if (tx, ty) in bomb_positions:
            digits.append(4)
            continue
        cell = int(map_grid[tx, ty])
        if cell == MAP_WALL:
            digits.append(0)
        elif cell == MAP_BOX:
            digits.append(1)
        elif cell in (MAP_ITEM_RADIUS, MAP_ITEM_CAPACITY):
            digits.append(3)
        else:
            digits.append(2)

    code = 0
    base = 5
    for digit in digits:
        code = code * base + digit
    return code


def _explosion_tiles(map_grid: np.ndarray, bomb: BombState, radius: int) -> set[Tuple[int, int]]:
    tiles = {(bomb.x, bomb.y)}
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for step in range(1, radius + 1):
            tx, ty = bomb.x + dx * step, bomb.y + dy * step
            if tx < 0 or ty < 0 or tx >= map_grid.shape[0] or ty >= map_grid.shape[1]:
                break
            cell = int(map_grid[tx, ty])
            if cell == MAP_WALL:
                break
            tiles.add((tx, ty))
            if cell == MAP_BOX:
                break
    return tiles


def _find_nearest_target(
    src: Tuple[int, int], candidates: List[Tuple[int, int]]
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda p: _manhattan(src, p))
    return nearest, _manhattan(src, nearest)


def _reward_shaping(prev_obs: dict, curr_obs: dict, agent_id: int) -> float:
    prev_players = np.asarray(prev_obs["players"])
    curr_players = np.asarray(curr_obs["players"])
    if int(prev_players[agent_id][2]) == 1 and int(curr_players[agent_id][2]) == 0:
        return -2.0

    reward = -0.01
    prev_alive_enemy = int(np.sum(prev_players[:, 2])) - int(prev_players[agent_id][2])
    curr_alive_enemy = int(np.sum(curr_players[:, 2])) - int(curr_players[agent_id][2])
    if curr_alive_enemy < prev_alive_enemy:
        reward += 1.0 * (prev_alive_enemy - curr_alive_enemy)
    if curr_alive_enemy == 0 and prev_alive_enemy > 0:
        reward += 2.0

    prev_pos = (int(prev_players[agent_id][0]), int(prev_players[agent_id][1]))
    curr_pos = (int(curr_players[agent_id][0]), int(curr_players[agent_id][1]))
    if prev_pos == curr_pos:
        reward -= 0.01

    prev_cell = int(prev_obs["map"][curr_pos[0], curr_pos[1]])
    if prev_cell in (MAP_ITEM_RADIUS, MAP_ITEM_CAPACITY):
        reward += 0.1

    return float(reward)


class QLearningCore:
    """State encoder + tabular Q-learning policy."""

    team_id = "QLearningV1"

    def __init__(
        self,
        agent_id: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        q_table_path: Optional[Path] = None,
    ) -> None:
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table: Dict[str, List[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)
        self.prev_obs: Optional[dict] = None
        self.prev_state: Optional[str] = None
        self.prev_action: Optional[int] = None
        self.bomb_radius_cache: Dict[Tuple[int, int, int], int] = {}
        self.prev_bomb_keys: set[Tuple[int, int, int]] = set()

        if self.q_table_path and self.q_table_path.exists():
            self._load_q_table(self.q_table_path)

    def _load_q_table(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        parsed: Dict[str, List[float]] = {}
        for key, values in data.items():
            if isinstance(values, list) and len(values) == N_ACTIONS:
                parsed[key] = [float(v) for v in values]
        self.q_table = defaultdict(lambda: [0.0] * N_ACTIONS, parsed)

    def save_q_table(self, path: Path) -> None:
        serializable = {key: values for key, values in self.q_table.items()}
        path.write_text(json.dumps(serializable), encoding="utf-8")

    def _update_bomb_cache(self, obs: dict) -> List[BombState]:
        bombs = _to_bomb_states(obs)
        players = np.asarray(obs["players"])
        current_keys: set[Tuple[int, int, int]] = set()

        for bomb in bombs:
            key = (bomb.x, bomb.y, bomb.owner_id)
            current_keys.add(key)
            if key not in self.prev_bomb_keys:
                owner_radius_bonus = int(players[bomb.owner_id][4])
                self.bomb_radius_cache[key] = 1 + owner_radius_bonus

        stale_keys = [key for key in self.bomb_radius_cache.keys() if key not in current_keys]
        for key in stale_keys:
            del self.bomb_radius_cache[key]

        self.prev_bomb_keys = current_keys
        return bombs

    def _safe_and_timer(self, obs: dict, pos: Tuple[int, int], bombs: List[BombState]) -> Tuple[bool, Optional[int]]:
        map_grid = np.asarray(obs["map"])
        min_timer: Optional[int] = None
        for bomb in bombs:
            key = (bomb.x, bomb.y, bomb.owner_id)
            radius = self.bomb_radius_cache.get(key, 1)
            if pos in _explosion_tiles(map_grid, bomb, radius):
                min_timer = bomb.timer if min_timer is None else min(min_timer, bomb.timer)
        return min_timer is None, min_timer

    @staticmethod
    def _timer_to_bucket(min_timer: Optional[int]) -> int:
        if min_timer is None:
            return 0
        if min_timer >= 5:
            return 1
        if min_timer >= 3:
            return 2
        return 3

    def encode_state(self, obs: dict) -> str:
        players = np.asarray(obs["players"])
        map_grid = np.asarray(obs["map"])
        bombs = self._update_bomb_cache(obs)

        me = players[self.agent_id]
        my_pos = (int(me[0]), int(me[1]))
        bombs_left = int(me[3])

        alive_enemies = [
            (int(p[0]), int(p[1]))
            for idx, p in enumerate(players)
            if idx != self.agent_id and int(p[2]) == 1
        ]
        boxes = list(zip(*np.where(map_grid == MAP_BOX)))

        nearest_enemy, enemy_dist = _find_nearest_target(my_pos, alive_enemies)
        nearest_box, box_dist = _find_nearest_target(my_pos, boxes)

        is_safe, min_timer = self._safe_and_timer(obs, my_pos, bombs)
        state_tuple = (
            1 if is_safe else 0,
            1 if bombs_left > 0 else 0,
            self._timer_to_bucket(min_timer),
            _direction_bucket(my_pos, nearest_enemy),
            _bucket_distance(enemy_dist),
            _direction_bucket(my_pos, nearest_box),
            _bucket_distance(box_dist),
            _adjacency_code(map_grid, my_pos, bombs),
        )
        return "|".join(str(v) for v in state_tuple)

    def choose_action(self, state: str, legal_actions: Optional[List[int]] = None) -> int:
        action_values = self.q_table[state]
        if legal_actions is None:
            legal_actions = list(range(N_ACTIONS))
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        best = max(legal_actions, key=lambda a: action_values[a])
        return int(best)

    def update(self, reward: float, next_state: str, done: bool) -> None:
        if self.prev_state is None or self.prev_action is None:
            return
        prev_q = self.q_table[self.prev_state][self.prev_action]
        next_max = 0.0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        self.q_table[self.prev_state][self.prev_action] = prev_q + self.alpha * (target - prev_q)

    def step_train(self, obs: dict) -> int:
        state = self.encode_state(obs)
        reward = 0.0
        if self.prev_obs is not None:
            reward = _reward_shaping(self.prev_obs, obs, self.agent_id)
            done = int(np.asarray(obs["players"])[self.agent_id][2]) == 0
            self.update(reward=reward, next_state=state, done=done)

        action = self.choose_action(state)
        self.prev_obs = obs
        self.prev_state = state
        self.prev_action = action
        return action

    def reset_episode(self) -> None:
        self.prev_obs = None
        self.prev_state = None
        self.prev_action = None
        self.bomb_radius_cache.clear()
        self.prev_bomb_keys.clear()


class Agent:
    """Competition runtime agent (inference only)."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        q_table_path = Path(__file__).parent / "q_table.json"
        self.core = QLearningCore(
            agent_id=agent_id,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.0,
            q_table_path=q_table_path if q_table_path.exists() else None,
        )

    def act(self, obs: dict) -> int:
        state = self.core.encode_state(obs)
        return self.core.choose_action(state)


def train_qlearning(
    episodes: int,
    enemy_type: str,
    seed: int,
    output_path: Path,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
) -> None:
    # Local imports keep submission runtime minimal.
    from engine import BomberEnv
    from agent import BoxFarmerAgent, GeniusRuleAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent

    random.seed(seed)
    np.random.seed(seed)

    enemy_map = {
        "simple": SimpleRuleAgent,
        "smarter": SmarterRuleAgent,
        "genius": GeniusRuleAgent,
        "tactical": TacticalRuleAgent,
        "box_farmer": BoxFarmerAgent,
    }
    enemy_cls = enemy_map[enemy_type]

    env = BomberEnv(max_steps=500, seed=seed)
    learner = QLearningCore(
        agent_id=0,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        q_table_path=None,
    )
    enemy_agent = enemy_cls(1)

    for episode in range(episodes):
        obs = env.reset(seed=seed + episode)
        learner.reset_episode()
        done = False

        while not done:
            action_user = learner.step_train(obs)
            action_enemy = enemy_agent.act(obs)
            actions = [action_user, action_enemy, ACTION_STOP, ACTION_STOP]
            next_obs, terminated, truncated = env.step(actions)
            done = bool(terminated or truncated)

            next_state = learner.encode_state(next_obs)
            reward = _reward_shaping(obs, next_obs, learner.agent_id)
            learner.update(reward=reward, next_state=next_state, done=done)
            learner.prev_obs = next_obs
            learner.prev_state = next_state
            learner.prev_action = learner.choose_action(next_state)
            obs = next_obs

        learner.epsilon = max(epsilon_min, learner.epsilon * epsilon_decay)

    learner.save_q_table(output_path)
    print(f"Saved Q-table to: {output_path}")
    print(f"Total learned states: {len(learner.q_table)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLearning V1 agent training utility.")
    parser.add_argument("--train", action="store_true", help="Enable local training mode.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument(
        "--enemy_type",
        choices=["simple", "smarter", "genius", "tactical", "box_farmer"],
        default="simple",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.999)
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent / "q_table.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.train:
        train_qlearning(
            episodes=args.episodes,
            enemy_type=args.enemy_type,
            seed=args.seed,
            output_path=Path(args.output),
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
        )
    else:
        print("No action selected. Use --train to run local Q-learning training.")
