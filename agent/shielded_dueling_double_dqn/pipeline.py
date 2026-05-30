from __future__ import annotations

import random
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import MAX_PLAYERS, TrainingConfig
from core import (
    ACTION_DIM,
    SPATIAL_CHANNELS,
    VECTOR_FEATURES,
    DuelingDQN,
    NStepAccumulator,
    PrioritizedReplayBuffer,
    RawTransition,
    SafetyShield,
    encode_observation,
)
from opponents import spawn_opponents
from reward import compute_reward, final_rank_bonus
from weight_store import checkpoint_path_for_episode, create_run_dir, last_checkpoint_path
from utils import plot_loss, plot_moving_average, plot_rewards, plot_win_rates, save_checkpoint, seed_everything


def ensure_torch_single_thread() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def epsilon_by_step(
    global_step: int,
    epsilon_start: float,
    epsilon_final: float,
    epsilon_decay_steps: int,
) -> float:
    if global_step >= epsilon_decay_steps:
        return float(epsilon_final)
    progress = float(global_step) / float(max(1, epsilon_decay_steps))
    return float(epsilon_start - (epsilon_start - epsilon_final) * progress)


def beta_by_step(
    global_step: int,
    beta_start: float,
    beta_final: float,
    beta_growth_steps: int,
) -> float:
    if global_step >= beta_growth_steps:
        return float(beta_final)
    progress = float(global_step) / float(max(1, beta_growth_steps))
    return float(beta_start + (beta_final - beta_start) * progress)


@dataclass(slots=True)
class TrainingHistory:
    losses: list[float]
    rewards: list[float]
    wins: list[float]


@dataclass(slots=True)
class EpisodeRecord:
    episode: int
    reward: float
    win: float
    loss: float | None
    steps: int
    epsilon: float
    replay_size: int


class Learner:
    def __init__(
        self,
        spatial_channels: int = SPATIAL_CHANNELS,
        vector_dim: int = VECTOR_FEATURES,
        action_dim: int = ACTION_DIM,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str | None = None,
        load_model: str | None = None,
    ) -> None:
        ensure_torch_single_thread()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = float(gamma)
        self.model = DuelingDQN(spatial_channels=spatial_channels, vector_dim=vector_dim, action_dim=action_dim).to(self.device)
        self.target = DuelingDQN(spatial_channels=spatial_channels, vector_dim=vector_dim, action_dim=action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction="none")
        self.shield = SafetyShield()
        self.global_step = 0
        self.epsilon = 1.0
        if load_model:
            self.load(load_model)

    def load(self, path: str | Path) -> tuple[int, int]:
        copied, partial = self.model.load_flexible_state_dict(path, map_location=self.device)
        self.target.load_state_dict(self.model.state_dict())
        return copied, partial

    def act(self, bundle, epsilon: float) -> int:
        safe_actions = self.shield.safe_actions(bundle)
        if not safe_actions:
            return self.shield.fallback_escape_actions(bundle)[0]

        if random.random() < epsilon:
            return int(random.choice(safe_actions))

        spatial = torch.from_numpy(bundle.spatial).unsqueeze(0).to(self.device)
        vector = torch.from_numpy(bundle.vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(spatial, vector).squeeze(0)
        return self.shield.select_action(bundle, q_values)

    def optimize(self, batch: dict[str, np.ndarray], max_grad_norm: float = 10.0) -> tuple[float, np.ndarray]:
        spatial = torch.as_tensor(batch["spatial"], dtype=torch.float32, device=self.device)
        vector = torch.as_tensor(batch["vector"], dtype=torch.float32, device=self.device)
        next_spatial = torch.as_tensor(batch["next_spatial"], dtype=torch.float32, device=self.device)
        next_vector = torch.as_tensor(batch["next_vector"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        q_values = self.model(spatial, vector).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_spatial, next_vector).argmax(dim=1, keepdim=True)
            next_q = self.target(next_spatial, next_vector).gather(1, next_actions).squeeze(1)
            target = rewards + discounts * next_q * (1.0 - dones)

        td_errors = target - q_values
        loss = (weights * self.loss_fn(q_values, target)).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return float(loss.item()), td_errors.detach().abs().cpu().numpy()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        save_checkpoint(
            self.model,
            self.optimizer,
            self.global_step,
            self.epsilon,
            float(self.optimizer.param_groups[0]["lr"]),
            (SPATIAL_CHANNELS, 13, 13),
            VECTOR_FEATURES,
            ACTION_DIM,
            path,
            metadata=metadata,
        )


class TrainingPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        ensure_torch_single_thread()
        seed_everything(config.seed)
        self.learner = Learner(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            load_model=config.pretrained_model,
        )
        self.replay = PrioritizedReplayBuffer(
            capacity=config.replay_capacity,
            spatial_shape=(SPATIAL_CHANNELS, 13, 13),
            vector_dim=VECTOR_FEATURES,
            alpha=0.6,
        )
        self.nstep = NStepAccumulator(n_step=config.n_step, gamma=config.gamma)
        self.history = TrainingHistory(losses=[], rewards=[], wins=[])
        self.episode_records: list[EpisodeRecord] = []
        self.global_step = 0
        self.run_dir = create_run_dir()
        self.model_path = last_checkpoint_path(self.run_dir)
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

    def run(self) -> Path | None:
        from engine.game import BomberEnv

        env = BomberEnv(max_steps=self.config.max_steps, seed=self.config.seed)
        episode_bar = tqdm(
            total=self.config.num_episodes,
            desc="Training",
            unit="episode",
            dynamic_ncols=True,
            position=0,
        )
        try:
            for episode in range(self.config.num_episodes):
                obs = env.reset(seed=self.config.seed + episode)
                prev_obs = None
                done = False
                episode_reward = 0.0
                episode_steps = 0
                last_loss: float | None = None
                survival_steps = np.zeros(MAX_PLAYERS, dtype=np.float32)
                opponents = spawn_opponents(self.config.enemy_type, self.config.opponent_paths)

                step_bar = tqdm(
                    total=self.config.max_steps,
                    desc=f"Episode {episode + 1}/{self.config.num_episodes}",
                    unit="step",
                    leave=False,
                    dynamic_ncols=True,
                    position=1,
                )
                try:
                    while not done:
                        bundle = encode_observation(
                            obs,
                            agent_id=0,
                            step_index=self.global_step,
                            max_steps=self.config.max_steps,
                        )
                        epsilon = epsilon_by_step(
                            self.global_step,
                            self.config.epsilon_start,
                            self.config.epsilon_final,
                            self.config.epsilon_decay_steps,
                        )
                        action = self.learner.act(bundle, epsilon)

                        actions = [action]
                        for pid in range(1, MAX_PLAYERS):
                            actions.append(opponents[pid].act(obs))

                        next_obs, terminated, truncated = env.step(actions)
                        done = terminated or truncated
                        reward = compute_reward(prev_obs, next_obs, agent_id=0)
                        survival_steps += np.asarray(next_obs["players"], dtype=np.int32)[:, 2].astype(np.float32)
                        if done:
                            reward += final_rank_bonus(next_obs, agent_id=0, survival_steps=survival_steps)

                        next_bundle = encode_observation(
                            next_obs,
                            agent_id=0,
                            step_index=self.global_step + 1,
                            max_steps=self.config.max_steps,
                        )
                        for entry in self.nstep.push(
                            RawTransition(
                                spatial=bundle.spatial,
                                vector=bundle.vector,
                                action=int(action),
                                reward=float(reward),
                                next_spatial=next_bundle.spatial,
                                next_vector=next_bundle.vector,
                                done=bool(done),
                            )
                        ):
                            self.replay.add(entry)

                        episode_reward += reward
                        episode_steps += 1
                        self.global_step += 1
                        self.learner.global_step = self.global_step
                        self.learner.epsilon = epsilon

                        if len(self.replay) >= self.config.learning_starts and self.global_step % self.config.train_freq == 0:
                            beta = beta_by_step(
                                self.global_step,
                                self.config.beta_start,
                                self.config.beta_final,
                                self.config.beta_growth_steps,
                            )
                            batch = self.replay.sample(batch_size=self.config.batch_size, beta=beta)
                            loss, td_errors = self.learner.optimize(batch, max_grad_norm=self.config.max_grad_norm)
                            self.replay.update_priorities(batch["indices"], td_errors)
                            self.history.losses.append(loss)
                            last_loss = loss

                        if self.global_step % self.config.target_update_interval == 0:
                            self.learner.sync_target()

                        prev_obs = obs
                        obs = next_obs
                        step_bar.update(1)
                        if done:
                            break
                finally:
                    step_bar.close()

                self.history.rewards.append(float(episode_reward))
                final_players = np.asarray(obs["players"], dtype=np.int32)
                win = 1.0 if int(final_players[0][2]) == 1 and int(np.sum(final_players[:, 2])) == 1 else 0.0
                self.history.wins.append(win)
                self.episode_records.append(
                    EpisodeRecord(
                        episode=episode + 1,
                        reward=float(episode_reward),
                        win=win,
                        loss=last_loss,
                        steps=episode_steps,
                        epsilon=epsilon,
                        replay_size=len(self.replay),
                    )
                )

                self.writer.add_scalar("train/episode_reward", float(episode_reward), episode + 1)
                self.writer.add_scalar("train/win", float(win), episode + 1)
                self.writer.add_scalar("train/episode_steps", float(episode_steps), episode + 1)
                self.writer.add_scalar("train/epsilon", float(epsilon), episode + 1)
                self.writer.add_scalar("train/replay_size", float(len(self.replay)), episode + 1)
                if last_loss is not None:
                    self.writer.add_scalar("train/loss", float(last_loss), episode + 1)

                episode_bar.set_postfix(
                    reward=f"{episode_reward:.3f}",
                    loss="-" if last_loss is None else f"{last_loss:.4f}",
                    win=f"{win:.0f}",
                    eps=f"{epsilon:.3f}",
                    replay=len(self.replay),
                    steps=episode_steps,
                )
                episode_bar.update(1)

                if self.config.checkpoint_interval_episodes > 0 and (episode + 1) % self.config.checkpoint_interval_episodes == 0:
                    self._save_snapshot(
                        checkpoint_path_for_episode(self.run_dir, episode + 1),
                        metadata={
                            "enemy_type": self.config.enemy_type,
                            "num_episodes": self.config.num_episodes,
                            "max_steps": self.config.max_steps,
                            "seed": self.config.seed,
                            "episode": episode + 1,
                        },
                    )
        finally:
            episode_bar.close()
            self.writer.flush()
            self.writer.close()

        if self.config.save_model:
            self._save_snapshot(
                last_checkpoint_path(self.run_dir),
                metadata={
                    "enemy_type": self.config.enemy_type,
                    "num_episodes": self.config.num_episodes,
                    "max_steps": self.config.max_steps,
                    "seed": self.config.seed,
                    "final": True,
                },
            )
            self._save_plots()
            self._save_metrics()
            return self.model_path

        return None

    def _save_snapshot(self, path: Path, metadata: dict | None = None) -> Path:
        self.learner.save(path, metadata=metadata)
        self.model_path = path
        return path

    def _save_plots(self) -> None:
        plot_loss(self.history.losses, save_path=self.run_dir / "loss.png")
        plot_rewards(self.history.rewards, save_path=self.run_dir / "rewards.png")
        plot_win_rates(self.history.wins, save_path=self.run_dir / "win_rates.png")
        plot_moving_average(
            self.history.rewards,
            window_size=10,
            save_path=self.run_dir / "moving_average.png",
        )

    def _save_metrics(self) -> None:
        path = self.run_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["episode", "reward", "win", "loss", "steps", "epsilon", "replay_size"],
            )
            writer.writeheader()
            for record in self.episode_records:
                writer.writerow(
                    {
                        "episode": record.episode,
                        "reward": f"{record.reward:.6f}",
                        "win": f"{record.win:.0f}",
                        "loss": "" if record.loss is None else f"{record.loss:.6f}",
                        "steps": record.steps,
                        "epsilon": f"{record.epsilon:.6f}",
                        "replay_size": record.replay_size,
                    }
                )
