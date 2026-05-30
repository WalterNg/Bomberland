from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model,
    optimizer,
    global_step: int,
    epsilon: float,
    lr: float,
    spatial_shape,
    vector_dim: int,
    num_actions: int,
    path: str | os.PathLike[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "global_step": int(global_step),
        "epsilon": float(epsilon),
        "lr": float(lr),
        "input_dim": (spatial_shape, vector_dim),
        "input_shape": (spatial_shape, vector_dim),
        "input_spec": (spatial_shape, vector_dim),
        "num_actions": int(num_actions),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_loss(loss_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)


def plot_rewards(reward_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Rewards")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)


def plot_win_rates(win_history, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(win_history, label="Win Rate")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("DQN Training Win Rates")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)


def plot_moving_average(data, window_size=10, save_path=None):
    ma_data = moving_average(data, window_size)
    plt.figure(figsize=(10, 5))
    plt.plot(ma_data, label=f"Moving Average (window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("DQN Training Moving Average")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)

