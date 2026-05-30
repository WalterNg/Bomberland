from __future__ import annotations

import os
from pathlib import Path

import torch

from config import (
    CHECKPOINT_ENV_VAR,
    CONFIG_PATH,
    CURRENT_DIR,
    MODEL_FILENAME,
    load_agent_config,
    resolve_config_path,
)
from core import ACTION_DIM, SPATIAL_CHANNELS, VECTOR_FEATURES, Action, DuelingDQN, SafetyShield, encode_observation
from weight_store import latest_checkpoint_path


def ensure_torch_single_thread() -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class InferencePolicy:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        ensure_torch_single_thread()

        self.config, self.config_path = load_agent_config(config_path)
        self.config_base_dir = self.config_path.parent
        device_name = device or str(self.config.get("inference", {}).get("device", "cpu"))
        self.device = torch.device(device_name)
        self.shield = SafetyShield()
        self.model = DuelingDQN(
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_FEATURES,
            action_dim=ACTION_DIM,
        ).to(self.device)
        self.model.eval()

        paths_cfg = self.config.get("paths", {})
        self.model_filename = str(paths_cfg.get("model_filename", MODEL_FILENAME))

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.load_checkpoint(self.checkpoint_path)

    def _checkpoint_env_override(self) -> Path | None:
        value = os.environ.get(CHECKPOINT_ENV_VAR)
        if not value:
            return None
        return Path(value)

    def _resolve_checkpoint_path(self, checkpoint_path: str | Path | None) -> Path:
        explicit = checkpoint_path if checkpoint_path is not None else self._checkpoint_env_override()
        configured = self.config.get("inference", {}).get("checkpoint_path")

        for candidate in (
            explicit,
            configured,
            latest_checkpoint_path(config_path=self.config_path),
            CURRENT_DIR / self.model_filename,
        ):
            if candidate is None:
                continue

            if isinstance(candidate, Path):
                path = candidate if candidate.is_absolute() else resolve_config_path(candidate, self.config_base_dir)
            else:
                path = resolve_config_path(candidate, self.config_base_dir)
            if path is not None and Path(path).exists():
                return Path(path)

        return CURRENT_DIR / self.model_filename

    def load_checkpoint(self, checkpoint_path: str | Path) -> tuple[int, int]:
        path = Path(checkpoint_path)
        if not path.exists():
            return 0, 0
        copied, partial = self.model.load_flexible_state_dict(path, map_location=self.device)
        return copied, partial

    def act(self, obs: dict, agent_id: int, step_index: int = 0) -> int:
        bundle = encode_observation(obs, agent_id=agent_id, step_index=step_index)
        if not bundle.alive:
            return Action.STOP

        spatial = torch.from_numpy(bundle.spatial).unsqueeze(0).to(self.device)
        vector = torch.from_numpy(bundle.vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(spatial, vector).squeeze(0)
        return self.shield.select_action(bundle, q_values)
