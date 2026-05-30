from __future__ import annotations

from config import load_agent_config
from inference import InferencePolicy

__all__ = ["Agent"]


TEAM_ID = load_agent_config()[0]["agent"]["team_id"]


class Agent:
    team_id = TEAM_ID

    def __init__(self, agent_id: int) -> None:
        self.agent_id = int(agent_id)
        self.step_count = 0
        self.policy = InferencePolicy()

    def act(self, obs: dict) -> int:
        self.step_count += 1
        return self.policy.act(obs, self.agent_id, step_index=self.step_count)


if __name__ == "__main__":
    from train import main as training_main

    raise SystemExit(training_main())
