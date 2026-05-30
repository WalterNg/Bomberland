# Shielded Dueling Double DQN

This folder contains a standalone Bomberland agent with:

- Shielded deterministic inference
- Dueling Double DQN network
- Prioritized replay
- n-step returns
- Training and inference entrypoints

## Inference

Use this folder as the agent path in the competition or local match runner:

```bash
python -m scripts.participant.run_local_match --agent_paths agent/shielded_dueling_double_dqn None None None
```

The runtime entrypoint is `agent.py`, which exposes `Agent`.

The bundled `shielded_dueling_double_dqn.pth` is a bootstrap checkpoint so the
folder is runnable immediately. Training will overwrite it with a freshly
trained model.

## Training

Train the agent and save checkpoints into a timestamped run folder under
`runs/`:

```bash
python agent/shielded_dueling_double_dqn/train.py --save_model
```

The default `--enemy_type` is `league`, which samples from the baseline pool
and any custom opponents passed through `--opponent_paths`.

Training creates a folder like `runs/DDMMYY-HHMMSS/` and saves:

- periodic checkpoints every 100 episodes as `100.pth`, `200.pth`, ...
- the final checkpoint as `last.pth`
- training artifacts such as `loss.png`, `rewards.png`, `win_rates.png`, `moving_average.png`, and `metrics.csv`

You can override the checkpoint interval with `--checkpoint_interval_episodes`.
