from __future__ import annotations

import argparse

from config import TrainingConfig, load_agent_config
from pipeline import TrainingPipeline


DEFAULT_TRAINING_CONFIG = load_agent_config()[0]["training"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Shielded Dueling Double DQN Bomberland agent.")
    parser.set_defaults(save_model=DEFAULT_TRAINING_CONFIG["save_model"])
    parser.add_argument("--enemy_type", type=str, default=DEFAULT_TRAINING_CONFIG["enemy_type"], choices=["simple", "smarter", "tactical", "genius", "box_farmer", "random", "league"])
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_TRAINING_CONFIG["num_episodes"])
    parser.add_argument("--max_steps", type=int, default=DEFAULT_TRAINING_CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_CONFIG["seed"])
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--no-save_model", dest="save_model", action="store_false")
    parser.add_argument("--load_model", type=str, default=DEFAULT_TRAINING_CONFIG["pretrained_model"])
    parser.add_argument("--opponent_paths", nargs="*", default=tuple(DEFAULT_TRAINING_CONFIG["opponent_paths"]))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAINING_CONFIG["batch_size"])
    parser.add_argument("--learning_starts", type=int, default=DEFAULT_TRAINING_CONFIG["learning_starts"])
    parser.add_argument("--train_freq", type=int, default=DEFAULT_TRAINING_CONFIG["train_freq"])
    parser.add_argument("--target_update_interval", type=int, default=DEFAULT_TRAINING_CONFIG["target_update_interval"])
    parser.add_argument("--replay_capacity", type=int, default=DEFAULT_TRAINING_CONFIG["replay_capacity"])
    parser.add_argument("--n_step", type=int, default=DEFAULT_TRAINING_CONFIG["n_step"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_TRAINING_CONFIG["learning_rate"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_TRAINING_CONFIG["gamma"])
    parser.add_argument("--epsilon_start", type=float, default=DEFAULT_TRAINING_CONFIG["epsilon_start"])
    parser.add_argument("--epsilon_final", type=float, default=DEFAULT_TRAINING_CONFIG["epsilon_final"])
    parser.add_argument("--epsilon_decay_steps", type=int, default=DEFAULT_TRAINING_CONFIG["epsilon_decay_steps"])
    parser.add_argument("--beta_start", type=float, default=DEFAULT_TRAINING_CONFIG["beta_start"])
    parser.add_argument("--beta_final", type=float, default=DEFAULT_TRAINING_CONFIG["beta_final"])
    parser.add_argument("--beta_growth_steps", type=int, default=DEFAULT_TRAINING_CONFIG["beta_growth_steps"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_TRAINING_CONFIG["max_grad_norm"])
    parser.add_argument("--checkpoint_interval_episodes", type=int, default=DEFAULT_TRAINING_CONFIG["checkpoint_interval_episodes"])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = TrainingConfig(
        enemy_type=args.enemy_type,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        save_model=args.save_model,
        pretrained_model=args.load_model,
        opponent_paths=tuple(args.opponent_paths or ()),
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        replay_capacity=args.replay_capacity,
        n_step=args.n_step,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        beta_start=args.beta_start,
        beta_final=args.beta_final,
        beta_growth_steps=args.beta_growth_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_interval_episodes=args.checkpoint_interval_episodes,
    )

    pipeline = TrainingPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
