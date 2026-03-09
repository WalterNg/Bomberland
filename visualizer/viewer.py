import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.append(str(ROOT_DIR))

from engine.game import BomberEnv
from agent.random_agent import RandomAgent
from agent.simple_rule_agent import SimpleRuleAgent
from agent.smarter_rule_agent import SmarterRuleAgent
from agent.genius_rule_agent import GeniusRuleAgent

class Viewer:
	def __init__(self, width=13, height=13, cell_size=42, fps=8):
		self.width = width
		self.height = height
		self.cell_size = cell_size
		self.fps = fps

		self.top_bar = 46
		self.screen_width = width * cell_size
		self.screen_height = height * cell_size + self.top_bar

		pygame.init()
		self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		pygame.display.set_caption("Bomberland Simple Viewer")
		self.clock = pygame.time.Clock()
		self.font_info = pygame.font.SysFont(None, 24)
		self.font_small = pygame.font.SysFont(None, 20)

	def draw_grid(self, grid):
		for x in range(self.width):
			for y in range(self.height):
				rect = pygame.Rect(
					x * self.cell_size,
					y * self.cell_size + self.top_bar,
					self.cell_size,
					self.cell_size,
				)
				if grid[x, y] == 1:
					pygame.draw.rect(self.screen, (120, 120, 120), rect)
				else:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.rect(self.screen, (190, 190, 190), rect, 1)

	def draw_players(self, players):
		colors = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]
		for i, p in enumerate(players):
			if p[2] != 1:
				continue
			center = (
				int(p[0]) * self.cell_size + self.cell_size // 2,
				int(p[1]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, colors[i % len(colors)], center, self.cell_size // 3)
			img = self.font_small.render(str(i), True, (255, 255, 255))
			self.screen.blit(img, (center[0] - 5, center[1] - 8))

	def draw_bombs(self, bombs):
		for b in bombs:
			if b[2] <= 0:
				continue
			center = (
				int(b[0]) * self.cell_size + self.cell_size // 2,
				int(b[1]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 4)
			timer_img = self.font_small.render(str(int(b[2])), True, (255, 255, 255))
			self.screen.blit(timer_img, (center[0] - 5, center[1] - 8))

	def draw_header(self, episode_idx, total_episodes, step_idx, total_steps, paused):
		pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.screen_width, self.top_bar))
		status = "PAUSED" if paused else "PLAY"
		text = (
			f"Episode {episode_idx + 1}/{total_episodes} | "
			f"Step {step_idx}/{max(total_steps - 1, 0)} | {status}"
		)
		help_text = "[A/D] Prev/Next Step   [W/S] Prev/Next Episode   [SPACE] Pause/Play   [ESC] Quit"
		self.screen.blit(self.font_info.render(text, True, (245, 245, 245)), (10, 5))
		self.screen.blit(self.font_small.render(help_text, True, (210, 210, 210)), (10, 25))

	def render(self, obs, episode_idx, total_episodes, step_idx, total_steps, paused):
		self.screen.fill((245, 245, 245))
		self.draw_header(episode_idx, total_episodes, step_idx, total_steps, paused)
		self.draw_grid(obs["map"])
		self.draw_players(obs["players"])
		self.draw_bombs(obs["bombs"])
		pygame.display.flip()
		self.clock.tick(self.fps)

	def close(self):
		pygame.quit()


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = str(value).strip().lower()
	if value in {"true", "1", "yes", "y", "t"}:
		return True
	if value in {"false", "0", "no", "n", "f"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def make_agents(model_paths):
	agents = [None] * len(model_paths)
	names = [None] * len(model_paths)

	for i, path in enumerate(model_paths):
		if path != "None":
			names[i] = path.split("/")[-2] if "/" in path else f"ModelAgent{i}"
			pass # TODO: train models -> load models from model_paths and create agents for testing
		else:
			x = random.randint(0, 3)
			if x == 0:
				names[i] = "RandomAgent"
				agents[i] = RandomAgent(i)
			elif x == 1:
				names[i] = "SimpleRuleAgent"
				agents[i] = SimpleRuleAgent(i)
			elif x == 2:
				names[i] = "SmarterRuleAgent"
				agents[i] = SmarterRuleAgent(i)
			else:
				names[i] = "GeniusRuleAgent"
				agents[i] = GeniusRuleAgent(i)
	return agents, names


def clone_obs(obs):
	return {
		"map": np.array(obs["map"], copy=True),
		"players": np.array(obs["players"], copy=True),
		"bombs": np.array(obs["bombs"], copy=True),
	}


def simulate_episodes(model_paths, num_episodes=10, max_steps=100, seed=None):
	env = BomberEnv(max_steps=max_steps)
	agents, names = make_agents(model_paths)
	episodes = []

	for episode in range(num_episodes):
		episode_seed = None if seed is None else seed + episode
		obs = env.reset(seed=episode_seed)
		done = False
		step = 0
		trajectory = [clone_obs(obs)]

		while not done and step < max_steps:
			actions = [agent.act(obs) for agent in agents]
			obs, terminated, truncated = env.step(actions)
			trajectory.append(clone_obs(obs))
			done = terminated or truncated
			step += 1

		episodes.append(trajectory)

	return episodes, names


def run_simple_viewer(model_paths, num_episodes=10, max_steps=100, seed=None, autoplay=True):
	episodes, agent_names = simulate_episodes(
		model_paths=model_paths,
		num_episodes=num_episodes,
		max_steps=max_steps,
		seed=seed,
	)
	if not episodes:
		print("No episodes to display.")
		return

	first_obs = episodes[0][0]
	viewer = Viewer(width=first_obs["map"].shape[0], height=first_obs["map"].shape[1])

	print("Agents:", ", ".join(agent_names))
	print("Controls: A/D step, W/S episode, SPACE pause/play, ESC quit")

	episode_idx = 0
	step_idx = 0
	paused = not autoplay
	last_tick = time.time()

	running = True
	while running:
		now = time.time()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
				elif event.key == pygame.K_d:
					step_idx = min(step_idx + 1, len(episodes[episode_idx]) - 1)
					paused = True
				elif event.key == pygame.K_a:
					step_idx = max(step_idx - 1, 0)
					paused = True
				elif event.key == pygame.K_s:
					episode_idx = min(episode_idx + 1, len(episodes) - 1)
					step_idx = 0
				elif event.key == pygame.K_w:
					episode_idx = max(episode_idx - 1, 0)
					step_idx = 0

		if not paused and (now - last_tick) >= (1 / max(viewer.fps, 1)):
			if step_idx < len(episodes[episode_idx]) - 1:
				step_idx += 1
			last_tick = now

		current_obs = episodes[episode_idx][step_idx]
		viewer.render(
			obs=current_obs,
			episode_idx=episode_idx,
			total_episodes=len(episodes),
			step_idx=step_idx,
			total_steps=len(episodes[episode_idx]),
			paused=paused,
		)

	viewer.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_paths", nargs="+", default=["None", "None"])
	parser.add_argument("--num_episodes", type=int, default=10)
	parser.add_argument("--max_steps", type=int, default=100)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--autoplay", type=str2bool, default=True)
	args = parser.parse_args()

	run_simple_viewer(
		model_paths=args.model_paths,
		num_episodes=args.num_episodes,
		max_steps=args.max_steps,
		seed=args.seed,
		autoplay=args.autoplay,
	)
