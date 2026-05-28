from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class StrategyDecision:
    profile_name: str
    overrides: Dict[str, float]
    pause_bomb_spam: bool = False
    force_bomb: bool = False


class Strategy:
    """Dynamic phase selector for the tactical agent.

    The policy increases aggression as enemies and boxes disappear.
    In solo endgame it prefers a three-bomb burst pattern, but pauses
    bomb spam when a nearby item becomes a better short-term target.
    """

    def __init__(
        self,
        *,
        solo_bomb_limit: int = 3,
        item_pause_distance: int = 4,
        collector_threshold: float = 0.35,
        pressure_threshold: float = 0.7,
    ) -> None:
        self.solo_bomb_limit = max(1, int(solo_bomb_limit))
        self.item_pause_distance = max(1, int(item_pause_distance))
        self.collector_threshold = float(collector_threshold)
        self.pressure_threshold = float(pressure_threshold)
        self._initial_enemy_count: Optional[int] = None
        self._initial_box_count: Optional[int] = None
        self._solo_bomb_burst = 0

    def decide(
        self,
        *,
        alive_count: int,
        box_count: int,
        nearest_item_distance: Optional[int],
        item_targets_present: bool,
    ) -> StrategyDecision:
        enemy_count = max(0, int(alive_count) - 1)
        self._seed_initial_counts(enemy_count=enemy_count, box_count=box_count)

        if alive_count <= 2:
            pause_bomb_spam = self._should_pause_solo_bombing(
                nearest_item_distance=nearest_item_distance,
                item_targets_present=item_targets_present,
            )
            if pause_bomb_spam:
                self._solo_bomb_burst = 0

            overrides = self._solo_overrides()
            return StrategyDecision(
                profile_name="aggressive",
                overrides=overrides,
                pause_bomb_spam=pause_bomb_spam,
                force_bomb=not pause_bomb_spam and self.can_spam_bomb(),
            )

        aggression = self._aggression_index(enemy_count=enemy_count, box_count=box_count)
        phase = self._phase_for_aggression(aggression=aggression, enemy_count=enemy_count, box_count=box_count)
        self._last_phase = phase
        return StrategyDecision(
            profile_name="collector" if phase == "collector" else "aggressive",
            overrides=self._phase_overrides(phase),
        )

    def can_spam_bomb(self) -> bool:
        return self._solo_bomb_burst < self.solo_bomb_limit

    def record_bomb_drop(self, *, solo: bool) -> None:
        if solo:
            self._solo_bomb_burst += 1
        else:
            self._solo_bomb_burst = 0

    def reset_solo_bomb_burst(self) -> None:
        self._solo_bomb_burst = 0

    def _should_pause_solo_bombing(
        self,
        *,
        nearest_item_distance: Optional[int],
        item_targets_present: bool,
    ) -> bool:
        return (
            item_targets_present
            and nearest_item_distance is not None
            and nearest_item_distance <= self.item_pause_distance
        )

    def _seed_initial_counts(self, *, enemy_count: int, box_count: int) -> None:
        if self._initial_enemy_count is None:
            self._initial_enemy_count = max(1, int(enemy_count))
        if self._initial_box_count is None:
            self._initial_box_count = max(1, int(box_count))

    def _aggression_index(self, *, enemy_count: int, box_count: int) -> float:
        enemy_base = max(1, self._initial_enemy_count or 1)
        box_base = max(1, self._initial_box_count or 1)
        enemy_ratio = min(1.0, max(0.0, float(enemy_count) / float(enemy_base)))
        box_ratio = min(1.0, max(0.0, float(box_count) / float(box_base)))
        return (1.0 - enemy_ratio) * 0.55 + (1.0 - box_ratio) * 0.45

    def _phase_for_aggression(self, *, aggression: float, enemy_count: int, box_count: int) -> str:
        if enemy_count <= 1:
            return "solo_spam"
        if aggression < self.collector_threshold:
            return "collector"
        if aggression < self.pressure_threshold:
            return "pressure"
        if box_count <= max(6, (self._initial_box_count or 1) // 5):
            return "pressure"
        return "siege"

    def _phase_overrides(self, phase: str) -> Dict[str, float]:
        if phase == "collector":
            return {}
        if phase == "pressure":
            return {
                "ITEM_SCORE_MULTIPLIER": 1.7,
                "ITEM_PROGRESS_BONUS": 24.0,
                "ITEM_STEP_BONUS": 140.0,
                "BOMB_DECISION_MARGIN": 12.0,
                "BOMB_TACTICAL_BONUS": 220.0,
                "STOP_AVOID_MARGIN": 8.0,
                "WASTE_BOMB_PENALTY": -60.0,
                "STOP_PENALTY": -20.0,
            }
        if phase == "siege":
            return {
                "ITEM_SCORE_MULTIPLIER": 1.3,
                "ITEM_PROGRESS_BONUS": 18.0,
                "ITEM_STEP_BONUS": 110.0,
                "BOMB_DECISION_MARGIN": 0.0,
                "BOMB_TACTICAL_BONUS": 280.0,
                "STOP_AVOID_MARGIN": 6.0,
                "WASTE_BOMB_PENALTY": -40.0,
                "STOP_PENALTY": -24.0,
            }
        return {
            "BOMB_DECISION_MARGIN": -45.0,
            "SOLO_BOMB_DECISION_MARGIN": -75.0,
            "BOMB_TACTICAL_BONUS": 180.0,
            "SOLO_BOMB_TACTICAL_BONUS": 360.0,
            "ITEM_SCORE_MULTIPLIER": 1.3,
            "ITEM_PROGRESS_BONUS": 18.0,
            "ITEM_STEP_BONUS": 110.0,
            "STOP_AVOID_MARGIN": 4.0,
            "RECENT_BOMB_ESCAPE_BONUS": 45.0,
            "STOP_PENALTY": -28.0,
        }

    def _solo_overrides(self) -> Dict[str, float]:
        return self._phase_overrides("solo_spam")
