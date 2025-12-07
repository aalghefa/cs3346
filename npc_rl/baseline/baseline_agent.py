from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

Obs = Union[np.ndarray, Dict[str, Any]]


@dataclass(frozen=True)
class BaselineConfig:
    retreat_threshold: float = 0.34
    prefer_strafe: bool = True
    move_priority: Tuple[int, ...] = (1, 4, 2, 3, 0)


class BaselineAgent:
    def __init__(self, cfg: Optional[BaselineConfig] = None):
        self.cfg = cfg or BaselineConfig()

    def _parse_obs(self, obs: Obs) -> Tuple[float, float, float, float, int, int]:
        if isinstance(obs, dict):
            dx = float(obs["dx"])
            dy = float(obs["dy"])
            hA = float(obs["hA"])
            hO = float(obs["hO"])
            los = int(obs["los"])
            cooldown = int(obs["cooldown"])
            return dx, dy, hA, hO, los, cooldown

        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.shape[0] < 6:
            raise ValueError(f"Expected obs length >= 6, got {arr.shape[0]}")
        dx, dy, hA, hO, los, cooldown = arr[:6]
        return float(dx), float(dy), float(hA), float(hO), int(round(los)), int(round(cooldown))

    @staticmethod
    def _encode_action(move: int, fire: int) -> int:
        return 2 * int(move) + int(fire)

    @staticmethod
    def _move_vectors() -> Dict[int, Tuple[int, int]]:
        return {
            0: (0, 0),
            1: (0, -1),
            2: (0, 1),
            3: (-1, 0),
            4: (1, 0),
        }

    @staticmethod
    def _manhattan(dx: float, dy: float) -> float:
        return abs(dx) + abs(dy)

    def _best_move_increase_distance(self, dx: float, dy: float, valid_moves: Optional[Tuple[int, ...]]) -> int:
        candidates = valid_moves if valid_moves is not None else (0, 1, 2, 3, 4)
        mv = self._move_vectors()

        def score(move: int) -> Tuple[float, int]:
            vx, vy = mv[move]
            dx2 = dx - vx
            dy2 = dy - vy
            dist = self._manhattan(dx2, dy2)
            pri = self.cfg.move_priority.index(move) if move in self.cfg.move_priority else 999
            return (dist, -pri)

        best = None
        best_move = 0
        for m in candidates:
            sc = score(m)
            if best is None or sc > best:
                best = sc
                best_move = m
        return best_move

    def _best_move_reduce_alignment(self, dx: float, dy: float, valid_moves: Optional[Tuple[int, ...]]) -> int:
        candidates = valid_moves if valid_moves is not None else (0, 1, 2, 3, 4)
        mv = self._move_vectors()
        target_axis = "x" if abs(dx) > abs(dy) else "y"

        def score(move: int) -> Tuple[float, float, int]:
            vx, vy = mv[move]
            dx2 = dx - vx
            dy2 = dy - vy
            ax = abs(dx2) if target_axis == "x" else abs(dy2)
            dist = self._manhattan(dx2, dy2)
            pri = self.cfg.move_priority.index(move) if move in self.cfg.move_priority else 999
            return (-ax, -dist, -pri)

        best = None
        best_move = 0
        for m in candidates:
            sc = score(m)
            if best is None or sc > best:
                best = sc
                best_move = m
        return best_move

    def _strafe_move(self, dx: float, dy: float, valid_moves: Optional[Tuple[int, ...]]) -> int:
        candidates = valid_moves if valid_moves is not None else (0, 1, 2, 3, 4)
        options = [1, 2] if abs(dx) >= abs(dy) else [3, 4]

        for m in self.cfg.move_priority:
            if m in options and m in candidates:
                return m
        for m in self.cfg.move_priority:
            if m in candidates:
                return m
        return 0

    def act(self, obs: Obs, valid_moves: Optional[Tuple[int, ...]] = None) -> int:
        dx, dy, hA, hO, los, cooldown = self._parse_obs(obs)

        if hA <= self.cfg.retreat_threshold:
            if los == 1 and cooldown == 0:
                return self._encode_action(move=0, fire=1)
            move = self._best_move_increase_distance(dx, dy, valid_moves)
            return self._encode_action(move=move, fire=0)

        if los == 1:
            if cooldown == 0:
                return self._encode_action(move=0, fire=1)
            if self.cfg.prefer_strafe:
                move = self._strafe_move(dx, dy, valid_moves)
                return self._encode_action(move=move, fire=0)
            return self._encode_action(move=0, fire=0)

        move = self._best_move_reduce_alignment(dx, dy, valid_moves)
        return self._encode_action(move=move, fire=0)
