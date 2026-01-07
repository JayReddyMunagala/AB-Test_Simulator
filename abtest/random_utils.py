from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SeededRandom:
    """A tiny seeded RNG matching the original TS implementation.

    Linear Congruential Generator (LCG):
      seed = (seed * 9301 + 49297) % 233280
      next = seed / 233280

    This is not cryptographically secure; it exists only to make simulations
    reproducible and to match the behavior of the original project.
    """

    seed: int

    def next(self) -> float:
        self.seed = (self.seed * 9301 + 49297) % 233280
        return self.seed / 233280.0

    @staticmethod
    def from_string(seed_str: str) -> "SeededRandom":
        # Ported from TS: simple string hash
        h = 0
        for ch in seed_str:
            char = ord(ch)
            h = ((h << 5) - h) + char
            h = h & h  # keep 32-bit signed behavior
        return SeededRandom(abs(h))


def generate_seed() -> str:
    return str(int(random.random() * 1_000_000))
