"""DQN 经验回放数据结构。 / Experience replay data structures for DQN."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    """经验回放中的单条状态转移。 / One state transition stored in replay memory."""
    state: list[list[int]]
    action: int
    reward: float
    next_state: list[list[int]]
    done: bool
    legal_next_actions: list[int]


class ReplayBuffer:
    """固定容量随机采样经验回放池。 / Fixed-capacity replay buffer with random sampling."""
    def __init__(self, capacity: int, rng: random.Random | None = None):
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.data: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(list(self.data), batch_size)
