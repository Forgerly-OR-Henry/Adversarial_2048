"""随机种子设置工具。 / Random seed setup helper."""

from __future__ import annotations

import random


def set_seed(seed: int | None = None) -> random.Random:
    """设置 Python 随机数种子并返回 RNG。 / Set the Python random seed and return the RNG."""
    rng = random.Random(seed)
    random.seed(seed)
    return rng
