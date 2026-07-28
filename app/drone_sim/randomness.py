from __future__ import annotations

import numpy as np


def make_rng(random_seed: int) -> np.random.Generator:
    """Create the only random source used by one reproducible simulation run."""
    return np.random.default_rng(int(random_seed))
