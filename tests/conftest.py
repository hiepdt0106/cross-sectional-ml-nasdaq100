"""Pytest fixtures and global setup.

Pins numpy + Python `random` seeds at session start so test outcomes do not
drift across numpy/scipy versions or pytest reorderings.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

SEED = 42


@pytest.fixture(autouse=True)
def _pin_global_seeds():
    np.random.seed(SEED)
    random.seed(SEED)
    yield
