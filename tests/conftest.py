"""Pytest configuration for testing Stim backend."""

import pytest
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture
def fx_bg() -> PCG64:
    """Return a random number generator."""
    return PCG64(SEED)


@pytest.fixture
def fx_rng(fx_bg: PCG64) -> Generator:
    """Return a random number generator."""
    return Generator(fx_bg)
