import numpy as np
import pandas as pd

from src.glm import fit_with_background, fit_without_background
from src.weekly import WINDOW_DAYS


def simulate(beta, lambda0, n=40_000, seed=0):
    """Semanas em risco com casos sorteados pelo próprio modelo."""
    rng = np.random.default_rng(seed)
    exp = rng.lognormal(mean=3, sigma=1.5, size=n)
    p = 1 - np.exp(-(lambda0 * WINDOW_DAYS + beta * exp))
    return pd.DataFrame({"C": (rng.random(n) < p).astype(int), "exp": exp})


def test_without_background_recovers_beta():
    fit = fit_without_background(simulate(beta=2e-4, lambda0=0))
    assert fit["beta_low"] < 2e-4 < fit["beta_high"]


def test_with_background_recovers_both_rates():
    fit = fit_with_background(simulate(beta=2e-4, lambda0=1e-3))
    assert fit["beta_low"] < 2e-4 < fit["beta_high"]
    assert abs(fit["lambda0"] - 1e-3) / 1e-3 < 0.3
    assert fit["p_exposure"] < 0.001


def test_exposure_without_effect_is_not_significant():
    fit = fit_with_background(simulate(beta=0, lambda0=1e-3))
    assert fit["beta_low"] == 0
    assert fit["p_exposure"] > 0.05


def test_without_background_handles_tiny_exposures():
    weeks = simulate(beta=2e-4, lambda0=0)
    weeks.loc[:50, "exp"] = 1e-300
    fit = fit_without_background(weeks)
    assert np.isfinite(fit["loglik"]) and fit["beta"] > 0
