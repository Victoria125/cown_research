import numpy as np
import pandas as pd
import pytest

from src.barn import add_barn_exposure, barn_exposure_series


@pytest.mark.parametrize("mu", [0.01, 0.07, 1.0, 10.0])
def test_one_infected_day_in_clean_barn_gives_exposure_one(mu):
    # a escala do artigo: uma infectada num ambiente limpo gera exposição 1 no primeiro dia
    exposure = barn_exposure_series(np.array([0, 1, 0, 0]), mu)
    assert exposure[0] == 0
    assert exposure[1] == pytest.approx(1)
    assert exposure[2] > exposure[3] > 0  # a contaminação que sobrou decai


def test_fast_decay_is_direct_transmission():
    # o que sobra do dia anterior é cerca de 1 / (mu - 1) por infectada
    infected = np.array([2, 0, 1, 3])
    assert barn_exposure_series(infected, 1000.0) == pytest.approx(infected, abs=0.01)


def test_barn_starts_in_equilibrium():
    exposure = barn_exposure_series(np.array([2, 2, 2, 2]), 0.07)
    assert exposure == pytest.approx(np.full(4, exposure[0]))


def test_cows_in_same_barn_share_exposure_divided_by_n():
    daily = pd.DataFrame({
        "Agent": [1, 2, 3, 1, 2, 3],
        "barn": 60,
        "date": pd.to_datetime(["2024-06-01"] * 3 + ["2024-06-02"] * 3),
        "INFECTED": [0, 0, 0, 1, 0, 0],
        "Exp": np.nan,
    })
    out = add_barn_exposure(daily, mu=0.07).sort_values(["date", "Agent"])
    assert out["Exp"].tolist() == pytest.approx([0, 0, 0, 1 / 3, 1 / 3, 1 / 3])
