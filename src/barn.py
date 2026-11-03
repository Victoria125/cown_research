"""Exposição por baia (ambiente homogêneo), pelo método de Chang & de Jong (2023).

Cada baia tem uma contaminação E que cresce com as vacas infectadas e decai à taxa mu
(por dia). A exposição é escalada para que uma vaca infectada num ambiente limpo gere
exposição 1 no primeiro dia, qualquer que seja mu (eq. 2 e 3 do artigo, com tau = 1 dia).

Diferente do PeDViS-BDD, não usa a posição das vacas: toda vaca da baia recebe a mesma
exposição no dia, dividida pelo número de vacas da baia (beta * E * S / N).
"""

import numpy as np
import pandas as pd


def shedding_rate(mu: float) -> float:
    """phi(mu) = mu^2 / (mu - 1 + e^-mu), a escala do artigo (seção 2.2)."""
    return mu**2 / (mu - 1 + np.exp(-mu))


def barn_exposure_series(infected: np.ndarray, mu: float) -> np.ndarray:
    """Exposição de cada dia numa baia, dado o número de vacas infectadas em cada dia.

    A baia começa em equilíbrio com as infectadas do primeiro dia, porque já havia vacas
    infectadas antes do início dos dados.
    """
    phi = shedding_rate(mu)
    decay = np.exp(-mu)
    gain = (1 - decay) / mu
    contamination = phi * infected[0] / mu
    exposure = np.empty(len(infected))
    for day, n_infected in enumerate(infected):
        exposure[day] = n_infected + gain * contamination
        contamination = gain * phi * n_infected + decay * contamination
    return exposure


def add_barn_exposure(daily: pd.DataFrame, mu: float) -> pd.DataFrame:
    """Troca a coluna Exp do arquivo diário pela exposição por baia (exposição / N)."""
    per_day = daily.groupby(["barn", "date"]).agg(I=("INFECTED", "sum"), N=("Agent", "size"))
    parts = []
    for barn, g in per_day.reset_index("barn").groupby("barn"):
        # dias sem nenhuma vaca na baia contam como zero infectadas
        days = pd.date_range(g.index.min(), g.index.max(), name="date")
        g = g.reindex(days)
        g["barn"] = barn
        g["barn_exp"] = barn_exposure_series(g["I"].fillna(0).to_numpy(), mu)
        parts.append(g.dropna(subset=["N"]))
    barns = pd.concat(parts).reset_index()
    barns["Exp"] = barns["barn_exp"] / barns["N"]
    return daily.drop(columns="Exp").merge(
        barns[["barn", "date", "Exp"]], on=["barn", "date"], how="left", validate="many_to_one",
    )
