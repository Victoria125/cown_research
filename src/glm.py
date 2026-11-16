"""Regressão por avaliação: taxa de transmissão beta, com e sem taxa de fundo lambda0.

Probabilidade de uma vaca saudável virar caso na semana (Chang & de Jong 2023):
    sem fundo:  p = 1 - exp(-beta * exp)                    = GLM binomial, cloglog, offset log(exp)
    com fundo:  p = 1 - exp(-(lambda0 * 7 + beta * exp))
Os dois são ajustados por máxima verossimilhança.
O modelo sem fundo não admite caso com exposição zero, então essas semanas ficam de fora. O
modelo com fundo usa todas e mede, em lambda0, o risco que não vem da exposição no piso.

Uso: python -m src.glm   (grava os resultados em data/processed/glm_*.parquet)
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2

from src.barn import add_barn_exposure
from src.daily import build_daily
from src.load import PROCESSED_DIR, load_exposure
from src.weekly import WINDOW_DAYS, build_weekly

# decaimento por dia, de 0.01 a 10, mais o 0.07 do arquivo do Gustavo
MU_GRID = np.unique(np.append(np.round(np.geomspace(0.01, 10, 31), 4), 0.07))
CI_DROP = chi2.ppf(0.95, 1) / 2  # queda no log-likelihood para o IC de 95%


def at_risk_weeks(weekly: pd.DataFrame, complete: bool = True) -> pd.DataFrame:
    """Semanas em que a vaca estava saudável na véspera e tem exposição.

    complete=True exige os 7 dias de exposição; False soma os dias disponíveis.
    """
    keep = weekly["S"].eq(1) & weekly["exp_days"].gt(0)
    if complete:
        keep &= weekly["complete"]
    return weekly[keep].reset_index(drop=True)


def _loglik(lambda0: float, beta: float, c: np.ndarray, exp: np.ndarray) -> float:
    """Log-verossimilhança, em escala log para não zerar com exposições muito pequenas."""
    with np.errstate(divide="ignore"):
        log_hazard = np.logaddexp(np.log(lambda0 * WINDOW_DAYS), np.log(beta) + np.log(exp))
    hazard = np.exp(log_hazard)
    # log(1 - e^-h), que vira log(h) quando h é muito pequeno
    small = hazard < 1e-8
    log_p = np.where(small, log_hazard, np.log(-np.expm1(-np.where(small, 1.0, hazard))))
    case = c == 1
    return log_p[case].sum() - hazard[~case].sum()


def fit_without_background(weeks: pd.DataFrame) -> dict:
    """O GLM cloglog com offset log(exp) do artigo, só nas semanas com exposição > 0.

    Ajustado por máxima verossimilhança direta (mesmo resultado do GLM), porque o GLM do
    statsmodels estoura com exposições muito pequenas; IC de beta por perfil.
    """
    used = weeks[weeks["exp"] > 0]
    c, exp = used["C"].to_numpy(), used["exp"].to_numpy()
    loglik_of = lambda log_beta: _loglik(0.0, np.exp(log_beta), c, exp)
    best = minimize_scalar(lambda lb: -loglik_of(lb), bounds=(-60, 10), method="bounded",
                           options={"xatol": 1e-9})
    loglik = -best.fun
    above = lambda lb: loglik_of(lb) - (loglik - CI_DROP)
    return {
        "model": "sem fundo", "weeks": len(used), "cases": int(c.sum()),
        "beta": np.exp(best.x),
        "beta_low": np.exp(brentq(above, -60, best.x)),
        "beta_high": np.exp(brentq(above, best.x, 10)),
        "lambda0": np.nan, "loglik": loglik, "aic": -2 * loglik + 2,
        "p_exposure": np.nan,
    }


def _profile_beta(beta: float, c: np.ndarray, exp: np.ndarray) -> float:
    """Melhor log-likelihood com beta fixo, escolhendo lambda0."""
    best = minimize_scalar(
        lambda log_l0: -_loglik(np.exp(log_l0), beta, c, exp), bounds=(-25, 5), method="bounded",
    )
    return -best.fun


def fit_with_background(weeks: pd.DataFrame) -> dict:
    """Máxima verossimilhança de lambda0 e beta; IC de beta por perfil de verossimilhança.

    p_exposure é o teste da razão de verossimilhança contra o modelo só com lambda0
    (beta = 0): diz se a exposição explica alguma coisa além do risco de fundo.
    """
    c, exp = weeks["C"].to_numpy(), weeks["exp"].to_numpy()
    rate = c.mean()
    start = [np.log(-np.log(1 - rate) / WINDOW_DAYS), np.log(rate / max(exp.mean(), 1e-12))]
    fit = minimize(lambda p: -_loglik(*np.exp(p), c, exp), start, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    lambda0, beta = np.exp(fit.x)
    loglik = -fit.fun

    null = c.sum() * np.log(rate) + (len(c) - c.sum()) * np.log(1 - rate)
    if loglik < null:  # o ótimo está em beta = 0
        lambda0, beta, loglik = -np.log(1 - rate) / WINDOW_DAYS, 0.0, null
    lr = max(2 * (loglik - null), 0)

    target = loglik - CI_DROP
    above = lambda b: _profile_beta(b, c, exp) - target
    low = 0.0 if beta == 0 or above(0.0) > 0 else brentq(above, 0.0, beta)
    high = max(beta, 1e-12)
    while above(high) > 0:
        high *= 2
    high = brentq(above, max(beta, high / 2), high)
    return {
        "model": "com fundo", "weeks": len(c), "cases": int(c.sum()),
        "beta": beta, "beta_low": low, "beta_high": high, "lambda0": lambda0,
        "loglik": loglik, "aic": -2 * loglik + 4,
        "p_exposure": chi2.sf(lr, 1) / 2,  # beta >= 0: metade da chi2(1)
    }


def fit_both(weeks: pd.DataFrame, positive_rows: np.ndarray | None = None) -> pd.DataFrame:
    """Os dois modelos nas mesmas semanas. positive_rows fixa as semanas do modelo sem fundo."""
    without = weeks if positive_rows is None else weeks[positive_rows]
    return pd.DataFrame([fit_without_background(without), fit_with_background(weeks)])


def barn_profile(daily: pd.DataFrame, mus=MU_GRID, complete: bool = True) -> pd.DataFrame:
    """Ajusta os dois modelos com a exposição por baia para cada mu.

    As semanas são as mesmas para todo mu. No modelo sem fundo, entram só as semanas com
    exposição > 0 em todos os mu, para os AIC serem comparáveis.
    """
    weeks = {mu: at_risk_weeks(build_weekly(add_barn_exposure(daily, mu)), complete) for mu in mus}
    keys = [w[["Agent", "date"]] for w in weeks.values()]
    assert all(k.equals(keys[0]) for k in keys), "semanas diferentes entre os mu"
    positive = np.logical_and.reduce([w["exp"].to_numpy() > 0 for w in weeks.values()])
    fits = [fit_both(w, positive).assign(mu=mu) for mu, w in weeks.items()]
    return pd.concat(fits, ignore_index=True)


def pedvis_fits(mu="0.07") -> pd.DataFrame:
    """Os dois modelos com a exposição do PeDViS-BDD, com semanas completas e com os dias disponíveis."""
    weekly = build_weekly(build_daily(mu), load_exposure(mu))
    fits = [
        fit_both(at_risk_weeks(weekly, complete)).assign(weeks_used=label)
        for complete, label in [(True, "7 dias de exposição"), (False, "dias disponíveis")]
    ]
    return pd.concat(fits, ignore_index=True).assign(mu=float(mu))


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pedvis = pedvis_fits("0.07")
    pedvis.to_parquet(PROCESSED_DIR / "glm_pedvis_mu0.07.parquet", index=False)
    profile = barn_profile(build_daily("0.07"))
    profile.to_parquet(PROCESSED_DIR / "glm_barn_profile.parquet", index=False)

    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print("PeDViS-BDD, mu = 0.07\n", pedvis)
        best = profile.loc[profile.groupby("model")["aic"].idxmin()]
        print("\nModelo por baia, melhor mu por AIC\n", best)


if __name__ == "__main__":
    main()
