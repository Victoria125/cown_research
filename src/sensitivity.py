"""Definição de caso novo, atraso entre infecção e lesão, e análises de sensibilidade.

Tudo com a exposição do PeDViS-BDD (mu = 0.07) e o modelo com fundo de src/glm.py:
    p = 1 - exp(-(lambda0 * 7 + beta * exp))

- Definição de caso (min_healthy = k): um caso novo só conta se a vaca estava saudável nas k
  avaliações anteriores. Com k = 1 é a definição original; com k >= 2, uma única avaliação
  saudável entre duas infectadas não basta (a lesão pode só ter oscilado).
- Atraso (delay = L): a janela de exposição são os 7 dias que terminam L dias antes da
  avaliação. L é estimado por máxima verossimilhança, nas mesmas semanas para todo L, com
  intervalo de confiança pelo perfil. Como L é escolhido pelos dados, o teste da exposição
  usa uma permutação do máximo sobre todos os L.
- Incerteza por vaca: as semanas da mesma vaca não são independentes, então os intervalos
  de beta e de L vêm de um bootstrap que sorteia vacas (e reescolhe L em cada sorteio).

Uso: python -m src.sensitivity   (grava data/processed/sensitivity_*.parquet)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from src.daily import build_daily
from src.glm import CI_DROP, WINDOW_DAYS, at_risk_weeks, fit_with_background, fit_without_background
from src.load import PROCESSED_DIR, load_exposure
from src.weekly import build_weekly

MIN_HEALTHY = (1, 2, 3)
DELAYS = range(0, 22)                     # 0 a 21 dias de atraso
CONTRADICTORY_COWS = [2611, 2678, 4065]   # LA/RA e S/I/NC discordam no arquivo (verify/audit.py)


def null_loglik(c: np.ndarray) -> float:
    rate = c.mean()
    return c.sum() * np.log(rate) + (len(c) - c.sum()) * np.log(1 - rate)


def fit_background_fast(c: np.ndarray, exp: np.ndarray) -> tuple[float, float, float]:
    """Máxima verossimilhança do modelo com fundo, sem intervalos: (loglik, lambda0, beta).

    Usada nas permutações e no bootstrap, onde fit_with_background seria lento demais.
    Se o ótimo for beta = 0, devolve o modelo só com lambda0.
    """
    # A log-verossimilhança é côncava em (lambda0, beta): o hazard é linear nos dois e
    # log(1 - e^-h) é côncavo. Então há um único máximo e um otimizador com gradiente e
    # limites >= 0 basta. Os parâmetros são reescalados para ficarem perto de 1.
    case = c == 1
    rate = c.mean()
    lambda_null = -np.log(1 - rate) / WINDOW_DAYS
    h_scale = lambda_null * WINDOW_DAYS
    x = exp / max(exp.mean(), 1e-300)  # exposição com média 1

    def nll_and_grad(p):
        h = h_scale * (p[0] + p[1] * x)
        dh = np.where(case, 1 / np.expm1(h), -1.0)
        value = np.sum(np.log(-np.expm1(-h[case]))) - np.sum(h[~case])
        return -value, -h_scale * np.array([dh.sum(), (dh * x).sum()])

    fit = minimize(nll_and_grad, [1.0, 0.1], jac=True, method="L-BFGS-B",
                   bounds=[(1e-9, None), (0.0, None)], options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500})
    null = null_loglik(c)
    if -fit.fun <= null + 1e-9 or fit.x[1] <= 0:
        return null, lambda_null, 0.0
    return -fit.fun, float(fit.x[0] * lambda_null), float(fit.x[1] * h_scale / max(exp.mean(), 1e-300))


def weeks_by_delay(daily, exposure, min_healthy, delays=DELAYS) -> dict:
    """Semanas em risco com a janela de cada atraso, todas nas mesmas linhas.

    Usa só as semanas com exposição em todos os atrasos, para as verossimilhanças serem
    comparáveis entre atrasos.
    """
    tables = {L: build_weekly(daily, exposure, min_healthy=min_healthy, delay=L) for L in delays}
    tables = {L: w[(w["S"] == 1) & (w["exp_days"] > 0)] for L, w in tables.items()}
    keys = None
    for w in tables.values():
        keys = w[["Agent", "date"]] if keys is None else keys.merge(w[["Agent", "date"]])
    return {L: keys.merge(w, on=["Agent", "date"]).reset_index(drop=True) for L, w in tables.items()}


def delay_profile(weeks: dict, n_perm: int = 300, seed: int = 1) -> tuple[pd.DataFrame, dict]:
    """Perfil da verossimilhança em L, IC de 95% de L e teste da exposição corrigido pela escolha de L."""
    cases = next(iter(weeks.values()))["C"].to_numpy()
    exps = {L: w["exp"].to_numpy() for L, w in weeks.items()}
    null = null_loglik(cases)
    rows = []
    for L, e in exps.items():
        ll, l0, beta = fit_background_fast(cases, e)
        rows.append({"delay": L, "loglik": ll, "lambda0": l0, "beta": beta, "lr": 2 * (ll - null)})
    prof = pd.DataFrame(rows)
    best = prof.loc[prof["loglik"].idxmax()]
    inside = prof[prof["loglik"] >= best["loglik"] - CI_DROP]
    observed = prof["lr"].max()

    rng = np.random.default_rng(seed)
    beat = 0
    for _ in range(n_perm):
        perm = rng.permutation(cases)
        perm_null = null_loglik(perm)
        if max(2 * (fit_background_fast(perm, e)[0] - perm_null) for e in exps.values()) >= observed - 1e-9:
            beat += 1
    summary = {
        "weeks": len(cases), "cases": int(cases.sum()),
        "best_delay": int(best["delay"]), "delay_low": int(inside["delay"].min()), "delay_high": int(inside["delay"].max()),
        "beta": best["beta"], "lr": observed,
        "p_best_delay": chi2.sf(observed, 1) / 2,               # como se L fosse fixado antes
        "p_corrected": (beat + 1) / (n_perm + 1),                # corrigido pela escolha de L
        "dAIC_vs_constant": -observed + 2 + 2,                   # +2 do beta e +2 por estimar L
    }
    return prof, summary


def cow_bootstrap(weeks: dict, n_boot: int = 200, seed: int = 2) -> dict:
    """Bootstrap por vaca: sorteia vacas com reposição, reescolhe L e reestima beta."""
    first = next(iter(weeks.values()))
    cows = first["Agent"].to_numpy()
    unique = np.unique(cows)
    rows_of = {a: np.flatnonzero(cows == a) for a in unique}
    cases = first["C"].to_numpy()
    exps = {L: w["exp"].to_numpy() for L, w in weeks.items()}
    rng = np.random.default_rng(seed)
    delays, betas = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[a] for a in rng.choice(unique, len(unique), replace=True)])
        c = cases[idx]
        if c.sum() == 0:
            continue
        fits = {L: fit_background_fast(c, e[idx]) for L, e in exps.items()}
        L = max(fits, key=lambda k: fits[k][0])
        delays.append(L)
        betas.append(fits[L][2])
    betas, delays = np.array(betas), np.array(delays)
    return {
        "boot": len(betas),
        "delay_boot_low": int(np.percentile(delays, 2.5)), "delay_boot_high": int(np.percentile(delays, 97.5)),
        "beta_boot_low": float(np.percentile(betas, 2.5)), "beta_boot_high": float(np.percentile(betas, 97.5)),
        "share_beta_zero": float(np.mean(betas < 1e-12)),
    }


def case_definition_table(daily, exposure) -> pd.DataFrame:
    """Os modelos com a janela original (1 a 7 dias antes) para cada definição de caso."""
    rows = []
    for k in MIN_HEALTHY:
        weekly = build_weekly(daily, exposure, min_healthy=k)
        for complete, label in [(True, "7 dias de exposição"), (False, "dias disponíveis")]:
            weeks = at_risk_weeks(weekly, complete)
            bg, nb = fit_with_background(weeks), fit_without_background(weeks)
            c = weeks["C"].to_numpy()
            rows.append({
                "min_healthy": k, "weeks_used": label, "weeks": len(weeks), "cases": int(c.sum()),
                "beta_without_background": nb["beta"], "beta": bg["beta"], "beta_low": bg["beta_low"],
                "beta_high": bg["beta_high"], "lambda0": bg["lambda0"], "p_exposure": bg["p_exposure"],
                "dAIC_vs_constant": bg["aic"] - (-2 * null_loglik(c) + 2),
            })
    return pd.DataFrame(rows)


def delay_table(daily, exposure, n_perm=300, n_boot=200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfil do atraso e resumo para cada definição de caso (e sem as vacas contraditórias)."""
    summaries, profiles = [], []
    for k in MIN_HEALTHY:
        all_weeks = weeks_by_delay(daily, exposure, k)
        for drop in (False, True):
            weeks = all_weeks
            if drop:
                weeks = {L: w[~w["Agent"].isin(CONTRADICTORY_COWS)].reset_index(drop=True) for L, w in all_weeks.items()}
            prof, summary = delay_profile(weeks, n_perm=n_perm)
            summary |= cow_bootstrap(weeks, n_boot=n_boot)
            label = {"min_healthy": k, "without_contradictory_cows": drop}
            summaries.append(label | summary)
            profiles.append(prof.assign(**label))
            print(f"min_healthy={k}, sem vacas contraditórias={drop}: atraso {summary['best_delay']} dias, "
                  f"p corrigido {summary['p_corrected']:.3f}", flush=True)
    return pd.DataFrame(summaries), pd.concat(profiles, ignore_index=True)


def main():
    daily, exposure = build_daily("0.07"), load_exposure("0.07")
    cases = case_definition_table(daily, exposure)
    summary, profile = delay_table(daily, exposure)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(PROCESSED_DIR / "sensitivity_case_definition.parquet", index=False)
    summary.to_parquet(PROCESSED_DIR / "sensitivity_delay.parquet", index=False)
    profile.to_parquet(PROCESSED_DIR / "sensitivity_delay_profile.parquet", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print("Definição de caso, janela original\n", cases)
        print("\nAtraso estimado\n", summary)


if __name__ == "__main__":
    main()
