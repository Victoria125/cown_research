"""Compara a versão em R (R/run_analysis.R) com o Python, tabela por tabela.

Uso:
    Rscript R/run_analysis.R <pasta>
    .venv\\Scripts\\python -m verify.compare_r <pasta>

As tabelas diária e semanal têm de ser iguais; as estimativas, iguais até 1e-4 relativo.
Os p de permutação e os intervalos de bootstrap usam sorteios diferentes em R e em Python,
então só se confere que ficam próximos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.daily import build_daily
from src.load import PROCESSED_DIR, load_exposure
from src.weekly import build_weekly


def main(folder: str):
    out = Path(folder)
    failures = []

    def check(name, ok, detail=""):
        print(f"{'OK' if ok else 'FALHOU'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    daily = build_daily("0.07").sort_values(["Agent", "date"]).reset_index(drop=True)
    r_daily = pd.read_csv(out / "daily.csv", parse_dates=["date"]).sort_values(["Agent", "date"]).reset_index(drop=True)
    check("diário: mesmas vaca-dias", len(daily) == len(r_daily)
          and (daily["Agent"] == r_daily["Agent"]).all() and (daily["date"] == r_daily["date"]).all(), f"{len(daily):,} linhas")
    for col in ["barn", "LA", "RA", "INFECTED", "C", "I", "S", "N", "Ct", "barn_move", "first_day", "at_risk", "legs_at_risk"]:
        check(f"diário: coluna {col} igual", (daily[col].astype(int) == r_daily[col].astype(int)).all())
    check("diário: days_since_prev igual", daily["days_since_prev"].astype("float").equals(r_daily["days_since_prev"].astype("float")))
    check("diário: Exp igual (1e-12 relativo)", np.allclose(daily["Exp"].fillna(-1), r_daily["Exp"].fillna(-1), rtol=1e-12, atol=0))

    weekly = build_weekly(daily, load_exposure("0.07"))
    r_weekly = pd.read_csv(out / "weekly.csv", parse_dates=["date"]).sort_values(["Agent", "date"]).reset_index(drop=True)
    check("semanal: mesmas linhas", len(weekly) == len(r_weekly) and (weekly["Agent"] == r_weekly["Agent"]).all()
          and (weekly["date"] == r_weekly["date"]).all(), f"{len(weekly):,} linhas")
    for col in ["barn", "S", "C", "legs_at_risk", "Ct", "exp_days", "complete"]:
        check(f"semanal: coluna {col} igual", (weekly[col].astype(int) == r_weekly[col].astype(int)).all())
    check("semanal: exp igual (1e-12 relativo)", np.allclose(weekly["exp"].fillna(-1), r_weekly["exp"].fillna(-1), rtol=1e-12, atol=0))

    cases = pd.read_parquet(PROCESSED_DIR / "sensitivity_case_definition.parquet")
    r_cases = pd.read_csv(out / "case_definition.csv")
    check("definição de caso: semanas e casos iguais",
          (cases[["weeks", "cases"]].to_numpy() == r_cases[["weeks", "cases"]].to_numpy()).all())
    for col in ["beta_without_background", "beta_high", "lambda0", "p_exposure", "dAIC_vs_constant"]:
        check(f"definição de caso: {col} igual (1e-4 relativo)", np.allclose(cases[col], r_cases[col], rtol=1e-4, atol=1e-12))
    check("definição de caso: β com fundo é zero nos dois", (cases["beta"] < 1e-12).all() and (r_cases["beta"] < 1e-12).all())

    delay = pd.read_parquet(PROCESSED_DIR / "sensitivity_delay.parquet")
    r_delay = pd.read_csv(out / "delay.csv")
    for col in ["weeks", "cases", "best_delay", "delay_low", "delay_high"]:
        check(f"atraso: {col} igual", (delay[col].to_numpy() == r_delay[col].to_numpy()).all())
    for col in ["beta", "lr", "p_best_delay"]:
        check(f"atraso: {col} igual (1e-4 relativo)", np.allclose(delay[col], r_delay[col], rtol=1e-4, atol=1e-12))
    gap = (delay["p_corrected"] - r_delay["p_corrected"]).abs()
    check("atraso: p de permutação próximo (sorteios diferentes, 300 cada)", (gap < 0.08).all(),
          "; ".join(f"{p:.3f}/{r:.3f}" for p, r in zip(delay["p_corrected"], r_delay["p_corrected"])))

    print(f"\n{len(failures)} falha(s)" if failures else "\nR e Python concordam em tudo.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
