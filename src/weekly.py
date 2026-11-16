"""Tabela por avaliação (uma linha por vaca por quinta), a unidade da regressão.

Os casos só são observados nas avaliações semanais, então a regressão não usa as linhas
diárias. Cada linha aqui é uma vaca numa avaliação, com o status na véspera e a exposição
somada nos 7 dias antes da avaliação, como em Chang & de Jong (2023).

Uso: python -m src.weekly 0.07 [outros valores de mu]
"""

import argparse

import pandas as pd

from src.daily import build_daily
from src.load import PROCESSED_DIR, load_exposure

ASSESSMENT_WEEKDAY = 3  # quinta
WINDOW_DAYS = 7

COLUMNS = [
    "Agent", "barn", "date",
    # vaca: S = saudável na véspera (em risco), C = caso novo na avaliação
    "S", "C",
    # perna: quantas pernas estavam saudáveis na véspera, quantas viraram caso
    "legs_at_risk", "Ct",
    # exposição somada nos 7 dias antes da avaliação e quantos desses dias têm dado
    "exp", "exp_days", "complete",
]


def exposure_window(exposure: pd.DataFrame, days: int = WINDOW_DAYS) -> pd.DataFrame:
    """Soma de Exp nos `days` dias antes de cada data, por vaca (o próprio dia fica de fora).

    `exposure` é uma tabela com Agent, date e Exp. Dias sem exposição não entram na soma;
    exp_days conta os dias que entraram. A soma é feita direto sobre os 7 valores, porque a
    soma móvel do pandas deixa resíduos de arredondamento (ex.: -1e-17 em vez de 0) depois
    que um valor grande sai da janela.
    """
    shifted = [
        exposure[["Agent", "date", "Exp"]].assign(date=exposure["date"] + pd.Timedelta(days=lag))
        for lag in range(1, days + 1)
    ]
    window = pd.concat(shifted).groupby(["Agent", "date"])["Exp"]
    return pd.DataFrame({"exp": window.sum(min_count=1), "exp_days": window.count()})


def build_weekly(daily: pd.DataFrame, exposure: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por vaca por avaliação, a partir do arquivo diário de src.daily.

    `exposure` é de onde sai a janela de exposição (Agent, date, Exp); por padrão, a coluna
    Exp do próprio diário. Para o PeDViS-BDD, passe o arquivo de exposição inteiro
    (load_exposure): o diário só tem os dias do arquivo de infecção, e a vaca pode sumir
    dele por alguns dias com a exposição ainda registrada.
    """
    daily = daily.sort_values(["Agent", "date"]).reset_index(drop=True)
    prev = daily.groupby("Agent")[["LA", "RA"]].shift()
    changed = ~daily["first_day"] & (daily["LA"].ne(prev["LA"]) | daily["RA"].ne(prev["RA"]))
    if (changed & daily["date"].dt.dayofweek.ne(ASSESSMENT_WEEKDAY)).any():
        raise ValueError("Status de perna mudou fora do dia de avaliação")

    weekly = daily[daily["date"].dt.dayofweek.eq(ASSESSMENT_WEEKDAY) & ~daily["first_day"]]
    weekly = weekly.join(exposure_window(daily if exposure is None else exposure), on=["Agent", "date"])
    exp_days = weekly["exp_days"].fillna(0).astype(int)  # nenhum dia na janela
    weekly = weekly.assign(
        S=weekly["at_risk"].astype(int),
        exp_days=exp_days,
        complete=exp_days.eq(WINDOW_DAYS),
    )
    return weekly[COLUMNS].sort_values(["Agent", "date"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mu", nargs="*", default=["0.07"])
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for mu in args.mu:
        path = PROCESSED_DIR / f"weekly_mu{mu}.parquet"
        build_weekly(build_daily(mu), load_exposure(mu)).to_parquet(path, index=False)
        print(f"{path.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()
