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
    # vaca: S = em risco (saudável nas min_healthy avaliações anteriores), C = caso novo na avaliação
    "S", "C",
    # perna: quantas pernas estavam saudáveis na véspera, quantas viraram caso
    "legs_at_risk", "Ct",
    # exposição somada nos 7 dias antes da avaliação e quantos desses dias têm dado
    "exp", "exp_days", "complete",
]


def exposure_window(exposure: pd.DataFrame, days: int = WINDOW_DAYS, delay: int = 0) -> pd.DataFrame:
    """Soma de Exp nos `days` dias que terminam `delay` dias antes de cada data, por vaca.

    Com delay = 0, são os 7 dias antes da avaliação (o próprio dia fica de fora); com
    delay = 7, os dias 8 a 14 antes, e assim por diante. `exposure` é uma tabela com
    Agent, date e Exp. Dias sem exposição não entram na soma; exp_days conta os dias que
    entraram. A soma é feita direto sobre os 7 valores, porque a soma móvel do pandas deixa
    resíduos de arredondamento (ex.: -1e-17 em vez de 0) depois que um valor grande sai
    da janela.
    """
    shifted = [
        exposure[["Agent", "date", "Exp"]].assign(date=exposure["date"] + pd.Timedelta(days=lag))
        for lag in range(delay + 1, delay + days + 1)
    ]
    window = pd.concat(shifted).groupby(["Agent", "date"])["Exp"]
    return pd.DataFrame({"exp": window.sum(min_count=1), "exp_days": window.count()})


def healthy_before(daily: pd.DataFrame, min_healthy: int) -> pd.Series:
    """Para cada linha, se a vaca estava saudável nas `min_healthy` checagens anteriores.

    As checagens são as quintas e o primeiro dia da vaca: o status só muda nas quintas, então
    a checagem anterior dá o status da véspera. Com min_healthy = 1 é o at_risk do diário;
    com 2 ou mais, uma única avaliação saudável entre duas infectadas não basta para contar
    um caso novo (a lesão pode só ter oscilado).
    """
    checks = daily[daily["date"].dt.dayofweek.eq(ASSESSMENT_WEEKDAY) | daily["first_day"]]
    by_cow = checks.groupby("Agent")["INFECTED"]
    healthy = pd.concat([by_cow.shift(j).eq(0) for j in range(1, min_healthy + 1)], axis=1).all(axis=1)
    return healthy.reindex(daily.index, fill_value=False)


def build_weekly(
    daily: pd.DataFrame, exposure: pd.DataFrame | None = None, min_healthy: int = 1, delay: int = 0,
) -> pd.DataFrame:
    """Uma linha por vaca por avaliação, a partir do arquivo diário de src.daily.

    `exposure` é de onde sai a janela de exposição (Agent, date, Exp); por padrão, a coluna
    Exp do próprio diário. Para o PeDViS-BDD, passe o arquivo de exposição inteiro
    (load_exposure): o diário só tem os dias do arquivo de infecção, e a vaca pode sumir
    dele por alguns dias com a exposição ainda registrada.

    `min_healthy` é o número de avaliações saudáveis seguidas exigidas antes de um caso novo
    (S = 1). `delay` atrasa a janela de exposição, para o tempo entre a infecção e a lesão
    ser vista. Os padrões (1 e 0) são a definição original.
    """
    daily = daily.sort_values(["Agent", "date"]).reset_index(drop=True)
    prev = daily.groupby("Agent")[["LA", "RA"]].shift()
    changed = ~daily["first_day"] & (daily["LA"].ne(prev["LA"]) | daily["RA"].ne(prev["RA"]))
    if (changed & daily["date"].dt.dayofweek.ne(ASSESSMENT_WEEKDAY)).any():
        raise ValueError("Status de perna mudou fora do dia de avaliação")

    daily = daily.assign(S=healthy_before(daily, min_healthy).astype(int))
    weekly = daily[daily["date"].dt.dayofweek.eq(ASSESSMENT_WEEKDAY) & ~daily["first_day"]]
    weekly = weekly.join(
        exposure_window(daily if exposure is None else exposure, delay=delay), on=["Agent", "date"],
    )
    exp_days = weekly["exp_days"].fillna(0).astype(int)  # nenhum dia na janela
    weekly = weekly.assign(exp_days=exp_days, complete=exp_days.eq(WINDOW_DAYS))
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
