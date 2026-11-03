"""Arquivo diário (uma linha por vaca por dia) no formato do daily000000007.xlsx.

Uso: python -m src.daily 0.07 [outros valores de mu]
"""

import argparse

import pandas as pd

from src.load import PROCESSED_DIR, load_exposure, load_infection

COLUMNS = [
    # colunas do exemplo do Gustavo
    "Agent", "barn", "date", "Exp", "LA", "RA", "INFECTED", "C", "I", "S", "N", "Ct",
    # colunas extras, usadas para montar a tabela semanal (src/weekly.py)
    "barn_move", "first_day", "days_since_prev", "at_risk", "legs_at_risk",
]


def one_row_per_day(infection: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas repetidas e resolve os dias de troca de baia.

    No dia da troca a vaca aparece nas duas baias. Fica a baia de destino, que é a
    baia do próximo dia sem troca, e o dia é marcado em barn_move.
    """
    df = (
        infection.drop_duplicates()
        .sort_values(["Agent", "date"], kind="stable")
        .reset_index(drop=True)
    )
    if (df.groupby(["Agent", "date"])[["LA", "RA"]].nunique() > 1).any(axis=None):
        raise ValueError("LA/RA diferentes para a mesma vaca no mesmo dia")

    n_barns = df.groupby(["Agent", "date"])["barn"].transform("nunique")
    days = df.drop_duplicates(["Agent", "date"]).copy()
    days["barn_move"] = n_barns.loc[days.index] > 1

    next_barn = days["barn"].where(~days["barn_move"]).groupby(days["Agent"]).bfill()
    moved = days["barn_move"]
    dest = days.loc[moved, ["Agent", "date"]].assign(barn=next_barn[moved].astype("Int64"))
    found = dest.merge(df[["Agent", "date", "barn"]], how="left", indicator=True)
    if (found["_merge"] != "both").any():
        raise ValueError(f"Troca de baia sem destino claro:\n{found[found['_merge'] != 'both']}")
    days.loc[dest.index, "barn"] = dest["barn"].astype(int)
    return days.reset_index(drop=True)


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula INFECTED, C, I, S, N e Ct comparando cada dia com o dia anterior da mesma vaca.

    Regras do e-mail do Gustavo (16/10), mais as decisões para os casos que ele não cobre:
    - primeiro dia da vaca: não há dia anterior, então C=0 e Ct=0; se já estiver
      infectada, conta como I (caso que já existia, não caso novo);
    - dia depois de um buraco na sequência: compara com o último dia observado,
      e days_since_prev mostra o tamanho do buraco;
    - vaca que se recupera: volta a S e pode virar C de novo depois;
    - perna nova numa vaca que já estava infectada: Ct=1, mas C=0 e I=1;
    - troca de perna no mesmo dia (uma sara e a outra adoece): também Ct=1, C=0.
    """
    df = df.sort_values(["Agent", "date"]).reset_index(drop=True)
    df["INFECTED"] = (df["LA"] | df["RA"]).astype(int)

    prev = df.groupby("Agent")[["date", "LA", "RA", "INFECTED"]].shift()
    df["first_day"] = prev["date"].isna()
    df["days_since_prev"] = (df["date"] - prev["date"]).dt.days.astype("Int64")
    # em risco = saudável no dia anterior. Não é o denominador da regressão diária: os
    # casos só aparecem nas avaliações (quintas), por isso a regressão usa src/weekly.py
    df["at_risk"] = prev["INFECTED"].eq(0)
    df["legs_at_risk"] = sum(prev[leg].eq(0).astype(int) for leg in ["LA", "RA"])

    df["C"] = (df["at_risk"] & df["INFECTED"].eq(1)).astype(int)
    df["I"] = (df["INFECTED"].eq(1) & df["C"].eq(0)).astype(int)
    df["S"] = 1 - df["INFECTED"]
    df["N"] = 1
    df["Ct"] = sum((df[leg].eq(1) & prev[leg].eq(0)).astype(int) for leg in ["LA", "RA"])
    return df


def build_daily(mu=0.07) -> pd.DataFrame:
    """Junta o status de infecção com a exposição de um valor de mu.

    A base é o arquivo de infecção. Vaca-dia sem exposição fica com Exp vazio, para não
    quebrar a sequência de status; vacas só com exposição ficam de fora (não têm status).
    """
    daily = add_status(one_row_per_day(load_infection()))
    daily = daily.merge(load_exposure(mu), on=["Agent", "date"], how="left", validate="one_to_one")
    return daily[COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mu", nargs="*", default=["0.07"])
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for mu in args.mu:
        path = PROCESSED_DIR / f"daily_mu{mu}.parquet"
        build_daily(mu).to_parquet(path, index=False)
        print(f"{path.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()
