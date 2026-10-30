from pathlib import Path
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_infection(path: Path = RAW_DIR / "infectious data_NLAS2b.xlsx") -> pd.DataFrame:
    """Status diário de cada pata (LA/RA), repetido a partir da avaliação semanal.

    NC, S e I do arquivo não são usados: o status é recalculado a partir de LA/RA.
    """
    df = pd.read_excel(path)
    df = df.rename(columns={"ID": "Agent", "Barn": "barn"})
    return cast(pd.DataFrame, df.loc[:, ["Agent", "barn", "date", "LA", "RA"]])


def load_exposure(mu=0.07, raw_dir=RAW_DIR):
    """Exposição diária de cada vaca para um valor de mu (Tick 1440 = um dia)."""
    path = raw_dir / f"mu_{mu}_agent_exposure.csv"
    df = pd.read_csv(path, float_precision="round_trip")
    df["date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.rename(columns={"Contamination Load Droplet": "Exp"})
    return df[["Agent", "date", "Exp"]]
