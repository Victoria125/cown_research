"""As verificações de verify/audit.py como testes (o relatório completo sai de python -m verify.audit).

Os testes que precisam dos dados brutos são pulados se data/raw não existir.
"""

import hashlib
import io
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.daily import add_status, build_daily
from src.glm import at_risk_weeks, fit_without_background
from src.load import RAW_DIR, ROOT, load_exposure
from src.weekly import build_weekly
from verify.audit import (
    RAW_SHA256, exposure_lookup, independent_status, independent_window, read_raw_exposure,
    read_raw_infection,
)

needs_data = pytest.mark.skipif(not RAW_DIR.exists(), reason="data/raw não está nesta pasta")


# --------------------------------------------------------------- regras provadas em Lean
def lean_table():
    lake = shutil.which("lake")
    if lake is None:
        pytest.skip("Lean (lake) não está instalado")
    out = subprocess.run([lake, "exe", "status_table"], cwd=ROOT / "verify" / "lean_status",
                         capture_output=True, text=True, check=True, timeout=900)
    return pd.read_csv(io.StringIO(out.stdout))


def test_python_status_follows_the_rules_proved_in_lean():
    table = lean_table()
    assert len(table) == 20  # 4 primeiros dias + 16 pares (ontem, hoje)
    for row in table.itertuples():
        days = ([] if row.first_day else [(row.prev_la, row.prev_ra)]) + [(row.la, row.ra)]
        cow = pd.DataFrame({
            "Agent": 1, "barn": 60, "date": pd.date_range("2024-06-06", periods=len(days)),
            "LA": [la for la, _ in days], "RA": [ra for _, ra in days],
        })
        last = add_status(cow).iloc[-1]
        assert [last["INFECTED"], last["C"], last["I"], last["S"], last["Ct"]] == \
            [row.INFECTED, row.C, row.I, row.S, row.Ct], row


def test_case_definition_follows_the_rule_proved_in_lean():
    """min_healthy em src/weekly.py contra healthyRun em verify/lean_status/Status.lean."""
    lake = shutil.which("lake")
    if lake is None:
        pytest.skip("Lean (lake) não está instalado")
    out = subprocess.run([lake, "exe", "status_table", "healthy-run"], cwd=ROOT / "verify" / "lean_status",
                         capture_output=True, text=True, check=True, timeout=900)
    table = pd.read_csv(io.StringIO(out.stdout), dtype={"history": str})
    assert len(table) == 90
    for row in table.itertuples():
        # checagens anteriores, da mais antiga para a mais recente, uma por quinta
        statuses = [int(s) for s in reversed(row.history)]
        legs = [leg for s in statuses for leg in [(s, 0)] * 7] + [(1, 0)]
        cow = pd.DataFrame({
            "Agent": 1, "barn": 60, "date": pd.date_range("2024-05-30", periods=len(legs)),
            "LA": [la for la, _ in legs], "RA": [ra for _, ra in legs], "Exp": 1.0,
        })
        weekly = build_weekly(add_status(cow), min_healthy=row.k)
        assert weekly["S"].iloc[-1] == row.S, row


# --------------------------------------------------------------- dados reais
@pytest.fixture(scope="module")
def pipeline():
    daily = build_daily("0.07")
    return daily, build_weekly(daily, load_exposure("0.07"))


@needs_data
@pytest.mark.parametrize("name", sorted(RAW_SHA256))
def test_raw_file_is_the_email_attachment(name):
    assert hashlib.sha256((RAW_DIR / name).read_bytes()).hexdigest() == RAW_SHA256[name]


@needs_data
def test_no_cow_day_is_lost(pipeline):
    daily, _ = pipeline
    raw = read_raw_infection()[["Agent", "date"]].drop_duplicates()
    assert set(map(tuple, raw.to_numpy())) == set(zip(daily["Agent"], daily["date"]))


@needs_data
def test_exposure_values_are_copied_exactly(pipeline):
    daily, _ = pipeline
    m = read_raw_exposure().merge(daily[["Agent", "date", "Exp"]], on=["Agent", "date"], suffixes=("", "_d"))
    assert np.array_equal(m["Exp"].to_numpy(), m["Exp_d"].to_numpy())


@needs_data
def test_status_equals_independent_loop(pipeline):
    daily, _ = pipeline
    m = daily.merge(independent_status(read_raw_infection()), on=["Agent", "date"], suffixes=("", "_ind"))
    assert len(m) == len(daily)
    for col in ["INFECTED", "C", "I", "S", "Ct", "at_risk"]:
        assert (m[col].astype(int) == m[f"{col}_ind"].astype(int)).all(), col


@needs_data
def test_exposure_window_sums_the_whole_exposure_file(pipeline):
    """Inclui os dias em que a vaca some do arquivo de infecção (problema achado na auditoria)."""
    _, weekly = pipeline
    w = weekly.merge(independent_window(weekly, exposure_lookup(read_raw_exposure())),
                     on=["Agent", "date"], suffixes=("", "_ind"))
    assert (w["exp_days"] == w["exp_days_ind"]).all()
    assert np.allclose(w["exp"].fillna(-1), w["exp_ind"].fillna(-1), rtol=1e-12, atol=0)


@needs_data
@pytest.mark.parametrize("k", [1, 2, 3])
def test_case_definition_equals_independent_loop(pipeline, k):
    """S com k avaliações saudáveis, refeito com um laço simples a partir do bruto."""
    daily, _ = pipeline
    ind = independent_status(read_raw_infection()).sort_values(["Agent", "date"])
    expected = {}
    for agent, g in ind.groupby("Agent"):
        checks = []  # status nas checagens (primeiro dia e quintas)
        for i, (date, infected) in enumerate(zip(g["date"], g["INFECTED"])):
            is_thursday = date.dayofweek == 3
            if is_thursday and i > 0:
                expected[(agent, date)] = int(len(checks) >= k and all(s == 0 for s in checks[-k:]))
            if is_thursday or i == 0:
                checks.append(infected)
    weekly = build_weekly(daily, load_exposure("0.07"), min_healthy=k)
    got = dict(zip(zip(weekly["Agent"], weekly["date"]), weekly["S"]))
    assert got == expected


@needs_data
def test_glm_without_background_equals_statsmodels(pipeline):
    _, weekly = pipeline
    weeks = at_risk_weeks(weekly, complete=True)
    used = weeks[weeks["exp"] > 0]
    glm = sm.GLM(used["C"].to_numpy(), np.ones((len(used), 1)), offset=np.log(used["exp"].to_numpy()),
                 family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    ours = fit_without_background(weeks)
    assert np.isclose(np.exp(glm.params[0]), ours["beta"], rtol=1e-4)
    assert np.isclose(glm.llf, ours["loglik"], rtol=1e-6)
