import numpy as np
import pandas as pd
import pytest

from src.daily import add_status
from src.weekly import build_weekly

THURSDAY = "2024-05-30"


def cow(legs, exp, start=THURSDAY, agent=1):
    """Arquivo diário de uma vaca: um par (LA, RA) e um valor de Exp por dia, em dias seguidos."""
    dates = pd.date_range(start, periods=len(legs))
    df = pd.DataFrame({
        "Agent": agent, "barn": 60, "date": dates,
        "LA": [la for la, _ in legs], "RA": [ra for _, ra in legs], "Exp": exp,
    })
    return add_status(df)


def test_one_row_per_assessment_with_week_exposure():
    # quinta 30/05 saudável, caso na quinta 06/06 (perna esquerda), segue infectada até 13/06
    legs = [(0, 0)] * 7 + [(1, 0)] * 8
    w = build_weekly(cow(legs, exp=np.arange(1.0, 16.0)))
    assert w["date"].dt.strftime("%d/%m").tolist() == ["06/06", "13/06"]
    assert w["S"].tolist() == [1, 0]
    assert w["C"].tolist() == [1, 0]
    assert w["legs_at_risk"].tolist() == [2, 1]
    assert w["Ct"].tolist() == [1, 0]
    # 06/06 soma 30/05 a 05/06 (Exp 1 a 7); 13/06 soma 06/06 a 12/06 (Exp 8 a 14)
    assert w["exp"].tolist() == [28.0, 77.0]
    assert w["complete"].all()


def test_missing_exposure_days_are_counted():
    exp = [1.0, np.nan, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0]
    w = build_weekly(cow([(0, 0)] * 8, exp=exp))
    assert w["exp"].tolist() == [5.0]
    assert w["exp_days"].tolist() == [5]
    assert not w["complete"].any()


def test_week_without_exposure_sums_to_exactly_zero():
    # um valor grande na primeira semana não pode deixar resíduo na segunda
    exp = [1e6, 3.3, 1e-9, 7e5, 0.1, 2.0, 1e4] + [0.0] * 8
    w = build_weekly(cow([(0, 0)] * 15, exp=exp))
    assert w["exp"].tolist()[1] == 0.0


def test_cow_entering_mid_week_has_incomplete_window():
    # entra no domingo 02/06, avaliada na quinta 06/06
    w = build_weekly(cow([(0, 0)] * 5, exp=[1.0] * 5, start="2024-06-02"))
    assert w["exp_days"].tolist() == [4]
    assert not w["complete"].any()


def test_row_order_does_not_matter():
    daily = pd.concat([
        cow([(0, 0)] * 7 + [(1, 0)] * 8, exp=np.arange(1.0, 16.0), agent=1),
        cow([(0, 0)] * 15, exp=np.full(15, 2.0), agent=2),
    ])
    shuffled = daily.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(build_weekly(daily), build_weekly(shuffled))


def test_status_change_outside_assessment_day_fails():
    # muda numa sexta (31/05)
    with pytest.raises(ValueError, match="fora do dia de avaliação"):
        build_weekly(cow([(0, 0), (1, 0)] + [(1, 0)] * 7, exp=[1.0] * 9))
