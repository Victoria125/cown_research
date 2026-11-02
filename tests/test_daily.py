import pandas as pd
import pytest

from src.daily import add_status, one_row_per_day


def cow(legs, start="2024-06-01", agent=1, barn=60):
    """Uma vaca com um par (LA, RA) por dia, em dias seguidos."""
    dates = pd.date_range(start, periods=len(legs))
    return pd.DataFrame({
        "Agent": agent, "barn": barn, "date": dates,
        "LA": [la for la, _ in legs], "RA": [ra for _, ra in legs],
    })


def status(legs):
    return add_status(cow(legs))[["INFECTED", "C", "I", "S", "N", "Ct", "at_risk"]]


def test_new_case_then_infected():
    s = status([(0, 0), (1, 0), (1, 0)])
    assert s["C"].tolist() == [0, 1, 0]
    assert s["I"].tolist() == [0, 0, 1]
    assert s["S"].tolist() == [1, 0, 0]
    assert s["Ct"].tolist() == [0, 1, 0]
    assert s["N"].tolist() == [1, 1, 1]


def test_both_legs_same_day():
    s = status([(0, 0), (1, 1)])
    assert s["C"].tolist() == [0, 1]
    assert s["Ct"].tolist() == [0, 2]


def test_second_leg_is_not_a_new_cow_case():
    s = status([(0, 0), (1, 0), (1, 1)])
    assert s["C"].tolist() == [0, 1, 0]
    assert s["I"].tolist() == [0, 0, 1]
    assert s["Ct"].tolist() == [0, 1, 1]


def test_first_day_has_no_new_case():
    s = status([(1, 0), (1, 0)])
    assert s["C"].tolist() == [0, 0]
    assert s["I"].tolist() == [1, 1]
    assert s["at_risk"].tolist() == [False, False]


def test_recovery_and_reinfection():
    s = status([(1, 0), (1, 0), (0, 0), (0, 1)])
    assert s["S"].tolist() == [0, 0, 1, 0]
    assert s["C"].tolist() == [0, 0, 0, 1]
    assert s["at_risk"].tolist() == [False, False, False, True]


def test_gap_compares_with_last_observed_day():
    df = pd.concat([cow([(0, 0)]), cow([(1, 0)], start="2024-06-08")])
    s = add_status(df)
    assert s["days_since_prev"].tolist()[1] == 7
    assert s["C"].tolist() == [0, 1]


def test_cows_are_independent():
    df = pd.concat([cow([(1, 0)], agent=1), cow([(0, 0), (0, 0)], agent=2)])
    s = add_status(df)
    assert s["C"].sum() == 0
    assert s["first_day"].tolist() == [True, True, False]


def test_barn_move_keeps_destination():
    df = cow([(0, 0)] * 3, barn=70)
    move_day = df.iloc[[1]].assign(barn=71)
    df = pd.concat([df.iloc[[0]], move_day, df.iloc[[1]], df.iloc[[2]].assign(barn=71)])
    days = one_row_per_day(df)
    assert days["barn"].tolist() == [70, 71, 71]
    assert days["barn_move"].tolist() == [False, True, False]


def test_identical_rows_are_dropped():
    df = cow([(0, 0), (1, 0)])
    days = one_row_per_day(pd.concat([df, df.iloc[[1]]]))
    assert len(days) == 2
    assert not days["barn_move"].any()


def test_move_without_destination_fails():
    df = cow([(0, 0)] * 2, barn=70)
    df = pd.concat([df, df.iloc[[1]].assign(barn=71)])
    with pytest.raises(ValueError, match="sem destino"):
        one_row_per_day(df)
