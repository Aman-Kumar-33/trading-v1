"""The tolerance-study engine (backend/research/live_macd_replay.py): the fast run-based `simulate` must give exactly the
trades of the plain per-tick rule, on random data and on hand-made cases.

Run from second-system/:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.research.live_macd_replay import Rule, Ticks, chop_count, hug_ratio, reference_simulate, simulate  # noqa: E402


def make_ticks(seed: int, n: int = 4000, days: int = 8) -> Ticks:
    rng = np.random.default_rng(seed)
    gap = np.cumsum(rng.normal(0, 0.3, n)) * 0.2 + rng.normal(0, 0.25, n)
    gap -= np.convolve(gap, np.ones(300) / 300, mode="same")           # keep it crossing zero
    secs = np.cumsum(rng.integers(1, 25, n)).astype(float)
    px = 100 + np.cumsum(rng.normal(0, 0.05, n))
    atr = np.full(n, rng.uniform(0.2, 2.0))
    day = np.sort(rng.integers(0, days, n))
    state0 = {d: rng.choice([None, "up", "down"]) for d in range(days)}
    return Ticks(gap, atr, secs, px, day, state0, list(range(days)))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k,hold", [(0.0, 0), (0.0, 60), (0.02, 0), (0.05, 30), (0.1, 60), (0.1, 180), (0.3, 120), (1.0, 0)])
def test_fast_engine_matches_the_per_tick_rule(seed, k, hold):
    tk = make_ticks(seed)
    fast, slow = simulate(tk, k, hold), reference_simulate(tk, k, hold)
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow):
        assert a[0] == b[0] and a[1:] == pytest.approx(b[1:])


def _rules(tk):
    rng = np.random.default_rng(99)
    gate = rng.random(len(tk.px)) > 0.4
    return [Rule(exit_k=0.0), Rule(exit_k=0.02, exit_hold=0), Rule(gate=gate), Rule(cooldown=600.0), Rule(max_entries=1),
            Rule(exit_k=0.0, gate=gate, cooldown=900.0, max_entries=2)]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("which", range(6))
@pytest.mark.parametrize("k,hold", [(0.0, 0), (0.05, 30), (0.1, 90)])
def test_fast_engine_matches_the_per_tick_rule_with_the_extras(seed, which, k, hold):
    tk = make_ticks(seed)
    rule = _rules(tk)[which]
    fast, slow = simulate(tk, k, hold, rule=rule), reference_simulate(tk, k, hold, rule=rule)
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow):
        assert a[0] == b[0] and a[1:] == pytest.approx(b[1:])


def test_the_extras_really_remove_trades():
    tk = make_ticks(0)
    base = len(simulate(tk, 0.02, 0))
    assert len(simulate(tk, 0.02, 0, rule=Rule(max_entries=1))) < base
    assert len(simulate(tk, 0.02, 0, rule=Rule(cooldown=3600.0))) < base
    assert len(simulate(tk, 0.02, 0, rule=Rule(gate=np.zeros(len(tk.px), bool)))) == 0


def test_chop_count_and_hug_ratio():
    gap = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    assert list(chop_count(gap, 3)) == [0, 1, 2, 2, 2, 1, 2]   # sign flips at bars 1, 2, 4, 6, summed over 3 bars
    r = hug_ratio(np.array([0.1, 0.1, 2.0, 2.0]), np.array([1.0] * 4), 2)
    assert r[1] == pytest.approx(0.1) and r[3] == pytest.approx(2.0)


def test_random_cases_actually_trade():
    tk = make_ticks(0)
    assert len(simulate(tk, 0.0, 0)) > 5          # the check above is not vacuous
    assert len(simulate(tk, 0.0, 0)) > len(simulate(tk, 0.3, 120))


def _ticks(gaps, secs, prices=None, state="down"):
    n = len(gaps)
    return Ticks(np.array(gaps, float), np.zeros(n), np.array(secs, float), np.array(prices or [100.0] * n, float),
                 np.zeros(n, int), {0: state}, [0])


def test_a_cross_must_hold_for_the_hold_time_before_it_trades():
    # +0.1 (beyond a 0.02 floor) for 30 s only, back inside the band: no trade with a 60 s hold, a trade with 0
    tk = _ticks([-0.1, 0.1, 0.1, 0.0, -0.1], [0, 10, 40, 50, 60], [100, 101, 102, 103, 104])
    assert simulate(tk, 0.0, 60) == []
    assert len(simulate(tk, 0.0, 0)) == 1


def test_buy_at_the_tick_where_the_hold_is_reached_and_sell_on_the_opposite_cross():
    tk = _ticks([0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1], [0, 30, 60, 90, 100, 130, 160],
                [100, 101, 102, 103, 104, 105, 106])
    (day, entry, exit_, t_in, t_out, i_in, i_out), = simulate(tk, 0.0, 60)
    assert (i_in, i_out) == (2, 6)
    assert (entry, exit_, t_in, t_out) == (102.0, 106.0, 60.0, 160.0)


def test_nothing_is_bought_when_the_state_is_already_up():
    tk = _ticks([0.1, 0.1, 0.1], [0, 30, 60], state="up")
    assert simulate(tk, 0.0, 0) == []


def test_open_position_is_sold_at_the_last_price_of_the_day():
    tk = Ticks(np.array([0.1, 0.1, 0.1, 0.1]), np.zeros(4), np.array([0, 10, 100, 110.0]), np.array([100, 101, 102, 103.0]),
               np.array([0, 0, 1, 1]), {0: "down", 1: "down"}, [0, 1])
    t = simulate(tk, 0.0, 0)
    assert [(x[0], x[1], x[2]) for x in t] == [(0, 100.0, 101.0), (1, 102.0, 103.0)]
