"""Tests for ``nysa_risk.backtest``.

Synthetic relative-price paths with hand-constructed outcomes: one path
that must trigger the gap constraint within 3 days, one that must never
trigger, and one liquidation that must produce bad debt through a
trigger→realization gap larger than ``1 − LT``.

Tidy parameters throughout: ``LT = 0.8``, ``LTV = 0.4`` — the
liquidation threshold on the relative price is a drop to
``LTV/LT = 50 %`` of the entry price, and the bad-debt buffer from
trigger to realization is ``1 − LT = 20 %``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nysa_risk import backtest as bt
from nysa_risk.backtest import (
    NA_DASH,
    BacktestResult,
    SimulationStats,
    format_bad_debt_table,
    format_gap_table,
    format_severity_table,
    format_time_table,
    relative_price_stream,
    run_backtest,
    simulate_positions,
)
from nysa_risk.config import (
    AssetUniverse,
    Borrowable,
    Calibration,
    Collateral,
    Meta,
    OndoConfig,
    PairsPolicy,
)

LT = 0.8   # trigger ratio LTV/LT = 0.5 → liquidation at half the entry price
LTV = 0.4  # bad-debt buffer 1 − LT = 0.2


def _stream(days: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Interleaved open/close stream, same shape as ``relative_price_stream``."""
    ts, px, cl = [], [], []
    for d, o, c in days:
        base = pd.Timestamp(d)
        ts += [base + pd.Timedelta(hours=9, minutes=30), base + pd.Timedelta(hours=16)]
        px += [o, c]
        cl += [False, True]
    return pd.DataFrame({"price": px, "is_close": cl}, index=pd.DatetimeIndex(ts))


def _sim(days: list[tuple[str, float, float]], **kw) -> SimulationStats:
    kw.setdefault("ltv", LTV)
    kw.setdefault("lt", LT)
    kw.setdefault("t_user_days", 3.0)
    kw.setdefault("horizon_days", 90.0)
    return simulate_positions(_stream(days), **kw)


# ---------------------------------------------------------------------------
# (1) Gap constraint — hand-constructed hit and no-hit paths
# ---------------------------------------------------------------------------


def test_gap_hit_path_triggers_within_three_days() -> None:
    """A 60 % close-to-close crash MUST trigger the day after the open (1.0 days)."""
    stats = _sim([
        ("2024-01-01", 100.0, 100.0),  # position opens at this close (P0=100, trigger ≤ 50)
        ("2024-01-02", 100.0, 40.0),   # close 40 ≤ 50 → trigger, exactly 1.0 day after open
        ("2024-01-03", 40.0, 40.0),    # realization = next open (no further drop → no bad debt)
    ])
    assert stats.n_positions == 3      # one opening per close observation
    assert stats.gap_checked == 1      # later opens lack 3 days of forward coverage → censored
    assert stats.gap_hits == 1
    assert stats.liquidations == 1
    assert stats.days_to_liquidation == (1.0,)
    assert stats.bad_debt_events == 0  # 40 → 40: zero drop at realization


def test_flat_path_never_triggers() -> None:
    """A flat path MUST NOT trigger — every fully-covered opening is a no-hit."""
    days = [(f"2024-01-0{d}", 100.0, 100.0) for d in range(1, 7)]  # 6 flat days
    stats = _sim(days)
    assert stats.n_positions == 6
    # Openings with ≥ 3 days of forward coverage: Jan 1, 2, 3 closes.
    assert stats.gap_checked == 3
    assert stats.gap_hits == 0
    assert stats.liquidations == 0
    assert stats.bad_debt_events == 0


def test_trigger_on_overnight_gap_at_open() -> None:
    """A gap through the threshold at the OPEN triggers there (17.5 h = ~0.73 days)."""
    stats = _sim([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 45.0, 44.0),    # overnight gap 100 → 45 pierces the 50 threshold
        ("2024-01-03", 44.0, 44.0),
    ])
    assert stats.gap_checked == 1 and stats.gap_hits == 1
    assert stats.liquidations == 1
    assert stats.days_to_liquidation[0] == pytest.approx(17.5 / 24.0, abs=1e-12)
    # Realization is the same day's close: drop 1 − 44/45 ≈ 2.2 % < 20 % → no bad debt.
    assert stats.bad_debt_events == 0


# ---------------------------------------------------------------------------
# (2) Time to liquidation — horizon behaviour
# ---------------------------------------------------------------------------


def test_liquidation_beyond_horizon_is_not_counted() -> None:
    """Only the Jan-1 opening can ever trigger (Jan-6 open); shrinking the horizon drops it."""
    days = [
        ("2024-01-01", 100.0, 100.0),  # thr 50 — triggers at Jan-6 open (45)
        ("2024-01-02", 60.0, 60.0),    # thr 30 for this opening — 45 never reaches it
        ("2024-01-03", 60.0, 60.0),
        ("2024-01-04", 60.0, 60.0),
        ("2024-01-05", 60.0, 60.0),
        ("2024-01-06", 45.0, 45.0),
        ("2024-01-07", 45.0, 45.0),
    ]
    full = _sim(days)  # 90-day horizon: the Jan-1 opening liquidates at Jan-6 open
    assert full.liquidations == 1
    assert full.days_to_liquidation[0] == pytest.approx(4.0 + 17.5 / 24.0, abs=1e-12)
    assert full.gap_hits == 0          # ~4.73 days > 3 — not a gap hit, but still gap-checked
    short = _sim(days, horizon_days=3.0)  # 3-day horizon: trigger falls outside every window
    assert short.liquidations == 0


def test_trigger_at_final_observation_has_no_realization() -> None:
    stats = _sim([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 100.0, 40.0),   # trigger at the very last observation
    ])
    assert stats.gap_hits == 1         # the gap check still records the hit
    assert stats.liquidations == 0     # but with no realization price there is no event
    assert stats.bad_debt_events == 0


# ---------------------------------------------------------------------------
# (3) Bad debt — hand-constructed realization gaps
# ---------------------------------------------------------------------------


def test_liquidation_with_gap_through_buffer_produces_bad_debt() -> None:
    """Trigger at 48, realize at 30: drop 37.5 % > 1 − LT = 20 % → bad debt MUST be recorded."""
    stats = _sim([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 100.0, 48.0),   # trigger at close (48 ≤ 50)
        ("2024-01-03", 30.0, 30.0),    # realization = next open, gapping through the buffer
    ])
    assert stats.liquidations == 1
    assert stats.bad_debt_events == 1
    assert stats.gap_checked == 1 and stats.gap_hits == 1
    # Severity: excess beyond the buffer = 0.375 − 0.20 = 0.175 of position value.
    assert len(stats.bad_debt_excess) == 1
    assert stats.bad_debt_excess[0] == pytest.approx(0.175, abs=1e-12)
    # Unconditional denominator: only the liquidated opening is decidable at 90d.
    assert stats.horizon_checked == 1


def test_liquidation_within_buffer_produces_no_bad_debt() -> None:
    """Trigger at 48, realize at 44: drop ≈ 8.3 % < 20 % → liquidation without bad debt."""
    stats = _sim([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 100.0, 48.0),
        ("2024-01-03", 44.0, 44.0),
    ])
    assert stats.liquidations == 1
    assert stats.bad_debt_events == 0


# ---------------------------------------------------------------------------
# simulate_positions — validation and edge cases
# ---------------------------------------------------------------------------


def test_horizon_checked_counts_decidable_openings() -> None:
    """Unconditional denominator: openings with full-horizon coverage or a liquidation."""
    days = [(f"2024-01-0{d}", 100.0, 100.0) for d in range(1, 7)]  # 6 flat days
    # 3-day horizon: Jan 1–3 closes have full coverage → decidable no-bad-debt outcomes.
    stats = _sim(days, horizon_days=3.0)
    assert stats.horizon_checked == 3
    assert stats.bad_debt_excess == ()
    # 90-day horizon: nothing in 5 days of data is decidable.
    stats90 = _sim(days)
    assert stats90.horizon_checked == 0


def test_simulate_rejects_bad_parameters() -> None:
    s = _stream([("2024-01-01", 1.0, 1.0)])
    with pytest.raises(ValueError):
        simulate_positions(s, ltv=0.9, lt=0.8, t_user_days=3.0)   # ltv ≥ lt
    with pytest.raises(ValueError):
        simulate_positions(s, ltv=0.4, lt=1.2, t_user_days=3.0)   # lt ≥ 1
    with pytest.raises(ValueError):
        simulate_positions(s, ltv=0.4, lt=0.8, t_user_days=0.0)
    with pytest.raises(ValueError):
        simulate_positions(s, ltv=0.4, lt=0.8, t_user_days=3.0, horizon_days=1.0)


def test_simulate_empty_stream_returns_zeros() -> None:
    empty = pd.DataFrame(
        {"price": pd.Series(dtype=float), "is_close": pd.Series(dtype=bool)},
        index=pd.DatetimeIndex([]),
    )
    stats = simulate_positions(empty, ltv=LTV, lt=LT, t_user_days=3.0)
    assert stats == SimulationStats(0, 0, 0, 0, 0, ())


# ---------------------------------------------------------------------------
# relative_price_stream
# ---------------------------------------------------------------------------


def test_relative_price_stream_interleaves_and_divides() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    c = pd.DataFrame({"open": [100.0, 104.0], "close": [102.0, 106.0]}, index=dates)
    b = pd.DataFrame({"open": [2.0, 2.0], "close": [2.0, 2.0]}, index=dates)
    stream = relative_price_stream(c, b)
    assert list(stream["price"]) == [50.0, 51.0, 52.0, 53.0]
    assert list(stream["is_close"]) == [False, True, False, True]
    assert stream.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# CLI tables — PASS/FAIL rendering against the declared bounds
# ---------------------------------------------------------------------------


def _result(collateral: str, param_set: str, *, checked: int, hits: int,
            liqs: int, bad: int, days: tuple[float, ...] = (),
            horizon: int = 0, excess: tuple[float, ...] = ()) -> BacktestResult:
    return BacktestResult(
        collateral=collateral, param_set=param_set, binding_borrowable="WBTC",
        lt=LT, ltv=LTV,
        stats=SimulationStats(
            n_positions=checked, gap_checked=checked, gap_hits=hits,
            liquidations=liqs, bad_debt_events=bad,
            days_to_liquidation=days,
            horizon_checked=horizon, bad_debt_excess=excess,
        ),
    )


def test_gap_table_pass_fail_against_10pct_bound() -> None:
    rows = [
        _result("AAPLon", "base", checked=100, hits=5, liqs=0, bad=0),    # 5 % ≤ 10 % → PASS
        _result("TSLAon", "base", checked=100, hits=50, liqs=0, bad=0),   # 50 % → FAIL
    ]
    text = format_gap_table(rows, t_user_days=3.0)
    assert "P(liq ≤ 3d) (%)" in text.splitlines()[0]
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    tsla = next(l for l in text.splitlines() if l.startswith("TSLAon"))
    assert "5.00" in aapl and "PASS" in aapl and "FAIL" not in aapl
    assert "50.00" in tsla and "FAIL" in tsla


def test_bad_debt_table_pass_fail_and_dash_when_no_liquidations() -> None:
    rows = [
        _result("AAPLon", "base", checked=10, hits=0, liqs=200, bad=1),   # 0.5 % ≤ 1 % → PASS
        _result("TSLAon", "base", checked=10, hits=0, liqs=100, bad=5),   # 5 % → FAIL
        _result("SPYon", "base", checked=10, hits=0, liqs=0, bad=0),      # no events → dash
    ]
    text = format_bad_debt_table(rows)
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    tsla = next(l for l in text.splitlines() if l.startswith("TSLAon"))
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert "0.50" in aapl and "PASS" in aapl
    assert "5.00" in tsla and "FAIL" in tsla
    assert NA_DASH in spy and "PASS" not in spy and "FAIL" not in spy


def test_bad_debt_table_unconditional_column() -> None:
    """uncond (%) = bad-debt events / decidable openings, alongside the conditional rate."""
    r = _result("AAPLon", "base", checked=10, hits=0, liqs=100, bad=2, horizon=400)
    text = format_bad_debt_table([r])
    assert "uncond (%)" in text.splitlines()[0]
    line = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "2.00" in line       # conditional: 2/100 → FAIL vs 1 %
    assert "FAIL" in line
    assert "0.50" in line       # unconditional: 2/400
    # No decidable openings → dash in the uncond column.
    no_h = _result("SPYon", "base", checked=10, hits=0, liqs=100, bad=1, horizon=0)
    spy = next(l for l in format_bad_debt_table([no_h]).splitlines() if l.startswith("SPYon"))
    assert NA_DASH in spy


def test_severity_table_stats_and_aggregate() -> None:
    rows = [
        _result("AAPLon", "base", checked=10, hits=0, liqs=50, bad=2,
                horizon=100, excess=(0.10, 0.30)),
        _result("TSLAon", "base", checked=10, hits=0, liqs=20, bad=1,
                horizon=100, excess=(0.05,)),
        _result("SPYon", "base", checked=10, hits=0, liqs=5, bad=0, horizon=100),
    ]
    text = format_severity_table(rows)
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "20.00" in aapl      # mean of (10 %, 30 %)
    assert "28.00" in aapl      # p90
    assert "30.00" in aapl      # max
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert NA_DASH in spy       # no events → dashes
    # Aggregate over all pooled events: (0.05, 0.10, 0.30).
    assert "aggregate (3 events)" in text
    assert "mean 15.00%" in text
    assert "p50 10.00%" in text
    assert "max 30.00%" in text


def test_severity_table_no_events_aggregate_line() -> None:
    rows = [_result("SPYon", "base", checked=10, hits=0, liqs=0, bad=0)]
    assert "aggregate: no bad-debt events" in format_severity_table(rows)


def test_time_table_median_mean_p90() -> None:
    rows = [_result("AAPLon", "base", checked=1, hits=0, liqs=3, bad=0,
                    days=(1.0, 2.0, 10.0))]
    text = format_time_table(rows)
    line = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "2.00" in line    # median
    assert "4.33" in line    # mean = 13/3
    assert "8.40" in line    # p90 (linear interpolation)


def test_tables_handle_empty_results() -> None:
    assert format_gap_table([], t_user_days=3.0) == "(no results)"
    assert format_time_table([]) == "(no results)"
    assert format_bad_debt_table([]) == "(no results)"


# ---------------------------------------------------------------------------
# run_backtest — end-to-end on synthetic parquets
# ---------------------------------------------------------------------------


def _synthetic_universe() -> AssetUniverse:
    return AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD", price_history_years=1, include_overnight_gaps=True),
        collaterals=(
            Collateral(symbol="AAPLon", type="rwa", category="equity",
                       underlying_ticker="AAPL", use="collateral_only"),
        ),
        borrowables=(
            Borrowable(symbol="USDC", type="crypto", category="stable",
                       price_source="USDC-USD", use="lending_and_borrowing",
                       volatility_class="stable"),
        ),
        pairs=PairsPolicy(default_policy="all_collaterals_vs_all_borrowables"),
        ondo=OndoConfig(limits_api="https://x", status_page="https://y", api_key_env="ONDO_API_KEY"),
        calibration=Calibration(
            ewma_lambda=0.94, stress_quantile=0.95, gap_sigma_quantile=0.90,
            es_factor=3.5,
            t_liq_days=0.33, t_user_days=3.0, k_user=1.53,
            stressed_liquidatable_share=0.25, rf_theta=0.01, rf_horizon_years=0.83,
            emode_min_advantage=0.05,
            target_liq30_emode=(0.10, 0.15),
            target_liq30_std=(0.10, 0.15),
            minimum_gap=0.02,
        max_uncond_bad_debt=0.004, min_calibration_years=3.0, severity_review_threshold=0.05, max_loss_given_bad_debt=0.065,        ),
    )


def test_run_backtest_end_to_end_on_synthetic_parquets(tmp_path: Path) -> None:
    """Full plumbing: parquets → vol → LT → LTV → backtest, one row per param set."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(seed=7)
    aapl = pd.DataFrame(
        {"open": 100 + rng.normal(0, 1, size=len(idx)).cumsum(),
         "close": 100 + rng.normal(0, 1, size=len(idx)).cumsum()},
        index=idx,
    )
    usdc = pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx)
    aapl.to_parquet(tmp_path / "AAPL.parquet", index=True)
    usdc.to_parquet(tmp_path / "USDC-USD.parquet", index=True)

    results = run_backtest(universe=_synthetic_universe(), data_dir=tmp_path)
    assert [(r.collateral, r.param_set) for r in results] == [("AAPLon", "base")]
    r = results[0]
    assert r.binding_borrowable == "USDC"
    assert 0.0 < r.ltv < r.lt < 1.0
    assert r.stats.n_positions > 0
    assert r.stats.gap_checked > 0
    # A gentle ±1 % random walk never falls to LTV/LT of any entry price.
    assert r.stats.liquidations == 0


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def test_main_cli_prints_three_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bt, "load_universe", lambda *a, **k: _synthetic_universe())
    monkeypatch.setattr(
        bt, "run_backtest",
        lambda **kw: [_result("AAPLon", "base", checked=100, hits=5,
                              liqs=10, bad=0, days=(1.0, 2.0))],
    )
    rc = bt.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(a) Gap constraint" in out
    assert "(b) Time to liquidation" in out
    assert "(c) Bad debt" in out
    assert "(d) Bad-debt severity" in out
    assert "PASS" in out
    # t_user from the config lands in the gap-table header.
    assert "P(liq ≤ 3d) (%)" in out
