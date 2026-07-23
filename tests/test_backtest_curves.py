"""Tests for ``nysa_risk.backtest_curves``.

Synthetic paths with hand-known event days: a path with a known
LT-touch day and a known insolvency day, censoring at both horizons
(the 365-day liquidability horizon and the 30-day insolvency horizon,
shrunk to small values to keep the fixtures tiny), and a hand-verified
running-LTV trigger boundary.

Tidy parameters throughout: ``LT = 0.8``, ``LTV = 0.4`` — the LT touch
is a drop to ``LTV/LT = 50 %`` of the entry price; insolvency is a
further drop from the trigger below ``LT = 80 %`` of the trigger price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nysa_risk import backtest_curves as bc
from nysa_risk.backtest_curves import (
    DEFAULT_SCENARIOS,
    CalibrationResult,
    CurveResult,
    CurveStats,
    Scenario,
    calibrate_ltv,
    compute_curves,
    format_calibration_table,
    format_insolvency_table,
    format_liquidability_table,
    run_curves,
)
from nysa_risk.config import (
    AssetUniverse,
    Borrowable,
    Calibration,
    Collateral,
    EModeCategory,
    Meta,
    OndoConfig,
    PairsPolicy,
)

LT = 0.8   # LT touch at 50 % of entry; insolvency below 80 % of trigger
LTV = 0.4


def _stream(days: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Interleaved open/close stream, same shape as ``relative_price_stream``."""
    ts, px, cl = [], [], []
    for d, o, c in days:
        base = pd.Timestamp(d)
        ts += [base + pd.Timedelta(hours=9, minutes=30), base + pd.Timedelta(hours=16)]
        px += [o, c]
        cl += [False, True]
    return pd.DataFrame({"price": px, "is_close": cl}, index=pd.DatetimeIndex(ts))


def _curves(days: list[tuple[str, float, float]], **kw) -> CurveStats:
    kw.setdefault("ltv", LTV)
    kw.setdefault("lt", LT)
    return compute_curves(_stream(days), **kw)


# ---------------------------------------------------------------------------
# Known trigger day and known insolvency day
# ---------------------------------------------------------------------------


def test_known_trigger_and_insolvency_days() -> None:
    """LT touch exactly 4.0 days after opening; insolvency exactly 3.0 days later.

    Only the Jan-1 opening can ever touch its threshold (50): the later
    openings (entry 70 → threshold 35, entry ≤ 50 → threshold ≤ 25)
    never see a price that low. Insolvency threshold from the trigger:
    50 · LT = 40, first strictly-below print is the Jan-8 close (39).
    """
    stats = _curves([
        ("2024-01-01", 100.0, 100.0),  # opening A: threshold 50
        ("2024-01-02", 70.0, 70.0),    # openings at 70: threshold 35 — never reached
        ("2024-01-03", 70.0, 70.0),
        ("2024-01-04", 70.0, 70.0),
        ("2024-01-05", 55.0, 50.0),    # A touches LT at this close → 4.0 days
        ("2024-01-06", 50.0, 50.0),    # 50 is NOT below 40 — still solvent
        ("2024-01-07", 45.0, 45.0),
        ("2024-01-08", 40.0, 39.0),    # open 40 not < 40 (strict); close 39 < 40 → 3.0 days
        ("2024-01-09", 39.0, 39.0),
    ])
    assert stats.n_openings == 9
    assert stats.n_liquidatable == 1
    assert stats.days_to_liquidation == (4.0,)
    assert stats.n_insolvencies == 1
    assert stats.days_to_insolvency == (3.0,)
    # 8-day span << 365 → every non-triggering opening is censored, none "never".
    assert stats.n_never_liquidatable == 0
    assert stats.n_censored_openings == 8
    assert stats.n_solvent_after_horizon == 0 and stats.n_censored_events == 0


def test_hand_verified_running_ltv_trigger_boundary() -> None:
    """Running LTV = LTV·P_open/P_t: 50.01 stays below LT, 50.00 touches it exactly.

    At P = 50.01: 0.4·100/50.01 ≈ 0.79984 < 0.8 → no touch.
    At P = 50.00: 0.4·100/50.00 = 0.8 = LT exactly → touch (running LTV ≥ LT).
    """
    assert LTV * 100.0 / 50.01 < LT
    assert LTV * 100.0 / 50.00 == pytest.approx(LT, abs=1e-15)
    stats = _curves([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 60.0, 50.01),   # running LTV 0.79984 — no touch
        ("2024-01-03", 51.0, 50.0),    # close 50.0 → touch at exactly 2.0 days
    ])
    assert stats.n_liquidatable == 1
    assert stats.days_to_liquidation == (2.0,)


# ---------------------------------------------------------------------------
# Censoring at the liquidability horizon
# ---------------------------------------------------------------------------


def test_openings_censored_vs_never_liquidatable_split() -> None:
    """Flat path, 5-day horizon: full-coverage openings are 'never', the rest censored."""
    days = [(f"2024-01-0{d}", 100.0, 100.0) for d in range(1, 9)]  # 8 flat days
    stats = _curves(days, liq_horizon_days=5.0)
    assert stats.n_openings == 8
    # Coverage from close d to the last close (Jan 8): 7, 6, 5 days → full horizon.
    assert stats.n_never_liquidatable == 3
    assert stats.n_censored_openings == 5
    assert stats.n_liquidatable == 0


def test_lt_touch_beyond_liq_horizon_is_censored_as_never() -> None:
    """A touch that falls outside the horizon window does not count as liquidatable."""
    days = [
        ("2024-01-01", 100.0, 100.0),  # threshold 50 — first touch at Jan-5 open (45)
        ("2024-01-02", 60.0, 60.0),    # threshold 30 for these openings — never reached
        ("2024-01-03", 60.0, 60.0),
        ("2024-01-04", 60.0, 60.0),
        ("2024-01-05", 45.0, 45.0),
        ("2024-01-06", 45.0, 45.0),
    ]
    full = _curves(days)  # default 365-day horizon → the touch is seen
    assert full.n_liquidatable == 1
    assert full.days_to_liquidation[0] == pytest.approx(3.0 + 17.5 / 24.0, abs=1e-12)

    short = _curves(days, liq_horizon_days=3.0)  # Jan-1 window ends at Jan-4 close
    assert short.n_liquidatable == 0
    # Jan-1's opening observed its full 3-day window without a touch → "never".
    assert short.n_never_liquidatable >= 1


# ---------------------------------------------------------------------------
# Censoring at the insolvency horizon
# ---------------------------------------------------------------------------


def test_event_solvent_after_full_insolvency_horizon() -> None:
    """Price holds above the insolvency line for the whole window → 'solvent after'."""
    stats = _curves([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 50.0, 45.0),    # touch at the open (50 ≤ 50); insolvency line = 40
        ("2024-01-03", 45.0, 45.0),    # 45 never drops below 40 …
        ("2024-01-04", 45.0, 45.0),
        ("2024-01-05", 45.0, 45.0),
        ("2024-01-06", 45.0, 45.0),    # … and > 3 days of post-trigger coverage exist
    ], insolvency_horizon_days=3.0)
    assert stats.n_liquidatable == 1
    assert stats.n_insolvencies == 0
    assert stats.n_solvent_after_horizon == 1
    assert stats.n_censored_events == 0


def test_event_censored_when_history_ends_before_insolvency_horizon() -> None:
    """Same trigger, but history ends 1.3 days later → the event is censored, not 'solvent'."""
    stats = _curves([
        ("2024-01-01", 100.0, 100.0),
        ("2024-01-02", 50.0, 45.0),
        ("2024-01-03", 45.0, 45.0),
    ], insolvency_horizon_days=3.0)
    assert stats.n_liquidatable == 1
    assert stats.n_insolvencies == 0
    assert stats.n_solvent_after_horizon == 0
    assert stats.n_censored_events == 1


# ---------------------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------------------


def test_compute_curves_rejects_bad_parameters() -> None:
    s = _stream([("2024-01-01", 1.0, 1.0)])
    with pytest.raises(ValueError):
        compute_curves(s, ltv=0.9, lt=0.8)                       # ltv ≥ lt
    with pytest.raises(ValueError):
        compute_curves(s, ltv=0.4, lt=1.2)                       # lt ≥ 1
    with pytest.raises(ValueError):
        compute_curves(s, ltv=0.4, lt=0.8, liq_horizon_days=0.0)
    with pytest.raises(ValueError):
        compute_curves(s, ltv=0.4, lt=0.8, insolvency_horizon_days=-1.0)


def test_pct_liquidatable_within_window() -> None:
    stats = CurveStats(
        n_openings=10, n_liquidatable=4, n_never_liquidatable=3,
        n_censored_openings=3, days_to_liquidation=(5.0, 30.0, 31.0, 100.0),
        n_insolvencies=0, n_solvent_after_horizon=0, n_censored_events=0,
        days_to_insolvency=(),
    )
    # 5.0 and 30.0 qualify (inclusive bound); denominator = all openings.
    assert stats.pct_liquidatable_within(30.0) == pytest.approx(0.2, abs=1e-15)
    assert stats.pct_liquidatable_within(1.0) == 0.0
    empty = CurveStats(0, 0, 0, 0, (), 0, 0, 0, ())
    assert empty.pct_liquidatable_within(30.0) is None


def test_compute_curves_empty_stream_returns_zeros() -> None:
    empty = pd.DataFrame(
        {"price": pd.Series(dtype=float), "is_close": pd.Series(dtype=bool)},
        index=pd.DatetimeIndex([]),
    )
    stats = compute_curves(empty, ltv=LTV, lt=LT)
    assert stats == CurveStats(0, 0, 0, 0, (), 0, 0, 0, ())


# ---------------------------------------------------------------------------
# calibrate_ltv — solve LTV for the target P(liq ≤ 30d) band
# ---------------------------------------------------------------------------
#
# All calibration fixtures use one flat stretch plus a single crash whose
# depth/timing pins P(liq ≤ 30d) exactly: openings on the 30 days before
# the crash trigger on it iff their threshold P0·LTV/LT lies above the
# crash price; earlier openings have it outside their 30-day window and
# later openings enter at the post-crash price. The metric is therefore a
# clean step function of LTV with a hand-computable value.

BAND = (0.10, 0.15)


def _dated(prices: list[tuple[float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="D")
    return _stream([(d.strftime("%Y-%m-%d"), o, c) for d, (o, c) in zip(dates, prices)])


def _calibrate(prices: list[tuple[float, float]], current_ltv: float, **kw):
    kw.setdefault("lt", LT)
    kw.setdefault("band", BAND)
    kw.setdefault("t_user_days", 3.0)
    kw.setdefault("minimum_gap", 0.02)
    return calibrate_ltv(_dated(prices), current_ltv=current_ltv, **kw)


def test_calibrate_lowers_over_liquidating_ltv() -> None:
    """Crash to 60 on day 61: 30 of 70 openings (43 %) trigger at LTV 0.60 — far over
    the 15 % bound. The metric drops to 0 % below LTV = 0.48 (threshold ratio 0.6),
    so the solver must settle just under 0.48 (the band is jumped over — constraints
    dominate and the upper bound is what binds)."""
    prices = [(100.0, 100.0)] * 60 + [(60.0, 60.0)] * 10
    ltv, status = _calibrate(prices, current_ltv=0.60)
    assert status == "lowered"
    assert ltv < 0.60
    assert ltv == pytest.approx(0.48, abs=0.005)
    stats = compute_curves(_dated(prices), ltv=ltv, lt=LT, liq_horizon_days=30.0)
    assert stats.pct_liquidatable_within(30.0) <= BAND[1]


def test_calibrate_raises_under_liquidating_ltv() -> None:
    """One-day dip to 75 on day 61 of 200: 30/200 = 15 % of openings trigger once
    LTV ≥ 0.60 (threshold ratio 0.75), 0 % below. Current LTV 0.50 under-liquidates
    (0 % < 10 %), so the solver must raise to ~0.60 where the metric enters the band."""
    prices = [(100.0, 100.0)] * 60 + [(75.0, 100.0)] + [(100.0, 100.0)] * 139
    ltv, status = _calibrate(prices, current_ltv=0.50)
    assert status == "raised"
    assert ltv > 0.50
    assert ltv == pytest.approx(0.60, abs=0.005)
    stats = compute_curves(_dated(prices), ltv=ltv, lt=LT, liq_horizon_days=30.0)
    liq30 = stats.pct_liquidatable_within(30.0)
    assert BAND[0] <= liq30 <= BAND[1]
    # Reaction floor holds: only the 3 openings just before the dip trigger ≤ 3d.
    assert stats.pct_liquidatable_within(3.0) <= 0.10


def test_calibrate_keeps_ltv_already_in_band() -> None:
    """Same dip over 250 days: 30/250 = 12 % at LTV 0.65 — inside [10 %, 15 %]."""
    prices = [(100.0, 100.0)] * 60 + [(75.0, 100.0)] + [(100.0, 100.0)] * 189
    ltv, status = _calibrate(prices, current_ltv=0.65)
    assert status == "unchanged"
    assert ltv == 0.65


def test_calibrate_caps_at_lt_minus_minimum_gap() -> None:
    """A flat path never liquidates at any LTV — raising stops at LT − minimum_gap."""
    prices = [(100.0, 100.0)] * 50
    ltv, status = _calibrate(prices, current_ltv=0.40)
    assert status == "raised-capped"
    assert ltv == pytest.approx(LT - 0.02, abs=1e-12)


def test_calibrate_rejects_bad_inputs() -> None:
    prices = [(100.0, 100.0)] * 5
    with pytest.raises(ValueError):
        _calibrate(prices, current_ltv=0.4, band=(0.15, 0.10))   # inverted band
    with pytest.raises(ValueError):
        _calibrate(prices, current_ltv=0.4, minimum_gap=0.9)     # gap ≥ lt


# ---------------------------------------------------------------------------
# run_curves — scenario wiring on synthetic parquets
# ---------------------------------------------------------------------------


def _universe() -> AssetUniverse:
    return AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD", price_history_years=1, include_overnight_gaps=True),
        collaterals=(
            Collateral(symbol="AAPLon", type="rwa", category="equity",
                       underlying_ticker="AAPL", use="collateral_only"),
        ),
        borrowables=(
            Borrowable(symbol="USDT", type="crypto", category="stable",
                       price_source="USDT-USD", use="lending_and_borrowing",
                       volatility_class="stable"),
            Borrowable(symbol="BNB", type="crypto", category="native",
                       price_source="BNB-USD", use="lending_and_borrowing",
                       volatility_class="volatile"),
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
        emode_categories=(EModeCategory(name="stable", borrowables=("USDT",)),),
    )


def _walk_frame(rng: np.random.Generator, idx: pd.DatetimeIndex,
                level: float, step: float) -> pd.DataFrame:
    """Gentle random walk with correlated open/close (small intraday noise).

    Independent walks for open and close would fabricate huge intraday
    moves (the two walks drift apart), blowing up sigma_stress until
    LTV goes negative and the row is skipped — so keep one walk and add
    per-day noise for the close.
    """
    opens = level + rng.normal(0, step, size=len(idx)).cumsum()
    closes = opens + rng.normal(0, step, size=len(idx))
    return pd.DataFrame({"open": opens, "close": closes}, index=idx)


def test_run_curves_end_to_end_scenario_wiring(tmp_path: Path) -> None:
    """Each scenario rides its own pair with its own parameter row."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(seed=11)
    aapl = _walk_frame(rng, idx, level=100.0, step=0.2)
    usdt = pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx)
    bnb = _walk_frame(rng, idx, level=300.0, step=0.6)
    aapl.to_parquet(tmp_path / "AAPL.parquet", index=True)
    usdt.to_parquet(tmp_path / "USDT-USD.parquet", index=True)
    bnb.to_parquet(tmp_path / "BNB-USD.parquet", index=True)

    results = run_curves(universe=_universe(), data_dir=tmp_path)
    by_scen = {r.scenario.name: r for r in results}
    assert set(by_scen) == {"e-mode", "volatile"}

    emode, volatile = by_scen["e-mode"], by_scen["volatile"]
    assert emode.scenario.borrowable == "USDT"
    assert emode.scenario.param_set == "emode:stable"
    assert volatile.scenario.borrowable == "BNB"
    assert volatile.scenario.param_set == "base"
    # E-Mode parameters come from the stable row → strictly more permissive.
    assert emode.lt > volatile.lt
    for r in results:
        assert 0.0 < r.ltv < r.lt < 1.0
        assert r.stats.n_openings > 0
        # 60 days of history << 365 → nothing can be 'never liquidatable'.
        assert r.stats.n_never_liquidatable == 0


def test_run_curves_skips_scenario_when_param_set_missing(tmp_path: Path) -> None:
    """A universe without an emode category yields only the volatile scenario."""
    from dataclasses import replace
    universe = replace(_universe(), emode_categories=())

    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rng = np.random.default_rng(seed=3)
    _walk_frame(rng, idx, level=100.0, step=0.2).to_parquet(tmp_path / "AAPL.parquet", index=True)
    pd.DataFrame({"open": 1.0, "close": 1.0}, index=idx).to_parquet(tmp_path / "USDT-USD.parquet", index=True)
    _walk_frame(rng, idx, level=300.0, step=0.6).to_parquet(tmp_path / "BNB-USD.parquet", index=True)

    results = run_curves(universe=universe, data_dir=tmp_path)
    assert [r.scenario.name for r in results] == ["volatile"]


# ---------------------------------------------------------------------------
# CLI tables and main
# ---------------------------------------------------------------------------


def _result(collateral: str, scenario: Scenario, stats: CurveStats) -> CurveResult:
    return CurveResult(collateral=collateral, scenario=scenario, lt=LT, ltv=LTV, stats=stats)


def test_liquidability_table_percentiles_and_dashes() -> None:
    with_events = _result("AAPLon", DEFAULT_SCENARIOS[1], CurveStats(
        n_openings=100, n_liquidatable=40, n_never_liquidatable=50,
        n_censored_openings=10, days_to_liquidation=tuple(float(d) for d in range(1, 41)),
        n_insolvencies=0, n_solvent_after_horizon=0, n_censored_events=0,
        days_to_insolvency=(),
    ))
    no_events = _result("SPYon", DEFAULT_SCENARIOS[0], CurveStats(
        n_openings=100, n_liquidatable=0, n_never_liquidatable=90,
        n_censored_openings=10, days_to_liquidation=(),
        n_insolvencies=0, n_solvent_after_horizon=0, n_censored_events=0,
        days_to_insolvency=(),
    ))
    text = format_liquidability_table([with_events, no_events])
    header = text.splitlines()[0]
    assert "liq ≤ 30d (%)" in header
    assert "liq ≤ 365d (%)" in header
    for col in ("p10 (d)", "p25 (d)", "p50 (d)", "p75 (d)", "p90 (d)", "never (%)"):
        assert col in header
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "30.0" in aapl                      # % liquidatable ≤ 30d: days 1..30 of 100 openings
    assert "40.0" in aapl                      # % liquidatable
    assert "50.0" in aapl                      # % never
    assert "20.5" in aapl                      # p50 of 1..40
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert "0.0" in spy and "90.0" in spy
    assert bc.NA_DASH in spy                   # percentile cells dashed without events


def test_insolvency_table_solvent_share_and_dashes() -> None:
    r = _result("AAPLon", DEFAULT_SCENARIOS[1], CurveStats(
        n_openings=100, n_liquidatable=10, n_never_liquidatable=0,
        n_censored_openings=0, days_to_liquidation=tuple([5.0] * 10),
        n_insolvencies=2, n_solvent_after_horizon=7, n_censored_events=1,
        days_to_insolvency=(2.0, 6.0),
    ))
    text = format_insolvency_table([r])
    assert "solvent > 30d (%)" in text.splitlines()[0]
    line = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "70.0" in line                      # 7 of 10 events still solvent
    assert "4.0" in line                       # p50 of (2, 6)
    no_liq = _result("SPYon", DEFAULT_SCENARIOS[0], CurveStats(
        n_openings=10, n_liquidatable=0, n_never_liquidatable=10,
        n_censored_openings=0, days_to_liquidation=(),
        n_insolvencies=0, n_solvent_after_horizon=0, n_censored_events=0,
        days_to_insolvency=(),
    ))
    line2 = next(l for l in format_insolvency_table([no_liq]).splitlines() if l.startswith("SPYon"))
    assert bc.NA_DASH in line2                 # solvent % undefined without liquidations


def test_calibration_table_delta_and_unchanged_cells() -> None:
    results = [
        CalibrationResult(collateral="AAPLon", scenario=DEFAULT_SCENARIOS[0], lt=LT,
                          band=BAND, current_ltv=0.60, current_liq30=0.43,
                          calibrated_ltv=0.48, calibrated_liq30=0.0, status="lowered"),
        CalibrationResult(collateral="SPYon", scenario=DEFAULT_SCENARIOS[1], lt=LT,
                          band=BAND, current_ltv=0.65, current_liq30=0.12,
                          calibrated_ltv=0.65, calibrated_liq30=0.12, status="unchanged"),
        CalibrationResult(collateral="TLTon", scenario=DEFAULT_SCENARIOS[0], lt=LT,
                          band=BAND, current_ltv=0.40, current_liq30=0.0,
                          calibrated_ltv=0.78, calibrated_liq30=0.0, status="raised-capped"),
    ]
    text = format_calibration_table(results)
    header = text.splitlines()[0]
    for col in ("current LTV (%)", "current ≤30d (%)", "calibrated LTV (%)",
                "calibrated ≤30d (%)", "Δ"):
        assert col in header
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "-12.00pp" in aapl
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert "unchanged" in spy
    tlt = next(l for l in text.splitlines() if l.startswith("TLTon"))
    assert "+38.00pp (capped)" in tlt


def test_main_cli_calibrate_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bc, "load_universe", lambda *a, **k: _universe())
    monkeypatch.setattr(
        bc, "run_calibration",
        lambda **kw: [CalibrationResult(
            collateral="AAPLon", scenario=DEFAULT_SCENARIOS[0], lt=LT, band=BAND,
            current_ltv=0.60, current_liq30=0.30,
            calibrated_ltv=0.52, calibrated_liq30=0.14, status="lowered",
        )],
    )
    rc = bc.main(["--data-dir", str(tmp_path), "--calibrate", "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LTV calibration" in out
    assert "10–15%" in out              # band from the config lands in the title
    assert "-8.00pp" in out
    # The standard curve tables are NOT printed in calibrate mode.
    assert "(1) Time to liquidability" not in out


def test_main_cli_prints_both_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bc, "load_universe", lambda *a, **k: _universe())
    monkeypatch.setattr(
        bc, "run_curves",
        lambda **kw: [_result("AAPLon", DEFAULT_SCENARIOS[0], CurveStats(
            n_openings=10, n_liquidatable=2, n_never_liquidatable=3,
            n_censored_openings=5, days_to_liquidation=(4.0, 8.0),
            n_insolvencies=1, n_solvent_after_horizon=1, n_censored_events=0,
            days_to_insolvency=(3.0,),
        ))],
    )
    rc = bc.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(1) Time to liquidability" in out
    assert "(2) Liquidation zone → insolvency" in out
    assert "not a forecast" in out
    assert "e-mode" in out
