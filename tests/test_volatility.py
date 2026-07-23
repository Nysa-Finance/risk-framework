"""Tests for ``nysa_risk.volatility``.

The core numeric tests use tiny hand-computed synthetic series so the
EWMA recursion, the equity-calendar alignment, and the ``sigma_stress``
quantile can each be verified against numbers derived on paper.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nysa_risk import volatility as vol
from nysa_risk.config import (
    AssetUniverse,
    Borrowable,
    Calibration,
    Collateral,
    Meta,
    OndoConfig,
    PairsPolicy,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _daily_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def _bdays_index(dates: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d) for d in dates])


def _constant_prices(index: pd.DatetimeIndex, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"open": value, "close": value}, index=index)


# ---------------------------------------------------------------------------
# _leg_on_reference / align_pair
# ---------------------------------------------------------------------------


def test_leg_on_reference_returns_native_values_for_reference_leg() -> None:
    idx = _bdays_index(["2024-01-05", "2024-01-08"])
    leg = pd.DataFrame({"open": [100.0, 104.0], "close": [105.0, 110.0]}, index=idx)
    o, c = vol._leg_on_reference(leg, idx, is_reference=True)
    assert list(o) == [100.0, 104.0]
    assert list(c) == [105.0, 110.0]


def test_leg_on_reference_uses_most_recent_close_for_non_reference_leg() -> None:
    """Crypto → equity alignment: open @ d = last close < d, close @ d = last close ≤ d."""
    crypto_idx = _daily_index("2024-01-03", 6)  # Wed..Mon
    crypto = pd.DataFrame(
        {"open": [999.0] * 6, "close": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]},
        index=crypto_idx,
    )
    equity_dates = _bdays_index(["2024-01-05", "2024-01-08"])  # Fri, Mon

    o, c = vol._leg_on_reference(crypto, equity_dates, is_reference=False)
    # Fri: open = Thu close (1010), close = Fri close (1020).
    # Mon: open = Sun close (1040), close = Mon close (1050).
    assert list(o) == [1010.0, 1040.0]
    assert list(c) == [1020.0, 1050.0]


def test_align_pair_uses_equity_calendar_and_captures_weekend_gap() -> None:
    equity = pd.DataFrame(
        {"open": [100.0, 104.0], "close": [105.0, 110.0]},
        index=_bdays_index(["2024-01-05", "2024-01-08"]),
    )
    crypto = pd.DataFrame(
        {"open": [999.0] * 6, "close": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]},
        index=_daily_index("2024-01-03", 6),
    )
    aligned = vol.align_pair(equity, crypto)

    # Reference calendar is the equity leg (fewer dates in overlap).
    assert list(aligned.index) == list(_bdays_index(["2024-01-05", "2024-01-08"]))
    assert list(aligned["c_open"]) == [100.0, 104.0]
    assert list(aligned["c_close"]) == [105.0, 110.0]
    # Weekend gap for crypto: Mon's "open value" is Sunday's close (1040), NOT Friday's close.
    # This preserves the crypto's actual weekend move against the equity's Fri→Mon gap.
    assert list(aligned["b_open"]) == [1010.0, 1040.0]
    assert list(aligned["b_close"]) == [1020.0, 1050.0]


def test_align_pair_drops_leading_rows_with_no_prior_close() -> None:
    """Different calendars: the denser leg starts on the first ref date → its
    ``open`` at that date has no strictly-earlier close and the row is dropped."""
    equity = pd.DataFrame(
        {"open": [100.0, 104.0], "close": [105.0, 110.0]},
        index=_bdays_index(["2024-01-05", "2024-01-08"]),
    )
    crypto = pd.DataFrame(
        {"open": [999.0] * 4, "close": [1020.0, 1030.0, 1040.0, 1050.0]},
        index=_daily_index("2024-01-05", 4),  # Fri, Sat, Sun, Mon
    )
    aligned = vol.align_pair(equity, crypto)
    assert list(aligned.index) == list(_bdays_index(["2024-01-08"]))
    assert aligned.loc[pd.Timestamp("2024-01-08"), "b_open"] == 1040.0  # Sun close
    assert aligned.loc[pd.Timestamp("2024-01-08"), "b_close"] == 1050.0  # Mon close


def test_align_pair_raises_when_no_overlap() -> None:
    a = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=_bdays_index(["2024-01-01"]))
    b = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=_bdays_index(["2025-01-01"]))
    a.index, b.index  # concrete indexes
    a = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=_bdays_index(["2024-01-01"]))
    b = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=_bdays_index(["2025-06-01"]))
    with pytest.raises(ValueError, match="no overlapping window"):
        # Force no overlap by using disjoint ranges.
        vol.align_pair(a.loc[:"2024-06-01"], b.loc["2025-01-01":])


# ---------------------------------------------------------------------------
# relative_log_returns — hand-verified numbers
# ---------------------------------------------------------------------------


def test_relative_log_returns_against_constant_borrowable() -> None:
    """With B ≡ 1, log(C/B).diff() collapses to log(C).diff() = the log-returns of C."""
    # Construct C so the interleaved stream gives exactly the desired log-returns.
    r = [0.02, -0.03, 0.01]
    logp = np.cumsum([math.log(100.0)] + r)  # 4 log-prices
    p = np.exp(logp)  # [100, 100·e^0.02, 100·e^-0.01, 100]
    dates = _bdays_index(["2024-01-01", "2024-01-02"])
    c = pd.DataFrame({"open": [p[0], p[2]], "close": [p[1], p[3]]}, index=dates)
    b = _constant_prices(dates, value=1.0)

    rets = vol.relative_log_returns(c, b)
    assert len(rets) == 3
    np.testing.assert_allclose(rets.to_numpy(), r, atol=1e-12)


def test_relative_log_returns_matches_leg_difference() -> None:
    """Algebraic identity: rel_ret = r_C - r_B on the aligned stream."""
    dates = _bdays_index(["2024-01-05", "2024-01-08", "2024-01-09"])
    c = pd.DataFrame({"open": [100.0, 105.0, 108.0], "close": [102.0, 106.0, 110.0]}, index=dates)
    b = pd.DataFrame({"open": [200.0, 210.0, 215.0], "close": [204.0, 208.0, 220.0]}, index=dates)

    aligned = vol.align_pair(c, b)
    c_stream = vol._interleaved_stream(aligned, "c")
    b_stream = vol._interleaved_stream(aligned, "b")
    expected = (np.log(c_stream) - np.log(b_stream)).diff().dropna()

    rets = vol.relative_log_returns(c, b)
    np.testing.assert_allclose(rets.to_numpy(), expected.to_numpy(), atol=1e-14)


# ---------------------------------------------------------------------------
# EWMA — hand-computed numbers with lambda = 0.5
# ---------------------------------------------------------------------------


def test_ewma_volatility_matches_hand_computed_recursion() -> None:
    r = pd.Series([0.02, -0.03, 0.01])
    sigma = vol.ewma_volatility(r, lam=0.5)
    # var_1 = 0.02^2 = 4e-4
    # var_2 = 0.5·4e-4 + 0.5·9e-4 = 6.5e-4
    # var_3 = 0.5·6.5e-4 + 0.5·1e-4 = 3.75e-4
    expected = np.sqrt([4e-4, 6.5e-4, 3.75e-4])
    np.testing.assert_allclose(sigma.to_numpy(), expected, atol=1e-15)


def test_ewma_volatility_rejects_out_of_range_lambda() -> None:
    r = pd.Series([0.01, -0.01])
    with pytest.raises(ValueError):
        vol.ewma_volatility(r, lam=0.0)
    with pytest.raises(ValueError):
        vol.ewma_volatility(r, lam=1.0)


# ---------------------------------------------------------------------------
# sigma_stress — hand-verified quantile
# ---------------------------------------------------------------------------


def test_sigma_stress_median_and_max() -> None:
    sigma = pd.Series([0.02, 0.025495097567963924, 0.01936491673103708])
    # Median of three sorted values.
    assert vol.sigma_stress_value(sigma, quantile=0.5) == pytest.approx(0.02, abs=1e-15)
    # Max at q=1.0.
    assert vol.sigma_stress_value(sigma, quantile=1.0) == pytest.approx(0.025495097567963924, abs=1e-15)


def test_sigma_stress_empty_raises() -> None:
    with pytest.raises(ValueError):
        vol.sigma_stress_value(pd.Series(dtype=float), quantile=0.95)


# ---------------------------------------------------------------------------
# End-to-end pair computation
# ---------------------------------------------------------------------------


def test_compute_pair_from_prices_end_to_end() -> None:
    r = [0.02, -0.03, 0.01]
    logp = np.cumsum([math.log(100.0)] + r)
    p = np.exp(logp)
    dates = _bdays_index(["2024-01-01", "2024-01-02"])
    c = pd.DataFrame({"open": [p[0], p[2]], "close": [p[1], p[3]]}, index=dates)
    b = _constant_prices(dates, value=1.0)

    result = vol.compute_pair_from_prices(
        c, b,
        collateral="AAPLon", borrowable="USDC",
        collateral_ticker="AAPL", borrowable_ticker="USDC-USD",
        lam=0.5, quantile=1.0, gap_quantile=0.5,
    )
    # sigma series = sqrt([4e-4, 6.5e-4, 3.75e-4]); max is sqrt(6.5e-4).
    assert result.sigma_stress == pytest.approx(math.sqrt(6.5e-4), abs=1e-15)
    # Daily scale = ×√2 (two observations per day).
    assert result.sigma_stress_daily == pytest.approx(math.sqrt(6.5e-4) * math.sqrt(2), abs=1e-15)
    # sigma_gap comes off the SAME series at the gap quantile (median = sqrt(4e-4) = 0.02).
    assert result.sigma_gap == pytest.approx(0.02, abs=1e-15)
    assert result.sigma_gap_daily == pytest.approx(0.02 * math.sqrt(2), abs=1e-15)
    assert result.sigma_gap != result.sigma_stress
    assert result.n_observations == 3
    assert result.first_date == date(2024, 1, 1)
    assert result.last_date == date(2024, 1, 2)


def test_compute_pair_short_history_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    dates = _bdays_index(["2024-01-01", "2024-01-02"])
    c = pd.DataFrame({"open": [100.0, 101.0], "close": [102.0, 103.0]}, index=dates)
    b = _constant_prices(dates, value=1.0)
    caplog.set_level(logging.WARNING, logger=vol.LOGGER.name)
    vol.compute_pair_from_prices(
        c, b,
        collateral="CRCLon", borrowable="USDC",
        collateral_ticker="CRCL", borrowable_ticker="USDC-USD",
        lam=0.94, quantile=0.95, gap_quantile=0.90,
        requested_years=10.0,
    )
    assert any("short overlapping history" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# compute_all_pairs — with real universe shape + synthetic on-disk data
# ---------------------------------------------------------------------------


def _write_parquet(dir_: Path, ticker: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(dir_ / f"{ticker}.parquet", index=True)


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
            Borrowable(symbol="WBTC", type="crypto", category="wrapped",
                       price_source="BTC-USD", use="lending_and_borrowing",
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
    )


def test_compute_all_pairs_from_synthetic_parquets(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rng = np.random.default_rng(seed=42)
    # AAPL random walk.
    aapl = pd.DataFrame(
        {
            "open": 100 + rng.normal(0, 1, size=len(idx)).cumsum(),
            "close": 100 + rng.normal(0, 1, size=len(idx)).cumsum(),
        },
        index=idx,
    )
    # BTC-USD daily calendar (superset).
    btc_idx = pd.date_range("2024-01-01", periods=45, freq="D")
    btc = pd.DataFrame(
        {"open": 40000 + rng.normal(0, 200, size=len(btc_idx)).cumsum(),
         "close": 40000 + rng.normal(0, 200, size=len(btc_idx)).cumsum()},
        index=btc_idx,
    )
    # USDC-USD stable, ~1.0.
    usdc = pd.DataFrame(
        {"open": 1.0 + rng.normal(0, 1e-4, size=len(btc_idx)),
         "close": 1.0 + rng.normal(0, 1e-4, size=len(btc_idx))},
        index=btc_idx,
    )
    _write_parquet(tmp_path, "AAPL", aapl)
    _write_parquet(tmp_path, "BTC-USD", btc)
    _write_parquet(tmp_path, "USDC-USD", usdc)

    universe = _synthetic_universe()
    results = vol.compute_all_pairs(universe=universe, data_dir=tmp_path)
    pairs = {(r.collateral, r.borrowable) for r in results}
    # Only collateral_only assets sit on the left of a pair — no crypto/crypto entries.
    assert pairs == {("AAPLon", "USDC"), ("AAPLon", "WBTC")}
    for r in results:
        assert r.sigma_stress > 0
        assert r.sigma_stress_daily == pytest.approx(r.sigma_stress * math.sqrt(2))
        # Gap regime: lower quantile of the same series → never above sigma_stress.
        assert 0 < r.sigma_gap <= r.sigma_stress
        assert r.sigma_gap_daily == pytest.approx(r.sigma_gap * math.sqrt(2))


def test_compute_all_pairs_skips_missing_ticker(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    aapl = pd.DataFrame({"open": np.arange(1.0, 11.0), "close": np.arange(1.0, 11.0)}, index=idx)
    _write_parquet(tmp_path, "AAPL", aapl)
    _write_parquet(tmp_path, "USDC-USD", _constant_prices(idx, value=1.0))
    # BTC-USD deliberately missing.

    caplog.set_level(logging.ERROR, logger=vol.LOGGER.name)
    results = vol.compute_all_pairs(universe=_synthetic_universe(), data_dir=tmp_path)
    kept = {(r.collateral, r.borrowable) for r in results}
    assert ("AAPLon", "USDC") in kept
    assert ("AAPLon", "WBTC") not in kept
    assert any("skipped" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# CLI table
# ---------------------------------------------------------------------------


def _dummy_result(collateral: str, borrowable: str, sigma_daily: float) -> vol.PairResult:
    return vol.PairResult(
        collateral=collateral, borrowable=borrowable,
        collateral_ticker=collateral, borrowable_ticker=borrowable,
        n_observations=100,
        first_date=date(2020, 1, 1), last_date=date(2025, 1, 1),
        effective_years=5.0,
        sigma_stress=sigma_daily / math.sqrt(2),
        sigma_stress_daily=sigma_daily,
        sigma_gap=sigma_daily / math.sqrt(2),
        sigma_gap_daily=sigma_daily,
    )


def test_format_table_is_sorted_by_daily_sigma_descending() -> None:
    rows = [
        _dummy_result("AAPLon", "USDC", 0.02),
        _dummy_result("TSLAon", "WBTC", 0.08),
        _dummy_result("SPYon", "USDC", 0.01),
    ]
    text = vol.format_table(rows)
    lines = text.splitlines()
    assert lines[0].startswith("pair")
    # Order in the body (after header + separator) is TSLA → AAPL → SPY.
    body = lines[2:]
    assert body[0].startswith("TSLAon/WBTC")
    assert body[1].startswith("AAPLon/USDC")
    assert body[2].startswith("SPYon/USDC")


def test_main_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    for ticker, seed in [("AAPL", 1), ("BTC-USD", 2), ("USDC-USD", 3)]:
        rng = np.random.default_rng(seed=seed)
        frame = pd.DataFrame(
            {"open": 100 + rng.normal(0, 1, size=len(idx)).cumsum(),
             "close": 100 + rng.normal(0, 1, size=len(idx)).cumsum()},
            index=idx,
        )
        _write_parquet(tmp_path, ticker, frame)

    monkeypatch.setattr(vol, "load_universe", lambda *a, **k: _synthetic_universe())
    rc = vol.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sigma_stress (daily %)" in out
    assert "AAPLon/USDC" in out or "AAPLon/WBTC" in out
