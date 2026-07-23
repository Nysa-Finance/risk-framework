"""Tests for ``nysa_risk.extraction.underlying``.

No network calls: every yfinance interaction is replaced by an in-memory
fake that returns a pre-built pandas DataFrame per ticker.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from nysa_risk.extraction import underlying as mod


def _fake_history(start: date, end: date) -> pd.DataFrame:
    """Business-day OHLC frame mimicking a yfinance response (auto_adjust=True)."""
    idx = pd.bdate_range(start=start, end=end - timedelta(days=1))
    n = len(idx)
    return pd.DataFrame(
        {
            "Open": [100.0 + i * 0.1 for i in range(n)],
            "High": [101.0 + i * 0.1 for i in range(n)],
            "Low": [99.0 + i * 0.1 for i in range(n)],
            "Close": [100.5 + i * 0.1 for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        },
        index=idx,
    )


def _make_downloader(short_ticker: str | None = None, short_start: date | None = None):
    """Return a fake downloader; optionally truncate one ticker to a shorter window."""

    def _fake(ticker: str, start: date, end: date) -> pd.DataFrame:
        if short_ticker is not None and ticker == short_ticker and short_start is not None:
            effective_start = max(start, short_start)
            if effective_start >= end:
                return pd.DataFrame()
            return _fake_history(effective_start, end)
        return _fake_history(start, end)

    return _fake


def test_download_asset_writes_parquet_with_open_and_close(tmp_path: Path) -> None:
    result = mod.download_asset(
        "AAPL",
        years=2,
        out_dir=tmp_path,
        end=date(2026, 7, 21),
        downloader=_make_downloader(),
        sleeper=lambda _s: None,
    )
    assert result.path == tmp_path / "AAPL.parquet"
    assert result.path.exists()
    assert result.rows > 0
    assert result.short_history is False

    df = pd.read_parquet(result.path)
    assert list(df.columns) == ["open", "close"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing
    assert not df["open"].isna().any()
    assert not df["close"].isna().any()
    # Trading-day cadence — no weekends.
    assert (df.index.dayofweek < 5).all()


def test_short_history_logs_warning_and_still_writes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=mod.LOGGER.name)
    listing_date = date(2025, 6, 5)  # ~13 months of history when requesting 10y
    result = mod.download_asset(
        "CRCL",
        years=10,
        out_dir=tmp_path,
        end=date(2026, 7, 21),
        downloader=_make_downloader(short_ticker="CRCL", short_start=listing_date),
        sleeper=lambda _s: None,
    )
    assert result.short_history is True
    assert result.effective_years < 10 * mod.SHORT_HISTORY_TOLERANCE
    assert result.first_date is not None and result.first_date >= listing_date
    assert result.path.exists()
    assert any("shorter history than requested" in rec.message for rec in caplog.records)


def test_download_retries_on_transient_failures(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _flaky(ticker: str, start: date, end: date) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("simulated network hiccup")
        return _fake_history(start, end)

    waits: list[float] = []
    result = mod.download_asset(
        "MSFT",
        years=1,
        out_dir=tmp_path,
        end=date(2026, 7, 21),
        max_retries=4,
        backoff_seconds=0.5,
        downloader=_flaky,
        sleeper=waits.append,
    )
    assert calls["n"] == 3
    assert waits == [0.5, 1.0]  # exponential backoff between attempts 1→2, 2→3
    assert result.rows > 0


def test_download_gives_up_after_max_retries(tmp_path: Path) -> None:
    def _always_fail(ticker: str, start: date, end: date) -> pd.DataFrame:
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        mod.download_asset(
            "NVDA",
            years=1,
            out_dir=tmp_path,
            max_retries=2,
            backoff_seconds=0.01,
            downloader=_always_fail,
            sleeper=lambda _s: None,
        )


def test_normalize_handles_multiindex_columns() -> None:
    idx = pd.bdate_range(start=date(2026, 1, 5), end=date(2026, 1, 9))
    frame = pd.DataFrame(
        {
            ("Open", "AAPL"): [1.0, 2.0, 3.0, 4.0, 5.0],
            ("High", "AAPL"): [1.1, 2.1, 3.1, 4.1, 5.1],
            ("Low", "AAPL"): [0.9, 1.9, 2.9, 3.9, 4.9],
            ("Close", "AAPL"): [1.05, 2.05, 3.05, 4.05, 5.05],
        },
        index=idx,
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    out = mod._normalize(frame, "AAPL")
    assert list(out.columns) == ["open", "close"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.name == "date"
    assert len(out) == 5


def test_run_iterates_over_universe_tickers(tmp_path: Path) -> None:
    from nysa_risk.config import (
        AssetUniverse,
        Borrowable,
        Calibration,
        Collateral,
        Meta,
        OndoConfig,
        PairsPolicy,
    )

    universe = AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD", price_history_years=1, include_overnight_gaps=True),
        collaterals=(
            Collateral(symbol="AAPLon", type="rwa", category="equity", underlying_ticker="AAPL", use="collateral_only"),
            Collateral(symbol="NVDAon", type="rwa", category="equity", underlying_ticker="NVDA", use="collateral_only"),
            # Duplicate underlying to prove dedup.
            Collateral(symbol="AAPLon2", type="rwa", category="equity", underlying_ticker="AAPL", use="collateral_only"),
        ),
        borrowables=(
            Borrowable(symbol="WBTC", type="crypto", category="wrapped", price_source="BTC-USD", use="lending_and_borrowing", volatility_class="volatile"),
        ),
        pairs=PairsPolicy(default_policy="all_collaterals_vs_all_borrowables"),
        ondo=OndoConfig(limits_api="https://x", status_page="https://y", api_key_env="ONDO_API_KEY"),
        calibration=Calibration(
            ewma_lambda=0.94, stress_quantile=0.95, gap_sigma_quantile=0.90, es_factor=3.5,
            t_liq_days=0.33, t_user_days=3.0, k_user=1.53,
            stressed_liquidatable_share=0.25, rf_theta=0.01, rf_horizon_years=0.83,
            emode_min_advantage=0.05,
            target_liq30_emode=(0.10, 0.15),
            target_liq30_std=(0.10, 0.15),
            minimum_gap=0.02,
        max_uncond_bad_debt=0.004, min_calibration_years=3.0, severity_review_threshold=0.05, max_loss_given_bad_debt=0.065,        ),
    )

    seen: list[str] = []

    def _capture(ticker: str, start: date, end: date) -> pd.DataFrame:
        seen.append(ticker)
        return _fake_history(start, end)

    results = mod.run(
        universe=universe,
        out_dir=tmp_path,
        end=date(2026, 7, 21),
        downloader=_capture,
        sleeper=lambda _s: None,
    )

    # AAPL, NVDA, BTC-USD — AAPL deduped.
    assert seen == ["AAPL", "NVDA", "BTC-USD"]
    assert {r.ticker for r in results} == {"AAPL", "NVDA", "BTC-USD"}
    for t in ["AAPL", "NVDA", "BTC-USD"]:
        assert (tmp_path / f"{t}.parquet").exists()


def test_run_continues_when_one_ticker_fails(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from nysa_risk.config import (
        AssetUniverse, Borrowable, Calibration, Collateral, Meta, OndoConfig, PairsPolicy,
    )
    universe = AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD", price_history_years=1, include_overnight_gaps=True),
        collaterals=(
            Collateral(symbol="AAPLon", type="rwa", category="equity", underlying_ticker="AAPL", use="collateral_only"),
            Collateral(symbol="BROKEN", type="rwa", category="equity", underlying_ticker="ZZZZ", use="collateral_only"),
        ),
        borrowables=(),
        pairs=PairsPolicy(default_policy="all_collaterals_vs_all_borrowables"),
        ondo=OndoConfig(limits_api="https://x", status_page="https://y", api_key_env="ONDO_API_KEY"),
        calibration=Calibration(
            ewma_lambda=0.94, stress_quantile=0.95, gap_sigma_quantile=0.90, es_factor=3.5,
            t_liq_days=0.33, t_user_days=3.0, k_user=1.53,
            stressed_liquidatable_share=0.25, rf_theta=0.01, rf_horizon_years=0.83,
            emode_min_advantage=0.05,
            target_liq30_emode=(0.10, 0.15),
            target_liq30_std=(0.10, 0.15),
            minimum_gap=0.02,
        max_uncond_bad_debt=0.004, min_calibration_years=3.0, severity_review_threshold=0.05, max_loss_given_bad_debt=0.065,        ),
    )

    def _selective(ticker: str, start: date, end: date) -> pd.DataFrame:
        if ticker == "ZZZZ":
            raise ConnectionError("dead ticker")
        return _fake_history(start, end)

    caplog.set_level(logging.ERROR, logger=mod.LOGGER.name)
    results = mod.run(
        universe=universe,
        out_dir=tmp_path,
        end=date(2026, 7, 21),
        max_retries=2,
        backoff_seconds=0.01,
        downloader=_selective,
        sleeper=lambda _s: None,
    )
    assert [r.ticker for r in results] == ["AAPL"]
    assert any("giving up after retries" in rec.message for rec in caplog.records)


def test_cli_main_uses_real_config_with_mocked_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m nysa_risk.extraction.underlying` entrypoint smoke test."""
    monkeypatch.setattr(mod, "_yf_download", _make_downloader())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    rc = mod.main(["--out-dir", str(tmp_path), "--ticker", "AAPL", "--ticker", "BTC-USD"])
    assert rc == 0
    assert (tmp_path / "AAPL.parquet").exists()
    assert (tmp_path / "BTC-USD.parquet").exists()
