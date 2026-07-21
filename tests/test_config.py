"""Validates that config/assets.yaml parses into the typed dataclasses.

These tests deliberately hit the real config file so any structural
drift between the YAML and the loader is caught immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nysa_risk.config import (
    AssetUniverse,
    Borrowable,
    Calibration,
    Collateral,
    Meta,
    OndoConfig,
    PairsPolicy,
    load_universe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "assets.yaml"


@pytest.fixture(scope="module")
def universe() -> AssetUniverse:
    assert CONFIG_PATH.exists(), f"missing config file: {CONFIG_PATH}"
    return load_universe(CONFIG_PATH)


def test_load_universe_returns_asset_universe(universe: AssetUniverse) -> None:
    assert isinstance(universe, AssetUniverse)


def test_meta_fields(universe: AssetUniverse) -> None:
    meta = universe.meta
    assert isinstance(meta, Meta)
    assert meta.base_currency == "USD"
    assert meta.price_history_years >= 5
    assert meta.include_overnight_gaps is True
    # version is coerced to string in the loader even if YAML gives a float
    assert isinstance(meta.version, str)


def test_collaterals_shape(universe: AssetUniverse) -> None:
    assert len(universe.collaterals) > 0
    seen = set()
    for c in universe.collaterals:
        assert isinstance(c, Collateral)
        assert c.symbol and c.symbol not in seen, f"duplicate collateral: {c.symbol}"
        seen.add(c.symbol)
        assert c.underlying_ticker, f"{c.symbol} missing underlying_ticker"
        assert c.type in {"rwa", "crypto"}
        assert c.use in {"collateral_only", "lending_and_borrowing"}


def test_borrowables_shape(universe: AssetUniverse) -> None:
    assert len(universe.borrowables) > 0
    seen = set()
    for b in universe.borrowables:
        assert isinstance(b, Borrowable)
        assert b.symbol and b.symbol not in seen, f"duplicate borrowable: {b.symbol}"
        seen.add(b.symbol)
        assert b.price_source, f"{b.symbol} missing price_source"
        assert b.type in {"rwa", "crypto"}
        assert b.use in {"collateral_only", "lending_and_borrowing"}
        assert b.volatility_class in {"volatile", "stable"}


def test_calibration_ranges(universe: AssetUniverse) -> None:
    cal = universe.calibration
    assert isinstance(cal, Calibration)
    # §3.1 EWMA lambda in the RiskMetrics range and stress quantile a percentile.
    assert 0.0 < cal.ewma_lambda < 1.0
    assert 0.5 < cal.stress_quantile < 1.0
    # §3.2 fat-tailed ES multiplier is >1 by construction.
    assert cal.es_factor > 1.0
    # §4 horizons must be positive; §4.1 k_user is a positive quantile multiplier.
    assert cal.t_liq_days > 0
    assert cal.t_user_days > 0
    assert cal.k_user > 0
    # docs/nysa-lb-caps.md §2.1 — a fraction in (0, 1].
    assert 0.0 < cal.stressed_liquidatable_share <= 1.0
    # §6 reserve fund — theta a fraction of debt, horizon in years.
    assert 0.0 < cal.rf_theta < 1.0
    assert cal.rf_horizon_years > 0


def test_pairs_policy(universe: AssetUniverse) -> None:
    assert isinstance(universe.pairs, PairsPolicy)
    assert universe.pairs.default_policy == "all_collaterals_vs_all_borrowables"
    # Exclusion list is a tuple of pairs (possibly empty).
    for pair in universe.pairs.exclusions:
        assert isinstance(pair, tuple) and len(pair) == 2


def test_ondo_config(universe: AssetUniverse) -> None:
    ondo = universe.ondo
    assert isinstance(ondo, OndoConfig)
    assert ondo.limits_api.startswith("http")
    assert ondo.status_page.startswith("http")
    assert ondo.api_key_env == "ONDO_API_KEY"


def test_admissible_pairs_matches_default_policy(universe: AssetUniverse) -> None:
    pairs = universe.admissible_pairs()
    n_c, n_b = len(universe.collaterals), len(universe.borrowables)
    expected = n_c * n_b + n_b * (n_b - 1) - len(universe.pairs.exclusions)
    assert len(pairs) == expected
    # No self-pairs and no duplicates.
    assert all(a != b for a, b in pairs)
    assert len(set(pairs)) == len(pairs)


def test_dataclasses_are_frozen(universe: AssetUniverse) -> None:
    """Immutability guards against accidental mutation in the pipeline."""
    with pytest.raises(Exception):
        universe.meta.base_currency = "EUR"  # type: ignore[misc]
