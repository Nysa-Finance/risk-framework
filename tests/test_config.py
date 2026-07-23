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
    PairExclusion,
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
        # LT = 1 − C_T − S needs a well-defined S ∈ [0, 1).
        assert 0.0 <= c.execution_cost < 1.0


def test_borrowables_shape(universe: AssetUniverse) -> None:
    assert len(universe.borrowables) > 0
    seen = set()
    for b in universe.borrowables:
        assert isinstance(b, Borrowable)
        assert b.symbol and b.symbol not in seen, f"duplicate borrowable: {b.symbol}"
        seen.add(b.symbol)
        assert b.price_source, f"{b.symbol} missing price_source"
        assert b.type in {"rwa", "crypto"}
        # Borrowables are strictly lend/borrow-only — they never act as collateral.
        assert b.use == "lending_and_borrowing"
        assert b.volatility_class in {"volatile", "stable"}


def test_calibration_ranges(universe: AssetUniverse) -> None:
    cal = universe.calibration
    assert isinstance(cal, Calibration)
    # §3.1 EWMA lambda in the RiskMetrics range and stress quantile a percentile.
    assert 0.0 < cal.ewma_lambda < 1.0
    assert 0.5 < cal.stress_quantile < 1.0
    # §4.1 gap-regime quantile: a percentile, milder than (or equal to) the
    # stress quantile by design — G is not a crash-conditioned buffer.
    assert 0.5 < cal.gap_sigma_quantile < 1.0
    assert cal.gap_sigma_quantile <= cal.stress_quantile
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
    # Threshold for the LT summary CLI: strictly positive fraction of collateral value.
    assert 0.0 < cal.emode_min_advantage < 1.0
    # backtest_curves --calibrate: ordered fraction bands, positive LTV ceiling gap.
    for band in (cal.target_liq30_emode, cal.target_liq30_std):
        assert isinstance(band, tuple) and len(band) == 2
        lo, hi = band
        assert 0.0 < lo <= hi < 1.0
    assert 0.0 < cal.minimum_gap < 1.0
    # calibrate.py (unified engine): fractions and a positive year floor.
    assert 0.0 < cal.max_uncond_bad_debt < 1.0
    assert cal.min_calibration_years > 0.0
    assert 0.0 < cal.severity_review_threshold < 1.0
    # LT severity pass: the declared per-position loss cap, above the flag threshold.
    assert 0.0 < cal.max_loss_given_bad_debt < 1.0
    assert cal.max_loss_given_bad_debt >= cal.severity_review_threshold


def test_pairs_policy(universe: AssetUniverse) -> None:
    assert isinstance(universe.pairs, PairsPolicy)
    assert universe.pairs.default_policy == "all_collaterals_vs_all_borrowables"
    # Exclusions are PairExclusion matchers (possibly wildcarded on one side).
    for ex in universe.pairs.exclusions:
        assert isinstance(ex, PairExclusion)
        assert ex.collateral is not None or ex.borrowable is not None


def test_ondo_config(universe: AssetUniverse) -> None:
    ondo = universe.ondo
    assert isinstance(ondo, OndoConfig)
    assert ondo.limits_api.startswith("http")
    assert ondo.status_page.startswith("http")
    assert ondo.api_key_env == "ONDO_API_KEY"


def test_admissible_pairs_matches_default_policy(universe: AssetUniverse) -> None:
    """Enumerated pairs = (eligible collaterals × borrowables) minus everything
    matched by an exclusion (class-level or specific)."""
    pairs = set(universe.admissible_pairs())
    borrowable_symbols = {b.symbol for b in universe.borrowables}
    collateral_only_symbols = {c.symbol for c in universe.collaterals if c.use == "collateral_only"}

    expected = {
        (c, b)
        for c in collateral_only_symbols
        for b in borrowable_symbols
        if not any(ex.matches(c, b) for ex in universe.pairs.exclusions)
    }
    assert pairs == expected

    # No self-pairs, no duplicates, every left leg is collateral_only, every
    # right leg is a borrowable.
    assert all(a != b for a, b in pairs)
    for c_sym, b_sym in pairs:
        assert c_sym in collateral_only_symbols
        assert b_sym in borrowable_symbols


def test_default_config_admits_every_borrowable_for_rwa_collaterals(
    universe: AssetUniverse,
) -> None:
    """With the reverted config, RWA collaterals are admissible against every borrowable."""
    pairs = universe.admissible_pairs()
    borrowables_seen = {b for _, b in pairs}
    assert borrowables_seen == {b.symbol for b in universe.borrowables}


def test_emode_categories_loaded_from_config(universe: AssetUniverse) -> None:
    cats = {c.name: c for c in universe.emode_categories}
    assert "stable" in cats
    assert cats["stable"].borrowables == ("USDC", "USDT")
    # Every referenced borrowable exists in the universe.
    borrowable_symbols = {b.symbol for b in universe.borrowables}
    for cat in universe.emode_categories:
        for b_sym in cat.borrowables:
            assert b_sym in borrowable_symbols


def test_dataclasses_are_frozen(universe: AssetUniverse) -> None:
    """Immutability guards against accidental mutation in the pipeline."""
    with pytest.raises(Exception):
        universe.meta.base_currency = "EUR"  # type: ignore[misc]


def test_pair_exclusion_matcher_semantics() -> None:
    # Class-level: any collateral against BNB is excluded.
    class_bnb = PairExclusion(borrowable="BNB")
    assert class_bnb.matches("AAPLon", "BNB")
    assert class_bnb.matches("SLVon", "BNB")
    assert not class_bnb.matches("AAPLon", "USDC")

    # Class-level on the collateral side.
    class_slv = PairExclusion(collateral="SLVon")
    assert class_slv.matches("SLVon", "USDC")
    assert class_slv.matches("SLVon", "WBTC")
    assert not class_slv.matches("AAPLon", "USDC")

    # Specific pair — both fields set.
    specific = PairExclusion(collateral="SLVon", borrowable="BNB")
    assert specific.matches("SLVon", "BNB")
    assert not specific.matches("SLVon", "USDC")
    assert not specific.matches("AAPLon", "BNB")


def _minimal_yaml(**overrides) -> str:
    parts = {
        "meta": (
            "meta:\n"
            "  version: '0.1'\n  base_currency: USD\n"
            "  price_history_years: 1\n  include_overnight_gaps: true\n"
        ),
        "collaterals": "collaterals: []\n",
        "borrowables": (
            "borrowables:\n"
            "  - {symbol: USDC, type: crypto, category: stable, price_source: USDC-USD,"
            " use: lending_and_borrowing, volatility_class: stable}\n"
        ),
        "pairs": "pairs:\n  default_policy: all_collaterals_vs_all_borrowables\n  exclusions: []\n",
        "ondo": "ondo:\n  limits_api: https://x\n  status_page: https://y\n  api_key_env: ONDO_API_KEY\n",
        "calibration": (
            "calibration:\n"
            "  ewma_lambda: 0.94\n  stress_quantile: 0.95\n  gap_sigma_quantile: 0.90\n"
            "  es_factor: 3.5\n"
            "  t_liq_days: 0.33\n  t_user_days: 3.0\n  k_user: 1.53\n"
            "  stressed_liquidatable_share: 0.25\n  rf_theta: 0.01\n  rf_horizon_years: 0.83\n"
            "  emode_min_advantage: 0.05\n"
            "  target_liq30_emode: [0.10, 0.15]\n  target_liq30_std: [0.10, 0.15]\n"
            "  minimum_gap: 0.02\n"
            "  max_uncond_bad_debt: 0.004\n  min_calibration_years: 3.0\n"
            "  severity_review_threshold: 0.05\n  max_loss_given_bad_debt: 0.065\n"
        ),
        "emode": "",
    }
    parts.update(overrides)
    return "".join(parts.values())


def test_loader_rejects_emode_category_referencing_unknown_borrowable(tmp_path: Path) -> None:
    p = tmp_path / "assets.yaml"
    p.write_text(
        _minimal_yaml(emode="emode_categories:\n  bogus:\n    borrowables: [DOES_NOT_EXIST]\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown borrowables"):
        load_universe(p)


def test_loader_rejects_emode_category_with_empty_borrowables(tmp_path: Path) -> None:
    p = tmp_path / "assets.yaml"
    p.write_text(
        _minimal_yaml(emode="emode_categories:\n  empty_cat:\n    borrowables: []\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one borrowable"):
        load_universe(p)


def test_loader_rejects_empty_exclusion(tmp_path: Path) -> None:
    """An exclusion with neither side set would match every pair — reject at load time."""
    p = tmp_path / "assets.yaml"
    p.write_text(
        "meta:\n"
        "  version: '0.1'\n  base_currency: USD\n  price_history_years: 1\n  include_overnight_gaps: true\n"
        "collaterals: []\n"
        "borrowables: []\n"
        "pairs:\n  default_policy: all_collaterals_vs_all_borrowables\n"
        "  exclusions:\n    - {}\n"
        "ondo:\n  limits_api: https://x\n  status_page: https://y\n  api_key_env: ONDO_API_KEY\n"
        "calibration:\n"
        "  ewma_lambda: 0.94\n  stress_quantile: 0.95\n  gap_sigma_quantile: 0.90\n"
        "  es_factor: 3.5\n"
        "  t_liq_days: 0.33\n  t_user_days: 3.0\n  k_user: 1.53\n"
        "  stressed_liquidatable_share: 0.25\n  rf_theta: 0.01\n  rf_horizon_years: 0.83\n"
        "  emode_min_advantage: 0.05\n"
        "  target_liq30_emode: [0.10, 0.15]\n  target_liq30_std: [0.10, 0.15]\n"
        "  minimum_gap: 0.02\n"
        "  max_uncond_bad_debt: 0.004\n  min_calibration_years: 3.0\n"
        "  severity_review_threshold: 0.05\n  max_loss_given_bad_debt: 0.065\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one"):
        load_universe(p)
