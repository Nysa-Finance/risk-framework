"""Tests for ``nysa_risk.parameters.lt``.

Sign chain pinned explicitly: the binding pair for a collateral is the
one with the **max** ``C_T``, giving the **min** ``LT`` (§3.3).

E-Mode: one row per ``(collateral, param_set)``; ``param_set="base"``
binds over all admissible borrowables, ``param_set="emode:<cat>"`` binds
over only the category's borrowables. Hand-verified numbers below cover
the case where the base binding is a volatile borrowable and the E-Mode
(stable) binding is a stablecoin — base LT is strictly lower.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path

import pytest

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
from nysa_risk.parameters import lt as lt_mod
from nysa_risk.parameters.lt import (
    BASE_PARAM_SET,
    EMODE_DASH,
    CollateralLT,
    PairLT,
    compute_all_lt_from_pairs,
    compute_collateral_lt,
    compute_pair_lt,
    format_table,
)
from nysa_risk.volatility import PairResult


def _calibration(**overrides) -> Calibration:
    base = dict(
        ewma_lambda=0.94,
        stress_quantile=0.95,
        gap_sigma_quantile=0.90,
        es_factor=3.5,
        t_liq_days=0.25,     # sqrt(0.25) = 0.5 → tidy hand-numbers
        t_user_days=3.0,
        k_user=1.53,
        stressed_liquidatable_share=0.25,
        rf_theta=0.01,
        rf_horizon_years=0.83,
        emode_min_advantage=0.05,
        target_liq30_emode=(0.10, 0.15),
        target_liq30_std=(0.10, 0.15),
        minimum_gap=0.02,
        max_uncond_bad_debt=0.004, min_calibration_years=3.0, severity_review_threshold=0.05, max_loss_given_bad_debt=0.065,    )
    base.update(overrides)
    return Calibration(**base)


# ---------------------------------------------------------------------------
# per-pair
# ---------------------------------------------------------------------------


def test_compute_pair_lt_round_numbers() -> None:
    cal = _calibration(es_factor=3.5, t_liq_days=0.25)
    # C_T = 3.5 · 0.02 · sqrt(0.25) = 0.035
    # LT  = 1 − 0.035 − 0.02       = 0.945
    p = compute_pair_lt(
        collateral="AAPLon", borrowable="USDC",
        sigma_stress_daily=0.02, sigma_gap_daily=0.015, execution_cost=0.02,
        calibration=cal,
    )
    assert p.ct == pytest.approx(0.035, abs=1e-15)
    assert p.lt == pytest.approx(0.945, abs=1e-15)
    assert p.execution_cost == 0.02
    assert p.sigma_stress_daily == 0.02
    # sigma_gap is carried through untouched — C_T must NOT use it.
    assert p.sigma_gap_daily == 0.015


def test_compute_pair_lt_rejects_bad_execution_cost() -> None:
    cal = _calibration()
    with pytest.raises(ValueError):
        compute_pair_lt(collateral="X", borrowable="Y",
                        sigma_stress_daily=0.02, sigma_gap_daily=0.02,
                        execution_cost=-0.01, calibration=cal)
    with pytest.raises(ValueError):
        compute_pair_lt(collateral="X", borrowable="Y",
                        sigma_stress_daily=0.02, sigma_gap_daily=0.02,
                        execution_cost=1.0, calibration=cal)


# ---------------------------------------------------------------------------
# per-collateral min rule — the sign chain
# ---------------------------------------------------------------------------


def _pair(c: str, b: str, ct: float, S: float = 0.02) -> PairLT:
    sigma = ct / (3.5 * math.sqrt(0.25))  # inverted for symmetry, unused
    return PairLT(
        collateral=c, borrowable=b,
        sigma_stress_daily=sigma,
        sigma_gap_daily=sigma,
        ct=ct, execution_cost=S,
        lt=1.0 - ct - S,
    )


def test_collateral_lt_binding_pair_maximises_ct() -> None:
    """Binding pair = arg max_j C_T(i,j). LT = 1 − max C_T − S."""
    S = 0.02
    pairs = [
        _pair("AAPLon", "USDC", ct=0.02, S=S),  # LT = 0.960
        _pair("AAPLon", "WBTC", ct=0.05, S=S),  # LT = 0.930  ← binding
        _pair("AAPLon", "WETH", ct=0.03, S=S),  # LT = 0.950
    ]
    result = compute_collateral_lt("AAPLon", param_set=BASE_PARAM_SET,
                                   execution_cost=S, pair_lts=pairs)
    assert isinstance(result, CollateralLT)
    assert result.param_set == BASE_PARAM_SET
    assert result.binding_borrowable == "WBTC"
    assert result.binding_ct == pytest.approx(0.05, abs=1e-15)
    # Published LT = min over j of LT(i,j) = 0.930.
    assert result.lt == pytest.approx(0.93, abs=1e-15)
    # And this equals 1 − max_j C_T − S — the algebraic identity from the docstring.
    assert result.lt == pytest.approx(1.0 - max(p.ct for p in pairs) - S, abs=1e-15)
    # Sanity: min of per-pair LTs matches too.
    assert result.lt == pytest.approx(min(p.lt for p in pairs), abs=1e-15)


def test_collateral_lt_ignores_pairs_for_other_collaterals() -> None:
    S = 0.02
    pairs = [
        _pair("AAPLon", "USDC", ct=0.05, S=S),
        _pair("TSLAon", "USDC", ct=0.99, S=S),   # noise: different collateral
    ]
    result = compute_collateral_lt("AAPLon", param_set=BASE_PARAM_SET,
                                   execution_cost=S, pair_lts=pairs)
    assert result.binding_borrowable == "USDC"
    assert result.binding_ct == pytest.approx(0.05, abs=1e-15)


def test_collateral_lt_raises_on_no_pairs() -> None:
    with pytest.raises(ValueError):
        compute_collateral_lt("LONELY", param_set=BASE_PARAM_SET,
                              execution_cost=0.02, pair_lts=[])


def test_collateral_lt_uses_supplied_execution_cost_not_pairs_field() -> None:
    """S(i) is a property of the collateral, applied uniformly across its pairs."""
    S_config = 0.03
    # Pair-level LT here is inconsistent with S_config (uses S=0.10); collateral LT should
    # still be computed using the supplied per-collateral execution_cost.
    pairs = [_pair("AAPLon", "USDC", ct=0.05, S=0.10)]
    result = compute_collateral_lt("AAPLon", param_set=BASE_PARAM_SET,
                                   execution_cost=S_config, pair_lts=pairs)
    assert result.execution_cost == S_config
    assert result.lt == pytest.approx(1.0 - 0.05 - S_config, abs=1e-15)


# ---------------------------------------------------------------------------
# universe-wide driver from PairResult objects (base + E-Mode)
# ---------------------------------------------------------------------------


def _universe_minimal(with_emode: bool = False) -> AssetUniverse:
    return AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD",
                  price_history_years=1, include_overnight_gaps=True),
        collaterals=(
            Collateral(symbol="AAPLon", type="rwa", category="equity",
                       underlying_ticker="AAPL", use="collateral_only",
                       execution_cost=0.02),
        ),
        borrowables=(
            Borrowable(symbol="USDC", type="crypto", category="stable",
                       price_source="USDC-USD", use="lending_and_borrowing",
                       volatility_class="stable"),
            Borrowable(symbol="USDT", type="crypto", category="stable",
                       price_source="USDT-USD", use="lending_and_borrowing",
                       volatility_class="stable"),
            Borrowable(symbol="WBTC", type="crypto", category="wrapped",
                       price_source="BTC-USD", use="lending_and_borrowing",
                       volatility_class="volatile"),
        ),
        pairs=PairsPolicy(default_policy="all_collaterals_vs_all_borrowables"),
        ondo=OndoConfig(limits_api="https://x", status_page="https://y", api_key_env="ONDO_API_KEY"),
        calibration=_calibration(),
        emode_categories=(
            EModeCategory(name="stable", borrowables=("USDC", "USDT")),
        ) if with_emode else (),
    )


def _pair_result(collateral: str, borrowable: str, sigma_daily: float,
                 sigma_gap_daily: float | None = None) -> PairResult:
    sigma_per_obs = sigma_daily / math.sqrt(2)
    gap_daily = sigma_daily if sigma_gap_daily is None else sigma_gap_daily
    return PairResult(
        collateral=collateral, borrowable=borrowable,
        collateral_ticker=collateral, borrowable_ticker=borrowable,
        n_observations=250,
        first_date=date(2024, 1, 1), last_date=date(2024, 12, 31),
        effective_years=1.0,
        sigma_stress=sigma_per_obs,
        sigma_stress_daily=sigma_daily,
        sigma_gap=gap_daily / math.sqrt(2),
        sigma_gap_daily=gap_daily,
    )


def test_compute_all_lt_from_pairs_base_only_when_no_emode() -> None:
    universe = _universe_minimal(with_emode=False)
    # AAPLon has three pairs; WBTC has the highest sigma so it binds base.
    # C_T for WBTC = 3.5·0.04·√0.25 = 3.5·0.04·0.5 = 0.07; LT = 1 − 0.07 − 0.02 = 0.91.
    prs = [
        _pair_result("AAPLon", "USDC", sigma_daily=0.01),
        _pair_result("AAPLon", "USDT", sigma_daily=0.015),
        _pair_result("AAPLon", "WBTC", sigma_daily=0.04),
    ]
    results = compute_all_lt_from_pairs(prs, universe)
    assert [(r.collateral, r.param_set) for r in results] == [("AAPLon", "base")]
    base = results[0]
    assert base.binding_borrowable == "WBTC"
    assert base.binding_ct == pytest.approx(0.07, abs=1e-15)
    assert base.lt == pytest.approx(0.91, abs=1e-15)


def test_emode_binding_is_restricted_to_category_and_lt_is_higher_than_base() -> None:
    """Hand-verified: base binds on volatile (WBTC), emode:stable binds on the largest
    stable (USDT). Emode LT strictly exceeds base LT — the capital efficiency claim."""
    universe = _universe_minimal(with_emode=True)
    prs = [
        _pair_result("AAPLon", "USDC", sigma_daily=0.010),  # C_T = 3.5·0.010·0.5 = 0.0175
        _pair_result("AAPLon", "USDT", sigma_daily=0.015),  # C_T = 3.5·0.015·0.5 = 0.02625
        _pair_result("AAPLon", "WBTC", sigma_daily=0.040),  # C_T = 3.5·0.040·0.5 = 0.07
    ]
    results = compute_all_lt_from_pairs(prs, universe)
    by_ps = {r.param_set: r for r in results if r.collateral == "AAPLon"}
    assert set(by_ps) == {"base", "emode:stable"}

    # base — binding = WBTC (max C_T over ALL borrowables).
    base = by_ps["base"]
    assert base.binding_borrowable == "WBTC"
    assert base.binding_ct == pytest.approx(0.07, abs=1e-15)
    assert base.lt == pytest.approx(1.0 - 0.07 - 0.02, abs=1e-15)  # 0.91

    # emode:stable — binding = USDT (max C_T over {USDC, USDT} only).
    emode = by_ps["emode:stable"]
    assert emode.binding_borrowable == "USDT"
    assert emode.binding_ct == pytest.approx(0.02625, abs=1e-15)
    assert emode.lt == pytest.approx(1.0 - 0.02625 - 0.02, abs=1e-15)  # 0.95375

    # The capital-efficiency guarantee: restricting to a lower-vol category ⇒ higher LT.
    assert emode.lt > base.lt
    # And the E-Mode row does NOT include the out-of-category pair.
    emode_borrowables = {p.borrowable for p in emode.pairs}
    assert emode_borrowables == {"USDC", "USDT"}
    assert "WBTC" not in emode_borrowables


def test_emode_category_with_no_admissible_borrowables_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A collateral whose pairs miss every borrowable in a category → no row for that category."""
    universe = _universe_minimal(with_emode=True)
    # Only supply WBTC — no stable pairs → emode:stable has nothing to bind.
    prs = [_pair_result("AAPLon", "WBTC", sigma_daily=0.04)]
    caplog.set_level(logging.INFO, logger=lt_mod.LOGGER.name)
    results = compute_all_lt_from_pairs(prs, universe)
    assert {(r.collateral, r.param_set) for r in results} == {("AAPLon", "base")}
    assert any("no admissible pairs for emode category 'stable'" in rec.message
               for rec in caplog.records)


def test_compute_all_lt_skips_pair_results_with_non_collateral_left_leg(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Borrowables aren't collaterals — a stray (borrowable, borrowable) PairResult is skipped."""
    universe = _universe_minimal()
    prs = [
        _pair_result("AAPLon", "USDC", sigma_daily=0.01),
        _pair_result("USDC", "WBTC", sigma_daily=0.02),  # would-be crypto/crypto
    ]
    caplog.set_level(logging.WARNING, logger=lt_mod.LOGGER.name)
    results = compute_all_lt_from_pairs(prs, universe)
    assert {r.collateral for r in results} == {"AAPLon"}
    assert any("missing execution_cost" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# CLI table
# ---------------------------------------------------------------------------


def _row(collateral: str, param_set: str, binding: str, ct: float,
         S: float = 0.02, sigma: float = 0.0) -> CollateralLT:
    return CollateralLT(
        collateral=collateral, param_set=param_set,
        execution_cost=S, binding_borrowable=binding,
        binding_ct=ct, binding_sigma_stress_daily=sigma,
        binding_sigma_gap_daily=sigma,
        lt=1.0 - ct - S, pairs=(),
    )


def test_format_table_columns_and_layout() -> None:
    """Header is ``collateral | LT standard (%) | LT e-mode (%)`` — one row per collateral."""
    rows = [
        # AAPLon: base LT = 0.91, emode LT = 0.95375 → advantage 0.04375 (below 0.05 threshold → dash).
        _row("AAPLon", "base",         "WBTC", ct=0.07),
        _row("AAPLon", "emode:stable", "USDT", ct=0.02625),
        # TSLAon: base LT = 0.88, emode LT = 0.96  → advantage 0.08 (above threshold → shown).
        _row("TSLAon", "base",         "WBTC", ct=0.10),
        _row("TSLAon", "emode:stable", "USDC", ct=0.02),
    ]
    text = format_table(rows, emode_min_advantage=0.05)
    lines = text.splitlines()
    assert lines[0].startswith("collateral")
    assert "LT standard (%)" in lines[0]
    assert "LT e-mode (%)" in lines[0]
    body = lines[2:]
    assert len(body) == 2
    # Sorted alphabetically.
    assert body[0].startswith("AAPLon")
    assert body[1].startswith("TSLAon")


def test_format_table_above_threshold_populates_emode_column() -> None:
    """TSLAon: base LT = 0.88, emode LT = 0.96 → advantage 0.08 (> 0.05) → e-mode column shows the value."""
    rows = [
        _row("TSLAon", "base",         "WBTC", ct=0.10),  # LT = 0.88
        _row("TSLAon", "emode:stable", "USDC", ct=0.02),  # LT = 0.96
    ]
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("TSLAon")
    assert "88.0000" in line          # LT standard
    assert "96.0000" in line          # LT e-mode populated
    assert EMODE_DASH not in line


def test_format_table_below_threshold_shows_dash() -> None:
    """AAPLon: base LT = 0.91, emode LT = 0.95375 → advantage 0.04375 (< 0.05) → dash."""
    rows = [
        _row("AAPLon", "base",         "WBTC", ct=0.07),      # LT = 0.91
        _row("AAPLon", "emode:stable", "USDT", ct=0.02625),   # LT = 0.95375
    ]
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("AAPLon")
    assert "91.0000" in line          # LT standard
    assert EMODE_DASH in line
    assert "95.3750" not in line      # e-mode value must NOT be shown


def test_format_table_at_exact_threshold_shows_dash() -> None:
    """Strict inequality: advantage must EXCEED the threshold (not equal it)."""
    rows = [
        _row("EDGEon", "base",         "WBTC", ct=0.10),     # LT = 0.88
        _row("EDGEon", "emode:stable", "USDC", ct=0.05),     # LT = 0.93 → advantage 0.05 exactly
    ]
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert EMODE_DASH in line


def test_format_table_no_emode_row_shows_dash() -> None:
    rows = [_row("SPYon", "base", "WBTC", ct=0.05)]  # no emode row at all
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("SPYon")
    assert EMODE_DASH in line


def test_format_table_picks_best_lt_when_multiple_emode_categories() -> None:
    """If there are multiple E-Mode rows, the highest-LT one is used for the display."""
    rows = [
        _row("TSLAon", "base",         "WBTC", ct=0.10),   # LT = 0.88
        _row("TSLAon", "emode:mid",    "USDC", ct=0.04),   # LT = 0.94  (below threshold vs base? 0.06 > 0.05 → OK)
        _row("TSLAon", "emode:stable", "USDT", ct=0.02),   # LT = 0.96  ← best
    ]
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert "96.0000" in line     # best e-mode LT
    assert "94.0000" not in line


def test_main_cli_reads_threshold_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI must use ``universe.calibration.emode_min_advantage`` — not a hardcoded value.

    We construct one universe with a large threshold (0.10 → e-mode advantage of ~0.04 fails)
    and another with a small threshold (0.01 → same advantage passes). Same pair-results in
    both cases; only the config threshold differs.
    """
    prs = [
        _pair_result("AAPLon", "USDC", 0.010),   # C_T = 0.0175
        _pair_result("AAPLon", "USDT", 0.015),   # C_T = 0.02625
        _pair_result("AAPLon", "WBTC", 0.040),   # C_T = 0.07  → base binds here (LT = 0.91)
    ]
    monkeypatch.setattr(
        lt_mod, "compute_all_pairs",
        lambda universe=None, data_dir=None: prs,
    )

    # Strict threshold (0.10) — advantage of 0.04375 fails → dash.
    strict = _universe_minimal(with_emode=True)
    strict = _replace_calibration(strict, emode_min_advantage=0.10)
    monkeypatch.setattr(lt_mod, "load_universe", lambda *a, **k: strict)
    lt_mod.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    strict_out = capsys.readouterr().out
    assert EMODE_DASH in strict_out
    assert "95.3750" not in strict_out

    # Lenient threshold (0.01) — same advantage passes → value shown.
    lenient = _universe_minimal(with_emode=True)
    lenient = _replace_calibration(lenient, emode_min_advantage=0.01)
    monkeypatch.setattr(lt_mod, "load_universe", lambda *a, **k: lenient)
    lt_mod.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    lenient_out = capsys.readouterr().out
    assert "95.3750" in lenient_out
    # AAPLon row should NOT show a dash on the lenient run.
    aapl_line = next(l for l in lenient_out.splitlines() if l.startswith("AAPLon"))
    assert EMODE_DASH not in aapl_line


def _replace_calibration(universe: AssetUniverse, **overrides) -> AssetUniverse:
    """Return a new AssetUniverse with the calibration overridden."""
    from dataclasses import replace
    new_cal = replace(universe.calibration, **overrides)
    return replace(universe, calibration=new_cal)
