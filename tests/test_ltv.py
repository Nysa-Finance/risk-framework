"""Tests for ``nysa_risk.parameters.ltv``.

Hand-verified numbers throughout. To keep the arithmetic exact we pick
calibration constants that yield tidy square roots:

* ``t_liq_days = 0.25`` ⇒ √t_liq = 0.5
* ``t_user_days = 4.0`` ⇒ √t_user = 2.0
* ``k_user = 1.5`` ⇒ G = 1.5·σ·2 = **3·σ**
* ``es_factor = 3.5`` ⇒ C_T = 3.5·σ·0.5 = **1.75·σ**
* ``S = 0.02``
"""

from __future__ import annotations

import math
from dataclasses import replace
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
from nysa_risk.parameters import ltv as ltv_mod
from nysa_risk.parameters.lt import (
    BASE_PARAM_SET,
    EMODE_DASH,
    CollateralLT,
)
from nysa_risk.parameters.ltv import (
    CollateralLTV,
    compute_all_ltv,
    compute_g,
    compute_ltv_from_lt,
    format_table,
    g_from_calibration,
)


def _calibration(**overrides) -> Calibration:
    base = dict(
        ewma_lambda=0.94,
        stress_quantile=0.95,
        gap_sigma_quantile=0.90,
        es_factor=3.5,
        t_liq_days=0.25,     # sqrt = 0.5
        t_user_days=4.0,     # sqrt = 2.0 → G = 3·σ (with k_user=1.5)
        k_user=1.5,
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


def _lt_row(collateral: str, param_set: str, binding: str, sigma: float,
            S: float = 0.02, sigma_gap: float | None = None) -> CollateralLT:
    """Build a CollateralLT with LT computed from sigma (using the same tidy calibration).

    ``sigma_gap`` defaults to ``sigma`` so the hand-verified numbers below
    stay valid; pass it explicitly to pin the sigma_gap/sigma_stress split.
    """
    ct = 1.75 * sigma        # = 3.5 · σ · √0.25
    return CollateralLT(
        collateral=collateral,
        param_set=param_set,
        execution_cost=S,
        binding_borrowable=binding,
        binding_ct=ct,
        binding_sigma_stress_daily=sigma,
        binding_sigma_gap_daily=sigma if sigma_gap is None else sigma_gap,
        lt=1.0 - ct - S,
        pairs=(),
    )


# ---------------------------------------------------------------------------
# compute_g — formula and validation
# ---------------------------------------------------------------------------


def test_compute_g_round_numbers() -> None:
    # 1.5 · 0.04 · sqrt(4) = 1.5 · 0.04 · 2 = 0.12
    assert compute_g(sigma_gap_daily=0.04, k_user=1.5, t_user_days=4.0) == pytest.approx(0.12, abs=1e-15)
    # 2.0 · 0.01 · sqrt(9) = 0.06
    assert compute_g(sigma_gap_daily=0.01, k_user=2.0, t_user_days=9.0) == pytest.approx(0.06, abs=1e-15)


def test_compute_g_matches_formula() -> None:
    sigma, k, t = 0.03, 1.53, 3.0
    assert compute_g(sigma, k, t) == pytest.approx(k * sigma * math.sqrt(t), abs=1e-15)


def test_compute_g_zero_sigma_gives_zero() -> None:
    assert compute_g(sigma_gap_daily=0.0, k_user=1.5, t_user_days=4.0) == 0.0


def test_compute_g_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        compute_g(sigma_gap_daily=-0.01, k_user=1.5, t_user_days=4.0)
    with pytest.raises(ValueError):
        compute_g(sigma_gap_daily=0.02, k_user=0.0, t_user_days=4.0)
    with pytest.raises(ValueError):
        compute_g(sigma_gap_daily=0.02, k_user=1.5, t_user_days=0.0)


def test_g_from_calibration_pulls_constants() -> None:
    cal = _calibration(k_user=1.5, t_user_days=4.0)
    assert g_from_calibration(0.04, cal) == pytest.approx(0.12, abs=1e-15)
    # Real defaults from the shipped config.
    cal_real = _calibration(k_user=1.53, t_user_days=3.0)
    assert g_from_calibration(0.02, cal_real) == pytest.approx(1.53 * 0.02 * math.sqrt(3.0), abs=1e-15)


# ---------------------------------------------------------------------------
# compute_ltv_from_lt — per-collateral LTV
# ---------------------------------------------------------------------------


def test_compute_ltv_from_lt_standard_pair_hand_verified() -> None:
    """Standard binding: σ=0.04 → LT=0.91 → G=0.12 → LTV=0.79."""
    cal = _calibration()
    lt_row = _lt_row("AAPLon", "base", "WBTC", sigma=0.04)
    assert lt_row.lt == pytest.approx(0.91, abs=1e-15)

    ltv_row = compute_ltv_from_lt(lt_row, cal)
    assert isinstance(ltv_row, CollateralLTV)
    assert ltv_row.collateral == "AAPLon"
    assert ltv_row.param_set == "base"
    assert ltv_row.binding_borrowable == "WBTC"
    assert ltv_row.binding_sigma_stress_daily == 0.04
    assert ltv_row.lt == pytest.approx(0.91, abs=1e-15)
    assert ltv_row.g == pytest.approx(0.12, abs=1e-15)
    assert ltv_row.ltv == pytest.approx(0.79, abs=1e-15)


def test_compute_ltv_from_lt_emode_pair_uses_that_pairs_sigma() -> None:
    """E-Mode binding uses the E-Mode pair's σ (stable USDT-like), not the base one."""
    cal = _calibration()
    lt_row = _lt_row("AAPLon", "emode:stable", "USDT", sigma=0.010)
    # LT = 1 - 1.75·0.010 - 0.02 = 1 - 0.0175 - 0.02 = 0.9625
    # G  = 3·0.010 = 0.03
    # LTV = 0.9325
    assert lt_row.lt == pytest.approx(0.9625, abs=1e-15)

    ltv_row = compute_ltv_from_lt(lt_row, cal)
    assert ltv_row.g == pytest.approx(0.03, abs=1e-15)
    assert ltv_row.ltv == pytest.approx(0.9325, abs=1e-15)


def test_compute_all_ltv_maps_over_rows() -> None:
    cal = _calibration()
    lt_rows = [
        _lt_row("AAPLon", "base",         "WBTC", sigma=0.04),
        _lt_row("AAPLon", "emode:stable", "USDT", sigma=0.010),
    ]
    ltv_rows = compute_all_ltv(lt_rows, cal)
    by_ps = {r.param_set: r for r in ltv_rows if r.collateral == "AAPLon"}
    assert by_ps["base"].ltv == pytest.approx(0.79, abs=1e-15)
    assert by_ps["emode:stable"].ltv == pytest.approx(0.9325, abs=1e-15)


# ---------------------------------------------------------------------------
# sigma_gap vs sigma_stress — regression: G must use the gap-regime sigma
# ---------------------------------------------------------------------------


def test_ltv_uses_sigma_gap_not_sigma_stress_end_to_end() -> None:
    """On a synthetic series where the two quantiles differ, sigma_gap ≠
    sigma_stress and the LTV gap buffer G uses sigma_gap (sigma_stress
    stays reserved for C_T/LT).

    Series (hand-computed, same as the volatility tests): returns
    [0.02, -0.03, 0.01] with λ=0.5 give the EWMA sigma series
    sqrt([4e-4, 6.5e-4, 3.75e-4]). quantile=1.0 → sqrt(6.5e-4);
    gap_quantile=0.5 → median = 0.02. The two quantiles differ.
    """
    import numpy as np
    import pandas as pd

    from nysa_risk import volatility as vol
    from nysa_risk.parameters.lt import compute_collateral_lt, compute_pair_lt

    r = [0.02, -0.03, 0.01]
    logp = np.cumsum([math.log(100.0)] + r)
    p = np.exp(logp)
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    c = pd.DataFrame({"open": [p[0], p[2]], "close": [p[1], p[3]]}, index=dates)
    b = pd.DataFrame({"open": 1.0, "close": 1.0}, index=dates)

    pr = vol.compute_pair_from_prices(
        c, b,
        collateral="AAPLon", borrowable="USDC",
        collateral_ticker="AAPL", borrowable_ticker="USDC-USD",
        lam=0.5, quantile=1.0, gap_quantile=0.5,
    )
    # The two regimes genuinely differ on this series.
    assert pr.sigma_gap != pr.sigma_stress
    assert pr.sigma_gap_daily == pytest.approx(0.02 * math.sqrt(2), abs=1e-15)
    assert pr.sigma_stress_daily == pytest.approx(math.sqrt(6.5e-4) * math.sqrt(2), abs=1e-15)

    # Thread through LT → LTV with the tidy calibration (G = 3·σ).
    cal = _calibration()
    pair_lt = compute_pair_lt(
        collateral=pr.collateral, borrowable=pr.borrowable,
        sigma_stress_daily=pr.sigma_stress_daily,
        sigma_gap_daily=pr.sigma_gap_daily,
        execution_cost=0.02, calibration=cal,
    )
    lt_row = compute_collateral_lt("AAPLon", BASE_PARAM_SET, 0.02, [pair_lt])
    ltv_row = compute_ltv_from_lt(lt_row, cal)

    # G comes from sigma_gap — and provably NOT from sigma_stress.
    assert ltv_row.binding_sigma_gap_daily == pr.sigma_gap_daily
    assert ltv_row.g == pytest.approx(3.0 * pr.sigma_gap_daily, abs=1e-15)
    assert ltv_row.g != pytest.approx(3.0 * pr.sigma_stress_daily, abs=1e-15)
    assert ltv_row.ltv == pytest.approx(ltv_row.lt - 3.0 * pr.sigma_gap_daily, abs=1e-15)
    # ...while C_T (inside LT) keeps using sigma_stress.
    assert lt_row.binding_ct == pytest.approx(1.75 * pr.sigma_stress_daily, abs=1e-15)


# ---------------------------------------------------------------------------
# CLI format_table — hand-verified rows, threshold behaviour
# ---------------------------------------------------------------------------


def _both_rows(collateral: str, base_sigma: float, emode_sigma: float,
               S: float = 0.02) -> list[CollateralLTV]:
    cal = _calibration()
    return compute_all_ltv(
        [
            _lt_row(collateral, "base",         "WBTC", sigma=base_sigma, S=S),
            _lt_row(collateral, "emode:stable", "USDT", sigma=emode_sigma, S=S),
        ],
        cal,
    )


def test_format_table_emode_populated_when_lt_advantage_exceeds_threshold() -> None:
    """Base σ=0.05 → LT=0.8925, LTV=0.7425.
    E-Mode σ=0.010 → LT=0.9625, LTV=0.9325.
    LT advantage = 0.070 > 0.05 → both E-Mode columns populated."""
    rows = _both_rows("BIGgap", base_sigma=0.05, emode_sigma=0.010)
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("BIGgap")
    assert "89.2500" in line   # LT standard  = 0.8925
    assert "74.2500" in line   # LTV standard = 0.7425
    assert "96.2500" in line   # LT e-mode    = 0.9625
    assert "93.2500" in line   # LTV e-mode   = 0.9325
    assert EMODE_DASH not in line


def test_format_table_emode_dashed_when_lt_advantage_below_threshold() -> None:
    """Base σ=0.04 → LT=0.91, LTV=0.79.
    E-Mode σ=0.015 → LT=0.95375, LTV=0.90875.
    LT advantage = 0.04375 < 0.05 → BOTH E-Mode columns dashed together."""
    rows = _both_rows("SMLgap", base_sigma=0.04, emode_sigma=0.015)
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("SMLgap")
    assert "91.0000" in line   # LT standard
    assert "79.0000" in line   # LTV standard
    # E-Mode values must NOT appear even though we computed them internally.
    assert "95.3750" not in line
    assert "90.8750" not in line
    # Both E-Mode columns dashed.
    assert line.count(EMODE_DASH) == 2


def test_format_table_threshold_gate_is_strict_inequality() -> None:
    """Advantage exactly equal to threshold → dash (must strictly *exceed*).

    Uses LT values that are exact in binary float (halves and eighths) so
    the advantage comes out exactly at the threshold without rounding.
    """
    # LT_std = 0.5, LT_em = 0.625 → advantage = 0.125 exactly (all powers of two).
    rows = [
        CollateralLTV(collateral="EDGE", param_set=BASE_PARAM_SET,
                      binding_borrowable="WBTC", binding_sigma_stress_daily=0.0,
                      binding_sigma_gap_daily=0.0,
                      lt=0.5, g=0.0, ltv=0.5),
        CollateralLTV(collateral="EDGE", param_set="emode:stable",
                      binding_borrowable="USDC", binding_sigma_stress_daily=0.0,
                      binding_sigma_gap_daily=0.0,
                      lt=0.625, g=0.0, ltv=0.625),
    ]
    assert (rows[1].lt - rows[0].lt) == 0.125  # exact in float
    text = format_table(rows, emode_min_advantage=0.125)
    line = text.splitlines()[2]
    assert line.count(EMODE_DASH) == 2


def test_format_table_no_emode_row_dashes_both_columns() -> None:
    cal = _calibration()
    rows = compute_all_ltv([_lt_row("SPYon", "base", "WBTC", sigma=0.04)], cal)
    text = format_table(rows, emode_min_advantage=0.05)
    line = text.splitlines()[2]
    assert line.startswith("SPYon")
    assert line.count(EMODE_DASH) == 2


def test_format_table_headers() -> None:
    rows = _both_rows("AAPLon", base_sigma=0.05, emode_sigma=0.010)
    text = format_table(rows, emode_min_advantage=0.05)
    header = text.splitlines()[0]
    assert "collateral" in header
    assert "LT standard (%)" in header
    assert "LTV standard (%)" in header
    assert "LT e-mode (%)" in header
    assert "LTV e-mode (%)" in header


# ---------------------------------------------------------------------------
# CLI main — threshold + config plumbing
# ---------------------------------------------------------------------------


def _universe(with_emode: bool = True, **cal_overrides) -> AssetUniverse:
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
        ondo=OndoConfig(limits_api="https://x", status_page="https://y",
                        api_key_env="ONDO_API_KEY"),
        calibration=_calibration(**cal_overrides),
        emode_categories=(EModeCategory(name="stable", borrowables=("USDC", "USDT")),) if with_emode else (),
    )


def test_main_cli_reads_threshold_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same LT rows, two universes — advantage 0.04375 dashes under 0.05 threshold, shows under 0.01."""
    lt_rows_factory = lambda: [
        _lt_row("SMLgap", "base",         "WBTC", sigma=0.04),
        _lt_row("SMLgap", "emode:stable", "USDT", sigma=0.015),
    ]

    monkeypatch.setattr(
        ltv_mod, "compute_all_lt",
        lambda universe=None, data_dir=None: lt_rows_factory(),
    )

    # Strict threshold (0.05) — advantage 0.04375 → dashed.
    strict = _universe(emode_min_advantage=0.05)
    monkeypatch.setattr(ltv_mod, "load_universe", lambda *a, **k: strict)
    ltv_mod.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    strict_out = capsys.readouterr().out
    strict_row = next(l for l in strict_out.splitlines() if l.startswith("SMLgap"))
    assert strict_row.count(EMODE_DASH) == 2
    assert "90.8750" not in strict_row   # LTV e-mode value hidden

    # Lenient threshold (0.01) — same advantage → shown.
    lenient = _universe(emode_min_advantage=0.01)
    monkeypatch.setattr(ltv_mod, "load_universe", lambda *a, **k: lenient)
    ltv_mod.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    lenient_out = capsys.readouterr().out
    lenient_row = next(l for l in lenient_out.splitlines() if l.startswith("SMLgap"))
    assert EMODE_DASH not in lenient_row
    assert "95.3750" in lenient_row      # LT e-mode  = 0.95375
    assert "90.8750" in lenient_row      # LTV e-mode = 0.90875


def test_main_cli_smoke_prints_all_five_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        ltv_mod, "compute_all_lt",
        lambda universe=None, data_dir=None: [
            _lt_row("BIGgap", "base",         "WBTC", sigma=0.05),
            _lt_row("BIGgap", "emode:stable", "USDT", sigma=0.010),
        ],
    )
    monkeypatch.setattr(ltv_mod, "load_universe", lambda *a, **k: _universe())
    rc = ltv_mod.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    # Header carries all five column names.
    header = out.splitlines()[0]
    for h in ("collateral", "LT standard", "LTV standard", "LT e-mode", "LTV e-mode"):
        assert h in header
    body = out.splitlines()[2]
    assert body.startswith("BIGgap")
    # Hand-verified values for the populated row.
    assert "89.2500" in body   # LT std
    assert "74.2500" in body   # LTV std
    assert "96.2500" in body   # LT em
    assert "93.2500" in body   # LTV em
