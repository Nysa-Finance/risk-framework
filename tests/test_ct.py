"""Tests for ``nysa_risk.parameters.ct`` — the C_T tail-loss charge.

Hand-verifiable numbers: ``C_T = k_ES · sigma_daily · sqrt(t_liq)``.
"""

from __future__ import annotations

import math

import pytest

from nysa_risk.config import Calibration
from nysa_risk.parameters.ct import compute_ct, ct_from_calibration


def _calibration(**overrides) -> Calibration:
    base = dict(
        ewma_lambda=0.94,
        stress_quantile=0.95,
        gap_sigma_quantile=0.90,
        es_factor=3.5,
        t_liq_days=0.3333,
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


def test_compute_ct_round_numbers() -> None:
    # 3.5 · 0.02 · sqrt(0.25) = 3.5 · 0.02 · 0.5 = 0.035
    assert compute_ct(sigma_stress_daily=0.02, es_factor=3.5, t_liq_days=0.25) == pytest.approx(0.035, abs=1e-15)
    # 2.5 · 0.04 · sqrt(4) = 2.5 · 0.04 · 2 = 0.2
    assert compute_ct(sigma_stress_daily=0.04, es_factor=2.5, t_liq_days=4.0) == pytest.approx(0.2, abs=1e-15)


def test_compute_ct_matches_formula() -> None:
    sigma, k, t = 0.031, 3.5, 0.3333
    assert compute_ct(sigma, k, t) == pytest.approx(k * sigma * math.sqrt(t), abs=1e-15)


def test_compute_ct_zero_sigma_gives_zero() -> None:
    assert compute_ct(sigma_stress_daily=0.0, es_factor=3.5, t_liq_days=0.5) == 0.0


def test_compute_ct_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        compute_ct(sigma_stress_daily=-0.01, es_factor=3.5, t_liq_days=1.0)
    with pytest.raises(ValueError):
        compute_ct(sigma_stress_daily=0.02, es_factor=0.0, t_liq_days=1.0)
    with pytest.raises(ValueError):
        compute_ct(sigma_stress_daily=0.02, es_factor=3.5, t_liq_days=0.0)
    with pytest.raises(ValueError):
        compute_ct(sigma_stress_daily=0.02, es_factor=3.5, t_liq_days=-1.0)


def test_ct_from_calibration_pulls_constants() -> None:
    cal = _calibration(es_factor=2.0, t_liq_days=1.0)
    # 2.0 · 0.03 · sqrt(1.0) = 0.06
    assert ct_from_calibration(sigma_stress_daily=0.03, calibration=cal) == pytest.approx(0.06, abs=1e-15)
    # Real defaults (es=3.5, t=1/3): 3.5 · 0.05 · sqrt(1/3)
    cal_default = _calibration(es_factor=3.5, t_liq_days=1.0 / 3.0)
    expected = 3.5 * 0.05 * math.sqrt(1.0 / 3.0)
    assert ct_from_calibration(sigma_stress_daily=0.05, calibration=cal_default) == pytest.approx(expected, abs=1e-15)


def test_compute_ct_is_linear_in_sigma() -> None:
    """C_T scales linearly with sigma → parity check for the risk-budget interpretation."""
    a = compute_ct(0.01, 3.5, 0.3333)
    b = compute_ct(0.05, 3.5, 0.3333)
    assert b == pytest.approx(5 * a, abs=1e-15)


def test_compute_ct_scales_as_sqrt_of_t_liq() -> None:
    """C_T at horizon 4T equals 2× C_T at horizon T."""
    a = compute_ct(0.02, 3.5, 0.25)
    b = compute_ct(0.02, 3.5, 1.0)
    assert b == pytest.approx(2 * a, abs=1e-15)
