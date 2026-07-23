"""Per-pair tail-loss charge ``C_T``.

Implements the ``C_T`` formula from
``docs/nysa-market-risk-framework.md`` §3.1–3.2:

.. math::

    C_T = k_{ES} \\cdot \\sigma_\\text{stress}^{daily} \\cdot \\sqrt{t_\\text{liq}}

where

* ``k_ES = calibration.es_factor`` — the fat-tailed ES multiplier at 99 %
  (§3.2, Student-t ≈ 4 dof).
* ``sigma_stress_daily`` — the daily-scale stress-vol produced by
  :mod:`nysa_risk.volatility` (already the per-observation quantile of
  the EWMA series times :math:`\\sqrt{\\text{OBS\\_PER\\_DAY}}`).
* ``t_liq_days = calibration.t_liq_days`` — the pessimistic liquidation
  window in days (§4.1).

All three inputs come from :mod:`nysa_risk.config`; nothing is
hard-coded here.
"""

from __future__ import annotations

import math

from ..config import Calibration


def compute_ct(
    sigma_stress_daily: float,
    es_factor: float,
    t_liq_days: float,
) -> float:
    """Return ``C_T`` for a single pair (fractional loss over the liquidation horizon)."""
    if sigma_stress_daily < 0:
        raise ValueError(f"sigma_stress_daily must be ≥ 0; got {sigma_stress_daily}")
    if es_factor <= 0:
        raise ValueError(f"es_factor must be > 0; got {es_factor}")
    if t_liq_days <= 0:
        raise ValueError(f"t_liq_days must be > 0; got {t_liq_days}")
    return es_factor * sigma_stress_daily * math.sqrt(t_liq_days)


def ct_from_calibration(sigma_stress_daily: float, calibration: Calibration) -> float:
    """Convenience wrapper: pull ``es_factor`` and ``t_liq_days`` from the loaded config."""
    return compute_ct(
        sigma_stress_daily=sigma_stress_daily,
        es_factor=calibration.es_factor,
        t_liq_days=calibration.t_liq_days,
    )
