"""Per-collateral loan-to-value ``LTV = LT − G``.

Implements ``docs/nysa-market-risk-framework.md`` §4:

.. math::

    G = k_\\text{user} \\cdot \\sigma_\\text{gap}^{daily}
        \\cdot \\sqrt{t_\\text{user}}

    LTV(i) = LT(i) - G(i)

``G`` is the borrower-reaction buffer between where a position is
originated (LTV) and where it gets liquidated (LT). Over the reaction
horizon ``t_user_days`` the position may drift by up to
``k_user × σ_gap × √t_user`` in adverse tail-move terms (``k_user`` is
the fat-tailed 90 % quantile multiplier from §4.1).

sigma_gap vs sigma_stress — why two vol regimes
-----------------------------------------------
``G`` uses ``sigma_gap`` — the ``gap_sigma_quantile`` (90th) percentile
of the binding pair's EWMA volatility series — a **separate quantity**
from ``sigma_stress``, which stays at ``stress_quantile`` (95th) and is
used only for C_T/LT. The design rationale: the gap constraint is a
90 % confidence bound (``k_user``) evaluated in a 90th-percentile
volatility regime (``gap_sigma_quantile``) — homogeneous by design. The
liquidation buffer ``C_T`` deliberately uses the harsher
95th-percentile regime because liquidations occur in crashes by
construction; the origination gap does not carry that conditioning, so
pairing the 90 % bound with a 95th-percentile regime would silently
double-stress it.

Which pair's sigma?
-------------------
The framework is **per parameter set**: standard LT uses the standard
binding pair (arg max_j C_T over ALL admissible borrowables), so the
standard LTV uses that same pair's ``sigma_gap_daily`` in ``G``. The
E-Mode LT is bound by the worst pair *within the category*, so the
E-Mode LTV uses that pair's sigma. Bookkeeping is carried by
:attr:`nysa_risk.parameters.lt.CollateralLT.binding_sigma_gap_daily`.

All calibration constants (``k_user``, ``t_user_days``,
``gap_sigma_quantile``, plus the downstream ``emode_min_advantage`` used
by the CLI's dashing rule) come from :mod:`nysa_risk.config`; nothing is
hard-coded here.

CLI
---

``python -m nysa_risk.parameters.ltv`` prints one row per collateral:

    collateral | LT standard (%) | LTV standard (%) | LT e-mode (%) | LTV e-mode (%)

The E-Mode columns follow the same ``> emode_min_advantage`` rule as the
LT summary (based on **LT** advantage, not LTV): if the best E-Mode
category doesn't beat the base LT by more than the configured threshold,
both E-Mode columns are printed as ``—`` — enabling E-Mode isn't worth
it for that collateral.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import AssetUniverse, Calibration, load_universe
from ..volatility import DEFAULT_DATA_DIR
from .lt import BASE_PARAM_SET, EMODE_DASH, CollateralLT, compute_all_lt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollateralLTV:
    collateral: str
    param_set: str
    binding_borrowable: str
    binding_sigma_stress_daily: float  # stress-regime sigma (C_T bookkeeping only)
    binding_sigma_gap_daily: float     # gap-regime sigma — the one G is computed from
    lt: float
    g: float                     # k_user · σ_gap · √t_user
    ltv: float                   # lt − g


# ---------------------------------------------------------------------------
# G — user-reaction buffer
# ---------------------------------------------------------------------------


def compute_g(
    sigma_gap_daily: float,
    k_user: float,
    t_user_days: float,
) -> float:
    """Return ``G = k_user · σ_gap_daily · √t_user`` (§4.1).

    ``sigma_gap_daily`` is the gap-regime sigma (``gap_sigma_quantile``
    of the binding pair's EWMA series) — never ``sigma_stress``.
    """
    if sigma_gap_daily < 0:
        raise ValueError(f"sigma_gap_daily must be ≥ 0; got {sigma_gap_daily}")
    if k_user <= 0:
        raise ValueError(f"k_user must be > 0; got {k_user}")
    if t_user_days <= 0:
        raise ValueError(f"t_user_days must be > 0; got {t_user_days}")
    return k_user * sigma_gap_daily * math.sqrt(t_user_days)


def g_from_calibration(sigma_gap_daily: float, calibration: Calibration) -> float:
    """Convenience wrapper: pull ``k_user`` and ``t_user_days`` from the loaded config."""
    return compute_g(
        sigma_gap_daily=sigma_gap_daily,
        k_user=calibration.k_user,
        t_user_days=calibration.t_user_days,
    )


# ---------------------------------------------------------------------------
# per-collateral LTV
# ---------------------------------------------------------------------------


def compute_ltv_from_lt(
    collateral_lt: CollateralLT,
    calibration: Calibration,
) -> CollateralLTV:
    """Lift a ``CollateralLT`` into ``CollateralLTV`` by subtracting ``G``.

    ``G`` uses the binding pair's **gap-regime** sigma from the underlying
    ``CollateralLT`` (``binding_sigma_gap_daily``, not the stress sigma),
    so standard and E-Mode rows each pick up their own ``σ_gap_binding``.
    """
    g = g_from_calibration(collateral_lt.binding_sigma_gap_daily, calibration)
    return CollateralLTV(
        collateral=collateral_lt.collateral,
        param_set=collateral_lt.param_set,
        binding_borrowable=collateral_lt.binding_borrowable,
        binding_sigma_stress_daily=collateral_lt.binding_sigma_stress_daily,
        binding_sigma_gap_daily=collateral_lt.binding_sigma_gap_daily,
        lt=collateral_lt.lt,
        g=g,
        ltv=collateral_lt.lt - g,
    )


def compute_all_ltv(
    collateral_lts: Iterable[CollateralLT],
    calibration: Calibration,
) -> list[CollateralLTV]:
    """Map ``compute_ltv_from_lt`` over every incoming ``CollateralLT``."""
    return [compute_ltv_from_lt(lt_row, calibration) for lt_row in collateral_lts]


def compute_all_ltv_from_universe(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
) -> list[CollateralLTV]:
    """End-to-end: load prices → volatility → LT → LTV."""
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    lt_rows = compute_all_lt(universe=universe, data_dir=data_dir)
    return compute_all_ltv(lt_rows, universe.calibration)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_table(
    ltv_rows: Iterable[CollateralLTV],
    emode_min_advantage: float,
) -> str:
    """Five-column summary: ``collateral | LT std | LTV std | LT emode | LTV emode``.

    The E-Mode columns are shown only when the best E-Mode category's
    LT exceeds the base LT by more than ``emode_min_advantage``. When
    the gate closes, **both** E-Mode columns are dashed together —
    dashing LT while showing LTV (or vice versa) would give a misleading
    picture of a category we've decided isn't worth enabling.
    """
    by_c: dict[str, dict[str, CollateralLTV]] = {}
    for r in ltv_rows:
        by_c.setdefault(r.collateral, {})[r.param_set] = r

    collaterals = sorted(by_c)
    if not collaterals:
        return "(no collaterals)"

    col_w = max(len("collateral"), max(len(c) for c in collaterals))
    lt_std_h = "LT standard (%)"
    ltv_std_h = "LTV standard (%)"
    lt_em_h = "LT e-mode (%)"
    ltv_em_h = "LTV e-mode (%)"
    ws = [max(len(h), 10) for h in (lt_std_h, ltv_std_h, lt_em_h, ltv_em_h)]

    header = (
        f"{'collateral'.ljust(col_w)}  "
        f"{lt_std_h:>{ws[0]}}  {ltv_std_h:>{ws[1]}}  "
        f"{lt_em_h:>{ws[2]}}  {ltv_em_h:>{ws[3]}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for c_sym in collaterals:
        rows = by_c[c_sym]
        base = rows.get(BASE_PARAM_SET)
        if base is None:
            LOGGER.warning("%s: no base LTV row — skipping from summary table", c_sym)
            continue
        emode_rows = [r for k, r in rows.items() if k != BASE_PARAM_SET]
        best_emode = max(emode_rows, key=lambda r: r.lt) if emode_rows else None

        lt_std_cell = f"{base.lt * 100:.4f}"
        ltv_std_cell = f"{base.ltv * 100:.4f}"
        if best_emode is not None and (best_emode.lt - base.lt) > emode_min_advantage:
            lt_em_cell = f"{best_emode.lt * 100:.4f}"
            ltv_em_cell = f"{best_emode.ltv * 100:.4f}"
        else:
            lt_em_cell = EMODE_DASH
            ltv_em_cell = EMODE_DASH

        lines.append(
            f"{c_sym.ljust(col_w)}  "
            f"{lt_std_cell:>{ws[0]}}  {ltv_std_cell:>{ws[1]}}  "
            f"{lt_em_cell:>{ws[2]}}  {ltv_em_cell:>{ws[3]}}"
        )

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.parameters.ltv",
        description="Compute per-collateral LT and LTV (standard + best E-Mode) and print a summary table.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()
    lt_rows = compute_all_lt(universe=universe, data_dir=args.data_dir)
    ltv_rows = compute_all_ltv(lt_rows, universe.calibration)
    print(format_table(ltv_rows, universe.calibration.emode_min_advantage))
    return 0 if ltv_rows else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
