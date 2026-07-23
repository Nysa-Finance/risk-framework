"""Per-pair and per-collateral liquidation threshold ``LT``.

Implements ``docs/nysa-market-risk-framework.md`` §3.3:

* Per pair ``(i, j)``: ``LT(i, j) = 1 − C_T(i, j) − S(i)`` where
  ``S(i)`` is the collateral's execution cost from
  :mod:`nysa_risk.config`.
* Published per collateral: ``LT(i) = min_j LT(i, j)``. Because
  ``LT = 1 − C_T − S`` is monotonically **decreasing** in ``C_T``, the
  ``min`` over pairs is equivalent to ``1 − max_j C_T(i, j) − S(i)``.
  The **binding borrowable** for collateral ``i`` is therefore the one
  that **maximises** ``C_T(i, j)``.

The sign chain matters — a bug that took ``min_j C_T`` (or, equivalently,
``max_j LT``) would publish a too-permissive threshold and violate the
worst-case guarantee. The tests pin the correct sign explicitly.

Parameter sets (base vs E-Mode)
-------------------------------
Every collateral emits multiple parameter rows:

* ``param_set="base"`` — binding taken over **all** admissible
  borrowables (the universe's ``admissible_pairs``).
* ``param_set="emode:<category>"`` — binding taken over **only** the
  borrowables listed in that E-Mode category
  (:class:`nysa_risk.config.EModeCategory`).

Why the min rule is sound per category
--------------------------------------
E-Mode positions are contractually restricted: a borrower who opts into
category ``k`` can only borrow assets in ``k.borrowables``. Their
worst-case tail loss is therefore the ``max`` of ``C_T`` **restricted to
that set** — pairs against out-of-category borrowables are, by
construction, unreachable from an E-Mode position and cannot bind it.
The min rule applies independently within each category, and gives a
category-specific LT that is monotone in the vol of the *worst
admissible borrowable in that category*. Restricting to a low-vol
category (e.g. ``stable``) generally yields a **higher** LT than base —
that is the capital efficiency the E-Mode contract is paying for.

The `base` row is still meaningful: it is the guarantee for borrowers
who have not opted into any category.

CLI
---

``python -m nysa_risk.parameters.lt`` prints one row per collateral:

    collateral | LT standard (%) | LT e-mode (%)

* **LT standard** is the base LT — binding taken over *all* admissible
  borrowables for that collateral.
* **LT e-mode** is the LT of the best-LT E-Mode category (i.e. the safest
  category available for this collateral, giving the highest LT). It is
  shown **only if it exceeds LT standard by more than
  ``calibration.emode_min_advantage``**; otherwise a dash ``—`` is
  printed, meaning enabling E-Mode isn't worth the operational cost for
  that collateral. The threshold is a config constant, never hard-coded.

Every underlying calibration constant — ``es_factor``, ``t_liq_days``,
``S(i)``, ``emode_min_advantage``, and the E-Mode category memberships —
comes from ``config/assets.yaml`` via :mod:`nysa_risk.config`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..config import AssetUniverse, Calibration, EModeCategory, load_universe
from ..volatility import DEFAULT_DATA_DIR, PairResult, compute_all_pairs
from .ct import compute_ct

LOGGER = logging.getLogger(__name__)

BASE_PARAM_SET = "base"


@dataclass(frozen=True, slots=True)
class PairLT:
    collateral: str
    borrowable: str
    sigma_stress_daily: float
    sigma_gap_daily: float       # gap-regime sigma of the same pair (carried for LTV's G term, §4)
    ct: float
    execution_cost: float
    lt: float  # 1 − ct − execution_cost


@dataclass(frozen=True, slots=True)
class CollateralLT:
    collateral: str
    param_set: str               # "base" or "emode:<category>"
    execution_cost: float
    binding_borrowable: str      # arg max_j C_T(i, j) within the param-set's borrowable subset
    binding_ct: float            # max_j C_T over the subset
    binding_sigma_stress_daily: float  # stress-regime sigma of the binding pair (C_T bookkeeping)
    binding_sigma_gap_daily: float     # gap-regime sigma of the binding pair (feeds LTV's G term, §4)
    lt: float                    # 1 − binding_ct − execution_cost
    pairs: tuple[PairLT, ...]    # every pair considered in this row (unsorted)


# ---------------------------------------------------------------------------
# per-pair
# ---------------------------------------------------------------------------


def compute_pair_lt(
    *,
    collateral: str,
    borrowable: str,
    sigma_stress_daily: float,
    sigma_gap_daily: float,
    execution_cost: float,
    calibration: Calibration,
) -> PairLT:
    """Compute ``C_T`` and ``LT`` for one pair.

    ``C_T`` uses ``sigma_stress_daily`` only; ``sigma_gap_daily`` is
    carried through untouched for the downstream LTV ``G`` term.
    """
    if not 0.0 <= execution_cost < 1.0:
        raise ValueError(f"{collateral}: execution_cost must be in [0, 1); got {execution_cost}")
    ct = compute_ct(
        sigma_stress_daily=sigma_stress_daily,
        es_factor=calibration.es_factor,
        t_liq_days=calibration.t_liq_days,
    )
    return PairLT(
        collateral=collateral,
        borrowable=borrowable,
        sigma_stress_daily=sigma_stress_daily,
        sigma_gap_daily=sigma_gap_daily,
        ct=ct,
        execution_cost=execution_cost,
        lt=1.0 - ct - execution_cost,
    )


# ---------------------------------------------------------------------------
# per-collateral min rule (within a param set)
# ---------------------------------------------------------------------------


def compute_collateral_lt(
    collateral: str,
    param_set: str,
    execution_cost: float,
    pair_lts: Iterable[PairLT],
) -> CollateralLT:
    """Take the ``min`` over pairs → the pair with **max** ``C_T`` is binding.

    Raises ``ValueError`` if no pairs remain for this ``(collateral,
    param_set)`` — a category with no admissible borrowables for this
    collateral has no publishable LT.
    """
    pairs_t = tuple(p for p in pair_lts if p.collateral == collateral)
    if not pairs_t:
        raise ValueError(f"{collateral}/{param_set}: no pair LTs supplied")
    binding = max(pairs_t, key=lambda p: p.ct)  # sign chain: max C_T ↔ min LT
    return CollateralLT(
        collateral=collateral,
        param_set=param_set,
        execution_cost=execution_cost,
        binding_borrowable=binding.borrowable,
        binding_ct=binding.ct,
        binding_sigma_stress_daily=binding.sigma_stress_daily,
        binding_sigma_gap_daily=binding.sigma_gap_daily,
        lt=1.0 - binding.ct - execution_cost,
        pairs=pairs_t,
    )


# ---------------------------------------------------------------------------
# universe-wide driver
# ---------------------------------------------------------------------------


def _execution_costs(universe: AssetUniverse) -> Mapping[str, float]:
    """Symbol → S(i). Only assets with ``use: collateral_only`` have S — borrowables never act as collateral."""
    return {c.symbol: c.execution_cost for c in universe.collaterals}


def compute_all_lt_from_pairs(
    pair_results: Iterable[PairResult],
    universe: AssetUniverse,
) -> list[CollateralLT]:
    """Emit one ``CollateralLT`` per ``(collateral, param_set)``.

    ``param_set`` is ``"base"`` (the min over every admissible borrowable
    for that collateral) plus one ``"emode:<category>"`` row per
    ``EModeCategory`` (the min restricted to the category's borrowables).
    """
    exec_costs = _execution_costs(universe)
    calibration = universe.calibration
    emode_categories: tuple[EModeCategory, ...] = universe.emode_categories

    by_collateral: dict[str, list[PairLT]] = {}
    for pr in pair_results:
        if pr.collateral not in exec_costs:
            LOGGER.warning("%s: missing execution_cost in config — skipping", pr.collateral)
            continue
        pair_lt = compute_pair_lt(
            collateral=pr.collateral,
            borrowable=pr.borrowable,
            sigma_stress_daily=pr.sigma_stress_daily,
            sigma_gap_daily=pr.sigma_gap_daily,
            execution_cost=exec_costs[pr.collateral],
            calibration=calibration,
        )
        by_collateral.setdefault(pr.collateral, []).append(pair_lt)

    out: list[CollateralLT] = []
    for c_sym, pair_lts in by_collateral.items():
        S = exec_costs[c_sym]
        # `base` — binding over every admissible pair for this collateral.
        out.append(compute_collateral_lt(c_sym, BASE_PARAM_SET, S, pair_lts))
        # One row per E-Mode category — binding over the restricted borrowable set.
        for cat in emode_categories:
            allowed = set(cat.borrowables)
            in_cat = [p for p in pair_lts if p.borrowable in allowed]
            if not in_cat:
                LOGGER.info(
                    "%s: no admissible pairs for emode category '%s' — skipping",
                    c_sym, cat.name,
                )
                continue
            out.append(
                compute_collateral_lt(c_sym, f"emode:{cat.name}", S, in_cat)
            )
    return out


def compute_all_lt(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
) -> list[CollateralLT]:
    """Load prices, run the volatility pipeline, and derive per-(collateral, param_set) LT."""
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    pair_results = compute_all_pairs(universe=universe, data_dir=data_dir)
    return compute_all_lt_from_pairs(pair_results, universe)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


EMODE_DASH = "—"


def format_table(
    collateral_lts: Iterable[CollateralLT],
    emode_min_advantage: float,
) -> str:
    """One row per collateral: ``collateral | LT standard (%) | LT e-mode (%)``.

    ``LT standard`` is the base row. ``LT e-mode`` is the highest-LT
    E-Mode row (over all configured categories) — but shown only if it
    exceeds the base LT by more than ``emode_min_advantage`` (in
    absolute fraction-of-collateral terms). Below that bar we print
    :data:`EMODE_DASH`, signalling that enabling E-Mode isn't worth it
    for that collateral. The threshold is passed in from the caller
    (which reads it from :class:`nysa_risk.config.Calibration`); this
    function never hard-codes it.
    """
    # Group rows by collateral: {collateral -> {param_set -> CollateralLT}}
    by_c: dict[str, dict[str, CollateralLT]] = {}
    for r in collateral_lts:
        by_c.setdefault(r.collateral, {})[r.param_set] = r

    collaterals = sorted(by_c)
    if not collaterals:
        return "(no collaterals)"

    col_w = max(len("collateral"), max(len(c) for c in collaterals))
    std_h = "LT standard (%)"
    em_h = "LT e-mode (%)"
    std_w = max(len(std_h), 10)
    em_w = max(len(em_h), 10)

    header = f"{'collateral'.ljust(col_w)}  {std_h:>{std_w}}  {em_h:>{em_w}}"
    sep = "-" * len(header)
    lines = [header, sep]

    for c_sym in collaterals:
        rows = by_c[c_sym]
        base = rows.get(BASE_PARAM_SET)
        if base is None:
            LOGGER.warning("%s: no base LT row — skipping from summary table", c_sym)
            continue
        emode_rows = [r for k, r in rows.items() if k != BASE_PARAM_SET]
        best_emode = max(emode_rows, key=lambda r: r.lt) if emode_rows else None

        std_cell = f"{base.lt * 100:.4f}"
        if best_emode is not None and (best_emode.lt - base.lt) > emode_min_advantage:
            em_cell = f"{best_emode.lt * 100:.4f}"
        else:
            em_cell = EMODE_DASH

        lines.append(f"{c_sym.ljust(col_w)}  {std_cell:>{std_w}}  {em_cell:>{em_w}}")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.parameters.lt",
        description="Compute per-collateral LT (base and per-E-Mode-category) and print binding pairs.",
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
    results = compute_all_lt(universe=universe, data_dir=args.data_dir)
    print(format_table(results, universe.calibration.emode_min_advantage))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
