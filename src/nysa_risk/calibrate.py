"""Unified LTV calibration engine — final LTV per collateral × scenario.

Produces the deployable LTV for each collateral under two scenarios
(``e-mode`` — path vs USDT with the ``emode:stable`` LT; ``standard`` —
path vs BNB with the ``base`` LT), starting from the **formula LTV as
prior** (the pipeline's ``LT − G``, :mod:`nysa_risk.parameters.ltv`).

For each row the engine bisects on LTV — daily openings at max LTV on
the same aligned interleaved open/close stream as the pipeline
(:func:`nysa_risk.backtest.relative_price_stream`) — to find the
**maximum LTV satisfying ALL** of:

(1) share of openings liquidated within 30 days ≤ the scenario band's
    upper edge (``calibration.target_liq30_emode`` /
    ``target_liq30_std``). The prior acts asymmetrically: if the share
    at the formula LTV is **below the band's lower edge** AND the pair
    has at least ``calibration.min_calibration_years`` of effective
    history, the engine may raise LTV toward the band; otherwise the
    formula LTV is a hard cap (the short-history asymmetry guard —
    lowering is conservative on any sample, raising needs data).
(2) P(liq ≤ ``t_user_days``) ≤ :data:`nysa_risk.backtest.GAP_HIT_BOUND`
    (the declared reaction bound, censoring-aware denominator).
(3) unconditional bad-debt rate — bad-debt events over openings with a
    decidable horizon outcome, under the conservative **next-price
    realization rule** (the liquidator is charged the full
    overnight/weekend gap to the next print; see
    :mod:`nysa_risk.backtest`) — ≤ ``calibration.max_uncond_bad_debt``.
(4) LTV ≤ LT − ``calibration.minimum_gap``.

All thresholds come from ``config/assets.yaml``; nothing is hard-coded.

LT severity pass (runs before the LTV pass)
-------------------------------------------
``calibration.max_loss_given_bad_debt`` is a **declared severity
bound**: the framework accepts rare bad debt but bounds its *depth* per
position — frequency stays governed by the 1 % conditional and 0.4 %
unconditional constraints. For each row whose worst-case excess loss
beyond the ``1 − LT`` buffer (at the formula configuration, next-price
realization rule) exceeds the cap, LT is bisected downward — re-running
the liquidation + realization simulation at each step, ceiling =
formula LT, floor = formula LT − :data:`LT_FLOOR_DELTA` (0.10) — until
the worst-case excess fits under the cap. If even the floor cannot
satisfy it, the row keeps the floor LT and carries an ``LT floor``
flag. During the pass the trigger LTV is
``min(formula LTV, LT − minimum_gap)``. The LTV pass then re-runs for
affected rows at the cut LT with the prior shifted by the same amount
(``formula LTV = LT − G`` with ``G`` independent of LT, so a ΔLT cut
moves the prior by ΔLT).

Severity below the cap is still **not** an LTV lever: if, at the final LTV, any bad-debt
event's excess loss beyond the ``1 − LT`` buffer exceeds
``calibration.severity_review_threshold``, the row gets an ``LT REVIEW``
flag carrying the empirical tail (p95 and max of the excess
distribution) — the LT, not the LTV, is the right knob for tail
severity, so the engine does not push LTV further down for it.

The constraint metrics are step functions of LTV over one realized
history and are **not** globally monotone — the bad-debt metric in
particular depends on *where* in a crash the trigger lands, so
infeasible pockets can sit below feasible regions. A plain bisection
can fall through such a pocket and return an LTV far below a feasible
prior. The engine therefore searches with a **descending LTV grid**
(``grid_step``, default 0.25 pp) from the upper bound, takes the
highest feasible grid point, and sharpens its edge by bisection; in
raise mode the search is anchored at the (verified-feasible) formula
prior, so the result never falls below it. The returned LTV is always
verified feasible; maximality holds to grid resolution. All rates are
per-opening frequencies over the historical window, not independent
probabilities (openings are overlapping, correlated paths).

CLI
---

``python -m nysa_risk.calibrate`` prints one row per collateral ×
scenario:

    collateral | scenario | formula LTV | final LTV | binding constraint
    | %≤30d | uncond. BD % | flags
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import (
    DEFAULT_HORIZON_DAYS,
    GAP_HIT_BOUND,
    NA_DASH,
    _render_table,
    relative_price_stream,
    simulate_positions,
)
from .backtest_curves import EARLY_LIQ_WINDOW_DAYS, Scenario
from .config import AssetUniverse, load_universe
from .parameters.ltv import compute_all_ltv_from_universe
from .volatility import DEFAULT_DATA_DIR, _ticker_index, load_prices

LOGGER = logging.getLogger(__name__)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="e-mode", borrowable="USDT", param_set="emode:stable"),
    Scenario(name="standard", borrowable="BNB", param_set="base"),
)

# Binding-constraint labels.
BIND_LIQ30 = "liq30"
BIND_GAP = "gap"
BIND_BAD_DEBT = "bad-debt"
BIND_CEILING = "ceiling"
BIND_FORMULA = "formula"

FLAG_SHORT_HISTORY = "short-history"

# LT severity pass: how far below the formula LT the search may go.
LT_FLOOR_DELTA = 0.10


@dataclass(frozen=True, slots=True)
class RowMetrics:
    """Constraint metrics at one LTV, all per-opening frequencies."""
    share30: float | None          # liquidations ≤ 30d / all openings
    gap_rate: float | None         # gap hits / gap-checked openings (censoring-aware)
    uncond_bad_debt: float | None  # bad-debt events / decidable openings
    excess: tuple[float, ...]      # per bad-debt event: drop − (1 − LT)

    @property
    def max_excess(self) -> float:
        return max(self.excess) if self.excess else 0.0


@dataclass(frozen=True, slots=True)
class LTVDecision:
    final_ltv: float
    binding: str                   # which constraint stopped the search
    metrics: RowMetrics            # evaluated at final_ltv
    effective_years: float
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LTDecision:
    """Outcome of the LT severity pass for one row."""
    formula_lt: float
    final_lt: float
    max_excess_before: float       # worst excess beyond 1 − LT at the formula configuration
    max_excess_after: float        # ... at the final LT (same trigger-LTV policy)
    floor_capped: bool             # the floor could not satisfy the cap

    @property
    def delta(self) -> float:
        return self.final_lt - self.formula_lt


@dataclass(frozen=True, slots=True)
class CalibratedRow:
    collateral: str
    scenario: Scenario
    lt: float                      # formula LT (pipeline prior)
    formula_ltv: float
    decision: LTVDecision
    lt_decision: LTDecision | None = None   # severity pass outcome (None = pass not run)


# ---------------------------------------------------------------------------
# Core per-row solver
# ---------------------------------------------------------------------------


def _row_metrics(
    stream: pd.DataFrame,
    ltv: float,
    lt: float,
    t_user_days: float,
    horizon_days: float,
    liq_window_days: float,
) -> RowMetrics:
    s = simulate_positions(stream, ltv=ltv, lt=lt,
                           t_user_days=t_user_days, horizon_days=horizon_days)
    share30 = (
        sum(1 for d in s.days_to_liquidation if d <= liq_window_days) / s.n_positions
        if s.n_positions else None
    )
    gap_rate = s.gap_hits / s.gap_checked if s.gap_checked else None
    uncond = s.bad_debt_events / s.horizon_checked if s.horizon_checked else None
    return RowMetrics(share30=share30, gap_rate=gap_rate,
                      uncond_bad_debt=uncond, excess=s.bad_debt_excess)


def calibrate_row(
    stream: pd.DataFrame,
    *,
    lt: float,
    formula_ltv: float,
    band: tuple[float, float],
    t_user_days: float,
    minimum_gap: float,
    bd_bound: float,
    min_years: float,
    severity_threshold: float,
    gap_bound: float = GAP_HIT_BOUND,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    liq_window_days: float = EARLY_LIQ_WINDOW_DAYS,
    tol: float = 1e-4,
    grid_step: float = 0.0025,
) -> LTVDecision:
    """Maximum LTV satisfying every constraint, formula LTV as prior.

    See the module docstring for the constraint set, the raise rule and
    the severity policy. Missing metrics (empty denominators) count as
    vacuously satisfied.
    """
    band_lo, band_hi = band
    if not 0.0 < band_lo <= band_hi < 1.0:
        raise ValueError(f"band must satisfy 0 < lo ≤ hi < 1; got {band}")
    if not 0.0 < minimum_gap < lt:
        raise ValueError(f"minimum_gap must be in (0, lt={lt}); got {minimum_gap}")
    if not 0.0 < formula_ltv < lt:
        raise ValueError(f"formula_ltv must be in (0, lt={lt}); got {formula_ltv}")
    if not 0.0 < bd_bound < 1.0:
        raise ValueError(f"bd_bound must be in (0, 1); got {bd_bound}")
    if min_years < 0.0 or severity_threshold <= 0.0:
        raise ValueError("min_years must be ≥ 0 and severity_threshold > 0")

    ceiling = lt - minimum_gap
    if len(stream.index) >= 2:
        effective_years = (stream.index[-1] - stream.index[0]).days / 365.25
    else:
        effective_years = 0.0

    cache: dict[float, RowMetrics] = {}

    def metrics(x: float) -> RowMetrics:
        key = round(x, 9)
        if key not in cache:
            cache[key] = _row_metrics(stream, x, lt, t_user_days,
                                      horizon_days, liq_window_days)
        return cache[key]

    def violations(m: RowMetrics) -> list[str]:
        out = []
        if m.share30 is not None and m.share30 > band_hi:
            out.append(BIND_LIQ30)
        if m.gap_rate is not None and m.gap_rate > gap_bound:
            out.append(BIND_GAP)
        if m.uncond_bad_debt is not None and m.uncond_bad_debt > bd_bound:
            out.append(BIND_BAD_DEBT)
        return out

    def feasible(x: float) -> bool:
        return x <= ceiling and not violations(metrics(x))

    def largest_feasible(lo_anchor: float, hi_bound: float) -> float:
        """Highest feasible LTV in [lo_anchor, hi_bound], to grid/tol resolution.

        Descending grid scan (robust to non-monotone pockets), then a
        local bisection to sharpen the edge of the feasible region.
        ``lo_anchor`` is returned as the fallback floor.
        """
        if feasible(hi_bound):
            return hi_bound
        x = hi_bound - grid_step
        while x > lo_anchor:
            if feasible(x):
                lo, up = x, min(x + grid_step, hi_bound)
                while up - lo > tol:
                    mid = 0.5 * (lo + up)
                    if feasible(mid):
                        lo = mid
                    else:
                        up = mid
                return lo
            x -= grid_step
        return lo_anchor

    flags: list[str] = []
    formula_capped = min(formula_ltv, ceiling)
    m_formula = metrics(formula_capped)
    under_band = m_formula.share30 is not None and m_formula.share30 < band_lo
    raise_allowed = under_band and effective_years >= min_years
    if under_band and not raise_allowed:
        flags.append(FLAG_SHORT_HISTORY)  # a raise was warranted but blocked

    if raise_allowed and feasible(formula_capped):
        # Anchored raise: the prior is feasible, never go below it.
        hi_search = ceiling
        final = largest_feasible(formula_capped, ceiling)
    else:
        # Formula is a cap (no raise warranted/allowed, or the prior itself
        # violates a constraint and must be lowered).
        hi_search = formula_capped
        final = formula_capped if feasible(formula_capped) else largest_feasible(tol, formula_capped)

    # Binding constraint.
    if final >= hi_search - tol:
        if hi_search == ceiling and (raise_allowed or formula_ltv > ceiling):
            binding = BIND_CEILING
        else:
            binding = BIND_FORMULA
    else:
        failing: list[str] = []
        for probe in (min(final + 2.0 * tol, hi_search), min(final + grid_step, hi_search)):
            failing = violations(metrics(probe))
            if failing:
                break
        if failing:
            binding = "+".join(failing)
        else:
            binding = BIND_FORMULA if abs(final - formula_capped) <= tol else BIND_CEILING

    # Severity is not an LTV lever — flag for LT review instead.
    m_final = metrics(final)
    if m_final.max_excess > severity_threshold:
        exc = np.asarray(m_final.excess, dtype=float)
        flags.append(
            f"LT REVIEW (max excess {m_final.max_excess * 100:.1f}%, "
            f"p95 {float(np.percentile(exc, 95)) * 100:.1f}%)"
        )

    return LTVDecision(
        final_ltv=final,
        binding=binding,
        metrics=m_final,
        effective_years=effective_years,
        flags=tuple(flags),
    )


# ---------------------------------------------------------------------------
# LT severity pass
# ---------------------------------------------------------------------------


def calibrate_lt(
    stream: pd.DataFrame,
    *,
    formula_lt: float,
    formula_ltv: float,
    severity_cap: float,
    t_user_days: float,
    minimum_gap: float,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
    floor_delta: float = LT_FLOOR_DELTA,
    tol: float = 1e-4,
) -> LTDecision:
    """Lower LT until the worst-case excess beyond ``1 − LT`` ≤ ``severity_cap``.

    Evaluated at the formula configuration with trigger LTV
    ``min(formula_ltv, lt − minimum_gap)``; realization follows the
    conservative next-price rule. Search range ``[formula_lt −
    floor_delta, formula_lt]``; if even the floor exceeds the cap, the
    floor is returned with ``floor_capped=True``. Lowering LT widens the
    buffer, so the excess is (near-)monotone decreasing in the cut; the
    returned LT is verified against the cap unless floor-capped.
    """
    if not 0.0 < formula_lt < 1.0:
        raise ValueError(f"formula_lt must be in (0, 1); got {formula_lt}")
    if not 0.0 < formula_ltv < formula_lt:
        raise ValueError(f"formula_ltv must be in (0, formula_lt); got {formula_ltv}")
    if severity_cap <= 0.0:
        raise ValueError(f"severity_cap must be > 0; got {severity_cap}")

    def max_excess(lt_x: float) -> float:
        ltv_eff = min(formula_ltv, lt_x - minimum_gap)
        if ltv_eff <= 0.0:
            return 0.0
        s = simulate_positions(stream, ltv=ltv_eff, lt=lt_x,
                               t_user_days=t_user_days, horizon_days=horizon_days)
        return max(s.bad_debt_excess) if s.bad_debt_excess else 0.0

    before = max_excess(formula_lt)
    if before <= severity_cap:
        return LTDecision(formula_lt, formula_lt, before, before, False)

    floor = max(formula_lt - floor_delta, tol)
    after_floor = max_excess(floor)
    if after_floor > severity_cap:
        return LTDecision(formula_lt, floor, before, after_floor, True)

    lo, hi = floor, formula_lt   # lo satisfies the cap, hi does not
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if max_excess(mid) <= severity_cap:
            lo = mid
        else:
            hi = mid
    return LTDecision(formula_lt, lo, before, max_excess(lo), False)


def calibrate_row_with_lt(
    stream: pd.DataFrame,
    *,
    formula_lt: float,
    formula_ltv: float,
    band: tuple[float, float],
    t_user_days: float,
    minimum_gap: float,
    bd_bound: float,
    min_years: float,
    severity_threshold: float,
    severity_cap: float,
    **calibrate_row_kwargs,
) -> tuple[LTDecision, LTVDecision]:
    """LT severity pass, then the LTV pass at the (possibly cut) LT.

    The LTV prior moves with the cut: ``formula LTV = LT − G`` and ``G``
    does not depend on LT, so the adjusted prior is
    ``formula_ltv + ΔLT`` (ΔLT ≤ 0). A floor-capped LT pass adds an
    ``LT floor`` flag to the returned LTV decision.
    """
    lt_dec = calibrate_lt(
        stream, formula_lt=formula_lt, formula_ltv=formula_ltv,
        severity_cap=severity_cap, t_user_days=t_user_days,
        minimum_gap=minimum_gap,
    )
    adjusted_formula_ltv = formula_ltv + lt_dec.delta
    if adjusted_formula_ltv <= 0.0:
        raise ValueError(
            f"LT cut of {-lt_dec.delta * 100:.1f}pp leaves no viable LTV prior "
            f"(formula {formula_ltv * 100:.1f}%)"
        )
    decision = calibrate_row(
        stream, lt=lt_dec.final_lt, formula_ltv=adjusted_formula_ltv,
        band=band, t_user_days=t_user_days, minimum_gap=minimum_gap,
        bd_bound=bd_bound, min_years=min_years,
        severity_threshold=severity_threshold, **calibrate_row_kwargs,
    )
    if lt_dec.floor_capped:
        decision = replace(decision, flags=decision.flags + (
            f"LT floor (excess {lt_dec.max_excess_after * 100:.1f}% "
            f"> cap {severity_cap * 100:.1f}%)",
        ))
    return lt_dec, decision


# ---------------------------------------------------------------------------
# Universe-wide driver
# ---------------------------------------------------------------------------


def run_calibrate(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
    *,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> list[CalibratedRow]:
    """One :class:`CalibratedRow` per collateral × scenario."""
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    cal = universe.calibration
    ltv_rows = compute_all_ltv_from_universe(universe=universe, data_dir=data_dir)
    by_key = {(r.collateral, r.param_set): r for r in ltv_rows}
    tickers = _ticker_index(universe)

    cache: dict[str, pd.DataFrame] = {}

    def _prices(symbol: str) -> pd.DataFrame:
        ticker = tickers[symbol]
        if ticker not in cache:
            cache[ticker] = load_prices(ticker, data_dir)
        return cache[ticker]

    out: list[CalibratedRow] = []
    for collateral in sorted({r.collateral for r in ltv_rows}):
        for scen in scenarios:
            row = by_key.get((collateral, scen.param_set))
            if row is None or scen.borrowable not in tickers:
                LOGGER.warning("%s/%s: missing parameter row or borrowable — skipping",
                               collateral, scen.name)
                continue
            if not 0.0 < row.ltv < row.lt:
                LOGGER.warning("%s/%s: formula LTV %.4f outside (0, LT=%.4f) — skipping",
                               collateral, scen.name, row.ltv, row.lt)
                continue
            try:
                stream = relative_price_stream(_prices(collateral), _prices(scen.borrowable))
            except (FileNotFoundError, ValueError) as exc:
                LOGGER.error("%s/%s (vs %s): skipped — %s",
                             collateral, scen.name, scen.borrowable, exc)
                continue
            band = (cal.target_liq30_emode if scen.param_set.startswith("emode")
                    else cal.target_liq30_std)
            try:
                lt_decision, decision = calibrate_row_with_lt(
                    stream,
                    formula_lt=row.lt,
                    formula_ltv=row.ltv,
                    band=band,
                    t_user_days=cal.t_user_days,
                    minimum_gap=cal.minimum_gap,
                    bd_bound=cal.max_uncond_bad_debt,
                    min_years=cal.min_calibration_years,
                    severity_threshold=cal.severity_review_threshold,
                    severity_cap=cal.max_loss_given_bad_debt,
                )
            except ValueError as exc:
                LOGGER.error("%s/%s: calibration failed — %s", collateral, scen.name, exc)
                continue
            out.append(CalibratedRow(
                collateral=collateral, scenario=scen,
                lt=row.lt, formula_ltv=row.ltv, decision=decision,
                lt_decision=lt_decision,
            ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_table(results: Iterable[CalibratedRow]) -> str:
    """One row per collateral × scenario, LT pass columns included."""
    headers = ["collateral", "scenario", "formula LT (%)", "final LT (%)", "ΔLT (pp)",
               "formula LTV (%)", "final LTV (%)", "binding", "≤30d (%)",
               "uncond BD (%)", "exc pre (%)", "exc post (%)", "flags"]
    rows: list[list[str]] = []
    for r in results:
        d = r.decision
        m = d.metrics
        ld = r.lt_decision
        if ld is None:
            final_lt, dlt, exc_pre, exc_post = f"{r.lt * 100:.2f}", "0.00", NA_DASH, NA_DASH
        else:
            final_lt = f"{ld.final_lt * 100:.2f}"
            dlt = f"{ld.delta * 100:+.2f}"
            exc_pre = f"{ld.max_excess_before * 100:.2f}"
            exc_post = f"{ld.max_excess_after * 100:.2f}"
        share = NA_DASH if m.share30 is None else f"{m.share30 * 100:.1f}"
        bd = NA_DASH if m.uncond_bad_debt is None else f"{m.uncond_bad_debt * 100:.2f}"
        flags = ", ".join(d.flags) if d.flags else NA_DASH
        rows.append([r.collateral, r.scenario.name,
                     f"{r.lt * 100:.2f}", final_lt, dlt,
                     f"{r.formula_ltv * 100:.2f}", f"{d.final_ltv * 100:.2f}",
                     d.binding, share, bd, exc_pre, exc_post, flags])
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def results_frame(results: Iterable[CalibratedRow]) -> pd.DataFrame:
    """Flatten calibration rows into a DataFrame (percent units, one row per collateral × scenario).

    Feeds the ``--csv`` export; column order mirrors the CLI table with
    the extra diagnostics (LT, delta, gap rate, max excess, history).
    """
    records = []
    for r in results:
        d = r.decision
        m = d.metrics
        ld = r.lt_decision
        records.append({
            "collateral": r.collateral,
            "scenario": r.scenario.name,
            "borrowable": r.scenario.borrowable,
            "param_set": r.scenario.param_set,
            "formula_lt_pct": round(r.lt * 100, 4),
            "final_lt_pct": round((ld.final_lt if ld else r.lt) * 100, 4),
            "delta_lt_pp": round((ld.delta if ld else 0.0) * 100, 4),
            "max_excess_before_pct": None if ld is None else round(ld.max_excess_before * 100, 4),
            "max_excess_after_pct": None if ld is None else round(ld.max_excess_after * 100, 4),
            "formula_ltv_pct": round(r.formula_ltv * 100, 4),
            "final_ltv_pct": round(d.final_ltv * 100, 4),
            "delta_pp": round((d.final_ltv - r.formula_ltv) * 100, 4),
            "binding": d.binding,
            "liq30_pct": None if m.share30 is None else round(m.share30 * 100, 4),
            "gap_rate_pct": None if m.gap_rate is None else round(m.gap_rate * 100, 4),
            "uncond_bad_debt_pct": None if m.uncond_bad_debt is None else round(m.uncond_bad_debt * 100, 4),
            "max_excess_pct": round(m.max_excess * 100, 4),
            "effective_years": round(d.effective_years, 2),
            "flags": "; ".join(d.flags),
        })
    return pd.DataFrame.from_records(records)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.calibrate",
        description="Unified LTV calibration: max LTV satisfying the liq-30d band, "
                    "the reaction bound, the unconditional bad-debt bound and the LT gap ceiling.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None,
                   help="also write the results as CSV to this path")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()
    cal = universe.calibration
    results = run_calibrate(universe=universe, data_dir=args.data_dir)
    print("Unified LTV calibration — constraints: "
          f"liq ≤ 30d in band (e-mode {cal.target_liq30_emode[0] * 100:g}–{cal.target_liq30_emode[1] * 100:g}%, "
          f"std {cal.target_liq30_std[0] * 100:g}–{cal.target_liq30_std[1] * 100:g}%); "
          f"P(liq ≤ {cal.t_user_days:g}d) ≤ {GAP_HIT_BOUND * 100:g}%; "
          f"uncond bad debt ≤ {cal.max_uncond_bad_debt * 100:g}%; "
          f"LTV ≤ LT − {cal.minimum_gap * 100:g}pp; "
          f"raises need ≥ {cal.min_calibration_years:g}y history; "
          f"LT REVIEW above {cal.severity_review_threshold * 100:g}% excess; "
          f"LT cut when max excess > {cal.max_loss_given_bad_debt * 100:g}% "
          f"(floor formula LT − {LT_FLOOR_DELTA * 100:g}pp)")
    print(format_table(results))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        results_frame(results).to_csv(args.csv, index=False)
        print(f"\nwrote {len(results)} rows to {args.csv}")
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
