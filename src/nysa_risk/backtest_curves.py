"""Survival-style time-to-event summaries for the published LTV/LT parameters.

Complements the constraint backtest (:mod:`nysa_risk.backtest`, which
validates declared exceedance bounds) with two duration questions,
answered per collateral under two fixed scenarios:

* ``e-mode``   — path = collateral vs **USDT**, parameters = the
  ``emode:stable`` LT/LTV row;
* ``volatile`` — path = collateral vs **BNB**, parameters = the
  standard (``base``) LT/LTV row.

Both run on the same aligned interleaved open/close relative-price
stream as the pipeline (:func:`nysa_risk.backtest.relative_price_stream`
over :func:`nysa_risk.volatility.align_pair` — overnight and weekend
gaps included). A position is opened at the scenario's maximum LTV at
the close of every day in the pair's overlapping history; with relative
price ``P_t`` the running LTV is ``LTV · P_open / P_t``.

(1) **Time to liquidability** — days from opening until the running LTV
    first touches LT, i.e. the first observation with
    ``P_t ≤ P_open · LTV / LT``; censored at :data:`LIQ_HORIZON_DAYS`
    (365 days). Openings with less than 365 days of remaining history
    that never trigger are censored at their available horizon — they
    are neither "liquidatable within 1y" nor "never liquidated", and
    appear in the ``censored`` column. Reported per collateral ×
    scenario: number of openings, % of openings becoming liquidatable
    within the horizon, the p10/p25/p50/p75/p90 percentiles of
    days-to-liquidation among those (p10 supports statements like
    "in 90 % of cases it takes more than X days to become
    liquidatable"), and the % that survived the full horizon without
    touching LT.

(2) **Time from liquidation zone to insolvency** — a counterfactual
    **liquidator-blackout tolerance diagnostic, not a forecast**: it
    assumes no liquidation occurs at the trigger, no interest accrues,
    and the borrower does not react — the position simply rides the
    realized path. For every liquidation event from (1), days from the
    LT touch until the collateral value falls below the debt: the first
    observation whose relative drop from the trigger exceeds ``1 − LT``
    (``P_t < P_trigger · LT``), censored at
    :data:`INSOLVENCY_HORIZON_DAYS` (30 days). Reported: percentiles of
    days-to-insolvency among insolvency events and the % of events
    still solvent after the full 30-day window (events with less than
    30 days of remaining history and no insolvency are censored).

Calibration mode (``--calibrate``)
----------------------------------
Solves, per collateral × scenario, for the LTV (holding LT fixed) whose
historical P(liq ≤ 30d) lands inside the declared target band
(``calibration.target_liq30_emode`` / ``target_liq30_std`` in
``config/assets.yaml``), by bisection on LTV re-running the
time-to-liquidability simulation. The band's upper bound binds when the
current LTV over-liquidates, the lower bound when it under-liquidates;
a current LTV already inside the band is kept and reported
``unchanged``. Two hard constraints dominate the band:

* the reaction bound stays a floor — the calibrated LTV must keep
  P(liq ≤ t_user_days) ≤ :data:`nysa_risk.backtest.GAP_HIT_BOUND`;
* the calibrated LTV never exceeds ``LT − calibration.minimum_gap``.

Because the metric is a step function of LTV over one realized history,
the band can be jumped over entirely; the solver then settles on the
largest LTV satisfying the constraints. Both probabilities here are
per-opening frequencies (same caveat as below).

Interpretation caveat
---------------------
Consecutive daily openings share most of their forward path — they are
**intentionally correlated**, not independent trials. Every percentage
here is a per-opening (or per-event) frequency over the one realized
historical window, NOT an independent probability; read the tables as
descriptive summaries of history, not as a probabilistic model.

CLI
---

``python -m nysa_risk.backtest_curves`` prints one table per question,
rows = collateral × scenario:

(1) ``collateral | scenario | openings | liq ≤ 30d (%) | liq ≤ 365d (%) | p10..p90 (d) | never (%) | censored``
(2) ``collateral | scenario | liquidations | insolvencies | p10..p90 (d) | solvent > 30d (%) | censored``
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import GAP_HIT_BOUND, NA_DASH, _render_table, relative_price_stream
from .config import AssetUniverse, load_universe
from .parameters.ltv import compute_all_ltv_from_universe
from .volatility import DEFAULT_DATA_DIR, _ticker_index, load_prices

LOGGER = logging.getLogger(__name__)

LIQ_HORIZON_DAYS = 365.0         # censoring horizon for time-to-liquidability
INSOLVENCY_HORIZON_DAYS = 30.0   # censoring horizon for liquidation-zone → insolvency
EARLY_LIQ_WINDOW_DAYS = 30.0     # short-window liquidability column (share of openings touching LT ≤ 30d)

PERCENTILES = (10, 25, 50, 75, 90)

_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str            # display name
    borrowable: str      # right leg of the simulated pair
    param_set: str       # published LT/LTV row supplying the parameters


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="e-mode", borrowable="USDT", param_set="emode:stable"),
    Scenario(name="volatile", borrowable="BNB", param_set="base"),
)


@dataclass(frozen=True, slots=True)
class CurveStats:
    # (1) time to liquidability
    n_openings: int
    n_liquidatable: int                     # touched LT within the liq horizon
    n_never_liquidatable: int               # survived the FULL horizon without touching LT
    n_censored_openings: int                # ran out of history before the horizon, no touch
    days_to_liquidation: tuple[float, ...]  # open → LT touch, calendar days
    # (2) liquidation zone → insolvency (per liquidation event)
    n_insolvencies: int                     # dropped beyond 1 − LT within the insolvency horizon
    n_solvent_after_horizon: int            # still solvent after the FULL insolvency horizon
    n_censored_events: int                  # ran out of history before the horizon, still solvent
    days_to_insolvency: tuple[float, ...]   # LT touch → insolvency, calendar days

    @property
    def pct_liquidatable(self) -> float | None:
        return self.n_liquidatable / self.n_openings if self.n_openings else None

    @property
    def pct_never_liquidatable(self) -> float | None:
        return self.n_never_liquidatable / self.n_openings if self.n_openings else None

    def pct_liquidatable_within(self, days: float) -> float | None:
        """Share of openings whose LT touch came within ``days`` (per-opening frequency).

        Same denominator convention as :attr:`pct_liquidatable` — all
        openings, including early-censored ones — so the columns remain
        directly comparable.
        """
        if not self.n_openings:
            return None
        return sum(1 for d in self.days_to_liquidation if d <= days) / self.n_openings

    @property
    def pct_solvent_after_horizon(self) -> float | None:
        return self.n_solvent_after_horizon / self.n_liquidatable if self.n_liquidatable else None


@dataclass(frozen=True, slots=True)
class CurveResult:
    collateral: str
    scenario: Scenario
    lt: float
    ltv: float
    stats: CurveStats


# ---------------------------------------------------------------------------
# Core time-to-event computation
# ---------------------------------------------------------------------------


def compute_curves(
    stream: pd.DataFrame,
    *,
    ltv: float,
    lt: float,
    liq_horizon_days: float = LIQ_HORIZON_DAYS,
    insolvency_horizon_days: float = INSOLVENCY_HORIZON_DAYS,
) -> CurveStats:
    """Measure both durations for a position opened on every close observation.

    ``stream`` has the :func:`nysa_risk.backtest.relative_price_stream`
    shape (``price``/``is_close`` on a DatetimeIndex). The LT touch is
    the first observation with ``P_t ≤ P_open · ltv / lt`` (running LTV
    ≥ LT); insolvency is the first later observation with
    ``P_t < P_trigger · lt`` (drop from the trigger exceeding 1 − LT).
    See the module docstring for the censoring rules.
    """
    if not 0.0 < lt < 1.0:
        raise ValueError(f"lt must be in (0, 1); got {lt}")
    if not 0.0 < ltv < lt:
        raise ValueError(f"ltv must be in (0, lt={lt}); got {ltv} — positions would open at/beyond LT")
    if liq_horizon_days <= 0:
        raise ValueError(f"liq_horizon_days must be > 0; got {liq_horizon_days}")
    if insolvency_horizon_days <= 0:
        raise ValueError(f"insolvency_horizon_days must be > 0; got {insolvency_horizon_days}")

    if stream.empty:
        return CurveStats(0, 0, 0, 0, (), 0, 0, 0, ())

    ts = stream.index.to_numpy()
    px = stream["price"].to_numpy(dtype=float)
    is_close = stream["is_close"].to_numpy(dtype=bool)
    liq_horizon = np.timedelta64(int(round(liq_horizon_days * 86_400)), "s")
    ins_horizon = np.timedelta64(int(round(insolvency_horizon_days * 86_400)), "s")
    threshold_ratio = ltv / lt        # LT touch when P_t / P_open ≤ LTV / LT

    n_openings = 0
    n_liquidatable = 0
    n_never = 0
    n_censored_openings = 0
    days_to_liquidation: list[float] = []
    n_insolvencies = 0
    n_solvent_after = 0
    n_censored_events = 0
    days_to_insolvency: list[float] = []

    for i in np.flatnonzero(is_close):
        n_openings += 1
        coverage_days = float((ts[-1] - ts[i]) / _DAY)
        end = int(np.searchsorted(ts, ts[i] + liq_horizon, side="right"))
        window = px[i + 1:end]
        hit_offsets = np.flatnonzero(window <= px[i] * threshold_ratio)

        if hit_offsets.size == 0:
            if coverage_days >= liq_horizon_days:
                n_never += 1            # full horizon observed, LT never touched
            else:
                n_censored_openings += 1
            continue

        # (1) LT touch within the horizon.
        j = i + 1 + int(hit_offsets[0])
        n_liquidatable += 1
        days_to_liquidation.append(float((ts[j] - ts[i]) / _DAY))

        # (2) Counterfactual ride from the trigger towards insolvency.
        ins_coverage_days = float((ts[-1] - ts[j]) / _DAY)
        ins_end = int(np.searchsorted(ts, ts[j] + ins_horizon, side="right"))
        ins_window = px[j + 1:ins_end]
        ins_offsets = np.flatnonzero(ins_window < px[j] * lt)
        if ins_offsets.size:
            k = j + 1 + int(ins_offsets[0])
            n_insolvencies += 1
            days_to_insolvency.append(float((ts[k] - ts[j]) / _DAY))
        elif ins_coverage_days >= insolvency_horizon_days:
            n_solvent_after += 1
        else:
            n_censored_events += 1

    return CurveStats(
        n_openings=n_openings,
        n_liquidatable=n_liquidatable,
        n_never_liquidatable=n_never,
        n_censored_openings=n_censored_openings,
        days_to_liquidation=tuple(days_to_liquidation),
        n_insolvencies=n_insolvencies,
        n_solvent_after_horizon=n_solvent_after,
        n_censored_events=n_censored_events,
        days_to_insolvency=tuple(days_to_insolvency),
    )


# ---------------------------------------------------------------------------
# Universe-wide driver
# ---------------------------------------------------------------------------


def run_curves(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
    *,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    liq_horizon_days: float = LIQ_HORIZON_DAYS,
    insolvency_horizon_days: float = INSOLVENCY_HORIZON_DAYS,
) -> list[CurveResult]:
    """One :class:`CurveResult` per collateral × scenario.

    Each scenario pins the simulated pair (``scenario.borrowable``) AND
    the published parameter row (``scenario.param_set``) — e.g. the
    ``volatile`` scenario rides the collateral/BNB path under the
    standard LT/LTV. Collaterals missing the scenario's parameter row
    or price data are skipped with a log message.
    """
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    ltv_rows = compute_all_ltv_from_universe(universe=universe, data_dir=data_dir)
    by_key = {(r.collateral, r.param_set): r for r in ltv_rows}
    tickers = _ticker_index(universe)

    cache: dict[str, pd.DataFrame] = {}

    def _prices(symbol: str) -> pd.DataFrame:
        ticker = tickers[symbol]
        if ticker not in cache:
            cache[ticker] = load_prices(ticker, data_dir)
        return cache[ticker]

    out: list[CurveResult] = []
    for collateral in sorted({r.collateral for r in ltv_rows}):
        for scen in scenarios:
            row = by_key.get((collateral, scen.param_set))
            if row is None:
                LOGGER.warning(
                    "%s/%s: no '%s' parameter row — skipping",
                    collateral, scen.name, scen.param_set,
                )
                continue
            if scen.borrowable not in tickers:
                LOGGER.warning(
                    "%s/%s: borrowable %s not in universe — skipping",
                    collateral, scen.name, scen.borrowable,
                )
                continue
            if not 0.0 < row.ltv < row.lt:
                LOGGER.warning(
                    "%s/%s: LTV %.4f outside (0, LT=%.4f) — skipping",
                    collateral, scen.name, row.ltv, row.lt,
                )
                continue
            try:
                stream = relative_price_stream(
                    _prices(collateral), _prices(scen.borrowable)
                )
            except (FileNotFoundError, ValueError) as exc:
                LOGGER.error("%s/%s (vs %s): skipped — %s",
                             collateral, scen.name, scen.borrowable, exc)
                continue
            stats = compute_curves(
                stream,
                ltv=row.ltv,
                lt=row.lt,
                liq_horizon_days=liq_horizon_days,
                insolvency_horizon_days=insolvency_horizon_days,
            )
            out.append(CurveResult(
                collateral=collateral, scenario=scen,
                lt=row.lt, ltv=row.ltv, stats=stats,
            ))
    return out


# ---------------------------------------------------------------------------
# Calibration mode — solve LTV for a target P(liq ≤ 30d) band
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    collateral: str
    scenario: Scenario
    lt: float
    band: tuple[float, float]
    current_ltv: float
    current_liq30: float
    calibrated_ltv: float
    calibrated_liq30: float
    status: str                  # "unchanged" | "lowered" | "raised" | "raised-capped"

    @property
    def delta(self) -> float:
        return self.calibrated_ltv - self.current_ltv


def _bisect_largest(lo: float, hi: float, ok, tol: float) -> float:
    """Largest ``x`` in ``[lo, hi]`` with ``ok(x)``, for monotone-decreasing ``ok``."""
    if ok(hi):
        return hi
    if not ok(lo):
        return lo  # nothing feasible in range — caller gets the floor
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def calibrate_ltv(
    stream: pd.DataFrame,
    *,
    lt: float,
    current_ltv: float,
    band: tuple[float, float],
    t_user_days: float,
    minimum_gap: float,
    gap_bound: float = GAP_HIT_BOUND,
    early_window_days: float = EARLY_LIQ_WINDOW_DAYS,
    tol: float = 1e-4,
) -> tuple[float, str]:
    """Bisect on LTV until P(liq ≤ 30d) lands inside ``band``.

    Both P(liq ≤ 30d) and the reaction metric P(liq ≤ t_user) are
    monotone non-decreasing step functions of LTV, so feasibility
    (P30 ≤ band_hi AND P_t_user ≤ gap_bound AND LTV ≤ LT − minimum_gap)
    is a lower set — bisection finds its edge. Returns
    ``(calibrated_ltv, status)``; see the module docstring for the
    behaviour when the band is jumped over or a constraint binds first.
    """
    band_lo, band_hi = band
    if not 0.0 < band_lo <= band_hi < 1.0:
        raise ValueError(f"band must satisfy 0 < lo ≤ hi < 1; got {band}")
    if not 0.0 < minimum_gap < lt:
        raise ValueError(f"minimum_gap must be in (0, lt={lt}); got {minimum_gap}")
    ceiling = lt - minimum_gap

    def metrics(x: float) -> tuple[float, float]:
        s = compute_curves(stream, ltv=x, lt=lt, liq_horizon_days=early_window_days)
        l30 = s.pct_liquidatable_within(early_window_days)
        gap = s.pct_liquidatable_within(t_user_days)
        if l30 is None or gap is None:
            raise ValueError("cannot calibrate on a stream with no openings")
        return l30, gap

    def feasible(x: float) -> bool:
        if x > ceiling:
            return False
        l30, gap = metrics(x)
        return l30 <= band_hi and gap <= gap_bound

    cur_l30, cur_gap = metrics(current_ltv)
    cur_feasible = current_ltv <= ceiling and cur_l30 <= band_hi and cur_gap <= gap_bound

    if cur_feasible and cur_l30 >= band_lo:
        return current_ltv, "unchanged"

    if not cur_feasible:
        # Over-liquidating (or constraint violation): largest feasible LTV below current.
        x = _bisect_largest(tol, min(current_ltv, ceiling), feasible, tol)
        return x, "lowered"

    # Under-liquidating: raise towards the band's lower edge, constraints permitting.
    max_ok = ceiling if feasible(ceiling) else _bisect_largest(current_ltv, ceiling, feasible, tol)
    if metrics(max_ok)[0] < band_lo:
        return max_ok, "raised-capped"  # constraints bind before the band is reached
    lo, hi = current_ltv, max_ok
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if metrics(mid)[0] >= band_lo:
            hi = mid
        else:
            lo = mid
    return hi, "raised"


def run_calibration(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
    *,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> list[CalibrationResult]:
    """One :class:`CalibrationResult` per collateral × scenario.

    Same pair/parameter wiring as :func:`run_curves`; the band comes from
    ``target_liq30_emode`` for e-mode parameter rows and
    ``target_liq30_std`` otherwise.
    """
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

    out: list[CalibrationResult] = []
    for collateral in sorted({r.collateral for r in ltv_rows}):
        for scen in scenarios:
            row = by_key.get((collateral, scen.param_set))
            if row is None or scen.borrowable not in tickers:
                LOGGER.warning("%s/%s: missing parameter row or borrowable — skipping", collateral, scen.name)
                continue
            if not 0.0 < row.ltv < row.lt:
                LOGGER.warning("%s/%s: LTV %.4f outside (0, LT=%.4f) — skipping", collateral, scen.name, row.ltv, row.lt)
                continue
            try:
                stream = relative_price_stream(_prices(collateral), _prices(scen.borrowable))
            except (FileNotFoundError, ValueError) as exc:
                LOGGER.error("%s/%s (vs %s): skipped — %s", collateral, scen.name, scen.borrowable, exc)
                continue
            band = cal.target_liq30_emode if scen.param_set.startswith("emode") else cal.target_liq30_std
            current = compute_curves(stream, ltv=row.ltv, lt=row.lt, liq_horizon_days=EARLY_LIQ_WINDOW_DAYS)
            if current.n_openings == 0:
                LOGGER.warning("%s/%s: no openings in overlap — skipping", collateral, scen.name)
                continue
            calibrated_ltv, status = calibrate_ltv(
                stream, lt=row.lt, current_ltv=row.ltv, band=band,
                t_user_days=cal.t_user_days, minimum_gap=cal.minimum_gap,
            )
            calibrated = compute_curves(stream, ltv=calibrated_ltv, lt=row.lt,
                                        liq_horizon_days=EARLY_LIQ_WINDOW_DAYS)
            out.append(CalibrationResult(
                collateral=collateral, scenario=scen, lt=row.lt, band=band,
                current_ltv=row.ltv,
                current_liq30=current.pct_liquidatable_within(EARLY_LIQ_WINDOW_DAYS),
                calibrated_ltv=calibrated_ltv,
                calibrated_liq30=calibrated.pct_liquidatable_within(EARLY_LIQ_WINDOW_DAYS),
                status=status,
            ))
    return out


def format_calibration_table(results: Iterable[CalibrationResult]) -> str:
    """``collateral | scenario | current LTV | current %≤30d | calibrated LTV | calibrated %≤30d | Δ``."""
    headers = ["collateral", "scenario", "current LTV (%)", "current ≤30d (%)",
               "calibrated LTV (%)", "calibrated ≤30d (%)", "Δ"]
    rows: list[list[str]] = []
    for r in results:
        if r.status == "unchanged":
            delta = "unchanged"
        else:
            delta = f"{r.delta * 100:+.2f}pp"
            if r.status == "raised-capped":
                delta += " (capped)"
        rows.append([r.collateral, r.scenario.name,
                     f"{r.current_ltv * 100:.2f}", f"{r.current_liq30 * 100:.1f}",
                     f"{r.calibrated_ltv * 100:.2f}", f"{r.calibrated_liq30 * 100:.1f}",
                     delta])
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


# ---------------------------------------------------------------------------
# CLI tables
# ---------------------------------------------------------------------------


def _percentile_cells(days: tuple[float, ...]) -> list[str]:
    if not days:
        return [NA_DASH] * len(PERCENTILES)
    d = np.asarray(days, dtype=float)
    return [f"{float(np.percentile(d, q)):.1f}" for q in PERCENTILES]


def _pct_cell(fraction: float | None) -> str:
    return NA_DASH if fraction is None else f"{fraction * 100:.1f}"


def format_liquidability_table(
    results: Iterable[CurveResult],
    liq_horizon_days: float = LIQ_HORIZON_DAYS,
    early_window_days: float = EARLY_LIQ_WINDOW_DAYS,
) -> str:
    """Table (1): openings, % liquidatable within 30d / the horizon, day percentiles, % never."""
    headers = (["collateral", "scenario", "openings",
                f"liq ≤ {early_window_days:g}d (%)", f"liq ≤ {liq_horizon_days:g}d (%)"]
               + [f"p{q} (d)" for q in PERCENTILES]
               + ["never (%)", "censored"])
    rows: list[list[str]] = []
    for r in results:
        s = r.stats
        rows.append(
            [r.collateral, r.scenario.name, str(s.n_openings),
             _pct_cell(s.pct_liquidatable_within(early_window_days)),
             _pct_cell(s.pct_liquidatable)]
            + _percentile_cells(s.days_to_liquidation)
            + [_pct_cell(s.pct_never_liquidatable), str(s.n_censored_openings)]
        )
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def format_insolvency_table(
    results: Iterable[CurveResult],
    insolvency_horizon_days: float = INSOLVENCY_HORIZON_DAYS,
) -> str:
    """Table (2): per liquidation event, day percentiles to insolvency, % still solvent."""
    headers = (["collateral", "scenario", "liquidations", "insolvencies"]
               + [f"p{q} (d)" for q in PERCENTILES]
               + [f"solvent > {insolvency_horizon_days:g}d (%)", "censored"])
    rows: list[list[str]] = []
    for r in results:
        s = r.stats
        rows.append(
            [r.collateral, r.scenario.name, str(s.n_liquidatable), str(s.n_insolvencies)]
            + _percentile_cells(s.days_to_insolvency)
            + [_pct_cell(s.pct_solvent_after_horizon), str(s.n_censored_events)]
        )
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.backtest_curves",
        description="Time-to-event (survival-style) summaries per collateral × scenario.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--liq-horizon-days", type=float, default=LIQ_HORIZON_DAYS,
                   help="censoring horizon for time-to-liquidability")
    p.add_argument("--insolvency-horizon-days", type=float, default=INSOLVENCY_HORIZON_DAYS,
                   help="censoring horizon for liquidation-zone → insolvency")
    p.add_argument("--calibrate", action="store_true",
                   help="solve per-row LTVs for the target P(liq ≤ 30d) band instead of printing the curves")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()

    if args.calibrate:
        cal = universe.calibration
        cal_results = run_calibration(universe=universe, data_dir=args.data_dir)
        print("LTV calibration — target P(liq ≤ 30d) band: "
              f"e-mode {cal.target_liq30_emode[0] * 100:g}–{cal.target_liq30_emode[1] * 100:g}%, "
              f"std {cal.target_liq30_std[0] * 100:g}–{cal.target_liq30_std[1] * 100:g}%  "
              f"(reaction floor P(liq ≤ {cal.t_user_days:g}d) ≤ {GAP_HIT_BOUND * 100:g}%, "
              f"ceiling LT − {cal.minimum_gap * 100:g}pp)")
        print(format_calibration_table(cal_results))
        return 0 if cal_results else 1

    results = run_curves(
        universe=universe,
        data_dir=args.data_dir,
        liq_horizon_days=args.liq_horizon_days,
        insolvency_horizon_days=args.insolvency_horizon_days,
    )

    print(f"(1) Time to liquidability — opening → first LT touch "
          f"(censored at {args.liq_horizon_days:g} days)")
    print(format_liquidability_table(results, args.liq_horizon_days))
    print()
    print(f"(2) Liquidation zone → insolvency — counterfactual blackout tolerance, "
          f"not a forecast (censored at {args.insolvency_horizon_days:g} days)")
    print(format_insolvency_table(results, args.insolvency_horizon_days))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
