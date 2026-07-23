"""Historical backtest of the published LTV/LT guarantees.

Validates the framework's two declared exceedance bounds against
realized history, using the SAME aligned pair data
(:func:`nysa_risk.volatility.align_pair` — interleaved open/close
observations, overnight and weekend gaps included) and the SAME
published parameters (:mod:`nysa_risk.parameters.lt` /
:mod:`nysa_risk.parameters.ltv`) as the calibration pipeline. Nothing
is re-estimated here — the backtest consumes the pipeline's outputs.

Method
------
For each collateral and each parameter set (``base`` plus every
``emode:<category>``), the simulation runs on that set's **binding
pair** — the pair whose ``C_T`` determined the published LT/LTV, i.e.
the pair the declared bounds are calibrated against. For every
historical day ``t`` in the pair's overlapping window, a position is
opened at the set's maximum LTV at the close of day ``t``. Debt is
denominated in the borrowable, so with relative price ``P_s``
(collateral value in borrowable terms) the position's running LTV is
``LTV · P_t / P_s``, and it reaches liquidation (running LTV ≥ LT) at
the first observation with::

    P_s ≤ P_t · LTV / LT

Checks
------
1. **Gap constraint** (bound :data:`GAP_HIT_BOUND` = 10 %) — does the
   position's LTV reach LT within ``t_user_days``? ``k_user`` is the
   fat-tailed 90 % quantile multiplier (§4.1), so no more than 10 % of
   openings should become liquidatable before the borrower's reaction
   horizon elapses. Openings whose remaining history is shorter than
   ``t_user_days`` and that never trigger are censored — excluded from
   the denominator rather than counted as no-hits.
2. **Time to liquidation** — for positions that do get liquidated
   within ``horizon_days`` (default 90), days from open to trigger:
   median / mean / p90.
3. **Bad debt** (bound :data:`BAD_DEBT_BOUND` = 1 %) — for every
   liquidation event the realization price is the **next available
   observation after the trigger**: the next open when the trigger
   lands on a close. This deliberately realizes through the
   overnight/weekend gap, which is conservative relative to the 8 h
   ``t_liq_days`` window assumed by ``C_T`` (§3.2) — the liquidator is
   charged the full gap to the next print instead of an intra-session
   execution. Bad debt occurs when the relative price drop from
   trigger to realization exceeds ``1 − LT`` (= ``C_T + S`` by
   construction, §3.3). ``es_factor`` is an Expected-Shortfall
   multiplier at 99 % (§3.2), hence the 1 % bound on the fraction of
   liquidations producing bad debt. A trigger on the final observation
   has no realization price and is excluded from the event count.
   Two companion metrics extend the check: the **unconditional rate**
   — bad-debt events over openings with a decidable horizon outcome
   (liquidated-with-realization, or full-horizon survivors), i.e. the
   share of max-LTV openings ending in bad debt within the horizon —
   and **severity**: per bad-debt event, the excess loss beyond the
   buffer, ``drop − (1 − LT)``, as a fraction of position value at the
   trigger.

The bounds themselves are structural to the declared framework (the
quantile/ES levels the constants were calibrated at), not tunables —
which is why they live here and not in ``config/assets.yaml``.

CLI
---

``python -m nysa_risk.backtest`` prints three tables:

(a) ``collateral | set | checked | hits | P(liq ≤ t_user) (%) | bound (%) | result``
(b) ``collateral | set | liquidations | median (d) | mean (d) | p90 (d)``
(c) ``collateral | set | liquidations | bad debt | observed (%) | bound (%) | result | uncond (%)``
(d) ``collateral | set | bad debt | mean excess (%) | p90 excess (%) | max excess (%)`` + aggregate line
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

from .config import AssetUniverse, load_universe
from .parameters.ltv import compute_all_ltv_from_universe
from .volatility import DEFAULT_DATA_DIR, _ticker_index, align_pair, load_prices

LOGGER = logging.getLogger(__name__)

# Declared exceedance bounds under validation (see module docstring).
GAP_HIT_BOUND = 0.10   # 1 − 0.90: k_user is the fat-tailed 90 % quantile multiplier (§4.1)
BAD_DEBT_BOUND = 0.01  # 1 − 0.99: es_factor is an ES multiplier at 99 % (§3.2)

DEFAULT_HORIZON_DAYS = 90.0  # forward window for the time-to-liquidation statistics

_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True, slots=True)
class SimulationStats:
    n_positions: int                       # every opening (one per close observation)
    gap_checked: int                       # openings with a decidable t_user outcome
    gap_hits: int                          # ... of which reached LT within t_user_days
    liquidations: int                      # triggers within horizon that have a realization price
    bad_debt_events: int                   # ... of which dropped beyond 1 − LT before realization
    days_to_liquidation: tuple[float, ...]  # open → trigger, in calendar days
    # Openings with a decidable horizon outcome (liquidated-with-realization
    # or full-horizon survivors) — the unconditional bad-debt denominator.
    horizon_checked: int = 0
    # Per bad-debt event: excess loss beyond the buffer, drop − (1 − LT),
    # as a fraction of position value at the trigger.
    bad_debt_excess: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class BacktestResult:
    collateral: str
    param_set: str               # "base" or "emode:<category>"
    binding_borrowable: str      # the pair the simulation ran on
    lt: float
    ltv: float
    stats: SimulationStats

    @property
    def gap_hit_rate(self) -> float | None:
        s = self.stats
        return s.gap_hits / s.gap_checked if s.gap_checked else None

    @property
    def bad_debt_rate(self) -> float | None:
        s = self.stats
        return s.bad_debt_events / s.liquidations if s.liquidations else None

    @property
    def unconditional_bad_debt_rate(self) -> float | None:
        """Share of decidable max-LTV openings ending in bad debt within the horizon."""
        s = self.stats
        return s.bad_debt_events / s.horizon_checked if s.horizon_checked else None


# ---------------------------------------------------------------------------
# Relative price stream
# ---------------------------------------------------------------------------


def relative_price_stream(
    collateral_prices: pd.DataFrame,
    borrowable_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Interleaved relative-price observations ``P = P_collateral / P_borrowable``.

    Same alignment and interleaving as the volatility pipeline
    (:func:`nysa_risk.volatility.align_pair`, nominal 09:30/16:00
    timestamps): one open and one close observation per reference date,
    overnight/weekend gaps included. Columns: ``price`` (float) and
    ``is_close`` (bool — positions only open on close observations).
    """
    aligned = align_pair(collateral_prices, borrowable_prices)
    if aligned.empty:
        return pd.DataFrame(
            {"price": pd.Series(dtype=float), "is_close": pd.Series(dtype=bool)},
            index=pd.DatetimeIndex([]),
        )
    opens = pd.DataFrame(
        {"price": (aligned["c_open"] / aligned["b_open"]).to_numpy(), "is_close": False},
        index=aligned.index + pd.Timedelta(hours=9, minutes=30),
    )
    closes = pd.DataFrame(
        {"price": (aligned["c_close"] / aligned["b_close"]).to_numpy(), "is_close": True},
        index=aligned.index + pd.Timedelta(hours=16),
    )
    return pd.concat([opens, closes]).sort_index()


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_positions(
    stream: pd.DataFrame,
    *,
    ltv: float,
    lt: float,
    t_user_days: float,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
) -> SimulationStats:
    """Open a position at ``ltv`` on every close observation and track it forward.

    Liquidation triggers at the first observation where the relative
    price falls to ``ltv / lt`` of the entry price (running LTV ≥ LT);
    the realization price is the next observation after the trigger.
    See the module docstring for the censoring rules.
    """
    if not 0.0 < lt < 1.0:
        raise ValueError(f"lt must be in (0, 1); got {lt}")
    if not 0.0 < ltv < lt:
        raise ValueError(f"ltv must be in (0, lt={lt}); got {ltv} — positions would open at/beyond liquidation")
    if t_user_days <= 0:
        raise ValueError(f"t_user_days must be > 0; got {t_user_days}")
    if horizon_days < t_user_days:
        raise ValueError(f"horizon_days ({horizon_days}) must cover t_user_days ({t_user_days})")

    if stream.empty:
        return SimulationStats(0, 0, 0, 0, 0, ())

    ts = stream.index.to_numpy()
    px = stream["price"].to_numpy(dtype=float)
    is_close = stream["is_close"].to_numpy(dtype=bool)
    n = len(px)
    horizon = np.timedelta64(int(round(horizon_days * 86_400)), "s")
    threshold_ratio = ltv / lt        # trigger when P_s / P_open ≤ LTV / LT
    buffer = 1.0 - lt                 # = C_T + S — the trigger→realization bad-debt buffer

    n_positions = 0
    gap_checked = 0
    gap_hits = 0
    liquidations = 0
    bad_debt_events = 0
    days_to_liquidation: list[float] = []
    horizon_checked = 0
    bad_debt_excess: list[float] = []

    for i in np.flatnonzero(is_close):
        n_positions += 1
        coverage_days = float((ts[-1] - ts[i]) / _DAY)
        end = int(np.searchsorted(ts, ts[i] + horizon, side="right"))
        window = px[i + 1:end]
        hit_offsets = np.flatnonzero(window <= px[i] * threshold_ratio)

        if hit_offsets.size == 0:
            # No trigger within horizon. The gap outcome is decidable only
            # if the full t_user window was observed; the horizon outcome
            # only if the full horizon was.
            if coverage_days >= t_user_days:
                gap_checked += 1
            if coverage_days >= horizon_days:
                horizon_checked += 1
            continue

        j = i + 1 + int(hit_offsets[0])
        days = float((ts[j] - ts[i]) / _DAY)

        # (1) Gap constraint — a trigger settles the outcome either way.
        gap_checked += 1
        if days <= t_user_days:
            gap_hits += 1

        # (2)+(3) Liquidation event — needs a realization observation.
        if j + 1 < n:
            liquidations += 1
            horizon_checked += 1
            days_to_liquidation.append(days)
            drop = 1.0 - px[j + 1] / px[j]
            if drop > buffer:
                bad_debt_events += 1
                bad_debt_excess.append(drop - buffer)

    return SimulationStats(
        n_positions=n_positions,
        gap_checked=gap_checked,
        gap_hits=gap_hits,
        liquidations=liquidations,
        bad_debt_events=bad_debt_events,
        days_to_liquidation=tuple(days_to_liquidation),
        horizon_checked=horizon_checked,
        bad_debt_excess=tuple(bad_debt_excess),
    )


# ---------------------------------------------------------------------------
# Universe-wide driver
# ---------------------------------------------------------------------------


def run_backtest(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
    *,
    horizon_days: float = DEFAULT_HORIZON_DAYS,
) -> list[BacktestResult]:
    """Run the pipeline (prices → vol → LT → LTV) then backtest every row.

    One :class:`BacktestResult` per ``(collateral, param_set)``, each
    simulated on that row's binding pair with that row's published
    LT/LTV. Rows whose LTV is not in ``(0, LT)`` (nothing to simulate)
    and pairs with missing price data are skipped with a log message.
    """
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    ltv_rows = compute_all_ltv_from_universe(universe=universe, data_dir=data_dir)
    tickers = _ticker_index(universe)
    t_user_days = universe.calibration.t_user_days

    cache: dict[str, pd.DataFrame] = {}

    def _prices(symbol: str) -> pd.DataFrame:
        ticker = tickers[symbol]
        if ticker not in cache:
            cache[ticker] = load_prices(ticker, data_dir)
        return cache[ticker]

    out: list[BacktestResult] = []
    for row in ltv_rows:
        if not 0.0 < row.ltv < row.lt:
            LOGGER.warning(
                "%s/%s: LTV %.4f outside (0, LT=%.4f) — skipping backtest",
                row.collateral, row.param_set, row.ltv, row.lt,
            )
            continue
        try:
            stream = relative_price_stream(
                _prices(row.collateral), _prices(row.binding_borrowable)
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.error(
                "%s/%s (vs %s): skipped — %s",
                row.collateral, row.param_set, row.binding_borrowable, exc,
            )
            continue
        stats = simulate_positions(
            stream,
            ltv=row.ltv,
            lt=row.lt,
            t_user_days=t_user_days,
            horizon_days=horizon_days,
        )
        out.append(
            BacktestResult(
                collateral=row.collateral,
                param_set=row.param_set,
                binding_borrowable=row.binding_borrowable,
                lt=row.lt,
                ltv=row.ltv,
                stats=stats,
            )
        )
    return out


# ---------------------------------------------------------------------------
# CLI tables
# ---------------------------------------------------------------------------

NA_DASH = "—"


def _render_table(headers: list[str], rows: list[list[str]], n_left: int = 2) -> str:
    """Fixed-width table: first ``n_left`` columns left-justified, the rest right-justified."""
    widths = [
        max(len(h), max((len(r[k]) for r in rows), default=0))
        for k, h in enumerate(headers)
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(w) if k < n_left else c.rjust(w)
            for k, (c, w) in enumerate(zip(cells, widths))
        )

    header = fmt(headers)
    lines = [header, "-" * len(header)]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def _sorted(results: Iterable[BacktestResult]) -> list[BacktestResult]:
    return sorted(results, key=lambda r: (r.collateral, r.param_set))


def format_gap_table(
    results: Iterable[BacktestResult],
    t_user_days: float,
    bound: float = GAP_HIT_BOUND,
) -> str:
    """Table (a): observed P(liquidation ≤ t_user) per (collateral, set) vs the bound."""
    headers = ["collateral", "set", "checked", "hits",
               f"P(liq ≤ {t_user_days:g}d) (%)", "bound (%)", "result"]
    rows: list[list[str]] = []
    for r in _sorted(results):
        rate = r.gap_hit_rate
        if rate is None:
            obs, verdict = NA_DASH, NA_DASH
        else:
            obs = f"{rate * 100:.2f}"
            verdict = "PASS" if rate <= bound else "FAIL"
        rows.append([r.collateral, r.param_set, str(r.stats.gap_checked),
                     str(r.stats.gap_hits), obs, f"{bound * 100:.0f}", verdict])
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def format_time_table(results: Iterable[BacktestResult]) -> str:
    """Table (b): time-to-liquidation stats (days) per (collateral, set)."""
    headers = ["collateral", "set", "liquidations", "median (d)", "mean (d)", "p90 (d)"]
    rows: list[list[str]] = []
    for r in _sorted(results):
        d = np.asarray(r.stats.days_to_liquidation, dtype=float)
        if d.size == 0:
            med = mean = p90 = NA_DASH
        else:
            med = f"{float(np.median(d)):.2f}"
            mean = f"{float(np.mean(d)):.2f}"
            p90 = f"{float(np.percentile(d, 90)):.2f}"
        rows.append([r.collateral, r.param_set, str(r.stats.liquidations), med, mean, p90])
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def format_bad_debt_table(
    results: Iterable[BacktestResult],
    bound: float = BAD_DEBT_BOUND,
) -> str:
    """Table (c): bad-debt fraction of liquidations per (collateral, set) vs the bound.

    The trailing ``uncond (%)`` column is the unconditional rate —
    bad-debt events over openings with a decidable horizon outcome.
    """
    headers = ["collateral", "set", "liquidations", "bad debt", "observed (%)",
               "bound (%)", "result", "uncond (%)"]
    rows: list[list[str]] = []
    for r in _sorted(results):
        rate = r.bad_debt_rate
        if rate is None:
            obs, verdict = NA_DASH, NA_DASH
        else:
            obs = f"{rate * 100:.2f}"
            verdict = "PASS" if rate <= bound else "FAIL"
        uncond = r.unconditional_bad_debt_rate
        uncond_cell = NA_DASH if uncond is None else f"{uncond * 100:.2f}"
        rows.append([r.collateral, r.param_set, str(r.stats.liquidations),
                     str(r.stats.bad_debt_events), obs, f"{bound * 100:.0f}", verdict,
                     uncond_cell])
    if not rows:
        return "(no results)"
    return _render_table(headers, rows)


def format_severity_table(results: Iterable[BacktestResult]) -> str:
    """Table (d): bad-debt severity — excess loss beyond the ``1 − LT`` buffer.

    Per row: mean / p90 / max of ``drop − (1 − LT)`` over that row's
    bad-debt events, as % of position value at trigger. A trailing line
    gives the aggregate distribution over all events pooled.
    """
    headers = ["collateral", "set", "bad debt",
               "mean excess (%)", "p90 excess (%)", "max excess (%)"]
    rows: list[list[str]] = []
    pooled: list[float] = []
    for r in _sorted(results):
        exc = np.asarray(r.stats.bad_debt_excess, dtype=float)
        pooled.extend(r.stats.bad_debt_excess)
        if exc.size == 0:
            mean = p90 = mx = NA_DASH
        else:
            mean = f"{float(exc.mean()) * 100:.2f}"
            p90 = f"{float(np.percentile(exc, 90)) * 100:.2f}"
            mx = f"{float(exc.max()) * 100:.2f}"
        rows.append([r.collateral, r.param_set, str(r.stats.bad_debt_events), mean, p90, mx])
    if not rows:
        return "(no results)"
    table = _render_table(headers, rows)
    if pooled:
        a = np.asarray(pooled, dtype=float)
        table += (
            f"\naggregate ({a.size} events): "
            f"mean {float(a.mean()) * 100:.2f}%  "
            f"p50 {float(np.percentile(a, 50)) * 100:.2f}%  "
            f"p90 {float(np.percentile(a, 90)) * 100:.2f}%  "
            f"max {float(a.max()) * 100:.2f}%"
        )
    else:
        table += "\naggregate: no bad-debt events"
    return table


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.backtest",
        description="Backtest the published LTV/LT parameters against realized pair history.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--horizon-days", type=float, default=DEFAULT_HORIZON_DAYS,
                   help="forward window for the time-to-liquidation statistics")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()
    results = run_backtest(
        universe=universe, data_dir=args.data_dir, horizon_days=args.horizon_days
    )
    t_user_days = universe.calibration.t_user_days

    print("(a) Gap constraint — P(liquidation ≤ t_user) vs declared bound")
    print(format_gap_table(results, t_user_days))
    print()
    print(f"(b) Time to liquidation (within {args.horizon_days:g}-day horizon)")
    print(format_time_table(results))
    print()
    print("(c) Bad debt — trigger→realization drop beyond 1 − LT vs declared bound")
    print(format_bad_debt_table(results))
    print()
    print("(d) Bad-debt severity — excess loss beyond the 1 − LT buffer")
    print(format_severity_table(results))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
