"""EWMA volatility and stressed-vol estimation.

Implements ``sigma_stress`` from ``docs/nysa-market-risk-framework.md``
§3.1: for every admissible ``(collateral, borrowable)`` pair, compute the
relative log-return series of the two underlyings, feed it into a
RiskMetrics-style EWMA variance recursion with ``ewma_lambda`` from
:mod:`nysa_risk.config`, and take the configured ``stress_quantile`` of
the resulting daily-sigma series. The same series also yields
``sigma_gap`` at the milder ``gap_sigma_quantile`` — the vol regime used
by the LTV gap buffer ``G`` (see :mod:`nysa_risk.parameters.ltv`), kept
distinct from ``sigma_stress`` which feeds C_T/LT only.

Return construction — two observations per reference date
---------------------------------------------------------
Each parquet in ``data/underlying/`` has an ``open`` and a ``close`` per
trading day. To honour ``meta.include_overnight_gaps`` we treat each day
as **two** price observations rather than folding gap risk into a single
close-to-close return. The interleaved stream per date ``d`` is
``[..., open(d), close(d), open(d+1), close(d+1), ...]`` and the log
differences of that stream give both:

* the intraday move ``log(close(d) / open(d))``, and
* the overnight gap ``log(open(d+1) / close(d))``.

Calendar alignment (mixed equity/crypto pairs)
----------------------------------------------
The two legs may trade on different calendars. Equities are closed on
weekends and holidays; crypto trades 24/7. **We anchor every pair to the
calendar with fewer trading dates in the overlap window** — typically the
equity collateral for RWA/crypto pairs, either leg for same-type pairs.

On each reference date ``d`` the equity leg contributes its own
``open(d), close(d)``. The 24/7 leg contributes:

* value at ``d.open``  = its **most recent close strictly before** ``d``
* value at ``d.close`` = its close on ``d`` (or last available close ≤ ``d``)

The consequence is deliberate: the equity's Friday-close→Monday-open gap
is measured against the crypto's *actual weekend move*
``log(crypto.close(Sun) / crypto.close(Fri))`` — not against a
zero-return interval or a same-day Monday snapshot. This preserves the
gap-risk asymmetry that ``t_liq_days`` and the ES multiplier in
``docs/nysa-market-risk-framework.md`` §3.2/§4 assume.

Short-history pairs
-------------------
For pairs where one leg has less history than the other (e.g. CRCL,
listed 2025-06), the returns are computed on the overlapping window
``[max(first_C, first_B), min(last_C, last_B)]`` and the effective depth
is logged.

CLI
---

::

    python -m nysa_risk.volatility

prints a table of ``pair | sigma_stress (daily %) | effective_years``
sorted by risk descending.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import AssetUniverse, Borrowable, Collateral, load_universe

LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "underlying"

# Two observations per date (open, close) → daily sigma is per-observation
# sigma × √2 (independent-increment approximation over the two half-days).
OBS_PER_DAY = 2
DAILY_SCALE = math.sqrt(OBS_PER_DAY)

# Warn threshold for short pair history (relative to universe.meta.price_history_years).
SHORT_HISTORY_TOLERANCE = 0.9


@dataclass(frozen=True, slots=True)
class PairResult:
    collateral: str
    borrowable: str
    collateral_ticker: str
    borrowable_ticker: str
    n_observations: int
    first_date: date
    last_date: date
    effective_years: float
    sigma_stress: float          # per-observation (stress_quantile of the EWMA series; C_T/LT only)
    sigma_stress_daily: float    # per-observation × √OBS_PER_DAY (daily-scale display value)
    sigma_gap: float             # per-observation (gap_sigma_quantile of the SAME series; LTV's G only)
    sigma_gap_daily: float       # sigma_gap × √OBS_PER_DAY


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


def load_prices(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load the parquet written by :mod:`nysa_risk.extraction.underlying`."""
    path = data_dir / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing price parquet for {ticker}: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: parquet index must be a DatetimeIndex, got {type(df.index).__name__}")
    if not {"open", "close"}.issubset(df.columns):
        raise ValueError(f"{ticker}: parquet must have open/close columns, got {list(df.columns)}")
    return df[["open", "close"]].sort_index()


# ---------------------------------------------------------------------------
# Alignment and relative returns
# ---------------------------------------------------------------------------


def _leg_on_reference(
    leg: pd.DataFrame, ref_dates: pd.DatetimeIndex, is_reference: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (open_values, close_values) of ``leg`` on ``ref_dates``.

    If ``leg`` is the reference itself, use its native open/close.
    Otherwise, use ``leg.close`` values with as-of lookup:

    * open @ d = last close with date **<** d
    * close @ d = last close with date **≤** d
    """
    if is_reference:
        return (
            leg.loc[ref_dates, "open"].to_numpy(),
            leg.loc[ref_dates, "close"].to_numpy(),
        )
    leg_idx = leg.index
    leg_close = leg["close"].to_numpy()
    ref_np = ref_dates.to_numpy()
    pos_lt = leg_idx.searchsorted(ref_np, side="left") - 1   # last < d
    pos_le = leg_idx.searchsorted(ref_np, side="right") - 1  # last ≤ d
    open_vals = np.where(pos_lt >= 0, leg_close[np.clip(pos_lt, 0, None)], np.nan)
    close_vals = np.where(pos_le >= 0, leg_close[np.clip(pos_le, 0, None)], np.nan)
    return open_vals, close_vals


def align_pair(
    collateral_prices: pd.DataFrame,
    borrowable_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build an aligned frame with columns ``c_open, c_close, b_open, b_close``.

    The reference calendar is whichever leg has fewer trading days in the
    overlapping window (equity for equity/crypto pairs). When both legs
    share the same calendar (equity/equity, crypto/crypto), both use their
    native open/close on each date. When they don't, the denser leg's
    ``open`` value on each reference date is its most recent close strictly
    before that date — the full (un-sliced) history is used for that
    lookup so the first reference date still has a prior close available.
    """
    first = max(collateral_prices.index[0], borrowable_prices.index[0])
    last = min(collateral_prices.index[-1], borrowable_prices.index[-1])
    if first > last:
        raise ValueError("no overlapping window between collateral and borrowable")

    c_in = collateral_prices.loc[first:last]
    b_in = borrowable_prices.loc[first:last]

    same_calendar = c_in.index.equals(b_in.index)
    c_is_ref = len(c_in.index) <= len(b_in.index)
    ref_dates = c_in.index if c_is_ref else b_in.index

    # Reference leg → native open/close on its own dates.
    # Non-reference leg → as-of lookup against its FULL history (so the first
    # reference date can still find a strictly-earlier close to fill its `open`).
    # When calendars are identical, both legs are effectively reference.
    c_open, c_close = _leg_on_reference(
        c_in if (c_is_ref or same_calendar) else collateral_prices,
        ref_dates,
        is_reference=c_is_ref or same_calendar,
    )
    b_open, b_close = _leg_on_reference(
        b_in if ((not c_is_ref) or same_calendar) else borrowable_prices,
        ref_dates,
        is_reference=(not c_is_ref) or same_calendar,
    )

    out = pd.DataFrame(
        {"c_open": c_open, "c_close": c_close, "b_open": b_open, "b_close": b_close},
        index=ref_dates,
    )
    return out.dropna()


def _interleaved_stream(pair: pd.DataFrame, prefix: str) -> pd.Series:
    """Return a monotonically-increasing series of open/close prices for one leg.

    Uses nominal timestamps (09:30 / 16:00) purely to make the open
    observation strictly precede the close observation of the same date.
    """
    dates = pair.index
    opens = pd.Series(
        pair[f"{prefix}_open"].to_numpy(),
        index=dates + pd.Timedelta(hours=9, minutes=30),
    )
    closes = pd.Series(
        pair[f"{prefix}_close"].to_numpy(),
        index=dates + pd.Timedelta(hours=16),
    )
    return pd.concat([opens, closes]).sort_index()


def relative_log_returns(
    collateral_prices: pd.DataFrame,
    borrowable_prices: pd.DataFrame,
) -> pd.Series:
    """Log returns of the relative price ``P_collateral / P_borrowable``.

    Equivalent to ``r_collateral - r_borrowable`` on the aligned observation
    stream. Includes both intraday and overnight/gap intervals.
    """
    aligned = align_pair(collateral_prices, borrowable_prices)
    if aligned.empty:
        return pd.Series(dtype=float)
    c_stream = _interleaved_stream(aligned, "c")
    b_stream = _interleaved_stream(aligned, "b")
    # log(c/b) diff = (log c - log b).diff — algebraically identical to r_c - r_b.
    return (np.log(c_stream) - np.log(b_stream)).diff().dropna()


# ---------------------------------------------------------------------------
# EWMA volatility
# ---------------------------------------------------------------------------


def ewma_volatility(returns: pd.Series, lam: float) -> pd.Series:
    """RiskMetrics EWMA sigma series.

    ``sigma_t^2 = lam · sigma_{t-1}^2 + (1 - lam) · r_t^2``, initialised
    with ``sigma_1^2 = r_1^2`` (pandas ``ewm(adjust=False)`` semantics).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be in (0, 1); got {lam}")
    if returns.empty:
        return pd.Series(dtype=float)
    var = returns.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var)


def sigma_stress_value(sigma_series: pd.Series, quantile: float) -> float:
    """Return the ``quantile``-th percentile of the EWMA sigma series."""
    if sigma_series.empty:
        raise ValueError("cannot compute sigma_stress on an empty series")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1]; got {quantile}")
    return float(sigma_series.quantile(quantile))


# ---------------------------------------------------------------------------
# Per-pair driver
# ---------------------------------------------------------------------------


def compute_pair_from_prices(
    collateral_prices: pd.DataFrame,
    borrowable_prices: pd.DataFrame,
    *,
    collateral: str,
    borrowable: str,
    collateral_ticker: str,
    borrowable_ticker: str,
    lam: float,
    quantile: float,
    gap_quantile: float,
    requested_years: float | None = None,
) -> PairResult:
    """Compute a ``PairResult`` from in-memory price frames.

    Both ``sigma_stress`` (at ``quantile``) and ``sigma_gap`` (at
    ``gap_quantile``) are read off the same EWMA sigma series.
    """
    returns = relative_log_returns(collateral_prices, borrowable_prices)
    if returns.empty:
        raise ValueError(f"{collateral}/{borrowable}: no relative returns (no overlap)")
    sigma_series = ewma_volatility(returns, lam)
    sigma_star = sigma_stress_value(sigma_series, quantile)
    sigma_gap_star = sigma_stress_value(sigma_series, gap_quantile)

    first_ts = returns.index[0]
    last_ts = returns.index[-1]
    first_dt = first_ts.date() if hasattr(first_ts, "date") else pd.Timestamp(first_ts).date()
    last_dt = last_ts.date() if hasattr(last_ts, "date") else pd.Timestamp(last_ts).date()
    effective_years = max((last_dt - first_dt).days / 365.25, 0.0)

    if requested_years is not None and effective_years < requested_years * SHORT_HISTORY_TOLERANCE:
        LOGGER.warning(
            "%s/%s: short overlapping history — %.2fy (requested %.1fy)",
            collateral,
            borrowable,
            effective_years,
            requested_years,
        )
    else:
        LOGGER.info(
            "%s/%s: computed on %.2fy of overlap (%d obs)",
            collateral,
            borrowable,
            effective_years,
            len(returns),
        )

    return PairResult(
        collateral=collateral,
        borrowable=borrowable,
        collateral_ticker=collateral_ticker,
        borrowable_ticker=borrowable_ticker,
        n_observations=len(returns),
        first_date=first_dt,
        last_date=last_dt,
        effective_years=effective_years,
        sigma_stress=sigma_star,
        sigma_stress_daily=sigma_star * DAILY_SCALE,
        sigma_gap=sigma_gap_star,
        sigma_gap_daily=sigma_gap_star * DAILY_SCALE,
    )


def compute_pair(
    collateral: str,
    borrowable: str,
    collateral_ticker: str,
    borrowable_ticker: str,
    data_dir: Path,
    *,
    lam: float,
    quantile: float,
    gap_quantile: float,
    requested_years: float | None = None,
) -> PairResult:
    """Load parquet files and compute the ``PairResult``."""
    c_prices = load_prices(collateral_ticker, data_dir)
    b_prices = load_prices(borrowable_ticker, data_dir)
    return compute_pair_from_prices(
        c_prices,
        b_prices,
        collateral=collateral,
        borrowable=borrowable,
        collateral_ticker=collateral_ticker,
        borrowable_ticker=borrowable_ticker,
        lam=lam,
        quantile=quantile,
        gap_quantile=gap_quantile,
        requested_years=requested_years,
    )


# ---------------------------------------------------------------------------
# Universe-wide driver
# ---------------------------------------------------------------------------


def _ticker_index(universe: AssetUniverse) -> Mapping[str, str]:
    """Map every symbol (collateral or borrowable) to its yfinance ticker."""
    idx: dict[str, str] = {}
    for c in universe.collaterals:
        idx[c.symbol] = c.underlying_ticker
    for b in universe.borrowables:
        idx[b.symbol] = b.price_source
    return idx


def compute_all_pairs(
    universe: AssetUniverse | None = None,
    data_dir: Path | None = None,
    *,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> list[PairResult]:
    """Compute ``PairResult`` for every admissible pair in the universe."""
    universe = universe or load_universe()
    data_dir = data_dir or DEFAULT_DATA_DIR
    tickers = _ticker_index(universe)
    lam = universe.calibration.ewma_lambda
    q = universe.calibration.stress_quantile
    gap_q = universe.calibration.gap_sigma_quantile
    requested_years = float(universe.meta.price_history_years)

    target_pairs = list(pairs) if pairs is not None else universe.admissible_pairs()
    out: list[PairResult] = []
    for c_sym, b_sym in target_pairs:
        try:
            result = compute_pair(
                collateral=c_sym,
                borrowable=b_sym,
                collateral_ticker=tickers[c_sym],
                borrowable_ticker=tickers[b_sym],
                data_dir=data_dir,
                lam=lam,
                quantile=q,
                gap_quantile=gap_q,
                requested_years=requested_years,
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.error("%s/%s: skipped — %s", c_sym, b_sym, exc)
            continue
        out.append(result)
    return out


# ---------------------------------------------------------------------------
# CLI: sorted table
# ---------------------------------------------------------------------------


def format_table(results: Iterable[PairResult]) -> str:
    """Render a fixed-width sorted table (sigma_stress descending)."""
    rows = sorted(results, key=lambda r: r.sigma_stress_daily, reverse=True)
    if not rows:
        return "(no pairs)"
    pair_w = max(len("pair"), max(len(f"{r.collateral}/{r.borrowable}") for r in rows))
    header = f"{'pair'.ljust(pair_w)}  {'sigma_stress (daily %)':>22}  {'effective_years':>16}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        pair = f"{r.collateral}/{r.borrowable}"
        lines.append(
            f"{pair.ljust(pair_w)}  {r.sigma_stress_daily * 100:>22.4f}  {r.effective_years:>16.2f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.volatility",
        description="Compute sigma_stress for every admissible pair and print a sorted table.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory with per-ticker parquets")
    p.add_argument("--config", type=Path, default=None, help="override path to assets.yaml")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()
    results = compute_all_pairs(universe=universe, data_dir=args.data_dir)
    print(format_table(results))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
