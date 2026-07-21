"""Download daily Open/Close history for every underlying in ``config/assets.yaml``.

The Nysa risk framework calibrates ``sigma_stress`` on long-horizon daily
returns of each underlying (§2 Data Sources of
``docs/nysa-market-risk-framework.md``). GM tokens track their underlying
NAV, so the underlying's yfinance history is used as the risk proxy.
Overnight (close-to-open) gaps are risk-relevant — hence both ``Open`` and
``Close`` are persisted (``meta.include_overnight_gaps``).

One parquet file per unique ticker is written to ``data/underlying/``:

    data/underlying/<TICKER>.parquet

with a ``date`` ``DatetimeIndex`` and columns ``open, close`` (sorted
ascending, trading days only).

Run as a module::

    python -m nysa_risk.extraction.underlying
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from ..config import AssetUniverse, load_universe

LOGGER = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "underlying"

# Warn when the effective history is materially shorter than requested.
SHORT_HISTORY_TOLERANCE = 0.9

# Retry policy for transient network failures.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class DownloadResult:
    ticker: str
    path: Path
    rows: int
    first_date: date | None
    last_date: date | None
    effective_years: float
    short_history: bool


def _yf_download(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Thin wrapper around ``yfinance.download`` — patch this in tests."""
    import yfinance as yf

    return yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )


def _download_with_retries(
    ticker: str,
    start: date,
    end: date,
    max_retries: int,
    backoff_seconds: float,
    downloader: Callable[[str, date, date], pd.DataFrame],
    sleeper: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = downloader(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 — yfinance surfaces many error types
            last_exc = exc
            wait = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "yfinance download failed for %s (attempt %d/%d): %s — retrying in %.1fs",
                ticker,
                attempt,
                max_retries,
                exc,
                wait,
            )
            if attempt < max_retries:
                sleeper(wait)
            continue
        if df is None or df.empty:
            # Yahoo returns an empty frame for unknown tickers or transient
            # outages. Retry a couple of times before giving up.
            last_exc = RuntimeError(f"empty response for {ticker}")
            wait = backoff_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "yfinance returned no rows for %s (attempt %d/%d) — retrying in %.1fs",
                ticker,
                attempt,
                max_retries,
                wait,
            )
            if attempt < max_retries:
                sleeper(wait)
            continue
        return df
    assert last_exc is not None
    raise last_exc


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reduce a yfinance frame to ``date, open, close`` sorted ascending."""
    # yfinance may return a MultiIndex on columns when multi_level_index=True;
    # we requested False, but be defensive in case a caller patches the flag.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=-1, drop_level=True)

    cols = {c.lower(): c for c in df.columns}
    if "open" not in cols or "close" not in cols:
        raise ValueError(
            f"{ticker}: response missing Open/Close columns (got {list(df.columns)})"
        )

    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df[cols["open"]], errors="coerce"),
            "close": pd.to_numeric(df[cols["close"]], errors="coerce"),
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None).normalize())
    out = out.dropna(how="any").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "date"
    return out


def _effective_years(first: date, last: date) -> float:
    return max((last - first).days / 365.25, 0.0)


def download_asset(
    ticker: str,
    years: int,
    out_dir: Path,
    *,
    end: date | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    downloader: Callable[[str, date, date], pd.DataFrame] = _yf_download,
    sleeper: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download one ticker's daily Open/Close and persist it as parquet.

    Returns a ``DownloadResult`` describing the effective coverage. Tickers
    with less history than requested (e.g. CRCL, listed 2025-06) are still
    written — a warning is logged with the effective depth.
    """
    end_date = end or datetime.now(timezone.utc).date()
    # +1 day: yfinance treats ``end`` as exclusive.
    start_date = end_date - timedelta(days=int(round(years * 365.25))) - timedelta(days=1)

    raw = _download_with_retries(
        ticker,
        start_date,
        end_date + timedelta(days=1),
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        downloader=downloader,
        sleeper=sleeper,
    )
    frame = _normalize(raw, ticker)
    if frame.empty:
        raise RuntimeError(f"{ticker}: no usable rows after normalization")

    first = frame.index[0].date()
    last = frame.index[-1].date()
    effective = _effective_years(first, last)
    short = effective < years * SHORT_HISTORY_TOLERANCE
    if short:
        LOGGER.warning(
            "%s: shorter history than requested — got %.2fy (first %s, last %s), requested %dy",
            ticker,
            effective,
            first.isoformat(),
            last.isoformat(),
            years,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.parquet"
    frame.to_parquet(path, index=True)
    LOGGER.info(
        "%s: wrote %d rows (%.2fy, %s → %s) to %s",
        ticker,
        len(frame),
        effective,
        first.isoformat(),
        last.isoformat(),
        path,
    )

    return DownloadResult(
        ticker=ticker,
        path=path,
        rows=len(frame),
        first_date=first,
        last_date=last,
        effective_years=effective,
        short_history=short,
    )


def _unique_tickers(universe: AssetUniverse) -> list[str]:
    """Union of collateral ``underlying_ticker`` and borrowable ``price_source``, deduped, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for c in universe.collaterals:
        if c.underlying_ticker not in seen:
            seen.add(c.underlying_ticker)
            out.append(c.underlying_ticker)
    for b in universe.borrowables:
        if b.price_source not in seen:
            seen.add(b.price_source)
            out.append(b.price_source)
    return out


def run(
    universe: AssetUniverse | None = None,
    out_dir: Path | None = None,
    *,
    end: date | None = None,
    tickers: Iterable[str] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    downloader: Callable[[str, date, date], pd.DataFrame] = _yf_download,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[DownloadResult]:
    """Download every unique ticker in the universe."""
    universe = universe or load_universe()
    out_dir = out_dir or DEFAULT_OUT_DIR
    target_tickers = list(tickers) if tickers is not None else _unique_tickers(universe)
    years = universe.meta.price_history_years

    results: list[DownloadResult] = []
    for ticker in target_tickers:
        try:
            results.append(
                download_asset(
                    ticker,
                    years=years,
                    out_dir=out_dir,
                    end=end,
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                    downloader=downloader,
                    sleeper=sleeper,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad ticker mustn't kill the batch
            LOGGER.error("%s: giving up after retries — %s", ticker, exc)
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.extraction.underlying",
        description="Download underlying daily Open/Close history for the Nysa asset universe.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory for parquet files")
    p.add_argument("--config", type=Path, default=None, help="override path to assets.yaml")
    p.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="restrict to one or more tickers (repeatable); default: all from config",
    )
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe = load_universe(args.config) if args.config else load_universe()
    results = run(
        universe=universe,
        out_dir=args.out_dir,
        tickers=args.tickers,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )
    short = [r.ticker for r in results if r.short_history]
    if short:
        LOGGER.warning("short-history tickers: %s", ", ".join(short))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
