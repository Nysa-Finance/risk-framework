"""Final parameter report — the deployable LT/LTV table (§7 Reporting).

Consumes the unified calibration engine (:mod:`nysa_risk.calibrate`:
severity-capped final LT, constraint-calibrated final LTV per
collateral × scenario) and publishes one row per collateral:

    Collateral | LT std (%) | LTV std (%) | LT e-mode (%) | LTV e-mode (%)

All values are the **final** engine outputs — never the raw formula
priors. The e-mode dash rule from the LT CLI is kept: the e-mode
columns are shown only when the final e-mode LT exceeds the final
standard LT by more than ``calibration.emode_min_advantage`` (absolute
fraction of collateral value); below that bar both e-mode columns print
``—`` together — enabling E-Mode isn't worth the operational
complexity for that collateral.

``--full`` appends the audit (traceability) columns per scenario —
binding constraint, %≤30d, unconditional bad debt, max excess,
effective years, flags. These always show raw engine values, even for
rows whose *published* e-mode columns are dashed: the dash rule governs
publication, not auditability.

Outputs
-------
(a) printed CLI table — ``python -m nysa_risk.report``
(b) ``data/report/parameters.md`` — markdown table; the header notes
    the generation date, the config hash (sha256 of ``assets.yaml``,
    first 12 hex chars) and that values are engine-calibrated
(c) ``data/report/parameters.csv`` — same columns, machine-readable
    (dashes become empty cells)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backtest import _render_table
from .calibrate import CalibratedRow, run_calibrate
from .config import DEFAULT_CONFIG_PATH, AssetUniverse, load_universe
from .parameters.lt import EMODE_DASH
from .volatility import DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "report"

BASE_HEADERS = ["Collateral", "LT std (%)", "LTV std (%)",
                "LT e-mode (%)", "LTV e-mode (%)"]
FULL_HEADERS = ["binding std", "≤30d std (%)", "uncond BD std (%)", "max exc std (%)",
                "binding e-mode", "≤30d e-mode (%)", "uncond BD e-mode (%)", "max exc e-mode (%)",
                "years", "flags"]


@dataclass(frozen=True, slots=True)
class ReportRow:
    collateral: str
    std: CalibratedRow | None
    emode: CalibratedRow | None
    show_emode: bool               # e-mode advantage rule verdict (publication only)


def _final_lt(r: CalibratedRow) -> float:
    return r.lt_decision.final_lt if r.lt_decision is not None else r.lt


def _final_ltv(r: CalibratedRow) -> float:
    return r.decision.final_ltv


def build_report_rows(
    results: Iterable[CalibratedRow],
    emode_min_advantage: float,
) -> list[ReportRow]:
    """Fold engine rows into one publication row per collateral.

    ``show_emode`` applies the LT CLI's rule on the **final** LTs: the
    e-mode columns are published only when
    ``final e-mode LT − final std LT > emode_min_advantage``.
    """
    by_coll: dict[str, dict[str, CalibratedRow]] = {}
    for r in results:
        kind = "emode" if r.scenario.param_set.startswith("emode") else "std"
        if kind in by_coll.setdefault(r.collateral, {}):
            LOGGER.warning("%s: duplicate %s row — keeping the first", r.collateral, kind)
            continue
        by_coll[r.collateral][kind] = r

    rows: list[ReportRow] = []
    for coll in sorted(by_coll):
        std = by_coll[coll].get("std")
        emode = by_coll[coll].get("emode")
        show = (
            std is not None and emode is not None
            and (_final_lt(emode) - _final_lt(std)) > emode_min_advantage
        )
        rows.append(ReportRow(collateral=coll, std=std, emode=emode, show_emode=show))
    return rows


# ---------------------------------------------------------------------------
# Cell builders (shared by CLI / markdown / CSV)
# ---------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 2) -> str:
    return EMODE_DASH if x is None else f"{x * 100:.{digits}f}"


def _trace_cells(r: CalibratedRow | None) -> list[str]:
    if r is None:
        return [EMODE_DASH] * 4
    m = r.decision.metrics
    return [
        r.decision.binding,
        EMODE_DASH if m.share30 is None else f"{m.share30 * 100:.1f}",
        EMODE_DASH if m.uncond_bad_debt is None else f"{m.uncond_bad_debt * 100:.2f}",
        f"{m.max_excess * 100:.2f}",
    ]


def _years_cell(row: ReportRow) -> str:
    years = [r.decision.effective_years for r in (row.std, row.emode) if r is not None]
    return EMODE_DASH if not years else f"{min(years):.2f}"


def _flags_cell(row: ReportRow) -> str:
    parts = []
    for label, r in (("std", row.std), ("e-mode", row.emode)):
        if r is not None and r.decision.flags:
            parts.append(f"{label}: {', '.join(r.decision.flags)}")
    return "; ".join(parts) if parts else EMODE_DASH


def _row_cells(row: ReportRow, full: bool) -> list[str]:
    std, emode = row.std, row.emode
    emode_published = emode is not None and row.show_emode
    cells = [
        row.collateral,
        _pct(_final_lt(std) if std else None),
        _pct(_final_ltv(std) if std else None),
        _pct(_final_lt(emode) if emode_published else None),
        _pct(_final_ltv(emode) if emode_published else None),
    ]
    if full:
        cells += _trace_cells(std) + _trace_cells(emode)
        cells += [_years_cell(row), _flags_cell(row)]
    return cells


def _headers(full: bool) -> list[str]:
    return BASE_HEADERS + FULL_HEADERS if full else list(BASE_HEADERS)


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


def format_cli(rows: Iterable[ReportRow], *, full: bool = False) -> str:
    body = [_row_cells(r, full) for r in rows]
    if not body:
        return "(no collaterals)"
    return _render_table(_headers(full), body, n_left=1)


def to_markdown(
    rows: Iterable[ReportRow],
    *,
    full: bool = False,
    generated: str,
    config_hash: str,
) -> str:
    headers = _headers(full)
    lines = [
        "# Nysa Risk — Final Lending Parameters",
        "",
        f"Generated {generated} · config `assets.yaml@{config_hash}` · "
        "values are engine-calibrated (`python -m nysa_risk.calibrate`: "
        "severity-capped LT, constraint-calibrated LTV).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_row_cells(r, full)) + " |")
    return "\n".join(lines) + "\n"


def to_csv_frame(rows: Iterable[ReportRow], *, full: bool = False) -> pd.DataFrame:
    """Machine-readable frame; published dashes become empty (None) cells."""
    def num(x: float | None) -> float | None:
        return None if x is None else round(x * 100, 4)

    records = []
    for row in rows:
        std, emode = row.std, row.emode
        emode_published = emode is not None and row.show_emode
        rec: dict[str, object] = {
            "collateral": row.collateral,
            "lt_std_pct": num(_final_lt(std) if std else None),
            "ltv_std_pct": num(_final_ltv(std) if std else None),
            "lt_emode_pct": num(_final_lt(emode) if emode_published else None),
            "ltv_emode_pct": num(_final_ltv(emode) if emode_published else None),
        }
        if full:
            for label, r in (("std", std), ("emode", emode)):
                m = r.decision.metrics if r is not None else None
                rec[f"binding_{label}"] = None if r is None else r.decision.binding
                rec[f"liq30_{label}_pct"] = None if m is None else num(m.share30)
                rec[f"uncond_bd_{label}_pct"] = None if m is None else num(m.uncond_bad_debt)
                rec[f"max_excess_{label}_pct"] = None if m is None else num(m.max_excess)
            years = [r.decision.effective_years for r in (std, emode) if r is not None]
            rec["effective_years"] = round(min(years), 2) if years else None
            flags = _flags_cell(row)
            rec["flags"] = "" if flags == EMODE_DASH else flags
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _config_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nysa_risk.report",
        description="Publish the final calibrated LT/LTV parameter table "
                    "(CLI + data/report/parameters.{md,csv}).",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--full", action="store_true",
                   help="append the audit columns (binding, %%≤30d, uncond BD, max excess, years, flags)")
    p.add_argument("--log-level", default="WARNING")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    universe: AssetUniverse = load_universe(args.config) if args.config else load_universe()
    results = run_calibrate(universe=universe, data_dir=args.data_dir)
    rows = build_report_rows(results, universe.calibration.emode_min_advantage)

    print(format_cli(rows, full=args.full))

    cfg_path = args.config if args.config else DEFAULT_CONFIG_PATH
    md = to_markdown(
        rows, full=args.full,
        generated=date.today().isoformat(),
        config_hash=_config_hash(cfg_path),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "parameters.md"
    csv_path = args.out_dir / "parameters.csv"
    md_path.write_text(md, encoding="utf-8")
    to_csv_frame(rows, full=args.full).to_csv(csv_path, index=False)
    print(f"\nwrote {md_path} and {csv_path}")
    return 0 if rows else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
