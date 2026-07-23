"""Tests for ``nysa_risk.report`` — the published parameter table.

A synthetic calibration output with hand-picked values drives all three
formats. AAPLon's e-mode LT beats std by 14pp (> the 5pp rule → shown,
using the LT-pass FINAL LT, not the formula); SPYon's beats it by only
4pp (→ both e-mode columns dashed together).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nysa_risk import report as rp
from nysa_risk.calibrate import (
    SCENARIOS,
    CalibratedRow,
    LTDecision,
    LTVDecision,
    RowMetrics,
)
from nysa_risk.parameters.lt import EMODE_DASH
from nysa_risk.report import (
    ReportRow,
    build_report_rows,
    format_cli,
    to_csv_frame,
    to_markdown,
)

EMODE, STD = SCENARIOS  # ("e-mode" / USDT / emode:stable), ("standard" / BNB / base)


def _crow(coll: str, scenario, lt: float, ltv: float, *,
          final_lt: float | None = None, binding: str = "formula",
          flags: tuple[str, ...] = ()) -> CalibratedRow:
    lt_dec = None
    if final_lt is not None:
        lt_dec = LTDecision(formula_lt=lt, final_lt=final_lt,
                            max_excess_before=0.08, max_excess_after=0.06,
                            floor_capped=False)
    return CalibratedRow(
        collateral=coll, scenario=scenario, lt=lt, formula_ltv=ltv,
        decision=LTVDecision(
            final_ltv=ltv, binding=binding,
            metrics=RowMetrics(share30=0.12, gap_rate=0.01,
                               uncond_bad_debt=0.002, excess=()),
            effective_years=8.5, flags=flags,
        ),
        lt_decision=lt_dec,
    )


def _synthetic() -> list[CalibratedRow]:
    return [
        # AAPLon: e-mode formula LT 0.92 CUT to 0.90 by the LT pass;
        # advantage vs std = 0.90 − 0.76 = 14pp > 5pp → published.
        _crow("AAPLon", STD, 0.76, 0.54, binding="liq30"),
        _crow("AAPLon", EMODE, 0.92, 0.82, final_lt=0.90, binding="bad-debt",
              flags=("LT REVIEW (max excess 6.5%, p95 6.5%)",)),
        # SPYon: e-mode advantage = 0.80 − 0.76 = 4pp < 5pp → dashed.
        _crow("SPYon", STD, 0.76, 0.55),
        _crow("SPYon", EMODE, 0.80, 0.75, binding="ceiling"),
    ]


def _rows() -> list[ReportRow]:
    return build_report_rows(_synthetic(), emode_min_advantage=0.05)


# ---------------------------------------------------------------------------
# Row building — dash rule on FINAL LTs
# ---------------------------------------------------------------------------


def test_dash_rule_uses_final_lt_and_min_advantage() -> None:
    rows = {r.collateral: r for r in _rows()}
    assert rows["AAPLon"].show_emode is True
    assert rows["SPYon"].show_emode is False
    # The rule must compare the CUT LT, not the formula, and the inequality
    # is strict (as in the LT CLI). Binary-exact values so the advantage
    # lands exactly on the threshold without rounding: 0.8125 − 0.75 = 0.0625.
    edge = build_report_rows([
        _crow("EDGEon", STD, 0.75, 0.5),
        _crow("EDGEon", EMODE, 0.875, 0.75, final_lt=0.8125),
    ], emode_min_advantage=0.0625)
    assert edge[0].show_emode is False   # exactly at threshold → dash
    # ... while the formula LT (0.875) would have cleared it — proving the
    # rule reads the LT-pass output.
    edge2 = build_report_rows([
        _crow("EDGEon", STD, 0.75, 0.5),
        _crow("EDGEon", EMODE, 0.875, 0.75),  # no LT pass → formula stands
    ], emode_min_advantage=0.0625)
    assert edge2[0].show_emode is True


# ---------------------------------------------------------------------------
# (a) CLI table
# ---------------------------------------------------------------------------


def test_cli_table_matches_expected_values() -> None:
    text = format_cli(_rows())
    header = text.splitlines()[0]
    for col in ("Collateral", "LT std (%)", "LTV std (%)",
                "LT e-mode (%)", "LTV e-mode (%)"):
        assert col in header
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "76.00" in aapl and "54.00" in aapl
    assert "90.00" in aapl and "82.00" in aapl   # final (cut) LT, not 92.00
    assert "92.00" not in aapl
    assert EMODE_DASH not in aapl
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert "76.00" in spy and "55.00" in spy
    assert spy.count(EMODE_DASH) == 2            # both e-mode columns dashed together
    assert "80.00" not in spy and "75.00" not in spy


def test_cli_full_appends_audit_columns_even_for_dashed_rows() -> None:
    text = format_cli(_rows(), full=True)
    header = text.splitlines()[0]
    for col in ("binding std", "binding e-mode", "≤30d std (%)",
                "uncond BD e-mode (%)", "max exc std (%)", "years", "flags"):
        assert col in header
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "liq30" in aapl and "bad-debt" in aapl
    assert "LT REVIEW" in aapl
    assert "8.50" in aapl
    # SPYon's e-mode is dashed in the published columns, but its audit
    # cells still show the raw engine values.
    spy = next(l for l in text.splitlines() if l.startswith("SPYon"))
    assert "ceiling" in spy


# ---------------------------------------------------------------------------
# (b) Markdown
# ---------------------------------------------------------------------------


def test_markdown_header_and_values() -> None:
    md = to_markdown(_rows(), generated="2026-07-23", config_hash="abc123def456")
    assert "Generated 2026-07-23" in md
    assert "assets.yaml@abc123def456" in md
    assert "engine-calibrated" in md
    assert "| Collateral | LT std (%) |" in md
    aapl = next(l for l in md.splitlines() if l.startswith("| AAPLon"))
    assert "90.00" in aapl and "82.00" in aapl
    spy = next(l for l in md.splitlines() if l.startswith("| SPYon"))
    assert spy.count(EMODE_DASH) == 2


# ---------------------------------------------------------------------------
# (c) CSV frame
# ---------------------------------------------------------------------------


def test_csv_frame_values_and_empty_dashes() -> None:
    df = to_csv_frame(_rows()).set_index("collateral")
    assert df.loc["AAPLon", "lt_std_pct"] == pytest.approx(76.0)
    assert df.loc["AAPLon", "lt_emode_pct"] == pytest.approx(90.0)
    assert df.loc["AAPLon", "ltv_emode_pct"] == pytest.approx(82.0)
    # Dashed publication → empty cells, not numbers.
    assert pd.isna(df.loc["SPYon", "lt_emode_pct"])
    assert pd.isna(df.loc["SPYon", "ltv_emode_pct"])
    assert df.loc["SPYon", "lt_std_pct"] == pytest.approx(76.0)


def test_csv_frame_full_traceability_columns() -> None:
    df = to_csv_frame(_rows(), full=True).set_index("collateral")
    assert df.loc["AAPLon", "binding_std"] == "liq30"
    assert df.loc["AAPLon", "binding_emode"] == "bad-debt"
    assert df.loc["SPYon", "binding_emode"] == "ceiling"   # audit survives the dash
    assert df.loc["AAPLon", "liq30_std_pct"] == pytest.approx(12.0)
    assert df.loc["AAPLon", "uncond_bd_emode_pct"] == pytest.approx(0.2)
    assert df.loc["AAPLon", "effective_years"] == pytest.approx(8.5)
    assert "LT REVIEW" in df.loc["AAPLon", "flags"]


# ---------------------------------------------------------------------------
# main — writes both files with the config hash of the loaded config
# ---------------------------------------------------------------------------


def test_main_writes_md_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "assets.yaml"
    cfg.write_text("fake config for hashing\n", encoding="utf-8")
    expected_hash = rp._config_hash(cfg)

    class _Uni:
        class calibration:
            emode_min_advantage = 0.05

    monkeypatch.setattr(rp, "load_universe", lambda *a, **k: _Uni())
    monkeypatch.setattr(rp, "run_calibrate", lambda **kw: _synthetic())

    out_dir = tmp_path / "report"
    rc = rp.main(["--config", str(cfg), "--out-dir", str(out_dir),
                  "--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0

    md = (out_dir / "parameters.md").read_text(encoding="utf-8")
    assert f"assets.yaml@{expected_hash}" in md
    assert "engine-calibrated" in md
    df = pd.read_csv(out_dir / "parameters.csv")
    assert list(df["collateral"]) == ["AAPLon", "SPYon"]
    assert df.set_index("collateral").loc["AAPLon", "lt_emode_pct"] == pytest.approx(90.0)

    out = capsys.readouterr().out
    assert "AAPLon" in out and "wrote" in out
