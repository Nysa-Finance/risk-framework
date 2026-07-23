"""Tests for ``nysa_risk.calibrate`` (unified LTV calibration engine).

Each constraint gets a synthetic path where it is the *individually*
binding one, with hand-computable cutoffs. ``LT = 0.8`` throughout, so
the liquidation-threshold ratio is ``LTV/0.8`` and the bad-debt buffer
from a trigger is ``1 − LT = 20 %``. Test-level constraint values are
chosen to keep fixtures small (they are engine parameters, sourced from
config in production).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nysa_risk import calibrate as ce
from nysa_risk.calibrate import (
    BIND_BAD_DEBT,
    BIND_CEILING,
    BIND_FORMULA,
    BIND_GAP,
    BIND_LIQ30,
    FLAG_SHORT_HISTORY,
    SCENARIOS,
    CalibratedRow,
    LTVDecision,
    RowMetrics,
    calibrate_row,
    format_table,
)

LT = 0.8
BAND = (0.10, 0.15)


def _stream(prices: list[tuple[float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    """Interleaved open/close stream over consecutive days."""
    dates = pd.date_range(start, periods=len(prices), freq="D")
    ts, px, cl = [], [], []
    for d, (o, c) in zip(dates, prices):
        ts += [d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=16)]
        px += [o, c]
        cl += [False, True]
    return pd.DataFrame({"price": px, "is_close": cl}, index=pd.DatetimeIndex(ts))


def _calibrate(prices: list[tuple[float, float]], formula_ltv: float, **kw) -> LTVDecision:
    kw.setdefault("lt", LT)
    kw.setdefault("band", BAND)
    kw.setdefault("t_user_days", 3.0)
    kw.setdefault("minimum_gap", 0.02)
    kw.setdefault("bd_bound", 0.004)
    kw.setdefault("min_years", 0.5)
    kw.setdefault("severity_threshold", 0.05)
    return calibrate_row(_stream(prices), formula_ltv=formula_ltv, **kw)


FLAT = (100.0, 100.0)


# ---------------------------------------------------------------------------
# Constraint (1): the liq-30d band binds
# ---------------------------------------------------------------------------


def test_liq30_band_binds() -> None:
    """Two one-day dips: 75 @ day 61 (30/200 = 15 % once LTV ≥ 0.60) and
    90 @ day 100 (another 30 openings once LTV ≥ 0.72, pushing the share
    to 30 % > 15 %). Formula 0.50 is under the band with enough history,
    so the engine raises — and must stop just below 0.72."""
    prices = ([FLAT] * 60 + [(75.0, 100.0)] + [FLAT] * 38
              + [(90.0, 100.0)] + [FLAT] * 100)
    d = _calibrate(prices, formula_ltv=0.50)
    assert d.binding == BIND_LIQ30
    assert d.final_ltv == pytest.approx(0.72, abs=0.005)
    assert d.metrics.share30 == pytest.approx(30 / 200, abs=1e-12)   # inside the band
    assert BAND[0] <= d.metrics.share30 <= BAND[1]
    assert not d.flags


# ---------------------------------------------------------------------------
# Constraint (2): the reaction (gap) bound binds
# ---------------------------------------------------------------------------


def test_gap_bound_binds() -> None:
    """Steady 2 %/day decline: every opening triggers after k days where
    0.98^k ≤ LTV/LT. At LTV ≥ 0.8·0.98³ ≈ 0.753 the trigger lands within
    3 days for every opening → gap rate ~100 % > 10 %. The liq-30d band is
    disabled (hi = 0.99) so the gap bound is the unique binder."""
    prices = [(100.0 * 0.98 ** i,) * 2 for i in range(60)]
    d = _calibrate(prices, formula_ltv=0.77, band=(0.01, 0.99), bd_bound=0.99)
    assert d.binding == BIND_GAP
    assert d.final_ltv == pytest.approx(0.8 * 0.98 ** 3, abs=0.005)
    assert d.metrics.gap_rate is not None and d.metrics.gap_rate <= 0.10


# ---------------------------------------------------------------------------
# Constraint (3): the unconditional bad-debt bound binds
# ---------------------------------------------------------------------------


def test_bad_debt_bound_binds() -> None:
    """Crash close 70 → next open 40 (drop 43 % > 20 % buffer). At
    LTV ≥ 0.56 the trigger lands on the day-51 close and every one of the
    50 pre-crash openings realizes through the gap → uncond BD ≈ 45 %.
    Below 0.56 the trigger is the day-52 open itself (realization drop 0,
    no bad debt). The engine must stop just below 0.56."""
    prices = [FLAT] * 50 + [(100.0, 70.0)] + [(40.0, 40.0)] * 149
    d = _calibrate(prices, formula_ltv=0.70, band=(0.01, 0.99))
    assert d.binding == BIND_BAD_DEBT
    assert d.final_ltv == pytest.approx(0.56, abs=0.005)
    assert d.metrics.uncond_bad_debt == 0.0   # no bad debt at the final LTV


def test_prior_anchor_survives_non_monotone_bad_debt() -> None:
    """Bad debt is NOT monotone in LTV: it depends on where in the crash the
    trigger lands. The ladder 100 → 90 → 60/50 → 45/30 → 17 makes an
    entry-100 opening realize benignly when it triggers at 60 or 50
    (drops ≤ 17 %) but gap through the buffer when it triggers at 90
    (→ 60, 33 %), at 45 (→ 30, 33 %) or at 30 (→ 17, 43 %). In
    threshold-ratio terms the infeasible pockets are r ∈ [0.3, 0.5) and
    r ≥ 0.9 — i.e. LTV ∈ [0.24, 0.40) and [0.72, ceiling] — with the
    formula prior (0.55, r ≈ 0.69) sitting feasible between them. A naive
    global bisection falls through the lower pocket and returns ~0.24,
    far below the feasible prior; the anchored grid search must instead
    climb to the top edge (~0.72)."""
    prices = ([FLAT] * 48 + [(100.0, 90.0)] + [(60.0, 50.0)] + [(45.0, 30.0)]
              + [(17.0, 17.0)] * 449)
    d = _calibrate(prices, formula_ltv=0.55, band=(0.50, 0.99))
    assert d.final_ltv >= 0.55                       # never below a feasible prior
    assert d.final_ltv == pytest.approx(0.72, abs=0.005)
    assert d.binding == BIND_BAD_DEBT
    # The one stray event (the entry-50 opening riding 30 → 17) stays under
    # the 0.4 % bound at the final point — verified feasible, not zero.
    assert d.metrics.uncond_bad_debt is not None
    assert 0.0 < d.metrics.uncond_bad_debt <= 0.004


# ---------------------------------------------------------------------------
# Constraint (4): the LT − minimum_gap ceiling binds
# ---------------------------------------------------------------------------


def test_ceiling_binds_when_raise_unconstrained() -> None:
    """Flat path: nothing ever liquidates, so a permitted raise runs into
    the LT − minimum_gap ceiling."""
    d = _calibrate([FLAT] * 300, formula_ltv=0.40)
    assert d.binding == BIND_CEILING
    assert d.final_ltv == pytest.approx(LT - 0.02, abs=1e-12)
    assert not d.flags


# ---------------------------------------------------------------------------
# Short-history asymmetry guard and the formula cap
# ---------------------------------------------------------------------------


def test_short_history_guard_blocks_raise() -> None:
    """Same flat path, but min_years above the stream's ~0.8y of history:
    the warranted raise is blocked, the formula stays, and the row is
    flagged."""
    d = _calibrate([FLAT] * 300, formula_ltv=0.40, min_years=3.0)
    assert d.binding == BIND_FORMULA
    assert d.final_ltv == 0.40
    assert FLAG_SHORT_HISTORY in d.flags


def test_formula_is_cap_when_share_already_in_band() -> None:
    """Single dip (75 @ day 61 of 200): share30 = 15 % at LTV 0.65 — inside
    the band, so no raise is attempted and the formula is kept."""
    prices = [FLAT] * 60 + [(75.0, 100.0)] + [FLAT] * 139
    d = _calibrate(prices, formula_ltv=0.65)
    assert d.binding == BIND_FORMULA
    assert d.final_ltv == 0.65
    assert FLAG_SHORT_HISTORY not in d.flags


# ---------------------------------------------------------------------------
# Severity: LT REVIEW flag, not an LTV lever
# ---------------------------------------------------------------------------


def test_lt_review_flag_does_not_lower_ltv() -> None:
    """One opening rides a 100 → 74 → 55 crash on day 2: excess beyond the
    buffer = 1 − 55/74 − 0.20 ≈ 5.7 % > 5 % threshold. With 400 days the
    unconditional rate (1 event / ~310 decidable openings ≈ 0.32 %) stays
    under the 0.4 % bound — so the LTV must still rise to the ceiling and
    the severity concern surfaces only as an LT REVIEW flag."""
    prices = [FLAT] + [(74.0, 55.0)] + [(55.0, 55.0)] * 398
    d = _calibrate(prices, formula_ltv=0.70)
    assert d.binding == BIND_CEILING
    assert d.final_ltv == pytest.approx(LT - 0.02, abs=1e-12)
    assert d.metrics.uncond_bad_debt is not None
    assert d.metrics.uncond_bad_debt <= 0.004
    assert d.metrics.max_excess == pytest.approx(1 - 55.0 / 74.0 - 0.2, abs=1e-12)
    assert any(f.startswith("LT REVIEW") for f in d.flags)


# ---------------------------------------------------------------------------
# LT severity pass — max_loss_given_bad_debt cap
# ---------------------------------------------------------------------------

CAP = 0.065


def _lt_pass(prices: list[tuple[float, float]], **kw):
    kw.setdefault("formula_lt", LT)
    kw.setdefault("formula_ltv", 0.40)
    kw.setdefault("severity_cap", CAP)
    kw.setdefault("t_user_days", 3.0)
    kw.setdefault("minimum_gap", 0.02)
    return ce.calibrate_lt(_stream(prices), **kw)


def test_lt_pass_cuts_lt_on_deep_gap() -> None:
    """Trigger at 50, realization gaps to 35: drop 30 %, excess at LT 0.8 is
    exactly 10 % > the 6.5 % cap. The drop is fixed, so excess = 0.30 − (1 − LT)
    and the pass must cut LT to ~0.765 where the excess meets the cap."""
    prices = [FLAT] * 50 + [(100.0, 50.0)] + [(35.0, 35.0)] * 69
    d = _lt_pass(prices)
    assert d.max_excess_before == pytest.approx(0.10, abs=1e-12)
    assert d.final_lt < LT
    assert d.final_lt == pytest.approx(0.765, abs=0.005)
    assert d.max_excess_after <= CAP
    assert not d.floor_capped


def test_lt_pass_untouched_when_cap_satisfied() -> None:
    """Realization at 38: drop 24 %, excess 4 % ≤ 6.5 % — LT must not move."""
    prices = [FLAT] * 50 + [(100.0, 50.0)] + [(38.0, 38.0)] * 69
    d = _lt_pass(prices)
    assert d.final_lt == LT
    assert d.max_excess_before == pytest.approx(0.04, abs=1e-12)
    assert d.max_excess_after == d.max_excess_before
    assert not d.floor_capped


def test_lt_pass_floor_capped_on_extreme_gap() -> None:
    """Realization at 30: drop 40 %, excess 20 %. Even the floor (LT − 10pp,
    buffer 30 %) leaves 10 % > cap → floor-capped, flag responsibility moves
    to the wrapper."""
    prices = [FLAT] * 50 + [(100.0, 50.0)] + [(30.0, 30.0)] * 69
    d = _lt_pass(prices)
    assert d.floor_capped
    assert d.final_lt == pytest.approx(LT - 0.10, abs=1e-12)
    assert d.max_excess_after == pytest.approx(0.10, abs=1e-12)


def test_wrapper_reruns_ltv_at_cut_lt_and_shifts_prior() -> None:
    """After a cut the LTV pass runs at the new LT with the prior shifted by
    ΔLT (formula LTV = LT − G, G independent of LT)."""
    prices = [FLAT] * 50 + [(100.0, 50.0)] + [(35.0, 35.0)] * 69
    lt_dec, decision = ce.calibrate_row_with_lt(
        _stream(prices), formula_lt=LT, formula_ltv=0.40, band=(0.01, 0.99),
        t_user_days=3.0, minimum_gap=0.02, bd_bound=0.99, min_years=0.5,
        severity_threshold=0.9, severity_cap=CAP,
    )
    assert lt_dec.final_lt == pytest.approx(0.765, abs=0.005)
    # The LTV decision respects the CUT LT's ceiling, not the formula LT's.
    assert decision.final_ltv <= lt_dec.final_lt - 0.02 + 1e-9
    assert not any(f.startswith("LT floor") for f in decision.flags)


def test_wrapper_flags_floor_capped_rows() -> None:
    prices = [FLAT] * 50 + [(100.0, 50.0)] + [(30.0, 30.0)] * 69
    lt_dec, decision = ce.calibrate_row_with_lt(
        _stream(prices), formula_lt=LT, formula_ltv=0.40, band=(0.01, 0.99),
        t_user_days=3.0, minimum_gap=0.02, bd_bound=0.99, min_years=0.5,
        severity_threshold=0.9, severity_cap=CAP,
    )
    assert lt_dec.floor_capped
    assert any(f.startswith("LT floor") for f in decision.flags)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_calibrate_row_rejects_bad_inputs() -> None:
    prices = [FLAT] * 5
    with pytest.raises(ValueError):
        _calibrate(prices, formula_ltv=0.9)                    # formula ≥ lt... 0.9 > 0.8
    with pytest.raises(ValueError):
        _calibrate(prices, formula_ltv=0.4, band=(0.15, 0.10))  # inverted band
    with pytest.raises(ValueError):
        _calibrate(prices, formula_ltv=0.4, minimum_gap=0.9)   # gap ≥ lt
    with pytest.raises(ValueError):
        _calibrate(prices, formula_ltv=0.4, bd_bound=0.0)      # degenerate bound


# ---------------------------------------------------------------------------
# Output table and CLI
# ---------------------------------------------------------------------------


def _row(collateral: str, scenario, formula: float, final: float,
         binding: str, flags: tuple[str, ...] = ()) -> CalibratedRow:
    return CalibratedRow(
        collateral=collateral, scenario=scenario, lt=LT, formula_ltv=formula,
        decision=LTVDecision(
            final_ltv=final, binding=binding,
            metrics=RowMetrics(share30=0.12, gap_rate=0.01,
                               uncond_bad_debt=0.001, excess=()),
            effective_years=8.0, flags=flags,
        ),
    )


def test_format_table_columns_and_flags() -> None:
    rows = [
        _row("AAPLon", SCENARIOS[0], 0.845, 0.822, BIND_LIQ30),
        _row("CRCLon", SCENARIOS[1], 0.258, 0.258, BIND_FORMULA,
             flags=(FLAG_SHORT_HISTORY,)),
    ]
    text = format_table(rows)
    header = text.splitlines()[0]
    for col in ("formula LT (%)", "final LT (%)", "ΔLT (pp)",
                "formula LTV (%)", "final LTV (%)", "binding",
                "≤30d (%)", "uncond BD (%)", "exc pre (%)", "exc post (%)", "flags"):
        assert col in header
    aapl = next(l for l in text.splitlines() if l.startswith("AAPLon"))
    assert "84.50" in aapl and "82.20" in aapl and BIND_LIQ30 in aapl
    # No LT pass on these rows → formula LT shown as final, excess dashed.
    assert "80.00" in aapl and "0.00" in aapl
    crcl = next(l for l in text.splitlines() if l.startswith("CRCLon"))
    assert FLAG_SHORT_HISTORY in crcl


def test_format_table_shows_lt_pass_columns() -> None:
    row = CalibratedRow(
        collateral="INTCon", scenario=SCENARIOS[0], lt=LT, formula_ltv=0.74,
        decision=LTVDecision(
            final_ltv=0.70, binding=BIND_FORMULA,
            metrics=RowMetrics(share30=0.12, gap_rate=0.01,
                               uncond_bad_debt=0.001, excess=()),
            effective_years=8.0, flags=(),
        ),
        lt_decision=ce.LTDecision(formula_lt=LT, final_lt=0.76,
                                  max_excess_before=0.109, max_excess_after=0.064,
                                  floor_capped=False),
    )
    line = next(l for l in format_table([row]).splitlines() if l.startswith("INTCon"))
    assert "80.00" in line     # formula LT
    assert "76.00" in line     # final LT
    assert "-4.00" in line     # ΔLT
    assert "10.90" in line     # exc pre
    assert "6.40" in line      # exc post


def test_results_frame_columns_and_values() -> None:
    rows = [_row("AAPLon", SCENARIOS[0], 0.845, 0.822, BIND_LIQ30,
                 flags=(FLAG_SHORT_HISTORY,))]
    df = ce.results_frame(rows)
    assert list(df["collateral"]) == ["AAPLon"]
    assert df.loc[0, "scenario"] == "e-mode"
    assert df.loc[0, "borrowable"] == "USDT"
    assert df.loc[0, "formula_ltv_pct"] == pytest.approx(84.5)
    assert df.loc[0, "final_ltv_pct"] == pytest.approx(82.2)
    assert df.loc[0, "delta_pp"] == pytest.approx(-2.3)
    assert df.loc[0, "binding"] == BIND_LIQ30
    assert df.loc[0, "flags"] == FLAG_SHORT_HISTORY
    # No LT pass on this row → final LT mirrors formula, delta 0.
    assert df.loc[0, "final_lt_pct"] == df.loc[0, "formula_lt_pct"]
    assert df.loc[0, "delta_lt_pp"] == 0.0
    for col in ("formula_lt_pct", "final_lt_pct", "delta_lt_pp",
                "max_excess_before_pct", "max_excess_after_pct",
                "liq30_pct", "gap_rate_pct", "uncond_bad_debt_pct",
                "max_excess_pct", "effective_years"):
        assert col in df.columns


def test_main_cli_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        ce, "run_calibrate",
        lambda **kw: [_row("AAPLon", SCENARIOS[0], 0.845, 0.822, BIND_LIQ30)],
    )
    out_csv = tmp_path / "out" / "ltvs.csv"
    rc = ce.main(["--data-dir", str(tmp_path), "--csv", str(out_csv),
                  "--log-level", "ERROR"])
    assert rc == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "collateral"] == "AAPLon"
    assert "wrote 1 rows" in capsys.readouterr().out


def test_main_cli_prints_constraint_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from nysa_risk.config import (AssetUniverse, Borrowable, Calibration,
                                  Collateral, EModeCategory, Meta, OndoConfig,
                                  PairsPolicy)
    universe = AssetUniverse(
        meta=Meta(version="0.1", base_currency="USD", price_history_years=1,
                  include_overnight_gaps=True),
        collaterals=(Collateral(symbol="AAPLon", type="rwa", category="equity",
                                underlying_ticker="AAPL", use="collateral_only"),),
        borrowables=(
            Borrowable(symbol="USDT", type="crypto", category="stable",
                       price_source="USDT-USD", use="lending_and_borrowing",
                       volatility_class="stable"),
            Borrowable(symbol="BNB", type="crypto", category="native",
                       price_source="BNB-USD", use="lending_and_borrowing",
                       volatility_class="volatile"),
        ),
        pairs=PairsPolicy(default_policy="all_collaterals_vs_all_borrowables"),
        ondo=OndoConfig(limits_api="https://x", status_page="https://y",
                        api_key_env="ONDO_API_KEY"),
        calibration=Calibration(
            ewma_lambda=0.94, stress_quantile=0.95, gap_sigma_quantile=0.90,
            es_factor=3.5, t_liq_days=0.33, t_user_days=3.0, k_user=1.53,
            stressed_liquidatable_share=0.25, rf_theta=0.01, rf_horizon_years=0.83,
            emode_min_advantage=0.05,
            target_liq30_emode=(0.10, 0.15), target_liq30_std=(0.10, 0.15),
            minimum_gap=0.02,
            max_uncond_bad_debt=0.004, min_calibration_years=3.0,
            severity_review_threshold=0.05, max_loss_given_bad_debt=0.065,
        ),
        emode_categories=(EModeCategory(name="stable", borrowables=("USDT",)),),
    )
    monkeypatch.setattr(ce, "load_universe", lambda *a, **k: universe)
    monkeypatch.setattr(
        ce, "run_calibrate",
        lambda **kw: [_row("AAPLon", SCENARIOS[0], 0.845, 0.822, BIND_LIQ30)],
    )
    rc = ce.main(["--data-dir", str(tmp_path), "--log-level", "ERROR"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Unified LTV calibration" in out
    assert "uncond bad debt ≤ 0.4%" in out
    assert "raises need ≥ 3y history" in out
    assert "AAPLon" in out and BIND_LIQ30 in out
