# Nysa Risk Engine

A risk-parameter engine that turns long price histories into the
per-collateral lending parameters — liquidation threshold (LT),
loan-to-value (LTV), lend/borrow caps and reserve-fund target — for the
asset universe declared in `config/assets.yaml`. It is a reference
implementation of the methodology in `docs/`: the formula-derived
parameters are treated as a **prior**, then empirically **calibrated**
against declared constraint bands and **validated** by historical
backtests, so every published number is traceable to both a design
document and a historical exceedance check.

The engine is market-agnostic. It carries no market-specific results —
those live on the per-market branches (e.g. `ondo-gm-market`), generated
by running the pipeline against that market's `config/assets.yaml` and
extracted data.

## Anchors

The code is intentionally traceable to the design documents. Every
module docstring points back to the section it implements.

- `docs/nysa-market-risk-framework.md` — sigma_stress → LT/LTV → RF
  calibration (§3 volatility, §4 horizons, §5 LT/LTV, §6 reserve, §7 reporting).
- `docs/nysa-lb-caps.md` — supply/borrow-cap sizing (§2 supply caps,
  §3 borrow caps, secondary-market depth constraints).
- `config/assets.yaml` — asset universe, admissible pairs, and the
  single source of truth for every calibration constant. Nothing that
  governs a parameter is hard-coded in the modules.

## Architecture

A single directional pipeline; each stage consumes the previous stage's
typed output and adds nothing the config does not sanction.

```
config/assets.yaml
      │  (typed loader: config.py)
      ▼
extraction ─▶ volatility ─▶ parameters ─▶ calibration engine ─▶ backtests ─▶ report
  price/NAV     EWMA          C_T→LT→LTV     LT + LTV passes       constraint    published
   history    sigma_stress   (formula prior)  (empirical)          validation    LT/LTV table
```

- **extraction** (`extraction/`) — pull the underlying price / GM-NAV /
  on-chain history for every symbol into per-ticker parquet files.
- **volatility** (`volatility.py`) — build the aligned, interleaved
  open/close relative-price stream per admissible pair (overnight and
  weekend gaps included), run the RiskMetrics EWMA recursion, and read
  off `sigma_stress` (stress quantile) and `sigma_gap` (gap quantile).
- **parameters** (`parameters/`) — the closed-form pipeline:
  `C_T` (`ct.py`) → per-collateral `LT = min_j (1 − C_T − S)` (`lt.py`,
  base + per-E-Mode-category) → `LTV = LT − G` (`ltv.py`). These are the
  **formula priors**.
- **calibration engine** (`calibrate.py`) — the unified two-pass engine.
  An **LT pass** caps per-position bad-debt depth (severity), then an
  **LTV pass** bisects each row's LTV to the largest value satisfying all
  constraints, formula LTV as prior. Everything downstream deploys these
  final values.
- **backtests** — historical validation. `backtest.py` checks the
  declared exceedance bounds (gap constraint, time-to-liquidation, bad
  debt + severity); `backtest_curves.py` produces survival-style
  time-to-event summaries and an LTV band solver (`--calibrate`).
- **report** (`report.py`) — publish the final calibrated LT/LTV table
  (CLI + `data/report/parameters.{md,csv}`), with a `--full` audit view.

All simulation stages share one construction — daily positions opened at
max LTV on the same aligned open/close stream, with the conservative
**next-price realization rule** (a liquidation realizes at the next
observation after the trigger, charging the full overnight/weekend gap;
this is deliberately harsher than the intra-session `t_liq` assumed by
`C_T`).

## Quickstart

```
source bin/activate
pip install -e ".[dev]"
pytest                              # full suite, validates config parsing too
```

The pipeline stages, each a `python -m nysa_risk.<module>` entry point:

```
# 1. extract underlying price histories → data/underlying/*.parquet
python -m nysa_risk.extraction.underlying

# 2. formula priors (closed form)
python -m nysa_risk.parameters.lt         # per-collateral LT (base + e-mode)
python -m nysa_risk.parameters.ltv        # per-collateral LTV = LT − G

# 3. calibration engine → final deployable LT/LTV
python -m nysa_risk.calibrate             # add --csv <path> to export

# 4. historical validation
python -m nysa_risk.backtest              # gap / time-to-liq / bad-debt + severity
python -m nysa_risk.backtest_curves       # survival curves; --calibrate for the LTV band solver

# 5. publish the final parameter table
python -m nysa_risk.report                # writes data/report/parameters.{md,csv}; --full for audit columns
```

## Design principles

**Formula as prior → empirical calibration → validation.** The
closed-form `LT`/`LTV` are never published directly. They seed a
calibration that pins each parameter to a historically-observed
frequency band, and the result is then re-validated by an independent
backtest. Each step is one command and reads every threshold from
`config/assets.yaml`.

**Calibration constraints** (all constants live in `calibration:` in
`config/assets.yaml`):

| # | Constraint | Config constant(s) |
|---|---|---|
| LT | Per-position bad-debt **depth** (max excess loss beyond the `1 − LT` buffer) is capped; LT is bisected down (floor = formula LT − 10pp) until it fits. | `max_loss_given_bad_debt` |
| 1 | Share of max-LTV openings liquidated within 30 days sits inside a target band; the formula LTV may be raised toward it only with enough history, else it is a cap. | `target_liq30_emode`, `target_liq30_std`, `min_calibration_years` |
| 2 | Borrower-reaction bound: `P(liq ≤ t_user_days)` stays under the declared quantile. | `t_user_days`, `k_user` (bound = `1 − 0.90`) |
| 3 | Unconditional bad-debt rate (bad-debt events / decidable openings) stays bounded. | `max_uncond_bad_debt` |
| 4 | Hard ceiling: `LTV ≤ LT − minimum_gap`. | `minimum_gap` |

**Severity is an LT lever, not an LTV lever.** Frequency of bad debt is
governed by constraints (2)–(3) on LTV; its per-position *depth* is
governed by the LT pass. Below the enforcement cap but above
`severity_review_threshold`, a row is flagged `LT REVIEW` for a human
rather than auto-adjusted.

**Conservatism is explicit.** Gap risk enters the return stream
(interleaved open/close), and liquidations realize at the next available
price — both documented, both deliberately pessimistic, so a passing
backtest is a lower bound on safety.

**Correlated-sample honesty.** Overlapping daily openings are correlated
paths, so every reported percentage is a per-opening frequency over one
realized history, not an independent probability. The docstrings say so
wherever a rate is produced.

## Layout

```
src/nysa_risk/
├── config.py            # typed loader over config/assets.yaml
├── volatility.py        # EWMA + stressed/gap-vol estimation (§3.1)
├── extraction/          # underlying price / GM-NAV / on-chain data
├── parameters/          # ct.py → lt.py → ltv.py  (formula priors)
├── calibrate.py         # unified LT + LTV calibration engine
├── backtest.py          # constraint validation (gap / time / bad debt / severity)
├── backtest_curves.py   # survival-style time-to-event + LTV band solver
└── report.py            # published LT/LTV parameter table (§7)

config/assets.yaml       # asset universe + calibration constants
tests/                   # pytest suite (hand-verified synthetic fixtures)
data/                    # extracted prices + computed outputs (gitignored)
```

## Setup

The project currently lives inside a Python 3.12 virtualenv rooted at the
repository directory itself (`bin/`, `lib/`, `pyvenv.cfg`).

```
source bin/activate
pip install -e ".[dev]"
```
<!--
  ondo-gm-market branch only.

  Append this section to the end of README.md on the `ondo-gm-market`
  branch (it extends the market-agnostic README from main). Do NOT
  merge it into main — main carries no market-specific results.

  Regenerate the table below with:  python -m nysa_risk.report
-->

## Results — Ondo GM market

Parameters below are the engine's final, deployable values for the Ondo
GM asset universe (`config/assets.yaml` on this branch), produced by
`python -m nysa_risk.calibrate` and published by `python -m nysa_risk.report`
(`data/report/parameters.md`). Two scenarios per collateral: **standard**
(borrow vs BNB, `base` parameter set) and **e-mode** (borrow vs USDT,
`emode:stable` set). All values are post-calibration — severity-capped LT,
constraint-calibrated LTV — not the closed-form priors.

| Collateral | LT std (%) | LTV std (%) | LT e-mode (%) | LTV e-mode (%) |
|:---|---:|---:|---:|---:|
| AAPLon | 76.22 | 54.05 | 91.61 | 82.21 |
| AMDon | 73.97 | 54.24 | 85.13 | 71.15 |
| AMZNon | 75.57 | 53.16 | 90.12 | 79.60 |
| CRCLon | 56.33 | 25.82 | 63.10 | 35.98 |
| GOOGLon | 75.78 | 53.59 | 91.63 | 82.64 |
| INTCon | 75.92 | 53.35 | 82.02 | 69.92 |
| METAon | 76.03 | 53.16 | 88.75 | 77.06 |
| MSFTon | 76.00 | 54.07 | 92.17 | 80.85 |
| MUon | 74.21 | 50.84 | 86.57 | 73.03 |
| NVDAon | 74.41 | 53.22 | 87.08 | 73.53 |
| QQQon | 76.21 | 54.35 | 93.00 | 85.81 |
| SLVon | 76.11 | 53.97 | 89.86 | 81.50 |
| SPYon | 76.24 | 54.49 | 93.90 | 88.41 |
| TLTon | 75.71 | 53.88 | 94.95 | 89.60 |
| TSLAon | 74.07 | 49.91 | 85.32 | 68.58 |

### Validation summary

Historical backtest of the final configuration (`python -m nysa_risk.backtest`,
`python -m nysa_risk.backtest_curves`), ~8.7 years of overlapping history per
pair (CRCLon excepted — see caveats):

- **Gap constraint (borrower reaction).** Observed `P(liq ≤ t_user_days)`
  ranges **0.3 %–1.4 %** across all collaterals — comfortably inside the
  declared **10 %** bound (`1 − k_user` at the 90 % quantile). Every row
  PASSes with wide margin.
- **Liquidation-within-30-days.** Calibrated into the target band
  **[10 %, 15 %]** (`target_liq30_std`, `target_liq30_emode`). E-mode rows
  bind on the 15 % upper edge (LTV trimmed ≈0.5–2.3pp from formula); most
  standard rows validate at their formula LTV inside the band.
- **Unconditional bad debt.** Bad-debt events / decidable openings stays
  **≤ 0.4 %** everywhere (`max_uncond_bad_debt`), i.e. at most ~1 in 250
  max-LTV openings ends in bad debt within the horizon.
- **Bad-debt severity.** Worst per-position excess loss beyond the
  `1 − LT` buffer capped at **6.5 %** (`max_loss_given_bad_debt`) by the LT
  pass, which cut LT on **three** rows to bring the tail inside the bound:

  | Row | Formula LT | Final LT | ΔLT | Max excess before → after |
  |:---|---:|---:|---:|---|
  | INTCon e-mode | 86.39 | 82.02 | −4.37pp | 10.87 % → 6.49 % |
  | AMDon e-mode | 86.22 | 85.13 | −1.09pp | 7.59 % → 6.50 % |
  | TLTon standard | 75.89 | 75.71 | −0.19pp | 6.68 % → 6.49 % |

### Known caveats

- **CRCLon — short history.** CRCL listed 2025-06, leaving only ~1.1 years
  of overlapping history (vs ~8.7y for the rest), below
  `min_calibration_years` (3.0). Its LTVs are held at the conservative
  formula prior (the short-history guard blocks any raise) and its
  empirical rates are not statistically meaningful. Treat CRCLon as an
  **isolation / exclusion candidate** until more history accrues rather
  than as a calibrated row.
- **Standard-set `LT REVIEW` flags are single-gap-day artifacts.** Several
  standard rows carry `LT REVIEW` (max excess ≈5.6–6.5 %, above the 5 %
  advisory `severity_review_threshold` but under the 6.5 % enforcement
  cap). Each traces to a **single historical gap day** surfaced by the
  conservative next-price realization rule — the liquidator charged the
  full overnight/weekend gap to the next print, harsher than the
  intra-session `t_liq` window `C_T` is sized for. They are advisory, not
  breaches: flagged for human review of LT, deliberately not auto-adjusted
  (severity is an LT lever, and these sit below the enforcement cap).
