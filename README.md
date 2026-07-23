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
