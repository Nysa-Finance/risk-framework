# Nysa Risk Engine

Reference implementation of the Nysa market-risk parameter methodology.
The engine calibrates per-pair risk parameters (liquidation threshold,
loan-to-value, lend/borrow caps, reserve-fund target) from long price
histories of the underlyings covered by `config/assets.yaml`.

## Anchors

The code is intentionally traceable to the design documents. Every
module carries a docstring pointing back to the section it implements.

- `docs/nysa-market-risk-framework.md` — sigma_stress → LT/LTV → RF
  calibration (§3 volatility, §4 horizons, §5 LT/LTV, §6 reserve).
- `docs/nysa-lb-caps.md` — supply/borrow-cap sizing (§2 supply caps,
  §3 borrow caps, secondary-market depth constraints).
- `config/assets.yaml` — asset universe, admissible pairs, and the
  single source of truth for calibration constants (EWMA lambda, stress
  quantile, ES multiplier, liquidation and user-reaction horizons, etc.).

## Layout

```
src/nysa_risk/
├── __init__.py
├── config.py            # typed loader over config/assets.yaml
├── volatility.py        # EWMA + stressed-vol estimation (§3.1)
├── report.py            # per-asset & per-pair parameter reports (§7)
├── extraction/          # underlying price / GM NAV / on-chain data
│   └── __init__.py
└── parameters/          # sigma_stress → LT → LTV → LB caps → RF pipeline
    └── __init__.py

config/assets.yaml       # asset universe + calibration constants
tests/                   # pytest suite
data/                    # extracted prices and computed outputs (gitignored)
```

## Setup

The project currently lives inside a Python 3.12 virtualenv rooted at
the repository directory itself (`bin/`, `lib/`, `pyvenv.cfg`).

```
source bin/activate
pip install -e ".[dev]"
```

## Commands

Run the test suite (validates config parsing against the real
`assets.yaml`):

```
pytest
```

Additional CLI entry points will be added as the extraction, parameter,
and reporting modules land.
