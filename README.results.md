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
