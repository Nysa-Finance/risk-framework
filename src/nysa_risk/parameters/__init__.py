"""Risk-parameter derivation subpackage.

Turns ``sigma_stress`` (from :mod:`nysa_risk.volatility`) into the
downstream parameters:

- Liquidation threshold (LT) and loan-to-value (LTV) —
  ``docs/nysa-market-risk-framework.md`` §5.
- Lend/borrow caps constrained by Ondo session limits and secondary
  market depth — ``docs/nysa-lb-caps.md`` §2 (supply caps) and §3
  (borrow caps).
- Reserve-fund target — ``docs/nysa-market-risk-framework.md`` §6.

Empty for now.
"""
