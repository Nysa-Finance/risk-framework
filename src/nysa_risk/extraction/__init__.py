"""Data extraction subpackage.

Implements the data-collection contract from
``docs/nysa-market-risk-framework.md`` §2 (Data Sources):

- Underlying price histories via yfinance (proxy for GM tokens whose
  on-chain history is short) — ``price_history_years`` from
  ``config/assets.yaml``.
- Ondo session limits from the API referenced in the ``ondo:`` block
  of the config, used as supply-cap inputs
  (``docs/nysa-lb-caps.md`` §2.1).
- Optional on-chain GM NAV / basis series used only to validate
  tracking against the underlying.

Empty for now.
"""
