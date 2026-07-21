"""EWMA volatility and stressed-vol estimation.

Implements the ``sigma_stress`` calculation from
``docs/nysa-market-risk-framework.md`` §3.1: the ``stress_quantile``
percentile (default 95th) of the EWMA daily-volatility series computed
with ``ewma_lambda`` (default 0.94). Both parameters are read from
``config/assets.yaml`` via :mod:`nysa_risk.config`.

Empty for now; implementation lands together with the extraction
pipeline in :mod:`nysa_risk.extraction`.
"""
