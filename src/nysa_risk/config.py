"""Typed configuration loader.

Reads ``config/assets.yaml`` and exposes the asset universe and the
calibration constants used by the parameter modules as immutable,
typed dataclasses.

Anchors:
    - ``config/assets.yaml`` — single source of truth.
    - ``docs/nysa-market-risk-framework.md`` §3 (calibration constants:
      ``ewma_lambda``, ``stress_quantile``, ``es_factor``), §4
      (``t_liq_days``, ``t_user_days``, ``k_user``), §6 (reserve fund
      ``rf_theta``, ``rf_horizon_years``).
    - ``docs/nysa-lb-caps.md`` §2.1 (``stressed_liquidatable_share``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

AssetType = Literal["rwa", "crypto"]
UsePolicy = Literal["collateral_only", "lending_and_borrowing"]
VolatilityClass = Literal["volatile", "stable"]

# Repo layout is <repo>/src/nysa_risk/config.py, so parents[2] == <repo>.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "assets.yaml"


@dataclass(frozen=True, slots=True)
class Meta:
    version: str
    base_currency: str
    price_history_years: int
    include_overnight_gaps: bool


@dataclass(frozen=True, slots=True)
class Collateral:
    symbol: str
    type: AssetType
    category: str
    underlying_ticker: str
    use: UsePolicy
    chain: Optional[str] = None
    address: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Borrowable:
    symbol: str
    type: AssetType
    category: str
    price_source: str
    use: UsePolicy
    volatility_class: VolatilityClass


@dataclass(frozen=True, slots=True)
class PairsPolicy:
    default_policy: str
    exclusions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class OndoConfig:
    limits_api: str
    status_page: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class Calibration:
    ewma_lambda: float
    stress_quantile: float
    es_factor: float
    t_liq_days: float
    t_user_days: float
    k_user: float
    stressed_liquidatable_share: float
    rf_theta: float
    rf_horizon_years: float


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    meta: Meta
    collaterals: tuple[Collateral, ...]
    borrowables: tuple[Borrowable, ...]
    pairs: PairsPolicy
    ondo: OndoConfig
    calibration: Calibration

    def admissible_pairs(self) -> list[tuple[str, str]]:
        """Enumerate (collateral, borrowable) pairs under the default policy.

        Every RWA/GM collateral is paired against every borrowable, and
        crypto borrowables also collateralize one another (Aave-style),
        minus anything listed in ``pairs.exclusions``.
        """
        excluded = {tuple(p) for p in self.pairs.exclusions}
        rwa_pairs = [
            (c.symbol, b.symbol)
            for c in self.collaterals
            for b in self.borrowables
            if (c.symbol, b.symbol) not in excluded
        ]
        crypto_pairs = [
            (b1.symbol, b2.symbol)
            for b1 in self.borrowables
            for b2 in self.borrowables
            if b1.symbol != b2.symbol
            and (b1.symbol, b2.symbol) not in excluded
        ]
        return rwa_pairs + crypto_pairs


def load_universe(path: Path | str | None = None) -> AssetUniverse:
    """Parse ``config/assets.yaml`` (or an override path) into an ``AssetUniverse``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    meta_raw = raw["meta"]
    meta = Meta(
        version=str(meta_raw["version"]),
        base_currency=str(meta_raw["base_currency"]),
        price_history_years=int(meta_raw["price_history_years"]),
        include_overnight_gaps=bool(meta_raw["include_overnight_gaps"]),
    )

    collaterals = tuple(Collateral(**c) for c in raw["collaterals"])
    borrowables = tuple(Borrowable(**b) for b in raw["borrowables"])

    pairs_raw = raw["pairs"]
    pairs = PairsPolicy(
        default_policy=str(pairs_raw["default_policy"]),
        exclusions=tuple(
            tuple(p) for p in pairs_raw.get("exclusions", []) or ()
        ),
    )

    ondo = OndoConfig(**raw["ondo"])
    calibration = Calibration(**{k: float(v) for k, v in raw["calibration"].items()})

    return AssetUniverse(
        meta=meta,
        collaterals=collaterals,
        borrowables=borrowables,
        pairs=pairs,
        ondo=ondo,
        calibration=calibration,
    )
