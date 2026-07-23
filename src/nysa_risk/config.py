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
    # Per-collateral execution cost S entering LT = 1 − C_T − S
    # (docs/nysa-market-risk-framework.md §3.3). Placeholder default; to
    # be replaced with redemption-channel values per collateral.
    execution_cost: float = 0.02


@dataclass(frozen=True, slots=True)
class Borrowable:
    symbol: str
    type: AssetType
    category: str
    price_source: str
    use: UsePolicy
    volatility_class: VolatilityClass


@dataclass(frozen=True, slots=True)
class PairExclusion:
    """Matcher over ``(collateral, borrowable)`` pairs.

    * ``collateral=None, borrowable=X`` — exclude every collateral against
      borrowable ``X`` (class-level exclusion).
    * ``collateral=Y, borrowable=None`` — exclude collateral ``Y`` against
      every borrowable.
    * both set — exclude that single specific pair.

    At least one field must be set; an all-``None`` exclusion would
    match everything and is rejected by the loader.
    """
    collateral: Optional[str] = None
    borrowable: Optional[str] = None

    def matches(self, collateral: str, borrowable: str) -> bool:
        if self.collateral is not None and self.collateral != collateral:
            return False
        if self.borrowable is not None and self.borrowable != borrowable:
            return False
        return self.collateral is not None or self.borrowable is not None


@dataclass(frozen=True, slots=True)
class PairsPolicy:
    default_policy: str
    exclusions: tuple[PairExclusion, ...] = ()


@dataclass(frozen=True, slots=True)
class OndoConfig:
    limits_api: str
    status_page: str
    api_key_env: str


@dataclass(frozen=True, slots=True)
class EModeCategory:
    """A contractually-scoped set of borrowables.

    Borrowers who opt into an E-Mode category can only borrow assets in
    ``borrowables`` — the per-collateral LT for that category is therefore
    computed with the min rule restricted to this set.
    """
    name: str
    borrowables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Calibration:
    ewma_lambda: float
    stress_quantile: float
    # Percentile of the EWMA vol series used for ``sigma_gap``, the vol
    # regime of the LTV gap buffer G (§4.1). Milder than ``stress_quantile``
    # by design: the gap constraint is a confidence bound on borrower
    # reaction, not a crash scenario — see ``nysa_risk.parameters.ltv``.
    gap_sigma_quantile: float
    es_factor: float
    t_liq_days: float
    t_user_days: float
    k_user: float
    stressed_liquidatable_share: float
    rf_theta: float
    rf_horizon_years: float
    # Minimum absolute LT advantage (as a fraction of collateral value) that
    # an E-Mode category must offer over the standard/base LT for that
    # category to be recommended in the summary report. Below this bar,
    # the CLI prints a dash — enabling E-Mode isn't worth the operational
    # complexity for that collateral.
    emode_min_advantage: float
    # --- backtest_curves --calibrate inputs ---
    # Acceptance band [lo, hi] for the historical share of openings that
    # become liquidatable within 30 days of opening at max LTV. The solver
    # lowers LTV when the observed share exceeds hi, raises it when below
    # lo, and keeps it when inside the band.
    target_liq30_emode: tuple[float, float]
    target_liq30_std: tuple[float, float]
    # Hard ceiling for any calibrated LTV: LTV ≤ LT − minimum_gap.
    minimum_gap: float
    # --- calibrate.py (unified engine) inputs ---
    # Constraint (3): unconditional bad-debt rate — bad-debt events over
    # openings with a decidable horizon outcome — must stay ≤ this.
    max_uncond_bad_debt: float
    # Short-history asymmetry guard: raising LTV above the formula prior
    # requires at least this much effective pair history (lowering is
    # always allowed — it is conservative on any sample size).
    min_calibration_years: float
    # Severity is not an LTV lever: a max excess loss beyond the 1 − LT
    # buffer above this threshold emits an LT REVIEW flag instead of
    # pushing LTV further down.
    severity_review_threshold: float
    # Declared severity bound enforced by the LT pass: the framework
    # accepts rare bad debt but bounds its per-position depth — LT is
    # cut until the worst-case excess beyond 1 − LT fits under this cap.
    max_loss_given_bad_debt: float


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    meta: Meta
    collaterals: tuple[Collateral, ...]
    borrowables: tuple[Borrowable, ...]
    pairs: PairsPolicy
    ondo: OndoConfig
    calibration: Calibration
    emode_categories: tuple[EModeCategory, ...] = ()

    def admissible_pairs(self) -> list[tuple[str, str]]:
        """Enumerate (collateral, borrowable) pairs under the default policy.

        Only assets flagged ``use: collateral_only`` are treated as
        collaterals. Assets with ``use: lending_and_borrowing`` are
        borrowables *only* — they never appear on the collateral side of
        a pair, so there are no Aave-style symmetric crypto/crypto
        entries here. Direction is strictly ``collateral → borrowable``.

        Anything matched by a ``pairs.exclusions`` entry is filtered
        out. Exclusions support wildcards on either side (see
        :class:`PairExclusion`) so a whole class of borrowables can be
        removed with a single entry.
        """
        exclusions = self.pairs.exclusions
        return [
            (c.symbol, b.symbol)
            for c in self.collaterals
            for b in self.borrowables
            if c.use == "collateral_only"
            and not any(ex.matches(c.symbol, b.symbol) for ex in exclusions)
        ]


def _parse_exclusion(entry: object) -> PairExclusion:
    """Accept either ``{collateral: X}`` / ``{borrowable: Y}`` / ``{collateral: X, borrowable: Y}``
    or a legacy 2-list ``[X, Y]``. Rejects an empty entry (which would match everything)."""
    if isinstance(entry, dict):
        exc = PairExclusion(
            collateral=entry.get("collateral"),
            borrowable=entry.get("borrowable"),
        )
    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
        exc = PairExclusion(collateral=str(entry[0]), borrowable=str(entry[1]))
    else:
        raise ValueError(f"unrecognised exclusion entry: {entry!r}")
    if exc.collateral is None and exc.borrowable is None:
        raise ValueError("exclusion must set at least one of `collateral`/`borrowable`")
    return exc


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
        exclusions=tuple(_parse_exclusion(e) for e in (pairs_raw.get("exclusions") or ())),
    )

    ondo = OndoConfig(**raw["ondo"])
    calibration = Calibration(**{k: _calibration_value(v) for k, v in raw["calibration"].items()})

    borrowable_symbols = {b.symbol for b in borrowables}
    emode_categories = tuple(
        _parse_emode(name, spec, borrowable_symbols)
        for name, spec in (raw.get("emode_categories") or {}).items()
    )

    return AssetUniverse(
        meta=meta,
        collaterals=collaterals,
        borrowables=borrowables,
        pairs=pairs,
        ondo=ondo,
        calibration=calibration,
        emode_categories=emode_categories,
    )


def _calibration_value(v: object) -> float | tuple[float, ...]:
    """Scalar calibration entries → float; lists (e.g. target bands) → tuple of floats."""
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return float(v)


def _parse_emode(name: str, spec: dict, borrowable_symbols: set[str]) -> EModeCategory:
    borrowables = tuple(spec.get("borrowables") or ())
    if not borrowables:
        raise ValueError(f"emode category '{name}' must list at least one borrowable")
    unknown = [b for b in borrowables if b not in borrowable_symbols]
    if unknown:
        raise ValueError(
            f"emode category '{name}' references unknown borrowables: {unknown}"
        )
    return EModeCategory(name=str(name), borrowables=borrowables)
