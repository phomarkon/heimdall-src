"""PyPSA-Eur-Sec technology cost-database loader.

Per docs/RESEARCH-PROPOSAL.md §2.5 and §4.8: Heimdall's physical envelope is
borrowed from PyPSA-Eur-Sec.  This module reads the canonical
``technology-data/outputs/costs_2030.csv`` from the PyPSA project
(https://github.com/PyPSA/technology-data) and exposes typed accessors.

The CSV is content-addressable; we pin a SHA-256 of the version we use
in ``data/raw/pypsa_eursec/costs_2030.csv.sha256``.  Reviewers can verify
chain of custody from the CSV → AssetSpec → verifier behaviour without
trusting any custom Heimdall constant.

Why a CSV-driven loader instead of a 6 GB `.nc` snapshot:
- The full PyPSA-Eur-Sec snapshot is ~6 GB and includes 8 760 hourly
  snapshots, hundreds of buses, all sector couplings.  We only need the
  P2H + thermal-storage envelope for DK1+DK2.
- The cost CSV is the *root* artefact PyPSA-Eur-Sec consumes for those
  numbers.  Loading it directly skips the dispatch-LP layer and lands the
  same parameters in our `HeimdallScenario`.
- This is the route the PyPSA team itself uses for "I just want the
  parameters" (see e.g. PyPSA-Eur's `add_existing_baseyear` snakemake
  rules, which read this CSV).

Reference rows we consume (exact `(technology, parameter)` keys from the
2030 vintage):
- ``central air-sourced heat pump`` — efficiency = 3.2 (COP)
- ``central water tank storage`` — standing losses 0.0077 %/h,
  energy-to-power ratio 60.34 h
- ``central water pit storage``  — standing losses 0.0078 %/h,
  energy-to-power ratio 30 h, lifetime 30 yr
- ``central resistive heater`` — efficiency 0.99 (alternative P2H route)

Two storage technologies are exposed because Danfoss's headline asset
class spans both: short-horizon water tanks (intra-day balancing) and
multi-horizon water-pit / borehole stores (seasonal arbitrage).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COSTS_CSV = REPO_ROOT / "data/raw/pypsa_eursec/costs_2030.csv"


@dataclass(frozen=True)
class CostEntry:
    technology: str
    parameter: str
    value: float
    unit: str
    source: str


def load_cost_table(path: Path | None = None) -> dict[tuple[str, str], CostEntry]:
    p = Path(path) if path is not None else DEFAULT_COSTS_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"PyPSA-Eur-Sec cost CSV not found at {p}. "
            "Pull via: curl -L -o data/raw/pypsa_eursec/costs_2030.csv "
            "https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2030.csv"
        )
    out: dict[tuple[str, str], CostEntry] = {}
    with p.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                value = float(row["value"])
            except (ValueError, KeyError):
                continue
            entry = CostEntry(
                technology=row["technology"],
                parameter=row["parameter"],
                value=value,
                unit=row.get("unit", ""),
                source=row.get("source", ""),
            )
            out[(entry.technology, entry.parameter)] = entry
    return out


def cost_csv_sha256(path: Path | None = None) -> str:
    p = Path(path) if path is not None else DEFAULT_COSTS_CSV
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convenience accessors used by the network builder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatPumpParams:
    """Central air-sourced heat pump parameters from PyPSA-Eur-Sec costs_2030.csv."""
    cop: float            # ``efficiency`` in PyPSA terminology (per-unit)
    investment_eur_per_kw: float
    fom_pct_per_year: float
    lifetime_years: float
    source: str


@dataclass(frozen=True)
class ThermalStorageParams:
    """District-heating thermal storage parameters from PyPSA-Eur-Sec.

    ``standing_loss_per_hour`` is unit-less per-hour fraction (the PyPSA
    convention is %/hour, divided by 100 here).  ``energy_to_power_h`` is
    the ratio between storage capacity (MWh) and charge/discharge power
    (MW) — PyPSA-Eur-Sec uses 60.34 h for central water tanks and 30 h
    for water pits.
    """
    standing_loss_per_hour: float
    energy_to_power_h: float
    investment_eur_per_kwh: float
    lifetime_years: float
    source: str


def heat_pump_params(table: dict[tuple[str, str], CostEntry] | None = None) -> HeatPumpParams:
    table = table or load_cost_table()
    cop = table[("central air-sourced heat pump", "efficiency")].value
    inv = table[("central air-sourced heat pump", "investment")].value
    fom = table[("central air-sourced heat pump", "FOM")].value
    life = table[("central air-sourced heat pump", "lifetime")].value
    src = table[("central air-sourced heat pump", "efficiency")].source
    return HeatPumpParams(cop=cop, investment_eur_per_kw=inv, fom_pct_per_year=fom,
                          lifetime_years=life, source=src)


def thermal_storage_params(
    technology: str = "central water tank storage",
    table: dict[tuple[str, str], CostEntry] | None = None,
) -> ThermalStorageParams:
    """Defaults to ``central water tank storage`` (60.34 h E/P; intra-day).

    Pass ``technology="central water pit storage"`` for the seasonal
    multi-horizon store (30 h E/P, lifetime 30 yr).
    """
    table = table or load_cost_table()
    standing_pct_h = table[(technology, "standing losses")].value
    e_to_p = table[(technology, "energy to power ratio")].value
    inv = table[(technology, "investment")].value
    life = table[(technology, "lifetime")].value
    return ThermalStorageParams(
        standing_loss_per_hour=standing_pct_h / 100.0,
        energy_to_power_h=e_to_p,
        investment_eur_per_kwh=inv,
        lifetime_years=life,
        source=table[(technology, "standing losses")].source,
    )


__all__ = [
    "CostEntry",
    "HeatPumpParams",
    "ThermalStorageParams",
    "DEFAULT_COSTS_CSV",
    "load_cost_table",
    "cost_csv_sha256",
    "heat_pump_params",
    "thermal_storage_params",
]
