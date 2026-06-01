"""Heimdall data ingestion package.

Per docs/RESEARCH-PROPOSAL.md §5.1, the canonical DK1 panel is built from:
  - ENTSO-E Transparency Platform v2 (cross-EU baseline; A85, A83, A65, A44)
  - Energinet Open Data API (DK-native, finer mFRR granularity, no key)

Both wrappers are thin and idempotent; tidy alignment lives in `loaders`.
"""

from heimdall_data.eds import EDSClient
from heimdall_data.energinet import EnerginetClient
from heimdall_data.entsoe import EntsoeClient
from heimdall_data.jao import JAOClient
from heimdall_data.loaders import load_dk1_panel
from heimdall_data.open_meteo import WeatherLocation
from heimdall_data.outages import UMMClient

__all__ = [
    "EDSClient",
    "EnerginetClient",
    "EntsoeClient",
    "JAOClient",
    "UMMClient",
    "WeatherLocation",
    "load_dk1_panel",
]
