from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from heimdall_ai_society.market_context import RealDataTools
from heimdall_data.eds import EDSClient, get_cached_eds_dataset, normalize_eds_table
from heimdall_data.entsoe import (
    get_cached_entsoe_flows,
    get_cached_entsoe_generation,
    get_cached_entsoe_loads,
    get_cached_entsoe_prices,
)
from heimdall_data.open_meteo import (
    WEATHER_VARIABLES,
    WeatherLocation,
    get_cached_weather_forecast,
)
from heimdall_data.outages import UMMClient, build_outage_events, write_outages

from packages.config import load_project_env


@dataclass(frozen=True)
class ExtractStatus:
    source: str
    zone: str
    kind: str
    rows: int
    ok: bool
    required: bool
    error: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real Heimdall context windows.")
    parser.add_argument("--month", default="2026-04", help="UTC month in YYYY-MM format.")
    parser.add_argument("--start", help="UTC inclusive start timestamp.")
    parser.add_argument("--end", help="UTC exclusive end timestamp.")
    parser.add_argument("--zones", nargs="+", default=["DK1", "DK2"], choices=["DK1", "DK2"])
    parser.add_argument("--dataset-id", help="Output dataset id under data/cache/real_context/.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eds-page-size", type=int, default=5000)
    parser.add_argument("--eds-delay-seconds", type=float, default=1.25)
    parser.add_argument("--outage-lookback-days", type=int, default=30)
    parser.add_argument("--outage-cap", type=int, default=100)
    parser.add_argument("--optional-weather", action="store_true",
                        help="Do not fail the whole pull if the open-meteo weather fetch errors.")
    args = parser.parse_args()

    load_project_env()
    start, end = _resolve_window(args.start, args.end, args.month)
    dataset_id = args.dataset_id or args.month.replace("-", "_")
    output_dir = args.output_dir or Path("data/cache/real_context") / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[ExtractStatus] = []
    eds_client = EDSClient(page_size=args.eds_page_size, delay_seconds=args.eds_delay_seconds)
    prices = _extract_prices(args.zones, start, end, args.refresh, output_dir, statuses)
    loads = _extract_loads(args.zones, start, end, args.refresh, output_dir, statuses)
    generation = _extract_generation(args.zones, start, end, args.refresh, output_dir, statuses)
    flows = _extract_flows(args.zones, start, end, args.refresh, output_dir, statuses)
    weather = _extract_weather(args.zones, start, end, args.refresh, output_dir, statuses, required=not args.optional_weather)
    eds_tables = _extract_eds(args.zones, start, end, args.refresh, output_dir, statuses, eds_client)
    outages = _extract_outages(start - timedelta(days=args.outage_lookback_days), args.refresh, output_dir, statuses, args.outage_cap)
    _raise_for_required_failures(statuses)

    _write_table(output_dir / "prices.parquet", prices)
    _write_table(output_dir / "loads.parquet", loads)
    _write_table(output_dir / "generation.parquet", generation)
    _write_table(output_dir / "flows.parquet", flows)
    _write_table(output_dir / "weather.parquet", weather)
    for name, frame in eds_tables.items():
        _write_table(output_dir / f"eds_{name}.parquet", frame)
    write_outages(output_dir / "outages.json", outages)

    windows = _context_windows(args.zones, start, end, prices, loads, generation, flows, weather)
    _write_table(output_dir / "context_windows.parquet", windows)

    manifest = {
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "dataset_id": dataset_id,
        "visibility": "agent_context",
        "schema_version": "1.1.0",
        "window_start_utc": start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": end.isoformat().replace("+00:00", "Z"),
        "zones": args.zones,
        "required_sources": ["entsoe_prices", "entsoe_loads", "entsoe_generation", "entsoe_flows", "open_meteo", "nordpool_umm"],
        "optional_sources": ["eds"],
        "throttling": {"eds_page_size": args.eds_page_size, "eds_delay_seconds": args.eds_delay_seconds},
        "statuses": [asdict(status) for status in statuses],
        "request_logs": {"eds": [asdict(item) for item in eds_client.request_log]},
        "completeness": _completeness(args.zones, start, end, prices, loads, generation, flows, weather, windows),
        "outage_count": len(outages),
        "sample_tool_calls": _sample_tool_calls(args.zones, windows, prices, loads, generation, flows, weather, [event.__dict__ for event in outages]),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["completeness"], indent=2, sort_keys=True))
    print(f"wrote real context: {output_dir}")
    return 0


def _extract_prices(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
) -> pd.DataFrame:
    frames = []
    for zone in zones:
        for price_type in ["day_ahead", "imbalance", "mfrr_up", "mfrr_down"]:
            frame = _capture(
                statuses,
                "entsoe_prices",
                zone,
                price_type,
                lambda zone=zone, price_type=price_type: get_cached_entsoe_prices(
                    zone=zone,
                    price_type=price_type,
                    start=start,
                    end=end,
                    refresh=refresh,
                    cache_dir=output_dir / "source_cache",
                ).frame,
                required=True,
                empty=pd.DataFrame(),
            )
            frames.append(frame)
    return _concat(frames)


def _extract_loads(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
) -> pd.DataFrame:
    frames = []
    for zone in zones:
        for kind in ["actual", "forecast"]:
            frame = _capture(
                statuses,
                "entsoe_loads",
                zone,
                kind,
                lambda zone=zone, kind=kind: get_cached_entsoe_loads(
                    zone=zone,
                    kind=kind,
                    start=start,
                    end=end,
                    refresh=refresh,
                    cache_dir=output_dir / "source_cache",
                ).frame,
                required=True,
                empty=pd.DataFrame(),
            )
            frames.append(frame)
    return _concat(frames)


def _extract_generation(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
) -> pd.DataFrame:
    frames = []
    for zone in zones:
        for generation_type in ["wind", "solar", "hydro", "thermal"]:
            required = generation_type != "hydro"
            frame = _capture(
                statuses,
                "entsoe_generation",
                zone,
                generation_type,
                lambda zone=zone, generation_type=generation_type: get_cached_entsoe_generation(
                    zone=zone,
                    generation_type=generation_type,
                    start=start,
                    end=end,
                    refresh=refresh,
                    cache_dir=output_dir / "source_cache",
                ).frame,
                required=required,
                empty=pd.DataFrame(),
            )
            frames.append(frame)
    return _concat(frames)


def _extract_flows(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
) -> pd.DataFrame:
    frames = []
    counterparties = {"DK1": ["DK_2"], "DK2": ["DK_1"]}
    for zone in zones:
        for counterparty in counterparties[zone]:
            frame = _capture(
                statuses,
                "entsoe_flows",
                zone,
                counterparty,
                lambda zone=zone, counterparty=counterparty: get_cached_entsoe_flows(
                    zone=zone,
                    counterparty=counterparty,
                    start=start,
                    end=end,
                    refresh=refresh,
                    cache_dir=output_dir / "source_cache",
                ).frame,
                required=True,
                empty=pd.DataFrame(),
            )
            frames.append(frame)
    return _concat(frames)


def _extract_weather(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
    required: bool = True,
) -> pd.DataFrame:
    locations = {
        "DK1": WeatherLocation("DK1", 56.2639, 9.5018),
        "DK2": WeatherLocation("DK2", 55.6761, 12.5683),
    }
    frames = []
    for zone in zones:
        frame = _capture(
            statuses,
            "open_meteo",
            zone,
            "weather",
            lambda zone=zone: get_cached_weather_forecast(
                locations[zone],
                variables=list(WEATHER_VARIABLES),
                start=start,
                end=end,
                refresh=refresh,
                cache_dir=output_dir / "source_cache",
            ).frame,
            required=required,
            empty=pd.DataFrame(),
        )
        frames.append(frame)
    return _concat(frames)


def _extract_eds(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
    client: EDSClient,
) -> dict[str, pd.DataFrame]:
    datasets = {
        "regulating_balance": ("RegulatingBalancePowerdata", "HourUTC"),
        "elspot": ("Elspotprices", "HourUTC"),
        "forecasts_hour": ("Forecasts_Hour", "HourUTC"),
        "production_consumption": ("ProductionConsumptionSettlement", "HourUTC"),
        "co2": ("CO2Emis", "Minutes5UTC"),
    }
    tables: dict[str, pd.DataFrame] = {}
    for slug, (dataset, time_column) in datasets.items():
        frames = []
        for zone in zones:
            frame = _capture(
                statuses,
                "eds",
                zone,
                dataset,
                lambda zone=zone, dataset=dataset, time_column=time_column: normalize_eds_table(
                    get_cached_eds_dataset(
                        dataset,
                        start=start,
                        end=end,
                        filters={"PriceArea": zone},
                        sort=f"{time_column} asc",
                        allow_empty=True,
                        refresh=refresh,
                        cache_dir=output_dir / "source_cache",
                        client=client,
                    ).frame,
                    dataset=dataset,
                ),
                required=False,
                empty=pd.DataFrame(),
            )
            frames.append(frame)
        tables[slug] = _concat(frames)
    return tables


def _extract_outages(
    publication_start: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[ExtractStatus],
    cap: int,
) -> list[Any]:
    cache_path = output_dir / "source_cache" / "nordpool_umm_messages.json"
    def fetch() -> list[Any]:
        if cache_path.exists() and not refresh:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = UMMClient().fetch_messages(publication_start=publication_start)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        events = build_outage_events(raw, cap=cap)
        if not events:
            raise RuntimeError("Nord Pool UMM returned no outage events after filtering")
        return events

    return _capture(
        statuses,
        "nordpool_umm",
        "ALL",
        "outages",
        fetch,
        required=True,
        empty=[],
    )


def _capture(
    statuses: list[ExtractStatus],
    source: str,
    zone: str,
    kind: str,
    fetch: Any,
    *,
    required: bool,
    empty: Any,
) -> Any:
    try:
        result = fetch()
        statuses.append(ExtractStatus(source, zone, kind, len(result), True, required))
        return result
    except Exception as exc:
        statuses.append(ExtractStatus(source, zone, kind, 0, False, required, f"{type(exc).__name__}: {exc}"))
        return empty


def _raise_for_required_failures(statuses: list[ExtractStatus]) -> None:
    failures = [status for status in statuses if status.required and not status.ok]
    if failures:
        details = "; ".join(f"{item.source}/{item.zone}/{item.kind}: {item.error}" for item in failures)
        raise RuntimeError(f"required data extraction failed: {details}")


def _context_windows(
    zones: list[str],
    start: datetime,
    end: datetime,
    prices: pd.DataFrame,
    loads: pd.DataFrame,
    generation: pd.DataFrame,
    flows: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    ticks = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    rows = []
    for zone in zones:
        base = pd.DataFrame({"timestamp_utc": ticks, "zone": zone})
        frame = base.copy()
        frame = _merge_price(frame, prices, zone)
        frame = _merge_load(frame, loads, zone)
        frame = _merge_generation(frame, generation, zone)
        frame = _merge_flows(frame, flows, zone)
        frame = _merge_weather(frame, weather, zone)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["zone", "timestamp_utc"])


def _merge_price(base: pd.DataFrame, prices: pd.DataFrame, zone: str) -> pd.DataFrame:
    out = base.copy()
    for price_type in ["day_ahead", "imbalance", "mfrr_up", "mfrr_down"]:
        subset = prices[(prices.get("zone") == zone) & (prices.get("price_type") == price_type)]
        out = _merge_asof(out, subset, f"price_{price_type}_eur_mwh", "price_eur_mwh")
    return out


def _merge_load(base: pd.DataFrame, loads: pd.DataFrame, zone: str) -> pd.DataFrame:
    out = base.copy()
    for kind in ["actual", "forecast"]:
        subset = loads[(loads.get("zone") == zone) & (loads.get("kind") == kind)]
        out = _merge_asof(out, subset, f"load_{kind}_mw", "load_mw")
    return out


def _merge_generation(base: pd.DataFrame, generation: pd.DataFrame, zone: str) -> pd.DataFrame:
    out = base.copy()
    if generation.empty:
        return out
    grouped = (
        generation[generation["zone"] == zone]
        .groupby(["timestamp_utc", "generation_type"], as_index=False)["generation_mw"]
        .sum()
    )
    for generation_type in ["wind", "solar", "hydro", "thermal"]:
        subset = grouped[grouped["generation_type"] == generation_type]
        out = _merge_asof(out, subset, f"generation_{generation_type}_mw", "generation_mw")
    return out


def _merge_flows(base: pd.DataFrame, flows: pd.DataFrame, zone: str) -> pd.DataFrame:
    if flows.empty:
        return base
    subset = (
        flows[flows["from_zone"] == zone]
        .groupby("timestamp_utc", as_index=False)["flow_mw"]
        .sum()
    )
    return _merge_asof(base, subset, "flow_net_export_mw", "flow_mw")


def _merge_weather(base: pd.DataFrame, weather: pd.DataFrame, zone: str) -> pd.DataFrame:
    out = base.copy()
    if weather.empty:
        return out
    subset = weather[weather["zone"] == zone]
    for variable in WEATHER_VARIABLES:
        out = _merge_asof(out, subset, f"weather_{variable}", variable)
    return out


def _merge_asof(base: pd.DataFrame, source: pd.DataFrame, output_column: str, value_column: str) -> pd.DataFrame:
    if source.empty or value_column not in source.columns:
        base[output_column] = pd.NA
        return base
    left = base.sort_values("timestamp_utc")
    right = source[["timestamp_utc", value_column]].copy()
    right["timestamp_utc"] = pd.to_datetime(right["timestamp_utc"], utc=True)
    right = right.sort_values("timestamp_utc").dropna(subset=[value_column])
    merged = pd.merge_asof(left, right, on="timestamp_utc", direction="backward")
    base[output_column] = merged[value_column].to_numpy()
    return base


def _completeness(
    zones: list[str],
    start: datetime,
    end: datetime,
    prices: pd.DataFrame,
    loads: pd.DataFrame,
    generation: pd.DataFrame,
    flows: pd.DataFrame,
    weather: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    expected_15m = len(pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC"))
    expected_hourly = len(pd.date_range(start, end, freq="h", inclusive="left", tz="UTC"))
    return {
        "expected_15m_per_zone": expected_15m,
        "expected_hourly_per_zone": expected_hourly,
        "source_rows": {
            "prices": len(prices),
            "loads": len(loads),
            "generation": len(generation),
            "flows": len(flows),
            "weather": len(weather),
            "context_windows": len(windows),
        },
        "context_non_null_fraction": {
            zone: {
                column: round(float(group[column].notna().mean()), 4)
                for column in group.columns
                if column.startswith(("price_", "load_", "generation_", "flow_", "weather_"))
            }
            for zone, group in windows.groupby("zone")
        },
        "tool_ready": {
            zone: bool(
                (windows[windows["zone"] == zone]["price_day_ahead_eur_mwh"].notna().any())
                and ("weather_wind_speed" in windows.columns)
                and (windows[windows["zone"] == zone]["weather_wind_speed"].notna().any())
            )
            for zone in zones
        },
    }


def _sample_tool_calls(
    zones: list[str],
    windows: pd.DataFrame,
    prices: pd.DataFrame,
    loads: pd.DataFrame,
    generation: pd.DataFrame,
    flows: pd.DataFrame,
    weather: pd.DataFrame,
    outages: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = {}
    for zone in zones:
        zone_windows = windows[windows["zone"] == zone]
        if zone_windows.empty:
            continue
        now = pd.Timestamp(zone_windows["timestamp_utc"].iloc[min(96, len(zone_windows) - 1)]).to_pydatetime()
        tools = RealDataTools(
            now=now,
            zone=zone,
            prices=prices,
            loads=loads,
            generation=generation,
            flows=flows,
            weather=weather,
            outages=outages,
            default_lookback_hours=24,
        )
        samples[zone] = {
            "get_last_prices": tools.get_last_prices(hours=24, zone=zone, price_type="day_ahead"),
            "get_last_loads": tools.get_last_loads(hours=24, zone=zone, kind="actual"),
            "get_last_generation": tools.get_last_generation(hours=24, zone=zone, generation_type="wind"),
            "get_crossborder_flows": tools.get_crossborder_flows(hours=24, zone=zone),
            "get_weather_today": tools.get_weather_today(zone=zone, variables=["wind_speed", "solar_radiation", "temperature"]),
            "get_outages": tools.get_outages(hours=24 * 7, zone=zone),
        }
    return samples


def _resolve_window(start: str | None, end: str | None, month: str) -> tuple[datetime, datetime]:
    if start or end:
        if not start or not end:
            raise ValueError("--start and --end must be provided together")
        parsed_start = pd.Timestamp(start)
        parsed_end = pd.Timestamp(end)
        if parsed_start.tzinfo is None or parsed_end.tzinfo is None:
            raise ValueError("--start/--end must be timezone-aware")
        return parsed_start.tz_convert("UTC").to_pydatetime(), parsed_end.tz_convert("UTC").to_pydatetime()
    return _month_window(month)


def _month_window(month: str) -> tuple[datetime, datetime]:
    start = pd.Timestamp(f"{month}-01T00:00:00Z")
    end = start + pd.DateOffset(months=1)
    return start.to_pydatetime(), end.to_pydatetime()


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    out = pd.concat(available, ignore_index=True, sort=False)
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    return out


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
