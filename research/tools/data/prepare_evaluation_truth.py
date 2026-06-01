from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from heimdall_data.eds import EDSClient, get_cached_eds_dataset, normalize_eds_table

from packages.config import load_project_env


@dataclass(frozen=True)
class TruthStatus:
    source: str
    zone: str
    kind: str
    rows: int
    ok: bool
    required: bool
    error: str | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare evaluation-only Heimdall truth data.")
    parser.add_argument("--month", default="2026-04", help="UTC month in YYYY-MM format.")
    parser.add_argument("--start", help="UTC inclusive start timestamp.")
    parser.add_argument("--end", help="UTC exclusive end timestamp.")
    parser.add_argument("--zones", nargs="+", default=["DK1", "DK2"], choices=["DK1", "DK2"])
    parser.add_argument("--dataset-id", help="Output dataset id under data/cache/evaluation_truth/.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-price-only-diagnostics", action="store_true")
    args = parser.parse_args()

    load_project_env()
    start, end = _resolve_window(args.start, args.end, args.month)
    dataset_id = args.dataset_id or args.month.replace("-", "_")
    output_dir = args.output_dir or Path("data/cache/evaluation_truth") / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[TruthStatus] = []
    eds_client = EDSClient()
    activations = _extract_activations(
        args.zones,
        start,
        end,
        args.refresh,
        output_dir,
        statuses,
        eds_client,
        required=not args.allow_price_only_diagnostics,
    )
    prices = _extract_prices(args.zones, start, end, args.refresh, output_dir, statuses, eds_client)
    _raise_for_required_failures(statuses)

    truth = build_activation_truth(
        activations,
        prices,
        allow_price_only_diagnostics=args.allow_price_only_diagnostics,
    )
    if truth.empty and not args.allow_price_only_diagnostics:
        raise RuntimeError("activation truth is empty; refusing to write evaluation truth")

    _write_table(output_dir / "activation_truth.parquet", truth)
    manifest = {
        "schema_version": "1.0.0",
        "visibility": "evaluation_only",
        "dataset_id": dataset_id,
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "window_start_utc": start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": end.isoformat().replace("+00:00", "Z"),
        "zones": args.zones,
        "allow_price_only_diagnostics": args.allow_price_only_diagnostics,
        "statuses": [asdict(status) for status in statuses],
        "row_count": len(truth),
        "coverage": _coverage(args.zones, start, end, truth),
    }
    (output_dir / "truth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["coverage"], indent=2, sort_keys=True))
    print(f"wrote evaluation truth: {output_dir}")
    return 0


def build_activation_truth(
    activations: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    allow_price_only_diagnostics: bool = False,
) -> pd.DataFrame:
    if activations.empty and not allow_price_only_diagnostics:
        raise RuntimeError("activation truth is required for evaluation datasets")
    price_pivot = _price_pivot(prices)
    if activations.empty:
        out = price_pivot.copy()
        out["activation_direction"] = "unknown"
        out["activated_volume_mwh"] = pd.NA
    else:
        out = activations.copy()
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        out = out.merge(price_pivot, on=["timestamp_utc", "zone"], how="left")
    out["settlement_price_eur_mwh"] = out.apply(_settlement_price, axis=1)
    out["truth_source"] = "energidataservice_mfrr_energy_activation_market"
    return out[
        [
            "timestamp_utc",
            "zone",
            "activation_direction",
            "activated_volume_mwh",
            "settlement_price_eur_mwh",
            "spot_price_eur_mwh",
            "imbalance_price_eur_mwh",
            "mfrr_up_price_eur_mwh",
            "mfrr_down_price_eur_mwh",
            "truth_source",
        ]
    ].sort_values(["timestamp_utc", "zone", "activation_direction"]).reset_index(drop=True)


def _extract_activations(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[TruthStatus],
    client: EDSClient,
    *,
    required: bool,
) -> pd.DataFrame:
    padded_start = start - timedelta(hours=3)
    padded_end = end + timedelta(hours=3)
    raw = _capture(
        statuses,
        "energidataservice",
        ",".join(zones),
        "MfrrEnergyActivationMarket",
        lambda: get_cached_eds_dataset(
            "MfrrEnergyActivationMarket",
            start=padded_start,
            end=padded_end,
            filters={"PriceArea": zones},
            allow_empty=False,
            refresh=refresh,
            cache_dir=output_dir / "source_cache",
            client=client,
        ).frame,
        required=required,
    )
    return _normalize_eds_mfrr_activations(raw, start=start, end=end)


def _extract_prices(
    zones: list[str],
    start: datetime,
    end: datetime,
    refresh: bool,
    output_dir: Path,
    statuses: list[TruthStatus],
    client: EDSClient,
) -> pd.DataFrame:
    padded_start = start - timedelta(hours=3)
    padded_end = end + timedelta(hours=3)
    raw = _capture(
        statuses,
        "energidataservice",
        ",".join(zones),
        "ImbalancePrice",
        lambda: get_cached_eds_dataset(
            "ImbalancePrice",
            start=padded_start,
            end=padded_end,
            filters={"PriceArea": zones},
            allow_empty=False,
            refresh=refresh,
            cache_dir=output_dir / "source_cache",
            client=client,
        ).frame,
        required=True,
    )
    return _normalize_eds_prices(raw, start=start, end=end)


def _normalize_eds_mfrr_activations(raw: pd.DataFrame, *, start: datetime, end: datetime) -> pd.DataFrame:
    frame = normalize_eds_table(raw, dataset="MfrrEnergyActivationMarket")
    if frame.empty:
        return pd.DataFrame(
            columns=["timestamp_utc", "zone", "activation_direction", "activated_volume_mwh"]
        )
    frame = _clip_window(frame, start=start, end=end)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        up_mwh = max(0.0, _float_or_zero(record.get("TotalmFRRUpMW")) * 0.25)
        down_mwh = max(0.0, _float_or_zero(record.get("TotalmFRRDownMW")) * 0.25)
        common = {"timestamp_utc": record["timestamp_utc"], "zone": str(record["zone"])}
        if up_mwh > 0:
            rows.append({**common, "activation_direction": "up", "activated_volume_mwh": round(up_mwh, 6)})
        if down_mwh > 0:
            rows.append({**common, "activation_direction": "down", "activated_volume_mwh": round(down_mwh, 6)})
        if up_mwh == 0 and down_mwh == 0:
            rows.append({**common, "activation_direction": "neutral", "activated_volume_mwh": 0.0})
    return pd.DataFrame(rows)


def _normalize_eds_prices(raw: pd.DataFrame, *, start: datetime, end: datetime) -> pd.DataFrame:
    frame = normalize_eds_table(raw, dataset="ImbalancePrice")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "price_type", "price_eur_mwh"])
    frame = _clip_window(frame, start=start, end=end)
    mappings = {
        "day_ahead": "SpotPriceEUR",
        "imbalance": "ImbalancePriceEUR",
        "mfrr_up": "mFRRMarginalPriceUpEUR",
        "mfrr_down": "mFRRMarginalPriceDownEUR",
    }
    rows = []
    for price_type, column in mappings.items():
        if column not in frame.columns:
            continue
        selected = frame[["timestamp_utc", "zone", column]].copy()
        selected["price_type"] = price_type
        selected["price_eur_mwh"] = pd.to_numeric(selected[column], errors="coerce")
        rows.append(selected[["timestamp_utc", "zone", "price_type", "price_eur_mwh"]])
    return _concat(rows).dropna(subset=["price_eur_mwh"])


def _clip_window(frame: pd.DataFrame, *, start: datetime, end: datetime) -> pd.DataFrame:
    return frame[
        (frame["timestamp_utc"] >= pd.Timestamp(start))
        & (frame["timestamp_utc"] < pd.Timestamp(end))
    ].copy()


def _float_or_zero(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(parsed) else float(parsed)


def _capture(
    statuses: list[TruthStatus],
    source: str,
    zone: str,
    kind: str,
    fetch: Any,
    *,
    required: bool,
) -> pd.DataFrame:
    try:
        frame = fetch()
        statuses.append(TruthStatus(source, zone, kind, len(frame), True, required))
        return frame
    except Exception as exc:
        statuses.append(
            TruthStatus(source, zone, kind, 0, False, required, f"{type(exc).__name__}: {exc}")
        )
        return pd.DataFrame()


def _raise_for_required_failures(statuses: list[TruthStatus]) -> None:
    failures = [status for status in statuses if status.required and not status.ok]
    if failures:
        details = "; ".join(f"{item.source}/{item.zone}/{item.kind}: {item.error}" for item in failures)
        raise RuntimeError(f"required evaluation truth extraction failed: {details}")


def _price_pivot(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "zone",
                "spot_price_eur_mwh",
                "imbalance_price_eur_mwh",
                "mfrr_up_price_eur_mwh",
                "mfrr_down_price_eur_mwh",
            ]
        )
    pivot = prices.pivot_table(
        index=["timestamp_utc", "zone"],
        columns="price_type",
        values="price_eur_mwh",
        aggfunc="last",
    ).reset_index()
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(pivot["timestamp_utc"], utc=True),
            "zone": pivot["zone"],
            "spot_price_eur_mwh": pivot.get("day_ahead"),
            "imbalance_price_eur_mwh": pivot.get("imbalance"),
            "mfrr_up_price_eur_mwh": pivot.get("mfrr_up"),
            "mfrr_down_price_eur_mwh": pivot.get("mfrr_down"),
        }
    )


def _settlement_price(row: pd.Series) -> float | None:
    direction = row.get("activation_direction")
    if direction == "up" and pd.notna(row.get("mfrr_up_price_eur_mwh")):
        return float(row["mfrr_up_price_eur_mwh"])
    if direction == "down" and pd.notna(row.get("mfrr_down_price_eur_mwh")):
        return float(row["mfrr_down_price_eur_mwh"])
    if pd.notna(row.get("imbalance_price_eur_mwh")):
        return float(row["imbalance_price_eur_mwh"])
    return None


def _coverage(zones: list[str], start: datetime, end: datetime, truth: pd.DataFrame) -> dict[str, Any]:
    expected = len(pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC"))
    return {
        "expected_15m_per_zone": expected,
        "rows": len(truth),
        "volume_backed_rows": int(truth["activated_volume_mwh"].notna().sum()) if "activated_volume_mwh" in truth.columns else 0,
        "zones": {zone: int((truth["zone"] == zone).sum()) if "zone" in truth.columns else 0 for zone in zones},
    }


def _resolve_window(start: str | None, end: str | None, month: str) -> tuple[datetime, datetime]:
    if start or end:
        if not start or not end:
            raise ValueError("--start and --end must be provided together")
        parsed_start = pd.Timestamp(start)
        parsed_end = pd.Timestamp(end)
        if parsed_start.tzinfo is None or parsed_end.tzinfo is None:
            raise ValueError("--start/--end must be timezone-aware")
        return parsed_start.tz_convert("UTC").to_pydatetime(), parsed_end.tz_convert("UTC").to_pydatetime()
    month_start = pd.Timestamp(f"{month}-01T00:00:00Z")
    return month_start.to_pydatetime(), (month_start + pd.DateOffset(months=1)).to_pydatetime()


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
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
