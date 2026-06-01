"""Data-access tools exposed to LLM agents during society ticks.

``RealDataTools`` wraps pre-loaded DataFrames behind causality-safe slicing
(``_now`` boundary) and returns JSON-ready dicts.  Extracted from
``market_context.py`` to keep that module focused on tick orchestration.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from heimdall_data.jao import empty_constraints_frame
from heimdall_data.open_meteo import validate_weather_variables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "timestamp_utc" not in frame.columns:
        return frame.copy()
    copy = frame.copy()
    copy["timestamp_utc"] = pd.to_datetime(copy["timestamp_utc"], utc=True)
    return copy


def _canonical_jao_constraints(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_constraints_frame()
    copy = frame.copy()
    for column in ["timestamp_utc", "publication_time_utc"]:
        if column not in copy.columns:
            copy[column] = pd.NaT
        copy[column] = pd.to_datetime(copy[column], utc=True, errors="coerce")
    for column in ["zone", "cnec_id", "constraint_name", "direction"]:
        if column not in copy.columns:
            copy[column] = ""
    for column in ["ram_mw", "shadow_price_eur_mw", "flow_mw"]:
        if column not in copy.columns:
            copy[column] = np.nan
        copy[column] = pd.to_numeric(copy[column], errors="coerce")
    if "source_url" not in copy.columns:
        copy["source_url"] = ""
    return copy


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _maybe_round(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _grid_signal_subset(grid: dict[str, Any]) -> dict[str, Any]:
    signals = grid.get("signals", {}) if isinstance(grid, dict) else {}
    if not isinstance(signals, dict):
        return {}
    return {
        f"jao_{key}": value
        for key, value in signals.items()
        if key in {"active_cnec_count", "tight_ram_count", "min_ram_mw", "max_shadow_price_eur_mw", "constrained_zone_intuition"}
    }


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "message": message, "rows": []}


def _tool_table(kind: str, frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    available = [column for column in columns if column in frame.columns]
    clean = frame[available].copy() if available else pd.DataFrame()
    if "timestamp_utc" in clean.columns:
        clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    clean = clean.tail(96)
    return {"ok": True, "kind": kind, "row_count": len(clean), "rows": clean.to_dict(orient="records")}


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# RealDataTools
# ---------------------------------------------------------------------------

class RealDataTools:
    def __init__(
        self,
        *,
        now: datetime,
        zone: str,
        prices: pd.DataFrame,
        loads: pd.DataFrame,
        generation: pd.DataFrame,
        flows: pd.DataFrame,
        weather: pd.DataFrame,
        outages: list[dict[str, Any]] | None,
        jao_constraints: pd.DataFrame | None = None,
        default_lookback_hours: int,
    ) -> None:
        self._now = now.astimezone(UTC)
        self._zone = zone
        self._prices = _canonical_time(prices)
        self._loads = _canonical_time(loads)
        self._generation = _canonical_time(generation)
        self._flows = _canonical_time(flows)
        self._weather = _canonical_time(weather)
        self._outages = outages or []
        self._jao_constraints = _canonical_jao_constraints(jao_constraints)
        self._default_lookback_hours = default_lookback_hours

    def with_observed_at(self, observed_at: datetime) -> RealDataTools:
        return RealDataTools(
            now=observed_at,
            zone=self._zone,
            prices=self._prices,
            loads=self._loads,
            generation=self._generation,
            flows=self._flows,
            weather=self._weather,
            outages=self._outages,
            jao_constraints=self._jao_constraints,
            default_lookback_hours=self._default_lookback_hours,
        )

    def get_last_prices(self, *, hours: int | None = None, zone: str | None = None, price_type: str = "day_ahead") -> dict[str, Any]:
        if price_type not in {"day_ahead", "imbalance", "mfrr_up", "mfrr_down"}:
            return _tool_error("unsupported_price_type", f"unsupported price_type={price_type!r}")
        frame = self._slice(self._prices, hours, zone or self._zone)
        frame = frame[frame["price_type"] == price_type]
        return _tool_table("prices", frame, ["timestamp_utc", "zone", "price_type", "price_eur_mwh"])

    def get_last_loads(self, *, hours: int | None = None, zone: str | None = None, kind: str = "actual") -> dict[str, Any]:
        if kind not in {"actual", "forecast"}:
            return _tool_error("unsupported_load_kind", f"unsupported load kind={kind!r}")
        frame = self._slice(self._loads, hours, zone or self._zone)
        frame = frame[frame["kind"] == kind]
        return _tool_table("loads", frame, ["timestamp_utc", "zone", "kind", "load_mw"])

    def get_last_generation(self, *, hours: int | None = None, zone: str | None = None, generation_type: str = "all") -> dict[str, Any]:
        if generation_type not in {"all", "wind", "solar", "hydro", "thermal"}:
            return _tool_error("unsupported_generation_type", f"unsupported generation_type={generation_type!r}")
        frame = self._slice(self._generation, hours, zone or self._zone)
        if generation_type != "all":
            frame = frame[frame["generation_type"] == generation_type]
        return _tool_table("generation", frame, ["timestamp_utc", "zone", "generation_type", "production_type", "generation_mw"])

    def get_crossborder_flows(self, *, hours: int | None = None, zone: str | None = None, counterparty: str | None = None) -> dict[str, Any]:
        frame = self._slice(self._flows, hours, zone or self._zone, zone_column="from_zone")
        if counterparty:
            frame = frame[frame["to_zone"] == counterparty]
        return _tool_table("flows", frame, ["timestamp_utc", "from_zone", "to_zone", "flow_mw"])

    def get_weather_today(self, *, zone: str | None = None, variables: list[str] | None = None) -> dict[str, Any]:
        return self._weather_table(
            zone=zone or self._zone,
            variables=variables,
            start=self._now.replace(hour=0, minute=0, second=0, microsecond=0),
            end=self._now,
        )

    def get_weather_forecast(self, *, zone: str | None = None, horizon_hours: int = 48, variables: list[str] | None = None) -> dict[str, Any]:
        horizon = max(1, min(int(horizon_hours), 168))
        return self._weather_table(
            zone=zone or self._zone,
            variables=variables,
            start=self._now,
            end=self._now + timedelta(hours=horizon),
        )

    def get_outages(self, *, hours: int | None = None, zone: str | None = None) -> dict[str, Any]:
        lookback = max(1, min(int(hours or self._default_lookback_hours), 168))
        selected_zone = zone or self._zone
        start = self._now - timedelta(hours=lookback)
        rows = []
        for event in self._outages:
            zones = set(event.get("zones") or [])
            if selected_zone not in zones:
                continue
            published = _parse_time(event.get("published_at_utc"))
            if published is not None and published > self._now:
                continue
            event_start = _parse_time(event.get("time_start_utc"))
            event_end = _parse_time(event.get("time_end_utc"))
            active_or_recent = (
                (published is not None and start <= published <= self._now)
                or (event_start is not None and event_start <= self._now and (event_end is None or event_end >= start))
            )
            if active_or_recent:
                rows.append(event)
        rows = sorted(rows, key=lambda item: (item.get("published_at_utc") or "", item.get("max_unavailable_capacity_mw") or 0), reverse=True)
        return {"ok": True, "kind": "outages", "row_count": len(rows[:50]), "rows": rows[:50]}

    def get_activation_context(self, *, hours: int | None = None, zone: str | None = None) -> dict[str, Any]:
        lookback = max(4, min(int(hours or self._default_lookback_hours), 168))
        selected_zone = zone or self._zone
        frame = self._slice(self._prices, lookback, selected_zone)
        if frame.empty:
            return {
                "ok": True,
                "kind": "activation_context",
                "observed_at": _iso_z(self._now),
                "zone": selected_zone,
                "lookback_hours": lookback,
                "row_count": 0,
                "watch_score": 0.0,
                "direction_hint": "neutral",
                "signals": {"reason": "missing_price_context"},
            }
        pivot = frame.pivot_table(index="timestamp_utc", columns="price_type", values="price_eur_mwh", aggfunc="last").sort_index()
        spot = pivot.get("day_ahead", pd.Series(dtype=float)).ffill()
        imbalance = pivot.get("imbalance", spot).fillna(spot)
        up = pivot.get("mfrr_up", imbalance).fillna(imbalance)
        down = pivot.get("mfrr_down", imbalance).fillna(imbalance)
        up_spread = (up - spot).dropna()
        down_spread = (spot - down).dropna()
        recent_spread = float(up_spread.tail(8).mean()) if not up_spread.tail(8).empty else 0.0
        volatility = float(imbalance.diff().abs().tail(16).mean()) if len(imbalance.dropna()) > 1 else 0.0
        up_positive = float((up_spread > 0).mean()) if not up_spread.empty else 0.0
        down_positive = float((down_spread > 0).mean()) if not down_spread.empty else 0.0
        direction_hint = "up" if up_positive >= down_positive else "down"
        watch_score = max(0.0, min(1.0, 0.45 * max(up_positive, down_positive) + 0.35 * min(volatility / 25.0, 1.0) + 0.20 * min(abs(recent_spread) / 50.0, 1.0)))
        return {
            "ok": True,
            "kind": "activation_context",
            "observed_at": _iso_z(self._now),
            "zone": selected_zone,
            "lookback_hours": lookback,
            "row_count": len(frame),
            "watch_score": round(watch_score, 6),
            "direction_hint": direction_hint,
            "signals": {
                "up_positive_share": round(up_positive, 6),
                "down_positive_share": round(down_positive, 6),
                "recent_up_spread_eur_mwh": round(recent_spread, 6),
                "mean_abs_imbalance_move_eur_mwh": round(volatility, 6),
            },
        }

    def get_market_regime_context(self, *, hours: int | None = None, zone: str | None = None) -> dict[str, Any]:
        activation = self.get_activation_context(hours=hours, zone=zone)
        selected_zone = zone or self._zone
        frame = self._slice(self._prices, hours, selected_zone)
        if frame.empty:
            return {
                "ok": True,
                "kind": "market_regime_context",
                "observed_at": _iso_z(self._now),
                "zone": selected_zone,
                "regime_label": "unavailable",
                "signals": {"reason": "missing_price_context"},
            }
        pivot = frame.pivot_table(index="timestamp_utc", columns="price_type", values="price_eur_mwh", aggfunc="last").sort_index()
        imbalance = pivot.get("imbalance", pivot.get("day_ahead", pd.Series(dtype=float))).dropna()
        volatility = float(imbalance.diff().abs().tail(16).mean()) if len(imbalance) > 1 else 0.0
        negative_share = float((imbalance.tail(32) < 0.0).mean()) if not imbalance.tail(32).empty else 0.0
        watch_score = float(activation.get("watch_score", 0.0) or 0.0)
        if negative_share >= 0.25:
            label = "negative_price_watch"
        elif volatility >= 35.0:
            label = "volatile"
        elif watch_score >= 0.6:
            label = "high_activation_watch"
        elif volatility <= 5.0 and watch_score < 0.25:
            label = "quiet"
        else:
            label = "normal_watch"
        return {
            "ok": True,
            "kind": "market_regime_context",
            "authority": "derived_non_leaking",
            "observed_at": _iso_z(self._now),
            "zone": selected_zone,
            "regime_label": label,
            "watch_score": round(watch_score, 6),
            "signals": {
                "mean_abs_imbalance_move_eur_mwh": round(volatility, 6),
                "recent_negative_price_share": round(negative_share, 6),
                "activation_direction_hint": activation.get("direction_hint", "neutral"),
            },
        }

    def get_border_pressure(self, *, hours: int | None = None, zone: str | None = None, counterparty: str | None = None) -> dict[str, Any]:
        selected_zone = zone or self._zone
        rows = self.get_crossborder_flows(hours=hours, zone=selected_zone, counterparty=counterparty).get("rows", [])
        flows = [float(row.get("flow_mw", 0.0) or 0.0) for row in rows if isinstance(row, dict)]
        grid = self.get_grid_constraints(hours=hours, zone=selected_zone)
        if not flows:
            return {
                "ok": True,
                "kind": "border_pressure",
                "authority": "derived_non_leaking",
                "observed_at": _iso_z(self._now),
                "zone": selected_zone,
                "row_count": 0,
                "pressure_label": "unavailable",
                "signals": {"reason": "missing_flow_context", **_grid_signal_subset(grid)},
            }
        latest = flows[-1]
        mean_abs = float(np.mean(np.abs(flows)))
        swing = max(flows) - min(flows)
        label = "high_export_pressure" if latest > mean_abs * 0.75 else "high_import_pressure" if latest < -mean_abs * 0.75 else "balanced_or_mixed"
        return {
            "ok": True,
            "kind": "border_pressure",
            "authority": "derived_non_leaking",
            "observed_at": _iso_z(self._now),
            "zone": selected_zone,
            "row_count": len(flows),
            "pressure_label": label,
            "signals": {
                "latest_flow_mw": round(latest, 6),
                "mean_abs_flow_mw": round(mean_abs, 6),
                "flow_swing_mw": round(swing, 6),
                **_grid_signal_subset(grid),
            },
        }

    def get_grid_constraints(self, *, hours: int | None = None, zone: str | None = None) -> dict[str, Any]:
        selected_zone = zone or self._zone
        lookback = max(1, min(int(hours or self._default_lookback_hours), 168))
        start = self._now - timedelta(hours=lookback)
        frame = self._jao_constraints
        if frame.empty:
            return {
                "ok": True,
                "kind": "grid_constraints",
                "authority": "jao_optional",
                "observed_at": _iso_z(self._now),
                "zone": selected_zone,
                "row_count": 0,
                "pressure_label": "unavailable",
                "signals": {"reason": "missing_jao_constraints"},
            }
        frame = frame[
            (frame["zone"] == selected_zone)
            & (frame["timestamp_utc"] >= pd.Timestamp(start))
            & (frame["timestamp_utc"] <= pd.Timestamp(self._now))
            & (frame["publication_time_utc"] <= pd.Timestamp(self._now))
        ]
        if frame.empty:
            return {
                "ok": True,
                "kind": "grid_constraints",
                "authority": "jao_optional",
                "observed_at": _iso_z(self._now),
                "zone": selected_zone,
                "row_count": 0,
                "pressure_label": "unavailable",
                "signals": {"reason": "no_visible_jao_rows"},
            }
        ram = pd.to_numeric(frame["ram_mw"], errors="coerce")
        shadow = pd.to_numeric(frame["shadow_price_eur_mw"], errors="coerce").fillna(0.0)
        flow = pd.to_numeric(frame["flow_mw"], errors="coerce")
        tight = frame[ram <= 200.0] if ram.notna().any() else frame.iloc[0:0]
        max_shadow = float(shadow.max()) if not shadow.empty else 0.0
        label = "binding_or_near_binding" if max_shadow > 0.0 or len(tight) else "monitored"
        return {
            "ok": True,
            "kind": "grid_constraints",
            "authority": "jao_derived_non_leaking",
            "observed_at": _iso_z(self._now),
            "zone": selected_zone,
            "row_count": int(len(frame)),
            "pressure_label": label,
            "top_constraints": [
                {
                    "cnec_id": str(row.get("cnec_id") or "unknown"),
                    "constraint_name": str(row.get("constraint_name") or "JAO constraint"),
                    "ram_mw": _maybe_round(row.get("ram_mw")),
                    "shadow_price_eur_mw": _maybe_round(row.get("shadow_price_eur_mw")),
                    "direction": row.get("direction"),
                }
                for row in frame.sort_values("shadow_price_eur_mw", ascending=False, na_position="last").head(5).to_dict("records")
            ],
            "signals": {
                "active_cnec_count": int(frame["cnec_id"].nunique()),
                "tight_ram_count": int(len(tight)),
                "min_ram_mw": _maybe_round(ram.min()) if ram.notna().any() else None,
                "max_shadow_price_eur_mw": round(max_shadow, 6),
                "mean_flow_mw": _maybe_round(flow.mean()) if flow.notna().any() else None,
                "constrained_zone_intuition": label,
            },
        }

    def get_outage_impact(self, *, hours: int | None = None, zone: str | None = None) -> dict[str, Any]:
        rows = self.get_outages(hours=hours, zone=zone).get("rows", [])
        scored = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            capacity = float(row.get("max_unavailable_capacity_mw") or row.get("unavailable_capacity_mw") or 0.0)
            score = min(1.0, capacity / 1000.0)
            scored.append(
                {
                    "summary": row.get("event_type") or row.get("reason") or row.get("message") or "outage",
                    "impact_score": round(score, 6),
                    "max_unavailable_capacity_mw": capacity,
                    "published_at_utc": row.get("published_at_utc"),
                    "time_start_utc": row.get("time_start_utc"),
                    "time_end_utc": row.get("time_end_utc"),
                }
            )
        scored = sorted(scored, key=lambda item: float(item["impact_score"]), reverse=True)
        return {
            "ok": True,
            "kind": "outage_impact",
            "authority": "derived_non_leaking",
            "observed_at": _iso_z(self._now),
            "zone": zone or self._zone,
            "impact_label": "high" if scored and scored[0]["impact_score"] >= 0.5 else "medium" if scored else "none",
            "row_count": len(scored),
            "top_events": scored[:5],
        }

    def _weather_table(self, *, zone: str, variables: list[str] | None, start: datetime, end: datetime) -> dict[str, Any]:
        selected = variables or ["temperature", "wind_speed", "solar_radiation", "cloud_cover", "precipitation"]
        try:
            validate_weather_variables(selected)
        except ValueError as exc:
            return _tool_error("unsupported_weather_variable", str(exc))
        frame = self._weather[
            (self._weather["zone"] == zone)
            & (self._weather["timestamp_utc"] >= pd.Timestamp(start))
            & (self._weather["timestamp_utc"] <= pd.Timestamp(end))
        ]
        return _tool_table("weather", frame, ["timestamp_utc", "zone", *selected])

    def _slice(self, frame: pd.DataFrame, hours: int | None, zone: str, *, zone_column: str = "zone") -> pd.DataFrame:
        lookback = max(1, min(int(hours or self._default_lookback_hours), 168))
        start = self._now - timedelta(hours=lookback)
        if frame.empty or zone_column not in frame.columns:
            return frame.iloc[0:0].copy()
        return frame[
            (frame[zone_column] == zone)
            & (frame["timestamp_utc"] >= pd.Timestamp(start))
            & (frame["timestamp_utc"] <= pd.Timestamp(self._now))
        ]
