"""Database infrastructure: connection factory, schema, and focused repository classes."""

from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class DatabaseUnavailableError(RuntimeError):
    pass


BUILTIN_AGENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "p2h",
        "label": "P2H operator",
        "category": "action",
        "archetype": "p2h",
        "is_builtin": True,
    },
    {
        "template_id": "ev",
        "label": "EV aggregator",
        "category": "action",
        "archetype": "ev",
        "is_builtin": True,
    },
    {
        "template_id": "wind",
        "label": "Wind BRP",
        "category": "action",
        "archetype": "wind",
        "is_builtin": True,
    },
    {
        "template_id": "generator",
        "label": "Generator",
        "category": "action",
        "archetype": "generator",
        "is_builtin": True,
    },
    {
        "template_id": "retailer",
        "label": "Retailer",
        "category": "action",
        "archetype": "retailer",
        "is_builtin": True,
    },
    {
        "template_id": "renewables",
        "label": "Renewables BRP",
        "category": "action",
        "archetype": "renewables",
        "is_builtin": True,
    },
    {
        "template_id": "risk-info",
        "label": "Trading risk monitor",
        "category": "information",
        "archetype": "risk-info",
        "is_builtin": True,
    },
]


SCHEMA_STATEMENTS = [
    """
    create table if not exists runs (
        run_id text primary key,
        status text not null default 'completed',
        started_at timestamptz,
        total_steps integer not null,
        config_json jsonb not null default '{}'::jsonb,
        summary_json jsonb not null default '{}'::jsonb,
        trace_sha256 text not null,
        source_run_dir text not null
    )
    """,
    """
    create table if not exists agents (
        run_id text not null references runs(run_id) on delete cascade,
        agent_id text not null,
        archetype text not null,
        display_name text not null,
        persona_json jsonb not null default '{}'::jsonb,
        asset_json jsonb not null default '{}'::jsonb,
        is_custom boolean not null default false,
        primary key (run_id, agent_id)
    )
    """,
    """
    create table if not exists trace_events (
        run_id text not null references runs(run_id) on delete cascade,
        event_index integer not null,
        step integer not null,
        timestamp timestamptz,
        agent_id text not null,
        event_json jsonb not null,
        primary key (run_id, event_index)
    )
    """,
    "create index if not exists trace_events_run_step_idx on trace_events(run_id, step)",
    """
    create table if not exists evaluations (
        run_id text primary key references runs(run_id) on delete cascade,
        summary_json jsonb not null default '{}'::jsonb,
        bid_evaluations_json jsonb not null default '[]'::jsonb,
        created_at timestamptz not null default now()
    )
    """,
    """
    create table if not exists agent_templates (
        template_id text primary key,
        template_json jsonb not null,
        created_at timestamptz not null default now()
    )
    """,
    """
    create table if not exists society_specs (
        society_id text primary key,
        spec_json jsonb not null,
        created_at timestamptz not null default now()
    )
    """,
]


class _BaseRepository:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("HEIMDALL_RUN_VIEW_DATABASE_URL")
        if not self.database_url:
            raise DatabaseUnavailableError("HEIMDALL_RUN_VIEW_DATABASE_URL is not set")

    def ensure_schema(self, conn: Any | None = None) -> None:
        owns_connection = conn is None
        conn = conn or self._connect()
        try:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            if owns_connection:
                conn.commit()
        finally:
            if owns_connection:
                conn.close()

    def _connect(self) -> Any:
        try:
            return psycopg.connect(self.database_url, row_factory=dict_row)
        except Exception as exc:
            raise DatabaseUnavailableError(str(exc)) from exc

    def _jsonb(self, value: Any) -> Any:
        return Jsonb(value)


class AgentTemplateRepository(_BaseRepository):
    def list(self) -> list[dict[str, Any]]:
        try:
            with self._connect() as conn:
                self.ensure_schema(conn)
                rows = conn.execute(
                    "select template_json from agent_templates order by template_id"
                ).fetchall()
        except DatabaseUnavailableError:
            return list(BUILTIN_AGENT_TEMPLATES)
        return [*BUILTIN_AGENT_TEMPLATES, *[row["template_json"] for row in rows]]

    def save(self, template: dict[str, Any]) -> dict[str, Any]:
        template_id = str(template.get("template_id") or template.get("id") or "").strip()
        if not template_id:
            raise ValueError("agent template requires template_id")
        template = {**template, "template_id": template_id, "is_builtin": False}
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                insert into agent_templates (template_id, template_json, created_at)
                values (%s, %s, now())
                on conflict (template_id) do update set template_json = excluded.template_json
                """,
                (template_id, self._jsonb(template)),
            )
            conn.commit()
        return template

    def delete(self, template_id: str) -> None:
        template_id = str(template_id or "").strip()
        if not template_id:
            raise ValueError("agent template requires template_id")
        with self._connect() as conn:
            self.ensure_schema(conn)
            cur = conn.execute("delete from agent_templates where template_id = %s", (template_id,))
            conn.commit()
            if cur.rowcount == 0:
                raise KeyError(template_id)


class SocietySpecRepository(_BaseRepository):
    def save(self, spec: dict[str, Any]) -> dict[str, Any]:
        society_id = str(spec.get("society_id") or spec.get("id") or "").strip()
        if not society_id:
            raise ValueError("society spec requires society_id")
        spec = {**spec, "society_id": society_id}
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                insert into society_specs (society_id, spec_json, created_at)
                values (%s, %s, now())
                on conflict (society_id) do update set spec_json = excluded.spec_json
                """,
                (society_id, self._jsonb(spec)),
            )
            conn.commit()
        return spec

    def get(self, society_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                "select spec_json from society_specs where society_id = %s", (society_id,)
            ).fetchone()
        if row is None:
            raise KeyError(society_id)
        return row["spec_json"]
