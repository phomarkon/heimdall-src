from __future__ import annotations

from pathlib import Path

import typer

from heimdall_run_view.database import DatabaseUnavailableError, PostgresRunStore

app = typer.Typer(help="Heimdall run-view utilities.")


@app.command()
def init_db() -> None:
    """Create the Postgres tables used by the run-view API."""
    try:
        PostgresRunStore().ensure_schema()
    except DatabaseUnavailableError as exc:
        typer.echo(f"Postgres unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo("heimdall-run-view database schema is ready")


@app.command()
def ingest(run_dir: Path) -> None:
    """Load one run directory into Postgres while keeping disk artifacts canonical."""
    try:
        run_id = PostgresRunStore().ingest_run(run_dir.resolve())
    except DatabaseUnavailableError as exc:
        typer.echo(f"Postgres unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"ingested {run_id}")


if __name__ == "__main__":
    app()
