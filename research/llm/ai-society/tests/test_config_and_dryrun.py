from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
from heimdall_ai_society.config import SocietyRunConfig, load_config
from heimdall_ai_society.llm_client import OpenAICompatibleLLMClient
from heimdall_ai_society.market_context import TickContext
from heimdall_ai_society.personas import build_personas
from heimdall_ai_society.runner import (
    _bid_budget_prompt_context,
    _decide_with_tools,
    _enforce_bid_budget,
    _apply_fill_selector_review,
    _initial_bid_budget_state,
    _update_bid_budget_state,
    run_society,
)
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tools import AgentToolExecutor

from packages.simulator.forecast import ForecastMarketState, ForecastSource


def test_local_dryrun_config_is_laptop_safe() -> None:
    config = load_config("research/llm/ai-society/configs/local-dryrun.yaml")
    assert config.llm.enabled is False
    assert config.verifier_mode == "mock"
    assert config.agent_count == 3
    assert config.forecaster_backend == "f0"


def test_config_accepts_ollama_provider_single_endpoint() -> None:
    config = SocietyRunConfig(
        tool_mode="json_response",
        llm={
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "model": "qwen3:235b",
            "max_concurrency": 2,
            "per_endpoint_max_concurrency": 2,
            "supports_response_format": False,
        },
    )
    assert config.llm.provider == "ollama"
    assert config.llm.endpoint_urls == ["http://127.0.0.1:11434/v1"]
    assert config.llm.supports_response_format is False


def test_openrouter_provider_uses_openrouter_env_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "openrouter.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: openrouter",
                "  model: anthropic/claude-3.5-haiku",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.test/heimdall")
    config = load_config(config_path)
    assert config.llm.provider == "openrouter"
    assert config.llm.base_url == "https://openrouter.ai/api/v1"
    assert config.llm.api_key == "or-test-key"
    assert config.llm.http_referer == "https://example.test/heimdall"
    assert config.llm.app_title == "Heimdall AI Society"
    assert config.llm.require_served_model_match is False


def test_config_preserves_vllm_dual_endpoint_defaults() -> None:
    config = SocietyRunConfig(
        llm={
            "base_urls": ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            "api_key": "heimdall-local",
            "model": "Qwen/Qwen3-32B",
        },
    )
    assert config.llm.provider == "vllm"
    assert config.llm.endpoint_urls == ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
    assert config.llm.require_served_model_match is True


def test_ollama_decision_falls_back_when_json_schema_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    decision = LLMBidDecision(action="watch", rationale="fallback worked").model_dump_json()

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": decision}}]}).encode("utf-8")

    def _urlopen(request, timeout):  # type: ignore[no-untyped-def]
        calls.append(json.loads(request.data.decode("utf-8")))
        if len(calls) == 1:
            raise HTTPError(request.full_url, 400, "bad response_format", hdrs=None, fp=None)
        return _Response()

    monkeypatch.setattr("heimdall_ai_society.llm_client.urlopen", _urlopen)
    client = OpenAICompatibleLLMClient(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="qwen3:235b",
        temperature=0.2,
        max_tokens=350,
        timeout_seconds=1,
        provider="ollama",
    )
    result = client._decide_sync("http://127.0.0.1:11434/v1", [{"role": "user", "content": "decide"}])
    assert result.action == "watch"
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert calls[1]["messages"][0]["role"] == "system"


def test_bid_budget_context_and_exhaustion_policy() -> None:
    state = _initial_bid_budget_state(1)
    _update_bid_budget_state(
        state,
        step=0,
        decision=LLMBidDecision(action="bid", side="up", quantity_mwh=1.0, limit_price_eur_mwh=100.0, rationale="accepted"),
        verifier_accepted=True,
        verifier_reason_codes=[],
        history_ticks=1,
    )
    context = _bid_budget_prompt_context(state, enabled=True, history_ticks=1)
    assert context is not None
    assert context["bids_remaining"] == 0
    assert len(context["recent_actions"]) == 1
    proposed = LLMBidDecision(action="bid", side="up", quantity_mwh=1.0, limit_price_eur_mwh=100.0, rationale="try")
    final, exhausted = _enforce_bid_budget(proposed, context)
    assert exhausted is True
    assert final.action == "watch"


def test_fill_selector_cannot_mutate_candidate() -> None:
    fallback = LLMBidDecision(action="bid", side="up", quantity_mwh=1.0, limit_price_eur_mwh=100.0, rationale="fallback", confidence=0.7)
    mutated = LLMBidDecision(action="bid", side="up", quantity_mwh=1.0, limit_price_eur_mwh=99.0, rationale="mutation", confidence=0.9)
    records: list[ToolCallRecord] = []
    final = _apply_fill_selector_review(
        fallback,
        mutated,
        [{"arguments": {"side": "up", "quantity_mwh": 1.0, "limit_price_eur_mwh": 100.0}, "clear_probability_proxy": 0.9, "worst_case_profit_eur": 10.0}],
        records,
    )
    assert final == fallback
    assert records[-1].result["selector_outcome"] == "ignored_mutation"
    assert records[-1].result["mutation_attempt"] is True


def test_archetype_tool_policy_and_authority_markers() -> None:
    wind, ev, retailer, p2h, generator, arbitrageur = build_personas(6)
    forecast = _forecast_state()
    p2h_executor = AgentToolExecutor(persona=p2h, forecast=forecast, data_tools=None, simulator_tool=None)
    feasibility = p2h_executor.execute(
        "get_bid_feasibility",
        {"side": "up", "quantity_mwh": 2.0, "limit_price_eur_mwh": 70.0},
    )
    assert feasibility.ok is True
    assert feasibility.result["kind"] == "bid_feasibility"
    assert feasibility.result["archetype"] == "p2h"
    assert feasibility.result["authority"] == "advisory"
    assert "score" in feasibility.result

    wind_executor = AgentToolExecutor(persona=wind, forecast=forecast, data_tools=None, simulator_tool=object())
    simulation = wind_executor.execute(
        "simulate_bid",
        {"side": "up", "quantity_mwh": 1.0, "limit_price_eur_mwh": 70.0},
    )
    assert simulation.ok is False
    assert simulation.result["error_code"] == "tool_not_allowed_for_archetype"

    ev_executor = AgentToolExecutor(persona=ev, forecast=forecast, data_tools=None, simulator_tool=None)
    ev_simulation = ev_executor.execute(
        "simulate_ev_bid",
        {"side": "up", "quantity_mwh": 0.5, "limit_price_eur_mwh": 70.0},
    )
    assert ev_simulation.ok is True
    assert ev_simulation.result["authority"] == "proxy_comparison"
    assert ev_simulation.result["backend"] == "proxy"
    assert ev_simulation.result["archetype"] == "ev"

    retailer_simulation = AgentToolExecutor(persona=retailer, forecast=forecast, data_tools=None, simulator_tool=None).execute(
        "simulate_ev_bid",
        {"side": "up", "quantity_mwh": 0.5, "limit_price_eur_mwh": 70.0},
    )
    assert retailer_simulation.ok is False
    assert retailer_simulation.result["error_code"] == "tool_not_allowed_for_archetype"

    wind_feasibility = wind_executor.execute(
        "get_wind_bid_feasibility",
        {"side": "up", "quantity_mwh": 0.5, "limit_price_eur_mwh": 70.0},
    )
    assert wind_feasibility.result["authority"] == "advisory"

    generator_feasibility = AgentToolExecutor(persona=generator, forecast=forecast, data_tools=None, simulator_tool=None).execute(
        "get_generator_bid_feasibility",
        {"side": "up", "quantity_mwh": 60.0, "limit_price_eur_mwh": 70.0},
    )
    assert generator_feasibility.result["authority"] == "advisory"
    assert "ramp_proxy_exceeded" in generator_feasibility.result["risk_flags"]

    spread = AgentToolExecutor(persona=arbitrageur, forecast=forecast, data_tools=None, simulator_tool=None).execute(
        "get_spread_opportunity",
        {"hours": 24, "zone": "DK1"},
    )
    assert spread.result["authority"] == "advisory"


def test_dryrun_writes_trace_and_summary(tmp_path: Path) -> None:
    config = SocietyRunConfig(
        run_id="pytest-dryrun",
        agent_count=4,
        ticks=2,
        forecaster_backend="ar1",
        verifier_mode="mock",
        output_dir=tmp_path,
        llm={"enabled": False},
    )
    run_dir = asyncio.run(run_society(config))
    trace_path = run_dir / "traces.jsonl"
    summary_path = run_dir / "summary.json"

    assert trace_path.exists()
    assert summary_path.exists()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    first = json.loads(lines[0])
    assert first["run_id"] == "pytest-dryrun"
    assert first["agent_id"] == "agent-000"
    assert first["observed_at"] <= first["timestamp"]
    assert first["agent_role"] == "action_agent"
    assert "watch_label" in first["decision"]
    assert first["tool_calls"] == []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "watched" in summary
    assert summary["accepted"] == 2
    assert summary["watched"] == 4
    assert summary["abstained"] == 2


def test_autonomous_unsupported_bid_is_downgraded() -> None:
    decision, records = _run_tool_decision(
        preprobe_mode="none",
        final_arguments={
            "action": "bid",
            "side": "up",
            "quantity_mwh": 1.0,
            "limit_price_eur_mwh": 70.0,
            "rationale": "bid without simulator",
            "confidence": 0.8,
        },
    )
    assert decision.action == "abstain"
    assert any(record.name == "selected_candidate_diagnostics" and record.provenance == "runner_diagnostic" for record in records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forecast_state() -> ForecastMarketState:
    return ForecastMarketState(
        delivery_timestamp="2025-03-04T03:00:00Z",
        zone="DK1",
        issued_at="2025-03-04T02:10:00Z",
        activation_direction="up",
        activation_volume_mwh=8.0,
        spot_price_eur_mwh=60.0,
        imbalance_price_lower_eur_mwh=55.0,
        imbalance_price_median_eur_mwh=75.0,
        imbalance_price_upper_eur_mwh=95.0,
        mfrr_up_price_lower_eur_mwh=55.0,
        mfrr_up_price_median_eur_mwh=75.0,
        mfrr_up_price_upper_eur_mwh=95.0,
        mfrr_down_price_lower_eur_mwh=45.0,
        mfrr_down_price_median_eur_mwh=55.0,
        mfrr_down_price_upper_eur_mwh=65.0,
        source=ForecastSource(kind="baseline_conformal", window_start="2025-03-01T00:00:00Z"),
    )


class _ScriptedToolLLM:
    def __init__(self, *, loop_calls: list[dict] | None = None, final_arguments: dict | None = None) -> None:
        self._loop_calls = list(loop_calls or [])
        self._final_arguments = final_arguments or {
            "action": "watch",
            "rationale": "forced final watch",
            "confidence": 0.5,
        }

    async def tool_round(self, messages, tools, tool_choice=None):  # type: ignore[no-untyped-def]
        if isinstance(tool_choice, dict):
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "forced-final",
                        "function": {
                            "name": "propose_action",
                            "arguments": json.dumps(self._final_arguments),
                        },
                    }
                ],
            }
        if self._loop_calls:
            call = self._loop_calls.pop(0)
            return {"role": "assistant", "tool_calls": [call]}
        return {"role": "assistant", "content": "", "tool_calls": []}


def _tick_context() -> TickContext:
    forecast = _forecast_state()
    return TickContext(
        timestamp=__import__("pandas").Timestamp(forecast.delivery_timestamp).to_pydatetime(),
        market_price_eur_mwh=forecast.spot_price_eur_mwh,
        forecast=forecast,
    )


def _run_tool_decision(
    *,
    preprobe_mode: str,
    objective: str = "bid_seeking",
    loop_calls: list[dict] | None = None,
    final_arguments: dict | None = None,
    persona=None,  # type: ignore[no-untyped-def]
):
    persona = persona or build_personas(1)[0]
    return asyncio.run(
        _decide_with_tools(
            persona,
            _tick_context(),
            _ScriptedToolLLM(loop_calls=loop_calls, final_arguments=final_arguments),  # type: ignore[arg-type]
            objective=objective,
            ablation_strategy="ranked_candidates",
            data_tools=None,
            simulator_tool=None,
            asset_simulator_mode="proxy",
            asset_proxy_style="market",
            asset_state_store=None,  # type: ignore[arg-type]
            simulator_semaphore=asyncio.Semaphore(1),
            max_tool_rounds=2,
            final_bid_guard="simulator_exact_match",
            safety_toolset="full",
            preprobe_mode=preprobe_mode,
        )
    )
