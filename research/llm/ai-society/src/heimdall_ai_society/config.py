from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from heimdall_contracts import PersonaArchetype
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.config import load_project_env


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    provider: Literal["vllm", "ollama", "openrouter", "openai_compatible"] = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    base_urls: list[str] | None = Field(default=None, min_length=1)
    api_key: str = "heimdall-local"
    http_referer: str | None = None
    app_title: str | None = None
    model: str = "Qwen/Qwen3-32B"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=350, ge=64, le=4096)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_concurrency: int = Field(default=4, ge=1, le=256)
    per_endpoint_max_concurrency: int | None = Field(default=None, ge=1, le=256)
    require_served_model_match: bool = True
    supports_response_format: bool = True
    supports_tools: bool = True

    @property
    def endpoint_urls(self) -> list[str]:
        return self.base_urls or [self.base_url]


class WeatherLocationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float
    longitude: float


class RAGConfig(BaseModel):
    """Retrieval-augmented generation over the leak-safe knowledge corpus.

    When ``enabled``, agents get a ``retrieve_knowledge`` tool whose temporal
    cutoff is forced by the runner to the current tick, so retrieval can never
    return a document dated after the decision (see ``rag.py``). Default off:
    when disabled the society behaves byte-identically to a no-RAG run.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    corpus_path: Path | None = None
    backend: Literal["dense", "tfidf"] = "dense"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    cache_dir: Path | None = None
    top_k: int = Field(default=4, ge=1, le=12)
    max_doc_chars: int = Field(default=700, ge=100, le=4000)


class SocietyRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    seed: int = 42
    forecaster_seed: int | None = None
    zone: Literal["DK1", "DK2"] = "DK1"
    agent_count: int = Field(default=5, ge=1, le=1000)
    archetype_cycle: list[PersonaArchetype] | None = None
    ticks: int = Field(default=4, ge=1, le=1344)
    start_timestamp: datetime = datetime(2025, 3, 4, 3, 0, tzinfo=UTC)
    forecaster_backend: Literal[
        "ar1",
        "f0",
        "f1_lgbm",
        "f2_blr",
        "f3",
        "f3_ensemble",
        "f3_lite",
        "f4_mc_dropout",
        "f5",
        "f6",
        "f7",
        "f7_optuna",
        "f7_patch4",
        "f7_patch8",
        "f7_patch16",
        "f8",
        "f8b",
        "f8c",
        "f8d",
        "f8e",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
    ] = "f0"
    forecaster_routing_mode: Literal["persona", "run_level"] = "persona"
    chooser_mode: Literal[
        "llm",
        "deterministic_best_accepted",
        "deterministic_high_fill_accepted",
        "llm_fill_selector",
        "deterministic_llm_critic",
        "deterministic_watch_threshold",
    ] = "llm"
    verifier_mode: Literal["mock", "simulator"] = "mock"
    market_context: Literal["synthetic", "real"] = "synthetic"
    tool_mode: Literal["json_response", "openai_tools"] = "json_response"
    preprobe_mode: Literal["full", "context_only", "specialist_context", "none"] = "full"
    objective: Literal["worth_bidding", "bid_seeking", "stress_test", "unverified_bid_seeking"] = "worth_bidding"
    final_bid_guard: Literal["simulator_exact_match", "schema_only_shadow"] = "simulator_exact_match"
    safety_toolset: Literal["full", "context_only"] = "full"
    ablation_strategy: Literal[
        "baseline",
        "direction_prior",
        "both_side_probes",
        "price_ladder",
        "ranked_candidates",
        "rejection_explain",
        "risk_trio",
        "price_styles",
        "committee_vote",
        "random_persona_10",
        "mixed_advisory",
        "ranked_committee",
        "cp01_aggressive_clear_ladder",
        "cp02_balanced_clear_ladder",
        "cp03_wide_price_ladder",
        "cp04_clearprob_ranked",
        "cp05_profit_clear_tradeoff",
        "cp06_side_clear_joint",
        "cp07_downside_first",
        "cp08_quantity_price_grid",
        "cp09_watch_threshold_low",
        "cp10_watch_threshold_high",
        "cp11_llm_suggest_candidates",
        "cp12_llm_suggest_plus_code_ladder",
        "cp12_delivery_risk_aware",
        "cp13_llm_probe_refine_frontier",
        "deterministic_rich",
        "diverse_action_society",
        "market_intelligence_society",
        "comm_broadcast_digest",
        "comm_broadcast_digest_risk_filter",
        "comm_broadcast_digest_priority_calibration",
        "comm_info_then_action",
        "comm_peer_signal",
        "comm_retry_council",
        "comm_deliberation_protocol",
        "comm_central_supervisor",
        "comm_society_chair",
        "comm_society_chair_2agree",
        "comm_society_chair_riskveto",
        "comm_society_chair_intel",
    ] = "baseline"
    persona_profile: Literal[
        "default",
        "risk_trio",
        "price_styles",
        "side_specialists",
        "committee",
        "random_p2h",
        "mixed_advisory",
        "diverse_action",
        "diverse_expert_action",
        "all_archetypes_v1",
        "all_archetypes_double_v1",
        "all_archetypes_double_homo",
        "all_archetypes_plus_info_v1",
        "all_archetypes_double_plus_info_v1",
        "action_core_8",
        "action_core_9_chair",
        "action_core_10_safety",
        "action_core_8_aggressive",
        "action_core_8_safety",
        "action_core_8_toolsplit",
        "balanced_intelligence",
        "crowd_intelligence",
        "market_expert_panel",
        "p2h_specialist_v2",
        "ev_specialist_v2",
        "action_core_8_plus_market_expert",
        "p2h_info_then_action_v2",
        "ev_info_then_action_v2",
        "market_experts_plus_action_core_6",
        "info_specialists_v1",
        "action_core_8_plus_info_specialists",
        "jao_grid_v1",
        "mixed_expert_18_sideaware",
        "mixed_expert_20_sideaware",
    ] = "default"
    scenario_id: Literal["p2h_dk1_pypsa"] = "p2h_dk1_pypsa"
    tool_policy: Literal["p2h_only_simulator", "proxy_simulator", "asset_simulator_v1"] = "p2h_only_simulator"
    asset_simulator_mode: Literal[
        "proxy",
        "scenario_envelope",
        "real",
        "pypsa_background",
        "dual_compare_proxy_controls",
        "dual_compare_real_controls",
        "dual_compare_pypsa_controls",
        "dual_compare_real_vs_pypsa",
    ] = "proxy"
    asset_proxy_style: Literal["market", "asset_light"] = "market"
    candidate_sizing_mode: Literal["current", "medium", "large"] = "current"
    candidate_sizing_cap_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    candidate_sizing_min_mwh: float = Field(default=0.25, gt=0.0, le=1000.0)
    candidate_sizing_max_candidates: int = Field(default=8, ge=1, le=32)
    bid_budget_enabled: bool = False
    bid_budget_per_agent: int = Field(default=4, ge=0, le=1344)
    bid_budget_scope: Literal["agent"] = "agent"
    bid_budget_history_ticks: int = Field(default=3, ge=0, le=24)
    max_tool_rounds: int = Field(default=4, ge=1, le=12)
    deliberation_inquiry_rounds: int = Field(default=1, ge=1, le=3)
    deliberation_action_rounds: int = Field(default=1, ge=1, le=3)
    deliberation_min_tool_calls: int = Field(default=1, ge=0, le=8)
    deliberation_require_action_probe: bool = True
    deliberation_max_peer_notes: int = Field(default=12, ge=1, le=64)
    supervisor_soft_quota_per_24_ticks: int = Field(default=6, ge=0, le=24)
    supervisor_max_orders_per_tick: int = Field(default=1, ge=1, le=4)
    simulator_max_concurrency: int = Field(default=8, ge=1, le=256)
    verifier_tau_eur: float = -100.0
    data_start: datetime | None = None
    data_end: datetime | None = None
    context_dataset_dir: Path | None = None
    default_lookback_hours: int = Field(default=24, ge=1, le=168)
    cache_refresh: bool = False
    data_cache_dir: Path | None = None
    weather_locations: dict[str, WeatherLocationConfig] = Field(default_factory=dict)
    output_dir: Path = Path("research/llm/ai-society/runs")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    memory_enabled: bool = False
    memory_bank_path: Path | None = None
    memory_scope_filter: Literal["all", "archetype", "agent", "synthesis"] = "all"
    memory_max_items_per_agent: int = Field(default=5, ge=1, le=50)
    memory_max_prompt_chars: int = Field(default=2000, ge=100, le=20000)
    reviewer_mode: Literal["code_only", "hybrid_llm"] = "code_only"
    seed_outage_context: bool = False
    # Optional extra instruction appended to the final propose_action prompt (rationale-shaping
    # experiments, e.g. "always state any present grid/border/outage constraint and its effect" or
    # "state why you rejected the opposite side"). Empty = unchanged behaviour.
    rationale_directive: str = ""

    @field_validator("start_timestamp")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("archetype_cycle")
    @classmethod
    def _non_empty_archetype_cycle(
        cls,
        value: list[PersonaArchetype] | None,
    ) -> list[PersonaArchetype] | None:
        if value == []:
            raise ValueError("archetype_cycle must be omitted or contain at least one archetype")
        return value

    @model_validator(mode="after")
    def _tool_policy_mode_aliases(self) -> SocietyRunConfig:
        if self.asset_simulator_mode == "real":
            self.asset_simulator_mode = "scenario_envelope"
        if self.tool_policy == "asset_simulator_v1" and self.asset_simulator_mode == "proxy":
            self.asset_simulator_mode = "scenario_envelope"
        elif self.tool_policy == "proxy_simulator":
            self.asset_simulator_mode = "proxy"
        if self.ablation_strategy == "comm_deliberation_protocol":
            if self.chooser_mode != "llm":
                raise ValueError("comm_deliberation_protocol requires chooser_mode='llm'")
            if self.tool_mode != "openai_tools":
                raise ValueError("comm_deliberation_protocol requires tool_mode='openai_tools'")
        if self.llm.enabled and self.tool_mode == "openai_tools" and not self.llm.supports_tools:
            raise ValueError("tool_mode='openai_tools' requires llm.supports_tools=true")
        if self.rag.enabled:
            if self.tool_mode != "openai_tools":
                raise ValueError("rag.enabled requires tool_mode='openai_tools'")
            if self.rag.corpus_path is None:
                raise ValueError("rag.enabled requires rag.corpus_path")
        return self


def load_config(path: str | Path) -> SocietyRunConfig:
    load_project_env(Path(path).resolve())
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    _apply_env_defaults(raw)
    return SocietyRunConfig.model_validate(raw)


def _apply_env_defaults(raw: dict) -> None:
    llm = raw.setdefault("llm", {})
    provider = llm.get("provider", "vllm")
    if provider == "openrouter":
        llm.setdefault("base_url", os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        llm.setdefault("api_key", os.environ.get("OPENROUTER_API_KEY", ""))
        llm.setdefault("http_referer", os.environ.get("OPENROUTER_HTTP_REFERER"))
        llm.setdefault("app_title", os.environ.get("OPENROUTER_APP_TITLE", "Heimdall AI Society"))
        llm.setdefault("require_served_model_match", False)
    else:
        llm.setdefault("base_url", os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
        llm.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "heimdall-local"))
    llm.setdefault("model", os.environ.get("HEIMDALL_LLM_MODEL", "Qwen/Qwen3-32B"))
    locations = raw.setdefault("weather_locations", {})
    locations.setdefault(
        "DK1",
        {
            "latitude": float(os.environ.get("HEIMDALL_DK1_LATITUDE", "56.2639")),
            "longitude": float(os.environ.get("HEIMDALL_DK1_LONGITUDE", "9.5018")),
        },
    )
    locations.setdefault(
        "DK2",
        {
            "latitude": float(os.environ.get("HEIMDALL_DK2_LATITUDE", "55.6761")),
            "longitude": float(os.environ.get("HEIMDALL_DK2_LONGITUDE", "12.5683")),
        },
    )
