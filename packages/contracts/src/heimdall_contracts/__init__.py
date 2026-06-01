"""Heimdall shared contracts.

Pydantic v2 schemas used across services. All exports are part of the public
API — services and the TypeScript frontend mirror these shapes.

Active Python consumers: BidAction, ConformalInterval, VerifierVerdict,
Persona, PersonaArchetype, RiskAttitude, QuantileForecast,
ActivationForecast, MarketClearing, MarketState, PhysicalViolation.

Frontend-only / reserved for future services: AgentMessage, AgentTrace,
BidDirection, LoadForecast, MarketKind, NegotiationKind, PriceForecast,
StageFailed, WeatherForecast.
"""

from heimdall_contracts.agents import (
    AgentMessage,
    AgentTrace,
    NegotiationKind,
    Persona,
    PersonaArchetype,
    RiskAttitude,
)
from heimdall_contracts.bids import BidAction, BidDirection, MarketKind
from heimdall_contracts.forecasts import (
    ActivationForecast,
    ConformalInterval,
    LoadForecast,
    PriceForecast,
    QuantileForecast,
    WeatherForecast,
)
from heimdall_contracts.market import MarketClearing, MarketState
from heimdall_contracts.verifier import (
    PhysicalViolation,
    StageFailed,
    VerifierVerdict,
)

__all__ = [
    "AgentMessage",
    "AgentTrace",
    "ActivationForecast",
    "BidAction",
    "BidDirection",
    "ConformalInterval",
    "LoadForecast",
    "MarketClearing",
    "MarketKind",
    "MarketState",
    "NegotiationKind",
    "Persona",
    "PersonaArchetype",
    "PhysicalViolation",
    "PriceForecast",
    "QuantileForecast",
    "RiskAttitude",
    "StageFailed",
    "VerifierVerdict",
    "WeatherForecast",
]
