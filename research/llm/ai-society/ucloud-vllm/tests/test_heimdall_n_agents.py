import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

BASE_DIR = Path(os.getenv("HEIMDALL_VLLM_BASE_DIR", Path(__file__).resolve().parents[1]))
load_dotenv(BASE_DIR / ".env")

host = os.getenv("HEIMDALL_VLLM_HOST", "127.0.0.1")
port = os.getenv("HEIMDALL_VLLM_PORT", "8000")
api_key = os.getenv("HEIMDALL_VLLM_API_KEY", "heimdall-local")
model = os.getenv("HEIMDALL_MODEL", "Qwen/Qwen3-0.6B")
base_url = f"http://{host}:{port}/v1"

client = AsyncOpenAI(base_url=base_url, api_key=api_key)

tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "propose_bid",
            "description": "Submit one structured Heimdall market bid proposal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {"type": "string", "enum": ["DA", "ID", "mFRR"]},
                    "direction": {"type": "string", "enum": ["buy", "sell"]},
                    "quantity_mw": {"type": "number", "minimum": 0},
                    "price_eur_per_mwh": {"type": "number"},
                    "delivery_quarter": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "market",
                    "direction",
                    "quantity_mw",
                    "price_eur_per_mwh",
                    "delivery_quarter",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    }
]


async def run_agent(agent_id: int) -> dict[str, Any]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are Heimdall BRP agent {agent_id}. You must produce exactly one "
                    "structured bid by calling the propose_bid tool. Keep the rationale short."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Market state: DK1 mFRR, next delivery quarter 2025-10-01T12:15:00Z, "
                    "forecast price interval [-20, 80] EUR/MWh. You operate a flexible P2H asset. "
                    f"Propose a cautious {agent_id + 1} MW bid."
                ),
            },
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "propose_bid"}},
        temperature=0,
        max_tokens=256,
    )

    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise AssertionError(f"agent {agent_id}: no tool_calls returned")

    call = msg.tool_calls[0]
    if call.function.name != "propose_bid":
        raise AssertionError(f"agent {agent_id}: unexpected tool {call.function.name}")

    args = json.loads(call.function.arguments)
    assert args["market"] in ["DA", "ID", "mFRR"]
    assert args["direction"] in ["buy", "sell"]
    assert isinstance(args["quantity_mw"], (int, float))
    assert args["quantity_mw"] >= 0
    assert isinstance(args["price_eur_per_mwh"], (int, float))
    assert "delivery_quarter" in args
    return args


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=5)
    args = parser.parse_args()

    start = time.perf_counter()
    results = await asyncio.gather(*(run_agent(i) for i in range(args.agents)))
    elapsed = time.perf_counter() - start
    print(f"HEIMDALL_{args.agents}_AGENT_TOOL_CALL_OK model={model} elapsed_s={elapsed:.2f}")
    for i, result in enumerate(results):
        print(f"agent={i} args={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    asyncio.run(main())
