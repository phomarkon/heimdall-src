import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(os.getenv("HEIMDALL_VLLM_BASE_DIR", Path(__file__).resolve().parents[1]))
load_dotenv(BASE_DIR / ".env")

host = os.getenv("HEIMDALL_VLLM_HOST", "127.0.0.1")
port = os.getenv("HEIMDALL_VLLM_PORT", "8000")
api_key = os.getenv("HEIMDALL_VLLM_API_KEY", "heimdall-local")
model = os.getenv("HEIMDALL_MODEL", "Qwen/Qwen3-0.6B")
base_url = f"http://{host}:{port}/v1"

client = OpenAI(base_url=base_url, api_key=api_key)

tools = [
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
                    "delivery_quarter": {
                        "type": "string",
                        "description": "ISO-8601 timestamp, e.g. 2025-10-01T12:15:00Z",
                    },
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

resp = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a Heimdall BRP agent. You must produce exactly one structured bid "
                "by calling the propose_bid tool. Keep the rationale short."
            ),
        },
        {
            "role": "user",
            "content": (
                "Market state: DK1 mFRR, next delivery quarter 2025-10-01T12:15:00Z, "
                "forecast price interval [-20, 80] EUR/MWh. You operate a flexible P2H asset. "
                "Propose a cautious 1 MW bid."
            ),
        },
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "propose_bid"}},
    temperature=0,
    max_tokens=256,
)

msg = resp.choices[0].message
print("Raw message:")
print(msg)

if not msg.tool_calls:
    raise AssertionError("No tool_calls returned. Heimdall named tool-call test failed.")

call = msg.tool_calls[0]
print("Tool call name:", call.function.name)
print("Tool call arguments:")
print(call.function.arguments)

args = json.loads(call.function.arguments)
assert args["market"] in ["DA", "ID", "mFRR"]
assert args["direction"] in ["buy", "sell"]
assert isinstance(args["quantity_mw"], (int, float))
assert args["quantity_mw"] >= 0
assert isinstance(args["price_eur_per_mwh"], (int, float))
assert "delivery_quarter" in args
print("HEIMDALL_TOOL_CALL_OK")
