import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(os.getenv("HEIMDALL_VLLM_BASE_DIR", Path(__file__).resolve().parents[1]))
load_dotenv(BASE_DIR / ".env")

host = os.getenv("HEIMDALL_VLLM_HOST", "127.0.0.1")
port = os.getenv("HEIMDALL_VLLM_PORT", "8000")
api_key = os.getenv("HEIMDALL_VLLM_API_KEY", "heimdall-local")
model = os.getenv("HEIMDALL_MODEL", "Qwen/Qwen3-0.6B")

base_url = f"http://{host}:{port}/v1"


def wait_for_models(timeout_s: int = 600) -> None:
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        try:
            r = httpx.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                print("Models endpoint OK:")
                print(r.text[:1000])
                return
            last_error = f"HTTP {r.status_code}: {r.text[:500]}"
        except Exception as e:
            last_error = repr(e)
        print(f"Waiting for vLLM... {last_error}")
        time.sleep(10)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s}s. Last error: {last_error}")


def test_chat() -> None:
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise test assistant."},
            {"role": "user", "content": "Reply with exactly: HEIMDALL_VLLM_OK"},
        ],
        temperature=0,
        max_tokens=32,
    )
    print("Chat completion OK:")
    print(resp.choices[0].message)


if __name__ == "__main__":
    print(f"Checking {base_url} model={model}")
    wait_for_models()
    test_chat()
