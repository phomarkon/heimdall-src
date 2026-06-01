# ollama-bigmodel-s06-20260522

Ollama-backed S06 big-model baseline screen over the three core April windows.

1. Generate/validate configs:
   `uv run python tools/experiments/generate_ollama_bigmodel_s06_20260522.py`
2. Install/start/pull Ollama models:
   `bash ai-society/configs/ollama-bigmodel-s06-20260522/setup_ollama.sh`
3. Launch smoke then full matrix:
   `bash ai-society/configs/ollama-bigmodel-s06-20260522/launch_ollama_matrix.sh`

The launcher writes `available-models.json`, `available-smoke.txt`, and `available-all.txt` after pull/serve checks. It skips models that cannot be pulled or listed by Ollama.
