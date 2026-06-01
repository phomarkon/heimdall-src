"""Ablation programme. Per docs/RESEARCH-PROPOSAL.md §5.4.

Codes A5, A8, A9 are tractable without the LLM agent stack. A1-A4, A6-A7,
A10-A12 require the agent runtime that is owned by the peer agent (Tim);
those files are intentionally absent from this directory.

Forecaster-side micro-ablations are labelled AF1..AF4:
  - AF1: patchTST patch size
  - AF2: encoder depth
  - AF3: univariate vs multivariate (F7 vs F8 head-to-head)
  - AF4: ACI on / off (Theorem 1b vs raw quantile heads)
"""
