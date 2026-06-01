"""D8: directional accuracy + capture by model (q32 vs q72), vs majority-class baseline."""
from __future__ import annotations
import argparse, glob, json
import pandas as pd
from tools.evaluation.evaluate_society_run import _load_truth, _load_traces, _score_bids
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--glob",action="append",required=True)
    ap.add_argument("--truth-dir",default="data/cache/evaluation_truth/april_2026"); ap.add_argument("--json-out")
    a=ap.parse_args()
    truth=_load_truth(f"{a.truth_dir}/activation_truth.parquet")
    td={(pd.Timestamp(r["timestamp_utc"]),r["zone"]):r["activation_direction"] for _,r in truth.iterrows() if r["activation_direction"] in ("up","down")}
    dirs=[]; 
    for g in a.glob: dirs+=glob.glob(g)
    agg={}
    for d in sorted(set(dirs)):
        arm="q72" if "-q72-" in d else "q32" if "-q32-" in d else "?"
        try: tr=_load_traces(__import__("pathlib").Path(d)/"traces.jsonl")
        except Exception: continue
        if tr.empty: continue
        x=agg.setdefault(arm,{"n":0,"cor":0,"bids":0,"runs":0,"profit":0.0})
        x["runs"]+=1
        for _,r in tr.iterrows():
            if r.get("action")=="bid" and r.get("side") in ("up","down"):
                k=(pd.Timestamp(r["timestamp_utc"]),r.get("zone","DK1"))
                if k in td: x["n"]+=1; x["cor"]+=(r["side"]==td[k]); x["bids"]+=1
        b=_score_bids(tr,truth)
        x["profit"]+=float(pd.to_numeric(b["realized_profit_eur"],errors="coerce").fillna(0).sum())
    # majority-class baseline over the union of evaluated ticks
    out={}
    print(f"\n{'arm':<6}{'runs':>5}{'sided_bids':>11}{'side_acc':>10}{'profit/run':>11}")
    for arm in ("q32","q72"):
        if arm not in agg: continue
        x=agg[arm]; acc=x["cor"]/x["n"] if x["n"] else float("nan")
        print(f"{arm:<6}{x['runs']:>5}{x['n']:>11}{acc:>10.3f}{x['profit']/max(1,x['runs']):>11.1f}")
        out[arm]={"runs":x["runs"],"sided_bids":x["n"],"side_accuracy":round(acc,4),"profit_per_run":round(x["profit"]/max(1,x["runs"]),2)}
    print("majority-class (always-down) baseline ~0.647; q72 must beat BOTH q32 and 0.647 to be a capability win.")
    if a.json_out:
        __import__("pathlib").Path(a.json_out).write_text(json.dumps(out,indent=2))
        print("wrote",a.json_out)
if __name__=="__main__": main()
