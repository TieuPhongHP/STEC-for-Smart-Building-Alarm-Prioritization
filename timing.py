"""Inference-cost comparison (single CPU thread): STEC features + HGB vs Event-GNN forward pass.
Timing does not depend on trained weights, so an untrained Event-GNN of the same size is used."""
import os, json, time, sys
os.environ.setdefault("STEC_REV_DIR", "/tmp/timing_tmp")
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision_experiments.py")).read()
head = src.split("# ---------------------------------------------------------------- E1")[0]
gnn_src = src[src.index("import torch, torch.nn as nn"):src.index("Ttr,Tva,Tte=tensorise")]
g = {"__file__": os.path.abspath("revision_experiments.py"), "__name__": "rx"}
exec(compile(head, "rx", "exec"), g)
subs = sorted({x for v in g["R"].NORMAL_SUBTYPES.values() for x in v} | {x[1] for t in g["R"].TEMPLATES.values() for x in t})
g["subs"] = subs
exec(compile(gnn_src.replace("torch.set_num_threads(2)", "torch.set_num_threads(1)"), "gnn", "exec"), g)
R = g["R"]; W = g["W_te"][:5000]; stec = g["stec"]
def t_stec():
    t = time.perf_counter(); X = pd.DataFrame([g["features"](w) for w in W]); stec.predict_proba(X[R.ALL_FEATURES]); return time.perf_counter() - t
model = g["EventGNN"](6 + len(subs) + 12 + 3, False); model.eval()
def t_gnn():
    t = time.perf_counter(); T = g["tensorise"](W)
    with torch.no_grad():
        for i in range(0, len(W), 1024): model(*[x[i:i+1024] for x in T[:5]])
    return time.perf_counter() - t
ts = min(t_stec() for _ in range(3)); tg = min(t_gnn() for _ in range(3))
nodes = sum(p.tree_.node_count if hasattr(p, "tree_") else p.nodes.shape[0] for pred in stec._predictors for p in pred)
res = {"n_windows": len(W), "stec_ms_per_1000": 1000 * ts / len(W) * 1000 / 1000 * 1, "gnn_ms_per_1000": 1000 * tg / len(W) * 1000 / 1000 * 1,
       "hgb_nodes": int(nodes), "gnn_params": int(sum(p.numel() for p in model.parameters()))}
res["stec_ms_per_1000"] = 1000 * ts * 1000 / len(W); res["gnn_ms_per_1000"] = 1000 * tg * 1000 / len(W)
print(res); json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing.json"), "w"), indent=2)
