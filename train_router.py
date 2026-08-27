#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 OSSP Router Team
# SPDX-License-Identifier: Apache-2.0

"""train_router.py — artifact 학습·저장. 저장소 루트에서 실행."""
import argparse, json, os, time
import numpy as np, joblib
import features as FE, core

def load(split, root):
    eps=json.load(open(f"{root}/data/materialized/{split}/inputs.json",encoding="utf-8"))["episodes"]
    out={e["episode_id"]:e for e in json.load(open(f"{root}/data/{split}/outcomes.json",encoding="utf-8"))["episodes"]}
    rows=[]
    for e in eps:
        o=out[e["episode_id"]]["models"]
        r=dict(id=e["episode_id"],p=FE.episode_text(e),ng=o["ax31"]["num_generations"],
               it=[o[m]["input_tokens"] for m in core.MODELS],
               ot=[o[m]["output_tokens"] for m in core.MODELS],
               S=[float(o[m]["score"]) for m in core.MODELS])
        r["C"]=[(r["it"][k]*core.RATE_IN[k]+r["ot"][k]*core.RATE_OUT[k])/core.TOKEN_UNIT for k in range(3)]
        rows.append(r)
    return rows

ap=argparse.ArgumentParser()
ap.add_argument("--repo-root",default=".")
ap.add_argument("--out",default="artifacts/router.joblib")
ap.add_argument("--splits",nargs="+",default=["train"])
ap.add_argument("--tier-params",default=None,help="JSON. 미지정 시 기본값")
a=ap.parse_args()
t0=time.time()
rows=[r for s in a.splits for r in load(s,a.repo_root)]
print(f"학습 문항 {len(rows)}",flush=True)
texts=[r["p"] for r in rows]
F,X,E,tags,cut=FE.build_matrices(texts)
ng=np.array([r["ng"] for r in rows]); IT=np.array([r["it"] for r in rows],float)
OT=np.array([r["ot"] for r in rows],float); S=np.array([r["S"] for r in rows])
Ctrue=np.array([r["C"] for r in rows])
folds=np.arange(len(rows))%5; np.random.default_rng(0).shuffle(folds)
art=core.train(F,X,E,tags,cut,ng,IT,OT,S,Ctrue,folds)
art.pop("oof_logot",None)
art["tier_params"]=json.loads(a.tier_params) if a.tier_params else {
    "fast":{"target":1.10,"inflate":1.00},
    "balanced":{"target":1.60,"inflate":1.00},
    "premium":{"target":3.20,"inflate":1.05}}
art["meta"]={"n_train":len(rows),"splits":a.splits,"hash":"fnv1a64"}
os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
joblib.dump(art,a.out,compress=3)
print(f"저장 {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB  t={time.time()-t0:.0f}s")
