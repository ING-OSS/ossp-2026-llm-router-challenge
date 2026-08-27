#!/usr/bin/env python3
"""router_run.py — 추론 전용. router-run --input ... --tier ... --output ..."""
import os
# BLAS/OpenMP 스레드 수를 먼저 고정한다. 공식 한도는 프로세스+스레드 합계 32개이며
# numpy/scipy/sklearn 은 import 시점에 스레드 풀 크기를 정하므로 import 보다 앞서야 한다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import argparse, json, sys, tempfile, time
import numpy as np, joblib
import features as FE, core

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--tier",required=True,
                    choices=["fast","balanced","premium"])
    ap.add_argument("--output",required=True)
    ap.add_argument("--artifact",default=os.environ.get("ROUTER_ARTIFACT","/challenge/artifacts/router.joblib"))
    a=ap.parse_args()
    try:
        t0=time.time()
        inp=json.load(open(a.input,encoding="utf-8"))
        eps=inp["episodes"]
        art=joblib.load(a.artifact)
        # 학습 시 n_jobs=-1 로 저장된 앙상블을 단일 프로세스로 강제한다.
        for _key in ("ot_et", "sc_et"):
            for _m in art.get(_key, []):
                _m.n_jobs = 1
        texts=[FE.episode_text(e) for e in eps]
        F,X,E,tags,cut=FE.build_matrices(texts)
        Pl,trees,C=core.predict(art,F,X,E,cut)
        q=art.get("robust_q",core.ROBUST_Q).get(a.tier)
        P=core.score_matrix(Pl,trees,q)
        tp=art["tier_params"][a.tier]
        sel=core.allocate(P,C,tags,a.tier,tp["target"],tp["inflate"])
        sub={"challenge_id":inp["challenge_id"],
             "decisions":[{"episode_id":e["episode_id"],"model_id":core.MODELS[int(s)]}
                          for e,s in zip(eps,sel)],
             "policy_id":"ossp-2026-prompt-router-v1",
             "schema_version":inp["schema_version"],
             "split":inp["split"],"tier":a.tier}
        d=os.path.dirname(os.path.abspath(a.output)) or "."
        fd,tmp=tempfile.mkstemp(dir=d,prefix=".sub",suffix=".json")
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(sub,f,ensure_ascii=False,indent=2,sort_keys=True)
        os.chmod(tmp,0o644); os.replace(tmp,a.output)
        print(f"OK: {a.tier} {len(eps)}문항 {time.time()-t0:.1f}s",file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"오류: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
