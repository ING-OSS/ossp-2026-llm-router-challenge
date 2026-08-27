# SPDX-FileCopyrightText: Copyright 2026 OSSP Router Team
# SPDX-License-Identifier: Apache-2.0

"""core.py — 비용모델 / score head / 배분기 / artifact 직렬화."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse import hstack, csr_matrix, vstack

MODELS = ["ax31-light", "ax31", "axk1-think"]
RATE_IN = np.array([1.0, 2.127, 6.565])
RATE_OUT = np.array([4.0, 8.509, 26.260])
TOKEN_UNIT = 1e6
TIER_LIMIT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
TIER_WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
NO_UP = ["ko_reason"]
NO_UP_FAST = ["ko_reason", "longctx"]
# think 는 premium 뿐 아니라 balanced 에서도 허용한다(오라클이 balanced 에서 K1 을 189문항 사용).
# fast 는 예산 여유(0.25)가 K1 1문항 비용에도 못 미쳐 차단한다.
THINK_TIERS = ("balanced", "premium")
# think 금지 클러스터: 승격 효율이 0 에 가깝거나(longctx 0.0008, ko_reason 0.001)
# think 가 light 보다 나쁘고(ruletaker), K1 1문항이 전체 예산의 35% 를 먹는(latexmath) 경우.
NO_THINK = ["longctx", "ko_reason", "latexmath", "ruletaker"]

KW_GBM = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=1.0, random_state=0)
ET_OT = dict(n_estimators=400, max_depth=20, min_samples_leaf=2,
             max_features=0.5, random_state=0, n_jobs=-1)
ET_SC = dict(n_estimators=220, max_depth=10, min_samples_leaf=5,
             max_features=0.7, random_state=0, n_jobs=-1)
# nested OOF(비용 보정까지 fold 밖) 판정: balanced/premium 승률 100% (w>=0.35).
# 기존 train OOF 는 ExtraTrees 입력(logot)이 train 전체 보정을 거쳐 ET 를 부당히
# 불리하게 평가했다. nested 와 dev 가 같은 방향을 가리켜 채택.
W_ET_SCORE = 0.40
# Robust routing (Markovic-Voronov et al. 2026, arXiv:2603.26796):
# 점추정 대신 예측구간 하한을 쓰면 불확실성이 큰 선택을 체계적으로 회피한다.
# 부트스트랩 재적합 대신 ExtraTrees 트리별 예측의 분위를 사용 (추가 비용 0).
# None = 점추정. tier 마다 다른 값을 쓴다.
# train OOF 부분표본(880x6) 승률 0~33% 로 기각 -> 전부 점추정 사용.
ROBUST_Q = {"fast": None, "balanced": None, "premium": None}
# 게이트 적용 여부. train OOF 승률: fast 해제 100%, balanced 해제 50%(무차별->유지), premium 해제 83%.
USE_GATE = {"fast": False, "balanced": True, "premium": False}
# K=300/prior=3 은 train OOF 승률이 높았으나 dev 총합 편향이 악화(light 1.078 -> 1.190).
# k-means 보정계수는 train 전체로 산출되므로 train OOF 판정이 in-sample 이다. 200/10 유지.
KM_K, KM_PRIOR = 200, 10.0


# ----------------------------------------------------------------- 학습
def train(F, X, E, tags, cut, ng, IT, OT, S, C_true, folds):
    art = {}
    m = len(F)
    Y = np.column_stack([np.log1p(OT[:, k] / ng) for k in range(3)])
    X1 = np.hstack([X, E])

    # out_tok: GBM(OOF) → ExtraTrees(다른 두 모델 예측을 특징으로) → 0.5/0.5 앙상블
    b_o = np.zeros((m, 3))
    gbm = []
    for k in range(3):
        for f in np.unique(folds):
            msk = folds != f
            b_o[folds == f, k] = HistGradientBoostingRegressor(**KW_GBM).fit(
                X1[msk], Y[msk, k]).predict(X1[folds == f])
        gbm.append(HistGradientBoostingRegressor(**KW_GBM).fit(X1, Y[:, k]))
    art["ot_gbm"] = gbm

    oof = np.zeros((m, 3))
    ets = []
    for k in range(3):
        o = [j for j in range(3) if j != k]
        Rt = np.hstack([F[:, 8:], E, b_o[:, [o[0]]], b_o[:, [o[1]]], b_o[:, [o[0]]] - b_o[:, [o[1]]]])
        e_o = np.zeros(m)
        for f in np.unique(folds):
            msk = folds != f
            e_o[folds == f] = ExtraTreesRegressor(**ET_OT).fit(Rt[msk], Y[msk, k]).predict(Rt[folds == f])
        ets.append(ExtraTreesRegressor(**ET_OT).fit(Rt, Y[:, k]))
        oof[:, k] = 0.5 * b_o[:, k] + 0.5 * e_o
    art["ot_et"] = ets

    art["ng"] = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(F, ng)
    art["it"] = [Ridge(alpha=1.0).fit(F[:, :8], IT[:, k] / ng) for k in range(3)]

    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         max_features=60000, sublinear_tf=True)
    A = normalize(vc.fit_transform(cut))
    km = MiniBatchKMeans(n_clusters=KM_K, random_state=0, n_init=5, batch_size=256).fit(A)
    ct = km.predict(A)
    art["km_vec"], art["km"] = vc, km

    OTg = OT / ng[:, None]
    gm = np.array([OTg[:, k].sum() / np.expm1(oof[:, k]).sum() for k in range(3)])
    SM = {}
    for c in set(ct.tolist()):
        msk = ct == c
        w = msk.sum() / (msk.sum() + KM_PRIOR)
        SM[int(c)] = (w * np.array([OTg[msk, k].sum() / max(np.expm1(oof[msk, k]).sum(), 1e-9)
                                    for k in range(3)]) + (1 - w) * gm)
    art["SM"], art["gm"] = SM, gm

    Ca = _cost_from(art, oof, F, ct)
    art["kappa"] = np.array([C_true[:, k].sum() / Ca[:, k].sum() for k in range(3)])
    Ca = Ca * art["kappa"]

    # score head
    Rt = _rich(art, oof, F, Ca)
    art["sd_scaler"] = StandardScaler().fit(F[:, 8:])
    art["cf_scaler"] = StandardScaler().fit(Rt)
    art["vw"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                max_features=60000, sublinear_tf=True)
    art["vc2"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=120000, sublinear_tf=True)
    Zt = hstack([art["vw"].fit_transform(cut), art["vc2"].fit_transform(cut),
                 csr_matrix(art["sd_scaler"].transform(F[:, 8:]) * 0.3),
                 csr_matrix(art["cf_scaler"].transform(Rt))]).tocsr()
    art["logit"] = []
    for k in range(3):
        y = S[:, k]; wp, wn = ng * y, ng * (1 - y)
        kp, kn = wp > 0, wn > 0
        ZZ = vstack([Zt[kp], Zt[kn]]).tocsr()
        yy = np.concatenate([np.ones(kp.sum()), np.zeros(kn.sum())])
        art["logit"].append(LogisticRegression(C=0.5, max_iter=2000, solver="liblinear").fit(
            ZZ, yy, sample_weight=np.concatenate([wp[kp], wn[kn]])))
    Rs = np.hstack([F[:, 8:], oof, oof[:, [2]] - oof[:, [0]], oof[:, [1]] - oof[:, [0]]])
    art["sc_et"] = [ExtraTreesRegressor(**ET_SC).fit(Rs, S[:, k]) for k in range(3)]
    art["oof_logot"] = oof
    return art


# ----------------------------------------------------------------- 추론
def _cost_from(art, logot, F, clusters):
    g = art["ng"].predict(F).astype(float)
    it = np.column_stack([np.maximum(art["it"][k].predict(F[:, :8]), 1) for k in range(3)])
    sm = np.array([art["SM"].get(int(c), art["gm"]) for c in clusters])
    ot = np.expm1(np.clip(logot, 0, 14)) * sm
    C = np.column_stack([g * (it[:, k] * RATE_IN[k] + ot[:, k] * RATE_OUT[k]) / TOKEN_UNIT
                         for k in range(3)])
    C[:, 2] = np.maximum(C[:, 2], C[:, 1])
    return C


def _rich(art, logot, F, C):
    g = art["ng"].predict(F).astype(float).reshape(-1, 1)
    it = np.column_stack([np.maximum(art["it"][k].predict(F[:, :8]), 1) for k in range(3)])
    Cc = np.maximum(C, 1e-12)
    return np.hstack([g, np.log(it), logot,
                      logot[:, [2]] - logot[:, [0]], logot[:, [1]] - logot[:, [0]],
                      np.log(Cc), np.log(Cc[:, [2]] / Cc[:, [0]]), np.log(Cc[:, [1]] / Cc[:, [0]])])


def predict(art, F, X, E, cut):
    X1 = np.hstack([X, E])
    b = np.column_stack([art["ot_gbm"][k].predict(X1) for k in range(3)])
    logot = np.zeros_like(b)
    for k in range(3):
        o = [j for j in range(3) if j != k]
        R = np.hstack([F[:, 8:], E, b[:, [o[0]]], b[:, [o[1]]], b[:, [o[0]]] - b[:, [o[1]]]])
        logot[:, k] = 0.5 * b[:, k] + 0.5 * art["ot_et"][k].predict(R)
    cl = art["km"].predict(normalize(art["km_vec"].transform(cut)))
    C = _cost_from(art, logot, F, cl) * art["kappa"]
    R = _rich(art, logot, F, C)
    Z = hstack([art["vw"].transform(cut), art["vc2"].transform(cut),
                csr_matrix(art["sd_scaler"].transform(F[:, 8:]) * 0.3),
                csr_matrix(art["cf_scaler"].transform(R))]).tocsr()
    Pl = np.clip(np.column_stack([m.predict_proba(Z)[:, 1] for m in art["logit"]]), 0, 1)
    Rs = np.hstack([F[:, 8:], logot, logot[:, [2]] - logot[:, [0]], logot[:, [1]] - logot[:, [0]]])
    trees = [np.column_stack([t.predict(Rs) for t in art["sc_et"][k].estimators_]) for k in range(3)]
    return Pl, trees, C


def score_matrix(Pl, trees, q):
    """q=None 이면 트리 평균(점추정), 아니면 트리 분포의 q% 분위(robust 하한)."""
    if q is None:
        Pe = np.column_stack([t.mean(axis=1) for t in trees])
    else:
        Pe = np.column_stack([np.percentile(t, q, axis=1) for t in trees])
    return np.clip((1 - W_ET_SCORE) * Pl + W_ET_SCORE * np.clip(Pe, 0, 1), 0, 1)


def allocate(P, C, tags, tier, target, inflate):
    n = len(P)
    X = C.copy(); X[:, 1:] = X[:, 1:] * inflate
    lt = C[:, 0].sum(); cap = lt * target
    mk = np.ones((n, 3), bool)
    if USE_GATE.get(tier, True):
        bad = np.isin(tags, NO_UP_FAST if tier == "fast" else NO_UP)
        mk[bad, 1] = False; mk[bad, 2] = False
        mk[np.isin(tags, NO_THINK), 2] = False
    if tier not in THINK_TIERS:
        mk[:, 2] = False
    mk[:, 0] = True
    ar = np.arange(n)

    def choose(lam):
        i = np.argmax(np.where(mk, P - lam * X / lt, -1e18) - 1e-12 * np.arange(3), axis=1)
        return i, X[ar, i].sum()

    i, t = choose(0.0)
    if t > cap:
        lo, hi = 0.0, 1.0
        i, t = choose(hi)
        while t > cap and hi < 2 ** 50:
            lo = hi; hi *= 2; i, t = choose(hi)
        for _ in range(90):
            mid = (lo + hi) / 2
            c, ct = choose(mid)
            if ct <= cap:
                hi = mid; i, t = c, ct
            else:
                lo = mid
    if t > cap:
        i = np.zeros(n, int)
    return i
