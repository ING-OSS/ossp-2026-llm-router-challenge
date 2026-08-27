# SPDX-FileCopyrightText: Copyright 2026 OSSP Router Team
# SPDX-License-Identifier: Apache-2.0

"""features.py — 프롬프트 → 특징. 학습·추론 공용. 결정적(FNV-1a)."""
from __future__ import annotations
import math, re
import numpy as np

TOKRX = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]")
LATEX = re.compile(r"\\[a-zA-Z]{2,}|\$[^$\n]{0,60}?[a-zA-Z\\^_{][^$\n]{0,60}?\$")
GSM = re.compile(r"\b(how many|how much|total|each|per|cost|dollars|minutes|hours|pounds|left|spend|sell)\b", re.I)
DMMATH = re.compile(r"^(What is|Calculate|Solve|Let |Simplify|Round|Sort|Divide|Multiply|Suppose|Find the|Differentiate|Factor|Evaluate|Convert|Put )")
TAGS = ["latexmath", "code", "longctx", "ko_mcq", "en_mcq", "ko_reason",
        "ruletaker", "dm_math", "gsm", "other"]
NUM = re.compile(r"-?\d+(?:\.\d+)?")
ANS = re.compile(r"(answer|정답|출력|Give your|round|나타내|express|modulo|remainder|nearest|형태로)", re.I)
IMP = re.compile(r"\b(prove|show that|explain why|derive|why|증명|설명)\b", re.I)
STEP = re.compile(r"\b(first|then|next|finally|step|각각|먼저|그다음)\b", re.I)

_FNV_OFF = 14695981039346656037
_FNV_P = 1099511628211
_M64 = (1 << 64) - 1


def fnv1a(s: str) -> int:
    d = _FNV_OFF
    for b in s.encode("utf-8"):
        d ^= b
        d = (d * _FNV_P) & _M64
    return d


def tag(p: str) -> str:
    ko = len(re.findall(r"[가-힣]", p)) / max(len(p), 1)
    if LATEX.search(p):
        return "latexmath"
    if "assert f(" in p or (p.lstrip().startswith("def ") and "assert" in p):
        return "code"
    if len(p) > 8000:
        return "longctx"
    if re.search(r"\n[A-D]\. ", p) and ko > 0.2:
        return "ko_mcq"
    if re.search(r"\n[A-D]\. ", p) or re.search(r"\n[A-B]\. ", p):
        return "en_mcq"
    if ko > 0.2:
        return "ko_reason"
    if "Question:" in p and re.search(r"\b(is|are)\b.*\.$", p.split("Question:")[-1].strip()):
        return "ruletaker"
    if DMMATH.match(p.strip()) and not GSM.search(p):
        return "dm_math"
    if GSM.search(p) and len(p) < 1200:
        return "gsm"
    return "other"


def dense(p: str) -> list:
    n = len(p); w = p.split(); ln = p.split("\n")
    a = sum(1 for c in p if ord(c) < 128)
    h = len(re.findall(r"[가-힣]", p))
    d = sum(c.isdigit() for c in p)
    ws = sum(c.isspace() for c in p)
    o = n - a - h
    base = [n, a, h, d, o, ws, len(w), len(ln),            # 앞 8개는 in_tok 회귀 전용
            math.log1p(n), math.log1p(len(w)), math.log1p(len(ln)),
            h / max(n, 1), d / max(n, 1), ws / max(n, 1),
            len(re.findall(r"[\+\-\*/=<>\^%]", p)) / max(n, 1),
            len(re.findall(r"\\[a-zA-Z]+", p)), p.count("$"),
            max((len(x) for x in ln), default=0),
            float(bool(re.search(r"\n[A-D]\. ", p))), float("Question:" in p),
            float(bool(GSM.search(p))), float(bool(DMMATH.match(p.strip()))),
            len(re.findall(r"\d+", p)), len(re.findall(r"[?？]", p)),
            len(TOKRX.findall(p[:20000]))]
    t = tag(p)
    return base + [1.0 if t == x else 0.0 for x in TAGS]


def hashv(p: str, bins: int = 1024) -> np.ndarray:
    tk = [x.casefold() if not x.isdecimal() else "<n>" for x in TOKRX.findall(p[:3000])]
    v = np.zeros(bins)
    for f in [f"1:{a}" for a in tk] + [f"2:{a}\x1f{b}" for a, b in zip(tk, tk[1:])]:
        h = fnv1a(f)
        v[h & (bins - 1)] += -1.0 if (h >> 63) & 1 else 1.0
    nr = np.linalg.norm(v)
    return v / nr if nr else v


def extra(p: str) -> list:
    n = max(len(p), 1)
    nums = NUM.findall(p[:4000])
    mags = [abs(float(x)) for x in nums[:200]] or [0]
    lines = p.split("\n")
    q = p.split("Question:")[-1] if "Question:" in p else (lines[-1] if lines else p)
    ops = len(re.findall(r"[\+\-\*/=<>\^%]", p[:4000]))
    return [math.log1p(len(nums)), math.log1p(max(mags)), math.log1p(float(np.mean(mags))),
            math.log1p(sum(len(x) for x in nums)), math.log1p(ops), ops / n,
            float(bool(ANS.search(p))), float(bool(IMP.search(p))),
            math.log1p(len(STEP.findall(p[:4000]))),
            math.log1p(len(q)), len(q) / n,
            math.log1p(p.count(",")), math.log1p(p.count(";")),
            math.log1p(len(re.findall(r"\b(and|or|not|if|then|모두|또는|아닌)\b", p[:4000], re.I))),
            math.log1p(len(re.findall(r"\bfor |\bwhile |\brange\(|\.append|\bif ", p))),
            math.log1p(len(set(re.findall(r"[A-Za-z]{3,}", p[:2000])))),
            math.log1p(len(re.findall(r"\\(frac|sum|prod|int|sqrt|log|lim)", p))),
            float(bool(re.search(r"\n[A-D]\. ", p))),
            math.log1p(len(re.findall(r"\?", p))),
            math.log1p(len(re.findall(r"[가-힣]{2,}", p[:2000])))]


def episode_text(ep: dict) -> str:
    if "prompt" in ep:
        return ep["prompt"]
    return "\n".join(m.get("content", "") for m in ep.get("messages", []))


def build_matrices(texts):
    """반환: F(dense), X(dense+hash), E(extra), tags, cut(앞 3000자)"""
    F = np.array([dense(t) for t in texts])
    X = np.hstack([F, np.array([hashv(t) for t in texts])])
    E = np.array([extra(t) for t in texts])
    tags = np.array([tag(t) for t in texts])
    cut = [t[:3000] for t in texts]
    return F, X, E, tags, cut
