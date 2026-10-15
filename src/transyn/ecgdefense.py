"""心電図の合成データの本人との紐づきへの対策（GHE issue 18）。

GHE issue 17 の作り方（ecglink）に対策を加える。

- E3-kanon: 本人ごとの合成（E3）で、本人の平均の波形の代わりに、波形が近い k 人（本人を含む）の平均を使う。
  合成の 1 人が特定の 1 人に由来しないようにする（k-匿名の考え方、microaggregation）
- E3-noise: E3 の拍に白色雑音（標準偏差 sigma mV）を足す（単純な対策の比較）
- E2-trim: 外れ具合が上位 frac の被験者を除いてから、集団の分布（E2）を推定する
- E2-dp: E2 を差分プライバシー付きで作る。基底はデータに依存しない離散コサイン変換（DCT）の先頭 DCT_K 個。
  被験者ごとの平均の係数（L2 ノルムを C_MEAN で切り詰め）と、拍ごとの残差（C_RESID で切り詰め）、心拍間隔の
  統計量の総和にガウス雑音を加える。隣接関係は被験者 1 人の入れ替え（人数は公開）、会計は zCDP（dpsynth と同じ）
"""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import dct, idct

from . import dpsynth, ecg, ecglink

DCT_K = 20
C_MEAN = 8.0    # 平均の波形の係数の L2 ノルムの上限（拍の値で約 ±0.7 mV の大きさに当たる、公開の定数）
C_RESID = 4.0   # 拍ごとの残差の係数の L2 ノルムの上限
RR_BOX = (0.3, 2.0)
RR_SD_MAX = 0.5
DP_SENSITIVITY = {
    "sum_mean": 2 * C_MEAN,
    "sum_mean_outer": 2 * C_MEAN**2,
    "sum_within": 2 * C_RESID**2,
    "sum_rr": RR_BOX[1] - RR_BOX[0],
    "sum_rr_sq": RR_BOX[1] ** 2 - RR_BOX[0] ** 2,
    "sum_rr_sd": RR_SD_MAX,
}


def trim_persons(train: ecglink.BeatSet, frac: float) -> ecglink.BeatSet:
    score = ecglink.person_outlier_scores(train)
    k = int(round(frac * len(score)))
    drop = score.sort_values().index[len(score) - k :] if k else []
    return train.select(person=[p for p in train.persons if p not in set(drop)])


def dp_noise_scales(epsilon: float, delta: float) -> dict[str, float]:
    rho = dpsynth.rho_from_eps(epsilon, delta)
    if math.isinf(rho):
        return {k: 0.0 for k in DP_SENSITIVITY}
    rho_j = rho / len(DP_SENSITIVITY)
    return {k: d / math.sqrt(2 * rho_j) for k, d in DP_SENSITIVITY.items()}


def _clip_norm(x: np.ndarray, c: float) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * np.minimum(1.0, c / np.maximum(n, 1e-12))


def _to_beats(coef: np.ndarray) -> np.ndarray:
    full = np.zeros((len(coef), ecg.BEAT_LEN))
    full[:, : coef.shape[1]] = coef
    return idct(full, norm="ortho", axis=1)


def _dp_e2(train: ecglink.BeatSet, n_persons: int, bpp: int, rng: np.random.Generator, epsilon: float, delta: float):
    coef = dct(train.beats, norm="ortho", axis=1)[:, :DCT_K]
    persons = train.persons
    means = np.array([coef[train.person == p].mean(0) for p in persons])
    means = _clip_norm(means, C_MEAN)
    within = []
    for p, m in zip(persons, means):
        r = _clip_norm(coef[train.person == p] - m, C_RESID)
        within.append(r.T @ r / len(r))
    rr = np.array([np.clip(train.rr[train.person == p].mean(), *RR_BOX) for p in persons])
    rr_sd = np.array([np.clip(train.rr[train.person == p].std(), 0, RR_SD_MAX) for p in persons])
    n = len(persons)
    s = dp_noise_scales(epsilon, delta)
    k = DCT_K
    sum_m = means.sum(0) + rng.normal(0, s["sum_mean"], k)
    sum_mm = means.T @ means + rng.normal(0, s["sum_mean_outer"], (k, k))
    sum_w = np.sum(within, axis=0) + rng.normal(0, s["sum_within"], (k, k))
    sum_rr = rr.sum() + rng.normal(0, s["sum_rr"])
    sum_rr2 = (rr**2).sum() + rng.normal(0, s["sum_rr_sq"])
    sum_sd = rr_sd.sum() + rng.normal(0, s["sum_rr_sd"])
    mu = sum_m / n
    between = dpsynth._psd(sum_mm / n - np.outer(mu, mu))
    w = dpsynth._psd(sum_w / n)
    rr_mu = float(np.clip(sum_rr / n, *RR_BOX))
    rr_between = math.sqrt(max(sum_rr2 / n - rr_mu**2, 1e-6))
    rr_within = float(np.clip(sum_sd / n, 0.005, RR_SD_MAX))
    beats, rrs = [], []
    for _ in range(n_persons):
        m = rng.multivariate_normal(mu, between)
        beats.append(_to_beats(rng.multivariate_normal(m, w, bpp)))
        rrs.append(np.clip(rng.normal(rng.normal(rr_mu, rr_between), rr_within, bpp), *RR_BOX))
    return beats, rrs


def _microaggregate(means: np.ndarray, k: int) -> np.ndarray:
    """被験者の平均の波形を、近い k 人ずつの組に分けて組の平均に置き換える（最初の主成分の順に並べて k 人ずつ）。
    最後の組が k 人に満たなければ、1 つ前の組に入れる。"""
    if k <= 1:
        return means.copy()
    order = np.argsort(means[:, 0])
    out = means.copy()
    groups = [order[i : i + k] for i in range(0, len(order), k)]
    if len(groups) > 1 and len(groups[-1]) < k:
        groups[-2] = np.r_[groups[-2], groups[-1]]
        groups = groups[:-1]
    for g in groups:
        out[g] = means[g].mean(0)
    return out


def generate(kind: str, train: ecglink.BeatSet, n_persons: int, beats_per_person: int, seed: int, **kw) -> ecglink.BeatSet:
    rng = np.random.default_rng(seed)
    if kind == "E2-trim":
        return ecglink.generate("E2", trim_persons(train, kw.get("frac", 0.2)), n_persons, beats_per_person, seed)
    if kind == "E2-pc":
        return ecglink.generate("E2", train, n_persons, beats_per_person, seed, fit_=ecglink.fit(train, k=kw.get("n_components", 5)))
    if kind == "E2-agg":
        f = ecglink.fit(train)
        f.means = _microaggregate(f.means, int(kw.get("k", 3)))
        from sklearn.covariance import LedoitWolf

        f.between = LedoitWolf().fit(f.means).covariance_
        return ecglink.generate("E2", train, n_persons, beats_per_person, seed, fit_=f)
    if kind == "E3-noise":
        syn = ecglink.generate("E3", train, n_persons, beats_per_person, seed)
        syn.beats = syn.beats + rng.normal(0, kw.get("sigma", 0.1), syn.beats.shape)
        return syn
    ids = [f"S{j:03d}" for j in range(n_persons)]
    source = {}
    if kind == "E3-kanon":
        f = ecglink.fit(train)
        k = int(kw.get("k", 3))
        d = ((f.means[:, None, :] - f.means[None, :, :]) ** 2).sum(-1)
        beats, rrs = [], []
        for j in range(n_persons):
            i = j % len(f.persons)
            nb = np.argsort(d[i])[:k]  # 本人を含む近い k 人
            m = f.means[nb].mean(0)
            cov = np.mean([f.own[f.persons[q]] for q in nb], axis=0)
            rr_m = f.rr["mean"].iloc[nb].mean()
            beats.append(f.pca.inverse_transform(rng.multivariate_normal(m, cov, beats_per_person)))
            rrs.append(np.clip(rng.normal(rr_m, f.rr["std"].mean(), beats_per_person), 0.3, 2.0))
            source[ids[j]] = f.persons[i]
    elif kind == "E2-dp":
        beats, rrs = _dp_e2(train, n_persons, beats_per_person, rng, kw.get("epsilon", 4.0), kw.get("delta", 1e-5))
    else:
        raise ValueError(kind)
    person = np.concatenate([[ids[j]] * len(b) for j, b in enumerate(beats)])
    return ecglink.BeatSet(np.vstack(beats), person, np.full(len(person), "syn"), np.concatenate(rrs),
                           np.full(len(person), "N"), source or None)


CONFIGS = (
    [(f"E3-kanon-k{k}", "E3-kanon", {"k": k}) for k in (2, 3, 5, 10)]
    + [(f"E3-noise-{s:g}", "E3-noise", {"sigma": s}) for s in (0.05, 0.1, 0.2)]
    + [(f"E2-trim-{f:g}", "E2-trim", {"frac": f}) for f in (0.1, 0.2)]
    + [(f"E2-dp-eps{e:g}", "E2-dp", {"epsilon": e}) for e in (1.0, 4.0, 8.0, float("inf"))]
    + [(f"E2-agg-k{k}", "E2-agg", {"k": k}) for k in (3, 5)]
    + [(f"E2-pc{c}", "E2-pc", {"n_components": c}) for c in (3, 5)]
)


def _utility(bs, members, nonmembers, kind, kw, seed, classes=("N", "V"), max_beats=100) -> float:
    """ecglink.utility_tstr と同じ手順を、対策つきの作り方で行う。"""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xs, ys = [], []
    for label, c in enumerate(classes):
        tr = bs.select(person=members, part="gen", symbol=c)
        counts = pd.Series(tr.person).value_counts()
        tr = tr.select(person=counts.index[counts >= 3])
        if len(tr.persons) < 2:
            return float("nan")
        per = int(min(max_beats, counts[counts >= 3].median()))
        kw2 = {**kw, "k": min(kw.get("k", 1), len(tr.persons))} if kind == "E3-kanon" else kw
        xs.append(generate(kind, tr, n_persons=len(tr.persons), beats_per_person=per, seed=seed, **kw2).beats)
        ys.append(np.full(len(xs[-1]), label))
    te = bs.select(person=nonmembers, symbol=list(classes))
    y_te = (te.symbol == classes[1]).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(np.vstack(xs), np.concatenate(ys))
    return float(roc_auc_score(y_te, model.predict_proba(te.beats)[:, 1]))


def experiment(bs: ecglink.BeatSet, seeds=(0, 1, 2, 3, 4), configs=CONFIGS, beats_per_person: int = 24,
               utility: bool = False) -> list[dict]:
    rows = []
    for seed in seeds:
        mem, non = ecglink.split_persons(bs, seed)
        train, enroll, ref = bs.select(person=mem, part="gen"), bs.select(part="enroll"), bs.select(person=non, part="gen")
        model = ecglink._identifier(enroll)
        for name, kind, kw in configs:
            syn = generate(kind, train, n_persons=len(mem), beats_per_person=beats_per_person, seed=seed, **kw)
            r = {"seed": seed, "config": name, **ecglink.linkage_attack(syn, enroll, mem, model=model),
                 **ecglink.realism(syn, ref, seed=seed)}
            if utility:
                r["utility"] = _utility(bs, mem, non, kind, kw, seed)
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return rows
