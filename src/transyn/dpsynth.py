"""差分プライバシー（DP）付きの見本用データ（GHE issue 14）。

見本用データの L3・L4（showcase）を、被験者ごとの寄与を有界にした統計量にガウス雑音を加えて作る。

- 隣接関係は「1 人を別の 1 人に入れ替える」（人数 n は公開とみなす）
- 被験者ごとの統計量 u（各変数の平均・log 標準偏差、年齢、死亡退院の 8 次元）を、データから決めない公開の範囲
  （STAY_BOX）で [0, 1] に正規化する
- 雑音を加えて出す量は 4 つ: u の総和、u uᵀ の総和、被験者内で標準化した系列のラグ 0・ラグ 1 の共分散行列の総和。
  標準化した系列の共分散の成分は |·| ≤ T/(T−1) に収まるので、切り詰めの上限を COV_BOUND とする
- プライバシの会計は zCDP。ε・δ から ρ を求め、4 つの量に等分してガウス機構の雑音の大きさを決める
- 生成後の値は、データの分位点ではなく公開の生理的な範囲（vitals.BOUNDS）で切り詰める（分位点は DP ではない）
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import showcase, vitals

OUT = Path("outputs") / "dpsynth"
K = len(vitals.VARS)
# 被験者ごとの統計量の公開の範囲（成人の一般的な目安として決めた値。データからは決めない）
STAY_BOX = np.array([
    (40.0, 160.0), (80.0, 100.0), (5.0, 40.0),                       # 平均: 心拍数・SpO2・呼吸数
    (math.log(0.2), math.log(25.0)), (math.log(0.05), math.log(10.0)), (math.log(0.2), math.log(15.0)),  # log 標準偏差
    (18.0, 90.0),                                                       # 年齢
    (0.0, 1.0),                                                         # 死亡退院
])
DIM = len(STAY_BOX)
COV_BOUND = 1.01
SENSITIVITY = {
    "sum_u": math.sqrt(DIM),
    "sum_uu": float(DIM),
    "c0": 2 * COV_BOUND * K,
    "c1": 2 * COV_BOUND * K,
}


def rho_from_eps(epsilon: float, delta: float) -> float:
    """(ε, δ)-DP を満たす zCDP の ρ: ε = ρ + 2√(ρ log(1/δ)) を ρ について解く。"""
    if math.isinf(epsilon):
        return float("inf")
    l = math.log(1 / delta)
    return (math.sqrt(l + epsilon) - math.sqrt(l)) ** 2


def noise_scales(epsilon: float, delta: float) -> dict[str, float]:
    """4 つの量に ρ を等分したときの、ガウス機構の雑音の標準偏差 σ = Δ / √(2ρ_j)。"""
    rho = rho_from_eps(epsilon, delta)
    if math.isinf(rho):
        return {k: 0.0 for k in SENSITIVITY}
    rho_j = rho / len(SENSITIVITY)
    return {k: d / math.sqrt(2 * rho_j) for k, d in SENSITIVITY.items()}


def _per_subject(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = vitals.to_array(train)
    static = vitals.static_of(train)
    mean, sd = x.mean(1), x.std(1) + 1e-6
    raw = np.column_stack([mean, np.log(sd), static["age"].to_numpy(float), static["died"].to_numpy(float)])
    u = np.clip((raw - STAY_BOX[:, 0]) / (STAY_BOX[:, 1] - STAY_BOX[:, 0]), 0, 1)
    z = (x - mean[:, None]) / sd[:, None]
    t = z.shape[1] - 1
    c0 = np.clip(np.einsum("nti,ntj->nij", z[:, :-1], z[:, :-1]) / t, -COV_BOUND, COV_BOUND)
    c1 = np.clip(np.einsum("nti,ntj->nij", z[:, :-1], z[:, 1:]) / t, -COV_BOUND, COV_BOUND)
    return u, c0, c1


def _psd(m: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    m = (m + m.T) / 2
    w, v = np.linalg.eigh(m)
    return (v * np.clip(w, floor, None)) @ v.T


def query_sums(train: pd.DataFrame) -> dict[str, np.ndarray]:
    """雑音を加える前の 4 つの総和と人数。各被験者の寄与は有界で、1 人の入れ替えによる変化は SENSITIVITY 以下。
    学習データを読むのはこの関数だけで、以降の処理はこの戻り値に雑音を加えたものだけを使う。"""
    u, c0, c1 = _per_subject(train)
    return {"n": len(u), "sum_u": u.sum(0), "sum_uu": (u[:, :, None] * u[:, None, :]).sum(0), "c0": c0.sum(0), "c1": c1.sum(0)}


def release(train: pd.DataFrame, epsilon: float, delta: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """雑音を加えた統計量（DP で公開してよい量）から、生成に使うパラメータを推定する。以降はこの戻り値だけを使う。"""
    q = query_sums(train)
    n = q["n"]
    s = noise_scales(epsilon, delta)
    sum_u = q["sum_u"] + rng.normal(0, s["sum_u"], DIM)
    sum_uu = q["sum_uu"] + rng.normal(0, s["sum_uu"], (DIM, DIM))
    sc0 = q["c0"] + rng.normal(0, s["c0"], (K, K))
    sc1 = q["c1"] + rng.normal(0, s["c1"], (K, K))
    mu = sum_u / n
    cov = _psd(sum_uu / n - np.outer(mu, mu))
    c0m = _psd(sc0 / n)
    a = np.linalg.solve(c0m, sc1 / n)
    radius = np.abs(np.linalg.eigvals(a)).max()
    if radius >= 0.99:  # 雑音で不安定になった場合は定常に戻す
        a = a * (0.99 / radius)
    noise_cov = _psd(c0m - a.T @ c0m @ a)
    return {"mu": mu, "cov": cov, "a": a, "c0": c0m, "noise_cov": noise_cov}


def _sample_stay(params: dict, level: str, n: int, rng: np.random.Generator) -> np.ndarray:
    mu, cov = params["mu"], params["cov"]
    p = float(np.clip(mu[-1], 0, 1))
    died = (rng.random(n) < p).astype(float)
    a_idx = np.arange(DIM - 1)
    if level == "L4":
        # 死亡退院の値で条件づけた多変量正規分布（年齢を含む 7 次元）
        s_ab, s_bb = cov[a_idx, -1], cov[-1, -1]
        cond_cov = _psd(cov[np.ix_(a_idx, a_idx)] - np.outer(s_ab, s_ab) / s_bb)
        out = np.empty((n, DIM - 1))
        for d in (0.0, 1.0):
            sel = died == d
            m = mu[a_idx] + s_ab / s_bb * (d - mu[-1])
            out[sel] = rng.multivariate_normal(m, cond_cov, sel.sum())
    elif level == "L3":
        # 被験者ごとの平均・ばらつき（6 次元）と年齢を、互いに・死亡退院と独立に引く
        b = np.arange(2 * K)
        out = np.empty((n, DIM - 1))
        out[:, b] = rng.multivariate_normal(mu[b], _psd(cov[np.ix_(b, b)]), n)
        out[:, -1] = rng.normal(mu[DIM - 2], math.sqrt(max(cov[DIM - 2, DIM - 2], 1e-6)), n)
    else:
        raise ValueError(level)
    u = np.clip(np.column_stack([out, died]), 0, 1)
    return STAY_BOX[:, 0] + u * (STAY_BOX[:, 1] - STAY_BOX[:, 0])


def generate_dp(level: str, train: pd.DataFrame, n_seqs: int, seed: int, epsilon: float, delta: float = 1e-5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    params = release(train, epsilon, delta, rng)
    stay = _sample_stay(params, level, n_seqs, rng)
    z = showcase._var_sample(params["a"], params["noise_cov"], n_seqs, rng)
    x = stay[:, None, :K] + np.exp(stay[:, None, K : 2 * K]) * z
    for j, v in enumerate(vitals.VARS):
        x[..., j] = np.clip(x[..., j], *vitals.BOUNDS[v])
    age = np.clip(np.round(stay[:, -2]), 18, 90).astype(int)
    return showcase._long(x, age, stay[:, -1].round().astype(int))[showcase.COLUMNS]


EPSILONS = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))


def audit(pool: pd.DataFrame, epsilons: tuple[float, ...], n: int, k: int, reps: int, seed: int,
          levels: tuple[str, ...] = ("L3", "L4")) -> list[dict]:
    """GHE issue 13 の狙い撃ちのメンバーシップ推論で、DP 付きの生成の経験的な ε の下限を測る。"""
    from . import attacks

    avg, out = attacks.choose_targets(pool, k)
    rows = []
    for level in levels:
        for eps in epsilons:
            def gen(lv, train, n_seqs, seed, eps=eps):
                return generate_dp(lv, train, n_seqs=n_seqs, seed=seed, epsilon=eps)

            for kind, targets in (("average", avg), ("outlier", out)):
                for t in targets:
                    r = attacks.targeted_mia(pool, t, level=level, n=n, reps=reps, seed=seed, generate=gen)
                    rows.append({**r, "epsilon": eps, "kind": kind})
                    print(rows[-1], flush=True)
    return rows


def evaluate(ns: tuple[int, ...] = (20, 50, 716), epsilons: tuple[float, ...] = EPSILONS,
             seeds: tuple[int, ...] = (0, 1, 2), levels: tuple[str, ...] = ("L3", "L4")) -> pd.DataFrame:
    """学習データ A（または A から n 人）で、段階・ε ごとに 716 人分を生成し、忠実度・関係・開示チェックを測る。"""
    a, b, c = (vitals.read_split(s) for s in "ABC")
    rows = []
    for n in ns:
        for seed in seeds:
            ids = np.random.default_rng(seed).choice(np.sort(a["seq_id"].unique()), size=min(n, a["seq_id"].nunique()), replace=False)
            train = a[a["seq_id"].isin(ids)].reset_index(drop=True)
            for level in levels:
                for eps in epsilons:
                    gen = generate_dp(level, train, n_seqs=716, seed=seed, epsilon=eps)
                    disc = showcase.disclosure_checks(gen, train, b)
                    rows.append({"level": level, "n": n, "epsilon": eps, "seed": seed,
                                 **vitals.fidelity(c, gen, seed), **showcase.relation_stats(gen),
                                 **{k: v for k, v in disc.items() if k != "small_groups"}})
                    print({k: rows[-1][k] for k in ("level", "n", "epsilon", "seed", "StayMean", "ACF", "HRDiffDied")}, flush=True)
    return pd.DataFrame(rows)
