"""合成データへのプライバシ攻撃（GHE issue 13）。

見本用データ（showcase の L1〜L4）を対象に、攻撃者が実際にやりそうな攻撃を試す。

- 狙い撃ちのメンバーシップ推論（targeted_mia）: 標的の被験者を学習データに入れた場合と入れない場合で生成を
  繰り返し、合成データから入っていたかを当てる（Stadler et al., USENIX Security 2022 のシャドーモデル型の考え方）。
  攻撃の手がかりは、標的の被験者ごとの平均・log 標準偏差（6 次元）に最も近い合成の被験者までの距離
- 属性推論（attribute_inference）: 年齢と前半 1 時間のバイタルを知る攻撃者が、合成データで学習した分類器で
  死亡退院を推測する。学習に使った人と使っていない人での当たり方の差を、漏れとみなす
- 部分系列による所属推定（partial_mia）: 実在の被験者の 1 時間分の系列を知る攻撃者が、合成データの中で
  最も近い区間までの距離から、その人が学習データに入っていたかを当てる

攻撃の強さは AUC と、真陽性率と偽陽性率の比から求める経験的な ε の下限（eps_lower_bound）で表す。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import showcase, vitals

OUT = Path("outputs") / "attacks"
WINDOW = 12  # 1 時間


def stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """被験者ごとの各変数の平均と log 標準偏差（6 次元）。"""
    g = df.groupby("seq_id")[vitals.VARS]
    mean, sd = g.mean(), g.std()
    return pd.concat([mean.add_suffix("_mean"), np.log(sd + 1e-6).add_suffix("_logsd")], axis=1)


def outlier_scores(df: pd.DataFrame) -> pd.Series:
    """被験者ごとの外れ具合: 被験者ごとの特徴量を集団で標準化した距離。"""
    f = stay_features(df)
    z = (f - f.mean()) / f.std()
    return np.sqrt((z**2).sum(axis=1))


def choose_targets(df: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """外れ具合が小さい k 人（平均的な被験者）と大きい k 人（外れた被験者）。"""
    s = outlier_scores(df).sort_values()
    return s.index[:k].to_numpy(), s.index[-k:].to_numpy()


def _cp(k: int, n: int, alpha: float) -> tuple[float, float]:
    """二項比率のクロッパー・ピアソン区間（両側 1 − alpha）。"""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def eps_lower_bound(s_in: np.ndarray, s_out: np.ndarray, alpha: float = 0.05) -> float:
    """経験的な ε の下限。しきい値ごとに、真陽性率の信頼下限と偽陽性率の信頼上限の比の対数をとり、最大値を返す
    （偽陰性率と真陰性率の側も同様）。どのしきい値でも区別できなければ 0。

    しきい値を同じ試行から選ぶので、厳密な下限ではなく目安（監査の研究では別の試行でしきい値を決める）。
    """
    s_in, s_out = np.asarray(s_in, float), np.asarray(s_out, float)
    best = 0.0
    for th in np.unique(np.r_[s_in, s_out]):
        tp, fp = int((s_in >= th).sum()), int((s_out >= th).sum())
        tpr_lo, _ = _cp(tp, len(s_in), alpha)
        _, fpr_hi = _cp(fp, len(s_out), alpha)
        tnr_lo, _ = _cp(len(s_out) - fp, len(s_out), alpha)
        _, fnr_hi = _cp(len(s_in) - tp, len(s_in), alpha)
        for num, den in ((tpr_lo, fpr_hi), (tnr_lo, fnr_hi)):
            if num > 0 and den > 0:
                best = max(best, float(np.log(num / den)))
    return best


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def targeted_mia(pool: pd.DataFrame, target, level: str, n: int, reps: int, seed: int,
                 generate=None, m_gen: int = 200, **gen_kwargs) -> dict:
    """標的を学習データ（n 人）に入れる試行と入れない試行を交互に reps 回行い、合成データ m_gen 人分から当てる。"""
    generate = generate or (lambda lv, train, n_seqs, seed: showcase.generate(lv, train, n_seqs=n_seqs, seed=seed, **gen_kwargs))
    feats = stay_features(pool)
    mu, sd = feats.mean(), feats.std()
    t = ((feats.loc[target] - mu) / sd).to_numpy()
    others = np.array([i for i in pool["seq_id"].unique() if i != target])
    rng = np.random.default_rng(seed)
    labels, scores = [], []
    for r in range(reps):
        member = r % 2 == 0
        chosen = rng.choice(others, size=n - 1 if member else n, replace=False)
        ids = np.r_[chosen, [target]] if member else chosen
        gen = generate(level, pool[pool["seq_id"].isin(ids)].reset_index(drop=True), n_seqs=m_gen, seed=seed * 100_000 + r)
        g = ((stay_features(gen) - mu) / sd).to_numpy()
        labels.append(int(member))
        scores.append(-float(np.sqrt(((g - t) ** 2).sum(1)).min()))
    labels, scores = np.array(labels), np.array(scores)
    return {"target": target, "level": level, "n": n, "reps": reps, "auc": _auc(labels, scores),
            "eps_lb": eps_lower_bound(scores[labels == 1], scores[labels == 0])}


def _first_hour_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    arr = vitals.to_array(df)[:, :WINDOW]
    static = vitals.static_of(df)
    x = np.concatenate([arr.mean(1), arr.std(1), arr.min(1), arr.max(1), static[["age"]].to_numpy(float)], axis=1)
    return x, static["died"].to_numpy()


def attribute_inference(gen: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame, seed: int = 0) -> dict:
    """合成データで学習した分類器で、年齢と前半 1 時間から死亡退院を推測する。学習に使った人（members）と
    使っていない人（nonmembers）での AUROC の差（advantage）が、学習データの属性の漏れの目安。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xg, yg = _first_hour_features(gen)
    if len(np.unique(yg)) < 2:
        return {"auroc_members": float("nan"), "auroc_nonmembers": float("nan"), "advantage": float("nan")}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed)).fit(xg, yg)
    out = {}
    for name, df in (("members", train), ("nonmembers", holdout)):
        x, y = _first_hour_features(df)
        out[f"auroc_{name}"] = _auc(y, model.predict_proba(x)[:, 1])
    out["advantage"] = out["auroc_members"] - out["auroc_nonmembers"]
    return out


def nearest_window_distance(window: np.ndarray, gen: pd.DataFrame) -> float:
    """window（length × 変数）に最も近い、合成系列の連続する区間までのユークリッド距離。"""
    arr = vitals.to_array(gen)
    w = np.lib.stride_tricks.sliding_window_view(arr, window.shape[0], axis=1)  # (n, T-L+1, 変数, L)
    w = np.moveaxis(w, -1, -2)
    return float(np.sqrt(((w - window) ** 2).sum(axis=(-1, -2))).min())


def partial_mia(gen: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame, seed: int = 0, k: int = 200) -> dict:
    """学習に使った人と使っていない人それぞれ k 人の、ランダムな 1 時間の区間について、合成データの最も近い区間までの
    距離で所属を当てる。"""
    rng = np.random.default_rng(seed)
    labels, scores = [], []
    for member, df in ((1, train), (0, holdout)):
        arr = vitals.to_array(df)
        for i in rng.choice(len(arr), size=min(k, len(arr)), replace=False):
            s = rng.integers(0, arr.shape[1] - WINDOW + 1)
            labels.append(member)
            scores.append(-nearest_window_distance(arr[i, s : s + WINDOW], gen))
    labels, scores = np.array(labels), np.array(scores)
    return {"partial_auc": _auc(labels, scores), "partial_eps_lb": eps_lower_bound(scores[labels == 1], scores[labels == 0])}


LEVELS = ("L1", "L2", "L3", "L4")
TAIL_CLIP = 0.005  # showcase-release の既定と同じ


def experiment_targeted(n: int, reps: int, k: int = 5, seed: int = 0, levels: tuple[str, ...] = LEVELS) -> pd.DataFrame:
    """A と B を合わせた 1,070 人から平均的な被験者 k 人・外れた被験者 k 人を標的に選び、段階ごとに狙い撃ちを試す。"""
    pool = pd.concat([vitals.read_split("A"), vitals.read_split("B")], ignore_index=True)
    avg, out = choose_targets(pool, k)
    rows = []
    for level in levels:
        for kind, targets in (("average", avg), ("outlier", out)):
            for t in targets:
                r = targeted_mia(pool, t, level=level, n=n, reps=reps, seed=seed, tail_clip=TAIL_CLIP)
                rows.append({**r, "kind": kind})
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def experiment_population(ns: tuple[int, ...] = (50, 716), seeds: tuple[int, ...] = (0, 1, 2),
                          levels: tuple[str, ...] = LEVELS) -> pd.DataFrame:
    """学習データ A（または A から n 人）で段階ごとに 716 人分を生成し、属性推論と部分系列による所属推定を試す。"""
    a, b = vitals.read_split("A"), vitals.read_split("B")
    rows = []
    for n in ns:
        for seed in seeds:
            ids = np.random.default_rng(seed).choice(np.sort(a["seq_id"].unique()), size=min(n, a["seq_id"].nunique()), replace=False)
            train = a[a["seq_id"].isin(ids)].reset_index(drop=True)
            for level in levels:
                gen = showcase.generate(level, train, n_seqs=716, seed=seed, tail_clip=TAIL_CLIP)
                rows.append({"level": level, "n": n, "seed": seed, **attribute_inference(gen, train, b, seed), **partial_mia(gen, train, b, seed)})
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)
