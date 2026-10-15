"""予測モデルの学習で、合成データは実データ何人分になるか（GHE issue 8）。

eICU Demo のバイタル（公開リポジトリの #15 と同じ前処理）で、前半 6 時間の系列から後半 6 時間のイベントを予測する
ロジスティック回帰を、実データ n 人と、そこに足すデータの組み合わせで学習し、学習曲線を比べる。

- 課題: desat（後半に SpO2 92% 未満が 30 分以上）、tachy（後半に心拍数 110 超が 30 分以上）
- 病院単位で外部・対象に分け（GHE issue 2 と同じ）、対象から評価用を固定で取り分け、残りから n 人を抜いて学習する
- 足すデータ: 単純な変換（ノイズと振幅の拡大縮小）、同じ n 人で学習した VAR(1) の合成データ、
  外部の病院で学習した生成モデルの合成データ、外部の実データ
- 合成データのラベルは、合成系列の後半から実データと同じ定義で決める
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import prognostic, syninf, vitals

OUT = Path("outputs") / "augment"
SPLIT = vitals.STEPS // 2
SPO2 = vitals.VARS.index("sao2")
HR = vitals.VARS.index("heartrate")
MIN_STEPS = 6  # 30 分

LABELS = {
    "desat": lambda x: ((x[:, SPLIT:, SPO2] < 92).sum(1) >= MIN_STEPS).astype(int),
    "tachy": lambda x: ((x[:, SPLIT:, HR] > 110).sum(1) >= MIN_STEPS).astype(int),
}


def _features(x_first: np.ndarray, age: np.ndarray) -> np.ndarray:
    return vitals.summary_features(x_first, pd.DataFrame({"age": age}))


@dataclass
class Data:
    """被験者ごとの前半 6 時間の系列・年齢・特徴量・各課題のラベル。並びは seq_id の昇順。"""

    ids: np.ndarray
    x_first: np.ndarray
    age: np.ndarray
    feats: np.ndarray
    y: dict[str, np.ndarray]

    @classmethod
    def of(cls, df: pd.DataFrame) -> "Data":
        arr = vitals.to_array(df)
        static = vitals.static_of(df)
        age = static["age"].to_numpy(float)
        x_first = arr[:, :SPLIT]
        return cls(static.index.to_numpy(), x_first, age, _features(x_first, age), {k: f(arr) for k, f in LABELS.items()})

    def take(self, idx: np.ndarray) -> "Data":
        return Data(self.ids[idx], self.x_first[idx], self.age[idx], self.feats[idx], {k: v[idx] for k, v in self.y.items()})


def jitter(x: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator, noise: float = 0.03, scale: float = 0.05):
    """単純な変換による拡張: 変数ごとに、系列全体の標準偏差 × noise のガウスノイズを足し、
    平均の周りで振幅を N(1, scale²) 倍する。1 件につき k 件作り、ラベルを引き継ぐ。"""
    sd = x.reshape(-1, x.shape[-1]).std(0)
    out = []
    for _ in range(k):
        mean = x.mean(1, keepdims=True)
        amp = rng.normal(1, scale, (len(x), 1, x.shape[-1]))
        out.append(mean + (x - mean) * amp + rng.normal(0, 1, x.shape) * sd * noise)
    return np.concatenate(out), np.tile(y, k)


def real_equivalent_n(value: float, ns: list[int], auc: list[float]) -> float:
    """実データのみの学習曲線（n と AUROC）を log n で線形補間し、value の AUROC に必要な実データの人数を返す。
    曲線の範囲外なら NaN。曲線は単調増加を仮定する。"""
    ns, auc = np.asarray(ns, float), np.asarray(auc, float)
    if not (auc.min() <= value <= auc.max()):
        return float("nan")
    return float(np.exp(np.interp(value, auc, np.log(ns))))


def _fit_auroc(x: np.ndarray, y: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y)) < 2:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed))
    model.fit(x, y)
    return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))


def run_rep(pool: pd.DataFrame, data: Data, train_idx: np.ndarray, test_idx: np.ndarray, n: int, ext: Data,
            syn_ext: dict[str, Data], rng: np.random.Generator, seed: int,
            k_jitter: int = 4, m_same: int = 4, m_ext: int = 400) -> list[dict]:
    sel = data.take(rng.choice(train_idx, size=n, replace=False))
    test = data.take(test_idx)
    sources: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = {}

    xa, _ = jitter(sel.x_first, np.zeros(n), k_jitter, rng)
    age_a = np.tile(sel.age, k_jitter)
    sources["jitter"] = [(_features(xa, age_a), {t: np.tile(v, k_jitter) for t, v in sel.y.items()})]

    gen = vitals.fit_generate_var1(pool[pool["seq_id"].isin(sel.ids)].reset_index(drop=True), n_seqs=m_same * n, epochs=0, seed=seed)
    g = Data.of(gen)
    sources["var1-same"] = [(g.feats, g.y)]

    for name, syn in syn_ext.items():
        s = syn.take(rng.choice(len(syn.ids), size=min(m_ext, len(syn.ids)), replace=False))
        sources[name] = [(s.feats, s.y)]
    e = ext.take(rng.choice(len(ext.ids), size=min(m_ext, len(ext.ids)), replace=False))
    sources["ext-real"] = [(e.feats, e.y)]

    rows = []
    for task in LABELS:
        rows.append({"method": "real", "task": task, "n": n, "positives": int(sel.y[task].sum()),
                     "auroc": _fit_auroc(sel.feats, sel.y[task], test.feats, test.y[task], seed)})
        for name, parts in sources.items():
            x = np.vstack([sel.feats] + [p[0] for p in parts])
            y = np.concatenate([sel.y[task]] + [p[1][task] for p in parts])
            rows.append({"method": name, "task": task, "n": n, "positives": int(sel.y[task].sum()),
                         "auroc": _fit_auroc(x, y, test.feats, test.y[task], seed)})
    return rows


def experiment(ns: list[int], reps: int, seed: int = 0, n_test: int = 300, par_epochs: int = 200,
               use_par: bool = True) -> pd.DataFrame:
    pool = syninf.load_pool()
    ids = pool["seq_id"].drop_duplicates().to_numpy()
    hosp = prognostic._hospitals(ids)
    ext_h, _ = prognostic.split_hospitals(hosp, 0.5, seed)
    ext_ids = ids[hosp.isin(ext_h).to_numpy()]
    ext_df, tgt_df = pool[pool["seq_id"].isin(ext_ids)], pool[~pool["seq_id"].isin(ext_ids)]
    ext, tgt = Data.of(ext_df), Data.of(tgt_df)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tgt.ids))
    test_idx, train_idx = order[:n_test], order[n_test:]

    syn_ext = {"var1-ext": Data.of(vitals.fit_generate_var1(ext_df, n_seqs=2 * len(ext.ids), epochs=0, seed=seed))}
    if use_par:
        syn_ext["par-ext"] = Data.of(vitals.fit_generate_par(ext_df, n_seqs=2 * len(ext.ids), epochs=par_epochs, seed=seed))
    info = {name: {t: float(d.y[t].mean()) for t in LABELS} for name, d in {"target": tgt, "external": ext, **syn_ext}.items()}
    print("陽性の割合", info, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(info).T.to_csv(OUT / f"prevalence_split{seed}.csv")
    rows = []
    for n in ns:
        for rep in range(reps):
            for r in run_rep(tgt_df, tgt, train_idx, test_idx, n, ext, syn_ext, rng, seed=seed * 100_000 + n * 1000 + rep):
                rows.append({**r, "rep": rep})
        print(f"n={n} 完了", flush=True)
    return pd.DataFrame(rows)


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    """課題・手法・n ごとの AUROC の平均と標準誤差、学習できなかった試行の割合、実データ換算の人数。"""
    g = reps.groupby(["task", "method", "n"], sort=False)["auroc"]
    table = pd.DataFrame({"auroc": g.mean(), "se": g.std() / np.sqrt(g.count()), "failed": g.apply(lambda s: s.isna().mean())}).reset_index()
    eq = []
    for r in table.itertuples():
        curve = table[(table["task"] == r.task) & (table["method"] == "real")].sort_values("n")
        eq.append(real_equivalent_n(r.auroc, curve["n"].tolist(), curve["auroc"].tolist()))
    table["real_equiv_n"] = eq
    return table
