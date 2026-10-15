"""学習人数が少ないときの生成モデルの忠実度とプライバシ（GHE issue 3）。

eICU Demo のバイタル（公開リポジトリの #15 と同じ前処理・A/B/C 分割）で、学習データ A から n 人を抜いて生成モデルを学習する。

- 生成数は n によらず N_GEN（#15 の学習データと同じ 716）にそろえ、指標のぶれを n の違いと切り分ける
- 忠実度は学習に使っていない実データ C と比べる（少人数の学習データと比べると、覚えたモデルほど良く見えるため）
- プライバシは学習に使った n 人（メンバー）と評価用 B（非メンバー）で測る。加えて、外れた値を持つ被験者の近くに
  合成系列が寄っていないか（outlier_proximity）を見る
- 基準として、学習に使っていない実データ n 人（B から抜く）を C と比べた値を並べる
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import vitals

OUT = Path("outputs") / "smalln"
N_GEN = 716
DGAN_BATCH = 64
DGAN_UPDATES = 24000  # #15 のバッチ 64・2,000 エポック（716 系列）と同じ更新回数
PAR_EPOCHS = 200


def choose_ids(a: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    ids = np.sort(a["seq_id"].unique())
    if n > len(ids):
        raise ValueError(f"n={n} が学習データの被験者数 {len(ids)} を超えている")
    return np.random.default_rng(seed).choice(ids, size=n, replace=False)


def epochs_for_updates(n: int, batch: int, updates: int) -> int:
    per_epoch = math.ceil(n / min(batch, n))
    return math.ceil(updates / per_epoch)


def _stay_scores(df: pd.DataFrame, mu: np.ndarray, sd: np.ndarray) -> pd.Series:
    """滞在ごとの各変数の平均を、参照データの滞在平均で標準化した距離（外れ具合）。"""
    m = df.groupby("seq_id")[vitals.VARS].mean()
    return pd.Series(np.sqrt((((m.to_numpy() - mu) / sd) ** 2).sum(1)), index=m.index)


def outlier_proximity(train: pd.DataFrame, holdout: pd.DataFrame, gen: pd.DataFrame, reference: pd.DataFrame, frac: float = 0.1) -> float:
    """外れた学習被験者から最も近い合成系列までの距離の中央値を、外れた非メンバーでの同じ値で割った比。

    外れ具合は滞在ごとの平均の標準化距離で決め、それぞれ上位 frac（最低 2 人）を「外れた被験者」とする。
    1 を大きく下回ると、外れた学習被験者の近くに合成系列が出ている（その人の特徴が合成データに残っている）ことを示す。
    """
    ref_means = reference.groupby("seq_id")[vitals.VARS].mean()
    mu, sd = ref_means.mean().to_numpy(), ref_means.std().to_numpy()
    x_ref = vitals.to_array(reference)
    s_mu, s_sd = vitals._scale(x_ref)

    def flat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        d = df.sort_values(["seq_id", "t"])
        return d["seq_id"].unique(), ((vitals.to_array(d) - s_mu) / s_sd).reshape(d["seq_id"].nunique(), -1)

    def extreme(df: pd.DataFrame) -> np.ndarray:
        s = _stay_scores(df, mu, sd)
        k = max(2, int(round(len(s) * frac)))
        return s.nlargest(k).index.to_numpy()

    nn = NearestNeighbors(n_neighbors=1).fit(flat(gen)[1])

    def median_dist(df: pd.DataFrame) -> float:
        ids, z = flat(df)
        sel = np.isin(ids, extreme(df))
        return float(np.median(nn.kneighbors(z[sel])[0][:, 0]))

    return median_dist(train) / median_dist(holdout)


def _model(name: str, n: int):
    if name == "var1":
        return vitals.fit_generate_var1, 0
    if name == "par":
        return vitals.fit_generate_par, PAR_EPOCHS
    if name == "dgan":
        return partial(vitals.fit_generate_dgan, batch_size=DGAN_BATCH), epochs_for_updates(n, DGAN_BATCH, DGAN_UPDATES)
    raise ValueError(name)


MODELS = ("var1", "par", "dgan")


def run(model: str, n: int, seed: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    a = vitals.read_split("A")
    ids = choose_ids(a, n, seed)
    fn, epochs = _model(model, n)
    gen = fn(a[a["seq_id"].isin(ids)].reset_index(drop=True), n_seqs=N_GEN, epochs=epochs, seed=seed)
    path = OUT / f"gen_{model}_n{n}_s{seed}.csv.gz"
    gen.to_csv(path, index=False)
    (OUT / f"ids_n{n}_s{seed}.json").write_text(json.dumps([int(i) for i in ids]))
    return path


def _row(train: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, gen: pd.DataFrame, reference: pd.DataFrame, seed: int) -> dict:
    return {
        **vitals.fidelity(c, gen, seed),
        "TSTR_AUROC": vitals.utility(gen, b, seed),
        **vitals.privacy(train, b, gen),
        "OutlierProx": outlier_proximity(train, b, gen, reference),
    }


def evaluate() -> pd.DataFrame:
    a, b, c = (vitals.read_split(k) for k in "ABC")
    rows = []
    for path in sorted(OUT.glob("gen_*_n*_s*.csv.gz")):
        model, rest = path.name.removeprefix("gen_").removesuffix(".csv.gz").split("_n")
        n, seed = (int(v) for v in rest.split("_s"))
        ids = json.loads((OUT / f"ids_n{n}_s{seed}.json").read_text())
        train = a[a["seq_id"].isin(ids)]
        rows.append({"model": model, "n": n, "seed": seed, **_row(train, b, c, pd.read_csv(path), a, seed)})
        print(rows[-1], flush=True)
    # 基準: 学習に使っていない実データ n 人（B から抜く）を、そのまま C と比べる。プライバシの列は意味を持たない
    for n in sorted({r["n"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows if r["n"] == n}):
            ids_b = choose_ids(b, min(n, b["seq_id"].nunique()), seed)
            real_n = b[b["seq_id"].isin(ids_b)]
            rows.append({"model": "real-n", "n": n, "seed": seed, **vitals.fidelity(c, real_n, seed),
                         "TSTR_AUROC": vitals.utility(real_n, c, seed)})
    return pd.DataFrame(rows)
