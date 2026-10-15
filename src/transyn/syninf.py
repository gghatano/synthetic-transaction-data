"""合成データで水増しして検定すると偽陽性が増えるかの検証（GHE issue 1）。

Decruyenaere et al.（UAI 2024）が表データで示した「合成データを実データとみなした推論は第 1 種の過誤が増え、
深層生成モデルでは標準誤差の補正をしても残る」を、被験者ごとの時系列（eICU Demo のバイタル）で確かめる。

1 回の試行:
1. 実データから被験者を n 人抜き、ランダムに 2 群のラベルを付ける（真の群間差は 0）。ラベルは died 列に入れ、
   vitals の生成モデルが条件（属性）として使う列をそのまま流用する
2. 群ラベルを条件に生成モデルを学習し、M 人分を生成する
3. 被験者ごとの平均心拍数の群間差を t 検定する（実データ・合成データ・補正つき合成データ）。
   実データでは、被験者内の相関を無視して記録単位で検定した場合も並べる

補正は、合成データの推定量の分散を σ²(1/n + 1/M) とみなし、素朴な標準誤差に √(1 + M/n) を掛ける。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import vitals

OUT = Path("outputs") / "syninf"
GROUP = "died"
VALUE = "heartrate"

MODELS = {
    "var1": vitals.fit_generate_var1,
    "par": vitals.fit_generate_par,
    "dgan": partial(vitals.fit_generate_dgan, batch_size=64),
}


@dataclass
class TTest:
    diff: float
    se: float
    dof: int
    p: float


def correction_factor(m: int, n: int) -> float:
    return math.sqrt(1 + m / n)


def assign_groups(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """被験者ごとにランダムな 2 群のラベルを付ける（人数の差は 1 以内）。"""
    ids = df["seq_id"].drop_duplicates().to_numpy()
    labels = np.arange(len(ids)) % 2
    rng.shuffle(labels)
    return df.assign(**{GROUP: df["seq_id"].map(dict(zip(ids, labels))).astype(int)})


def subject_means(df: pd.DataFrame, var: str = VALUE) -> pd.DataFrame:
    g = df.groupby("seq_id")
    return pd.DataFrame({"value": g[var].mean(), "group": g[GROUP].first()})


def _student(values: np.ndarray, groups: np.ndarray, se_factor: float = 1.0) -> TTest:
    a, b = values[groups == 1], values[groups == 0]
    if len(a) < 2 or len(b) < 2:
        return TTest(float("nan"), float("nan"), 0, float("nan"))
    dof = len(a) + len(b) - 2
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / dof
    se = math.sqrt(pooled * (1 / len(a) + 1 / len(b))) * se_factor
    diff = float(a.mean() - b.mean())
    return TTest(diff, se, dof, float(2 * stats.t.sf(abs(diff / se), dof)))


def ttest_subjects(df: pd.DataFrame, se_factor: float = 1.0, var: str = VALUE) -> TTest:
    """被験者ごとの平均を 1 人 1 値として 2 群を比べる（独立な単位は被験者）。"""
    m = subject_means(df, var)
    return _student(m["value"].to_numpy(float), m["group"].to_numpy(), se_factor)


def ttest_records(df: pd.DataFrame, var: str = VALUE) -> TTest:
    """記録を独立とみなして 2 群を比べる（被験者内の相関を無視した誤った解析）。"""
    return _student(df[var].to_numpy(float), df[GROUP].to_numpy())


def run_rep(pool: pd.DataFrame, n: int, m: int, model_fn, seed: int, epochs: int) -> dict:
    ids = pool["seq_id"].drop_duplicates().to_numpy()
    if n < 4 or m < 4:
        raise ValueError("n と m は 4 以上にすること")
    if n > len(ids):
        raise ValueError(f"n={n} が実データの被験者数 {len(ids)} を超えている")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=n, replace=False)
    train = assign_groups(pool[pool["seq_id"].isin(chosen)], rng).reset_index(drop=True)
    gen = model_fn(train, n_seqs=m, epochs=epochs, seed=seed)

    real, rec = ttest_subjects(train), ttest_records(train)
    naive = ttest_subjects(gen)
    corrected = ttest_subjects(gen, se_factor=correction_factor(m, n))
    sd_real = subject_means(train)["value"].std()
    sd_syn = subject_means(gen)["value"].std()
    return {
        "seed": seed,
        "n": n,
        "m": m,
        "diff_real": real.diff,
        "se_real": real.se,
        "p_real_subject": real.p,
        "p_real_record": rec.p,
        "diff_syn": naive.diff,
        "se_syn_naive": naive.se,
        "p_syn_naive": naive.p,
        "p_syn_corrected": corrected.p,
        "sd_ratio": float(sd_syn / sd_real),
    }


def load_pool() -> pd.DataFrame:
    """前処理済みの A/B/C を合わせた全滞在（vitals-prepare の出力）。seq_id は滞在 ID のまま。"""
    return pd.concat([vitals.read_split(k) for k in "ABC"], ignore_index=True)


def run(model: str, n: int, m: int, reps: int, epochs: int, start: int = 0) -> Path:
    """reps 回の試行を行い、1 回ごとに CSV に追記する。既にある seed は飛ばす（途中から再開できる）。

    出力はシードの開始番号ごとに別ファイルにするので、開始番号を変えて並行に実行できる。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"reps_{model}_n{n}_m{m}_s{start}.csv"
    done = set(pd.read_csv(path)["seed"]) if path.exists() else set()
    pool = load_pool()
    for seed in range(start, start + reps):
        if seed in done:
            continue
        row = {"model": model, "epochs": epochs, **run_rep(pool, n, m, MODELS[model], seed, epochs)}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(row, flush=True)
    return path


def summarize(paths: list[Path], alpha: float = 0.05) -> pd.DataFrame:
    """条件（model, n, m）ごとの第 1 種の過誤率と、標準誤差の過小評価の程度。"""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rows = []
    for (model, n, m), g in df.groupby(["model", "n", "m"], sort=False):
        k = len(g)
        row = {"model": model, "n": n, "m": m, "reps": k}
        for col in ["p_real_subject", "p_real_record", "p_syn_naive", "p_syn_corrected"]:
            rate = float((g[col] < alpha).mean())
            row[col.removeprefix("p_")] = rate
            row[col.removeprefix("p_") + "_se"] = math.sqrt(rate * (1 - rate) / k)
        # 合成データでの群間差の試行間の標準偏差（真のばらつき）と、素朴な標準誤差の平均の比
        row["se_naive_over_sd"] = float(g["se_syn_naive"].mean() / g["diff_syn"].std())
        row["sd_ratio"] = float(g["sd_ratio"].mean())
        rows.append(row)
    return pd.DataFrame(rows)
