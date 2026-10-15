"""見本用データの忠実度の段階と公開前チェック（GHE issue 9）。

英国統計局（ONS）の合成データの 6 段階（Bates et al., ONS methodology working paper series no.16, 2019）を、
被験者ごとの時系列に当てはめた 5 段階を作る。

| 段階 | 保つもの | ONS の段階との対応 |
|---|---|---|
| L0 | 項目・単位・間隔・記録長だけ。値は一般的な参考範囲から一様に引く（データを使わない） | synthetic structural |
| L1 | 変数ごとの値の分布（時点も被験者も独立） | synthetically-augmented plausible |
| L2 | L1 ＋ 被験者内の時間構造（全員が同じ平均・ばらつき） | multivariate plausible の一部 |
| L3 | L2 ＋ 被験者ごとの平均・ばらつきの分布（属性とは無関係） | multivariate plausible |
| L4 | L3 ＋ 年齢・死亡退院との関係（公開リポジトリの #15 の VAR(1)） | multivariate detailed |

公開前チェックは Raab, McCall, Cavin（Four checks for low-fidelity synthetic data, arXiv:2503.14211）の
表示・開示リスク・構造・文書化を、時系列向けに具体化したもの。
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import vitals
from .models import _seed

OUT = Path("outputs") / "showcase"
LEVELS = ("L0", "L1", "L2", "L3", "L4")
# L0 の値の範囲。データからではなく、成人の一般的な目安として決めた値
REFERENCE_RANGE = {"heartrate": (50.0, 120.0), "sao2": (88.0, 100.0), "respiration": (10.0, 30.0)}
REFERENCE_AGE = (18, 90)
REFERENCE_MORTALITY = 0.1
COLUMNS = ["seq_id", "t", *vitals.VARS, "age", "died"]

DESCRIPTION = {
    "L0": {
        "generator": "成人の一般的な参考範囲からの一様乱数（実データを使わない）",
        "preserved": ["項目と単位", "5分間隔・12時間（144点）の記録長"],
        "not_preserved": ["値の分布", "時間変化", "被験者ごとの違い", "変数間・属性との関係"],
    },
    "L1": {
        "generator": "変数ごとに、学習データの全時点の値から独立に復元抽出",
        "preserved": ["項目と単位", "記録長", "変数ごとの値の分布"],
        "not_preserved": ["時間変化（自己相関）", "被験者ごとの違い", "変数間・属性との関係"],
    },
    "L2": {
        "generator": "変数間のVAR(1)で作った変動を、全員共通の平均・ばらつきに載せる",
        "preserved": ["項目と単位", "記録長", "変数ごとの値のおおよその分布", "被験者内の時間変化と変数間の相関"],
        "not_preserved": ["被験者ごとの違い", "属性との関係"],
    },
    "L3": {
        "generator": "被験者ごとの平均・ばらつきを多変量正規分布から引き、被験者内の変動をVAR(1)で作る（属性とは無関係）",
        "preserved": ["項目と単位", "記録長", "値の分布", "被験者内の時間変化", "被験者ごとの違い"],
        "not_preserved": ["年齢・死亡退院との関係"],
    },
    "L4": {
        "generator": "L3を死亡退院の有無ごとに推定し、年齢も含めて引く（公開リポジトリの#15のVAR(1)）",
        "preserved": ["項目と単位", "記録長", "値の分布", "被験者内の時間変化", "被験者ごとの違い", "年齢・死亡退院との関係（平均・共分散の水準）"],
        "not_preserved": ["非線形・高次の関係", "欠測のしかた（学習データは補間済み）"],
    },
}
INTENDED_USE = "データの形と典型的な変動を説明するための見本。分析・検定・予測モデルの学習には使わない"


# ---------------------------------------------------------------- 段階ごとの生成


def _long(x: np.ndarray, age: np.ndarray, died: np.ndarray) -> pd.DataFrame:
    n = len(x)
    df = pd.DataFrame(x.reshape(-1, len(vitals.VARS)), columns=vitals.VARS)
    df.insert(0, "t", np.tile(np.arange(vitals.STEPS), n))
    df.insert(0, "seq_id", np.repeat(np.arange(n), vitals.STEPS))
    df["age"] = np.repeat(age, vitals.STEPS)
    df["died"] = np.repeat(died, vitals.STEPS).astype(int)
    return df


def _var_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """滞在内で標準化した変動に変数間の VAR(1) を当てる（vitals.fit_generate_var1 と同じ推定）。"""
    mean, std = x.mean(1), x.std(1) + 1e-6
    z = (x - mean[:, None]) / std[:, None]
    prev, cur = z[:, :-1].reshape(-1, z.shape[-1]), z[:, 1:].reshape(-1, z.shape[-1])
    coef, *_ = np.linalg.lstsq(prev, cur, rcond=None)
    return coef, np.cov((cur - prev @ coef).T)


def _var_sample(coef: np.ndarray, noise_cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    k = coef.shape[0]
    z = np.empty((n, vitals.STEPS, k))
    z[:, 0] = rng.multivariate_normal(np.zeros(k), np.eye(k), n)
    for t in range(1, vitals.STEPS):
        z[:, t] = z[:, t - 1] @ coef + rng.multivariate_normal(np.zeros(k), noise_cov, n)
    return z


def generate(level: str, train: pd.DataFrame, n_seqs: int, seed: int, tail_clip: float | None = None) -> pd.DataFrame:
    """段階 level の見本用データを n_seqs 人分作る。

    tail_clip を与えると、L1〜L4 の値を学習データの [tail_clip, 1 − tail_clip] 分位点で切り詰める。既定の
    vitals._finalize は学習データの最小・最大で切り詰めるので、実在の被験者の極値がそのまま現れることがある。
    """
    gen = _generate(level, train, n_seqs, seed)
    if tail_clip is not None and level != "L0":
        for v in vitals.VARS:
            lo, hi = train[v].quantile([tail_clip, 1 - tail_clip])
            gen[v] = gen[v].clip(lo, hi)
    return gen


def _generate(level: str, train: pd.DataFrame, n_seqs: int, seed: int) -> pd.DataFrame:
    _seed(seed)
    rng = np.random.default_rng(seed)
    if level == "L0":
        x = np.stack([rng.uniform(*REFERENCE_RANGE[v], (n_seqs, vitals.STEPS)) for v in vitals.VARS], axis=-1)
        age = rng.integers(REFERENCE_AGE[0], REFERENCE_AGE[1] + 1, n_seqs)
        died = (rng.random(n_seqs) < REFERENCE_MORTALITY).astype(int)
        return _long(x, age, died)[COLUMNS]
    if level == "L4":
        return vitals.fit_generate_var1(train, n_seqs=n_seqs, epochs=0, seed=seed)

    arr = vitals.to_array(train)
    static = vitals.static_of(train)
    age = rng.choice(static["age"].to_numpy(), n_seqs)
    died = rng.choice(static["died"].to_numpy(), n_seqs)
    flat = arr.reshape(-1, arr.shape[-1])
    if level == "L1":
        x = np.stack([rng.choice(flat[:, j], (n_seqs, vitals.STEPS)) for j in range(flat.shape[1])], axis=-1)
    else:
        coef, noise_cov = _var_fit(arr)
        z = _var_sample(coef, noise_cov, n_seqs, rng)
        if level == "L2":
            x = flat.mean(0) + flat.std(0) * z
        elif level == "L3":
            stay = np.column_stack([arr.mean(1), np.log(arr.std(1) + 1e-6)])
            s = rng.multivariate_normal(stay.mean(0), np.cov(stay.T), n_seqs)
            k = arr.shape[-1]
            x = s[:, None, :k] + np.exp(s[:, None, k:]) * z
        else:
            raise ValueError(level)
    return vitals._finalize(_long(x, age, died), train)


# ---------------------------------------------------------------- 公開前チェック


def label(gen: pd.DataFrame, level: str, source: str = "eICU Collaborative Research Database Demo v2.0.1") -> tuple[pd.DataFrame, dict]:
    """表示: 実在しない ID（SYN- で始まる）に付け替え、合成である旨と保つ性質を書いたメタデータを返す。"""
    out = gen.copy()
    out["seq_id"] = [f"SYN-{level}-{int(i):05d}" for i in pd.factorize(out["seq_id"])[0]]
    meta = {
        "synthetic": True,
        "level": level,
        "source": source,
        "created": datetime.date.today().isoformat(),
        "intended_use": INTENDED_USE,
        **DESCRIPTION[level],
    }
    return out, meta


def _windows(df: pd.DataFrame, length: int) -> dict[int, set[bytes]]:
    x = np.round(vitals.to_array(df), 3)
    ids = np.sort(df["seq_id"].unique())
    return {i: {x[k, s : s + length].tobytes() for s in range(x.shape[1] - length + 1)} for k, i in enumerate(ids)}


def exact_segment_matches(gen: pd.DataFrame, real: pd.DataFrame, length: int = 12) -> int:
    """合成系列のうち、連続する length 点（12 点 = 1 時間）の 3 変数の値が、実データのどこかの区間と完全に一致するものの数。"""
    real_set = set().union(*_windows(real, length).values())
    return sum(1 for w in _windows(gen, length).values() if w & real_set)


def disclosure_checks(gen: pd.DataFrame, real: pd.DataFrame, holdout: pd.DataFrame, min_group: int = 10) -> dict:
    """開示リスク: 区間の完全一致、実データの極値の再現、少数の群、距離に基づく指標。"""
    # 実データの最小・最大のうち、その値をとる被験者が min_group 人未満のもの（特定の人を表す極値）が、
    # 合成データにそのまま現れるか。SpO2 の 100% のように多くの人がとる天井の値は数えない
    extremes = 0
    for v in vitals.VARS:
        vals = set(np.round(gen[v].to_numpy(), 6))
        r = np.round(real[v].to_numpy(), 6)
        for x in (r.min(), r.max()):
            holders = real.loc[r == x, "seq_id"].nunique()
            # 公開の生理的な範囲の端（vitals.BOUNDS）と同じ値は、切り詰めで誰にでも出るので数えない
            at_bound = x in vitals.BOUNDS[v]
            extremes += int(holders < min_group and x in vals and not at_bound)
    groups = vitals.static_of(real)["died"].value_counts()
    small = [f"died={g}" for g, c in groups.items() if c < min_group]
    return {
        "exact_segments": exact_segment_matches(gen, real),
        "extreme_values_reproduced": extremes,
        "small_groups": small,
        **vitals.privacy(real, holdout, gen),
    }


def structure_checks(gen: pd.DataFrame, real: pd.DataFrame) -> dict[str, bool]:
    """構造: 列（並びは問わない）、系列ごとの時点数と時刻番号、生理的にありうる範囲、欠測なし。"""
    sizes = gen.groupby("seq_id")["t"].agg(["size", "min", "max"])
    in_range = all(gen[v].between(*vitals.BOUNDS[v]).all() for v in vitals.VARS)
    return {
        "columns": set(gen.columns) == set(real.columns),
        "steps": bool(((sizes["size"] == vitals.STEPS) & (sizes["min"] == 0) & (sizes["max"] == vitals.STEPS - 1)).all()),
        "ranges": bool(in_range),
        "no_missing": bool(gen[vitals.VARS].notna().all().all()),
    }


# ---------------------------------------------------------------- 段階ごとの評価


def relation_stats(df: pd.DataFrame) -> dict[str, float]:
    """見本から読み取れてしまう関係: 死亡退院の有無による平均心拍数の差、年齢と平均心拍数の相関。"""
    m = df.groupby("seq_id").agg(hr=("heartrate", "mean"), age=("age", "first"), died=("died", "first"))
    diff = m.loc[m["died"] == 1, "hr"].mean() - m.loc[m["died"] == 0, "hr"].mean() if m["died"].nunique() == 2 else float("nan")
    return {"HRDiffDied": float(diff), "AgeHRCorr": float(np.corrcoef(m["age"], m["hr"])[0, 1])}


def evaluate_levels(seeds: tuple[int, ...] = (0, 1, 2), tail_clip: float = 0.005) -> pd.DataFrame:
    """段階ごとに、既定（学習データの最小・最大で切り詰め）と、裾を tail_clip で切り詰めた版（-clip）を評価する。"""
    a, b, c = (vitals.read_split(k) for k in "ABC")
    rows = [{"level": "real-C", "seed": 0, **vitals.fidelity(a, c), **relation_stats(c), "TSTR_AUROC": vitals.utility(c, b)}]
    variants = [(lv, None) for lv in LEVELS] + [(f"{lv}-clip", tail_clip) for lv in LEVELS if lv != "L0"]
    for name, clip in variants:
        for seed in seeds:
            gen = generate(name.removesuffix("-clip"), a, n_seqs=a["seq_id"].nunique(), seed=seed, tail_clip=clip)
            rows.append({
                "level": name, "seed": seed,
                **vitals.fidelity(c, gen, seed), **relation_stats(gen),
                "TSTR_AUROC": vitals.utility(gen, b, seed),
                **{k: v for k, v in disclosure_checks(gen, a, b).items() if k != "small_groups"},
                **{f"struct_{k}": v for k, v in structure_checks(gen, a).items()},
            })
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)
