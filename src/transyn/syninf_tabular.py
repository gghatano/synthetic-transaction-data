"""Decruyenaere et al.（UAI 2024）の表データでのシミュレーションの追試（GHE issue 1 の前段）。

著者コード（https://github.com/syndara-lab/inferential-utility 、MIT ライセンス）の utils/disease.py と同じ
生成過程（年齢 → 病期 → バイオマーカー、治療 → 死亡）で実データを作り、生成モデルで同じ人数（M = n）の合成データを作る。
母集団の真値を帰無仮説にした検定で棄却した割合を「第 1 種の過誤率」とする（著者の定義）。補正は標準誤差に √(1 + M/n)。

著者との違い:
- 統計モデルとして著者は synthpop（R）とベイジアンネットワークを使う。ここでは同じ構造を持つ
  パラメトリックな逐次モデル（正しく特定されたモデル）で代える
- CTGAN・TVAE は SDV の既定のハイパーパラメータ（著者は既定版と、optuna で調整した版の両方）
- 推定量は年齢の平均（age_mean）と、死亡のロジスティック回帰の治療の係数（death_therapy）の 2 つに絞る
- 依存を増やさないため、ロジスティック回帰と順序ロジットは numpy / scipy で実装する
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit

OUT = Path("outputs") / "syninf_tabular"
COLUMNS = ["age", "stage", "biomarker", "therapy", "death"]
STAGE_INTERCEPTS = np.array([2.0, 3.0, 4.0])
STAGE_BETA_AGE = -0.05
BIOMARKER_BETA_STAGE = np.array([0.0, -1.0, -2.0, -3.0])
DEATH_BETA_STAGE = np.array([0.0, 0.5, 1.0, 1.5])
TRUTH = {"age_mean": 50.0, "death_therapy": -0.5}


@dataclass
class Est:
    value: float
    se: float
    dof: int | None  # t 分布の自由度（None なら標準正規）


def _sample_stage(age: np.ndarray, intercepts: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """比例オッズモデル: P(stage <= k) = expit(c_k + beta × age)。"""
    cum = expit(intercepts[None, :] + beta * age[:, None])
    return (rng.random(len(age))[:, None] > cum).sum(1)


def sample_disease(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    stage = _sample_stage(age, STAGE_INTERCEPTS, STAGE_BETA_AGE, rng)
    mean_bio = 1 / (4 + BIOMARKER_BETA_STAGE[stage])
    biomarker = rng.gamma(shape=25, scale=mean_bio / 25)
    therapy = (rng.random(n) < 0.5).astype(int)
    logit = -3 + 0.05 * age + DEATH_BETA_STAGE[stage] - 0.5 * therapy
    death = (rng.random(n) < expit(logit)).astype(int)
    return pd.DataFrame({"age": age, "stage": stage, "biomarker": biomarker, "therapy": therapy, "death": death})


def _design(df: pd.DataFrame, stages: list[int]) -> np.ndarray:
    cols = [np.ones(len(df)), df["age"].to_numpy(float)]
    cols += [(df["stage"] == s).to_numpy(float) for s in stages]
    cols.append(df["therapy"].to_numpy(float))
    return np.column_stack(cols)


def logistic(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray] | None:
    """ニュートン法によるロジスティック回帰。係数と Wald の標準誤差。収束しなければ None。"""
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = expit(x @ beta)
        hess = x.T @ (x * (p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(hess, x.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    else:
        return None
    if not np.isfinite(beta).all() or np.abs(beta).max() > 30:
        return None
    return beta, np.sqrt(np.diag(np.linalg.inv(hess)))


def _present_stages(df: pd.DataFrame) -> list[int]:
    """病期 II〜IV のダミーのうち、データに 1 件以上あるもの（無い列は推定できない）。"""
    return [s for s in (1, 2, 3) if (df["stage"] == s).any()]


def estimate(df: pd.DataFrame) -> dict[str, Est]:
    n = len(df)
    out = {"age_mean": Est(float(df["age"].mean()), float(df["age"].std(ddof=1) / math.sqrt(n)), n - 1)}
    stages = _present_stages(df)
    fit = logistic(_design(df, stages), df["death"].to_numpy(float))
    out["death_therapy"] = Est(float(fit[0][-1]), float(fit[1][-1]), None) if fit else Est(float("nan"), float("nan"), None)
    return out


def _fit_polr(age: np.ndarray, stage: np.ndarray) -> tuple[np.ndarray, float]:
    """比例オッズモデルの最尤推定。切片は c_1 と正の増分で表し、順序を保つ。"""

    def nll(theta: np.ndarray) -> float:
        c = np.cumsum(np.r_[theta[0], np.exp(theta[1:3])])
        cum = np.c_[np.zeros(len(age)), expit(c[None, :] + theta[3] * age[:, None]), np.ones(len(age))]
        prob = cum[np.arange(len(age)), stage + 1] - cum[np.arange(len(age)), stage]
        return -np.log(np.clip(prob, 1e-12, None)).sum()

    res = optimize.minimize(nll, x0=np.array([2.0, 0.0, 0.0, 0.0]), method="BFGS")
    c = np.cumsum(np.r_[res.x[0], np.exp(res.x[1:3])])
    return c, float(res.x[3])


def fit_generate_parametric(df: pd.DataFrame, m: int, seed: int) -> pd.DataFrame:
    """生成過程と同じ構造の逐次モデル（年齢: 正規、病期: 比例オッズ、バイオマーカー: 病期ごとのガンマ、
    治療: ベルヌーイ、死亡: ロジスティック）をデータから推定し、その順に生成する。"""
    rng = np.random.default_rng(seed + 10_000)
    age = rng.normal(df["age"].mean(), df["age"].std(ddof=1), m)
    c, b = _fit_polr(df["age"].to_numpy(float), df["stage"].to_numpy(int))
    stage = _sample_stage(age, c, b, rng)
    biomarker = np.empty(m)
    pooled = df["biomarker"]
    for s in range(4):
        x = df.loc[df["stage"] == s, "biomarker"]
        x = x if len(x) >= 3 else pooled
        shape, scale = x.mean() ** 2 / x.var(), x.var() / x.mean()
        biomarker[stage == s] = rng.gamma(shape, scale, (stage == s).sum())
    therapy = (rng.random(m) < df["therapy"].mean()).astype(int)
    stages = _present_stages(df)
    fit = logistic(_design(df, stages), df["death"].to_numpy(float))
    coef = fit[0] if fit else np.r_[math.log(df["death"].mean() / (1 - df["death"].mean())), np.zeros(len(stages) + 2)]
    gen = pd.DataFrame({"age": age, "stage": stage, "biomarker": biomarker, "therapy": therapy})
    # データに無かった病期は死亡への効果を 0 とする
    logit = coef[0] + coef[1] * age + coef[-1] * therapy
    for s, beta in zip(stages, coef[2:-1]):
        logit += beta * (stage == s)
    gen["death"] = (rng.random(m) < expit(logit)).astype(int)
    return gen[COLUMNS]


def _fit_generate_sdv(kind: str, df: pd.DataFrame, m: int, seed: int) -> pd.DataFrame:
    import torch
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

    np.random.seed(seed)
    torch.manual_seed(seed)
    data = df.assign(stage=df["stage"].astype(str), therapy=df["therapy"].astype(str), death=df["death"].astype(str))
    metadata = Metadata.detect_from_dataframe(data)
    for col in ["stage", "therapy", "death"]:
        metadata.update_column(col, sdtype="categorical")
    batch = max(10, min(500, len(df) // 10 * 10))
    if kind == "ctgan":
        model = CTGANSynthesizer(metadata, batch_size=batch, enforce_rounding=False)
    else:
        model = TVAESynthesizer(metadata, batch_size=batch, enforce_rounding=False)
    model.fit(data)
    gen = model.sample(m)
    return gen.assign(stage=gen["stage"].astype(int), therapy=gen["therapy"].astype(int), death=gen["death"].astype(int))[COLUMNS]


MODELS = {
    "parametric": fit_generate_parametric,
    "ctgan": lambda df, m, seed: _fit_generate_sdv("ctgan", df, m, seed),
    "tvae": lambda df, m, seed: _fit_generate_sdv("tvae", df, m, seed),
}


def _reject(e: Est, truth: float, factor: float = 1.0) -> float:
    if not np.isfinite(e.value):
        return float("nan")
    q = stats.t.ppf(0.975, e.dof) if e.dof else stats.norm.ppf(0.975)
    return float(abs(e.value - truth) > q * e.se * factor)


def run_rep(n: int, model_fn, seed: int, m: int | None = None) -> dict:
    m = m or n
    real = sample_disease(n, seed)
    gen = model_fn(real, m, seed)
    er, eg = estimate(real), estimate(gen)
    row = {"seed": seed, "n": n, "m": m}
    factor = math.sqrt(1 + m / n)
    for name, truth in TRUTH.items():
        row |= {
            f"{name}_real": er[name].value,
            f"{name}_syn": eg[name].value,
            f"{name}_se_syn": eg[name].se,
            f"{name}_reject_real": _reject(er[name], truth),
            f"{name}_reject_naive": _reject(eg[name], truth),
            f"{name}_reject_corrected": _reject(eg[name], truth, factor),
        }
    return row


def run(model: str, n: int, reps: int, start: int = 0) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"reps_{model}_n{n}.csv"
    done = set(pd.read_csv(path)["seed"]) if path.exists() else set()
    for seed in range(start, start + reps):
        if seed in done:
            continue
        row = {"model": model, **run_rep(n, MODELS[model], seed)}
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        print(row, flush=True)
    return path


def summarize(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rows = []
    for (model, n), g in df.groupby(["model", "n"], sort=False):
        for name, truth in TRUTH.items():
            ok = g[g[f"{name}_syn"].notna()]
            rows.append({
                "model": model, "n": n, "estimator": name, "reps": len(ok),
                # 合成データで推定できなかった（死亡が 1 種類しかない、分離が起きた）試行の割合
                "nonestimable": float(1 - len(ok) / len(g)),
                "bias": float((ok[f"{name}_syn"] - truth).mean()),
                "type1_real": float(ok[f"{name}_reject_real"].mean()),
                "type1_naive": float(ok[f"{name}_reject_naive"].mean()),
                "type1_corrected": float(ok[f"{name}_reject_corrected"].mean()),
                # 合成データの推定値の試行間の標準偏差（真のばらつき）に対する、補正後の標準誤差の平均の比
                "se_corrected_over_sd": float(ok[f"{name}_se_syn"].mean() * math.sqrt(1 + ok["m"].iloc[0] / n) / ok[f"{name}_syn"].std()),
            })
    return pd.DataFrame(rows)
