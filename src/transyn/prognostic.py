"""外部データで学習した予後予測を共変量にして、必要な被験者数を減らせるかの検証（GHE issue 2）。

PROCOVA（Schuler et al., Int J Biostat 2021）の考え方を、被験者ごとの時系列で半合成実験として確かめる。

- 12 時間のバイタル系列を、前半 6 時間（ベースライン）と後半 6 時間（結果）に分ける
- 結果は後半の平均心拍数（mean）か、心拍数の log 標準偏差（logsd）
- 病院単位で「外部」と「対象」に分ける。外部データで、ベースラインの要約特徴量から結果を予測するモデルを学習する
- 対象から n 人を抜いてランダムに 2 群に分け、片方の結果に既知の効果 delta を足す（結果の尺度での加法的な効果）。
  予後予測はベースラインだけから作るので、効果の影響を受けない
- 調整なし、ベースラインの同じ量で調整した ANCOVA、予後予測で調整した ANCOVA を比べる
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import syninf, vitals

OUT = Path("outputs") / "prognostic"
SPLIT = vitals.STEPS // 2  # 前半 6 時間（72 点）がベースライン
HR = vitals.VARS.index("heartrate")
OUTCOMES = ("mean", "logsd")


def outcome(x: np.ndarray, kind: str) -> np.ndarray:
    return _stat(x[:, SPLIT:, HR], kind)


def baseline(x: np.ndarray, kind: str) -> np.ndarray:
    return _stat(x[:, :SPLIT, HR], kind)


def baseline_mean(x: np.ndarray) -> np.ndarray:
    return baseline(x, "mean")


def _stat(hr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "mean":
        return hr.mean(1)
    if kind == "logsd":
        return np.log(hr.std(1) + 1e-6)
    raise ValueError(kind)


@dataclass
class Features:
    """被験者ごとの、ベースラインの要約特徴量（x）、結果（y）、ベースラインの同じ量（base）。"""

    ids: np.ndarray
    x: np.ndarray
    y: pd.DataFrame
    base: pd.DataFrame

    @classmethod
    def of(cls, df: pd.DataFrame) -> "Features":
        arr = vitals.to_array(df)
        static = vitals.static_of(df)
        x = vitals.summary_features(arr[:, :SPLIT], static)
        y = pd.DataFrame({k: outcome(arr, k) for k in OUTCOMES})
        base = pd.DataFrame({k: baseline(arr, k) for k in OUTCOMES})
        return cls(static.index.to_numpy(), x, y, base)

    def take(self, idx: np.ndarray) -> "Features":
        return Features(self.ids[idx], self.x[idx], self.y.iloc[idx].reset_index(drop=True), self.base.iloc[idx].reset_index(drop=True))


def fit_prognostic(feats: Features, outcome_kind: str, kind: str = "linear"):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "linear":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    elif kind == "gbm":
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    else:
        raise ValueError(kind)
    return model.fit(feats.x, feats.y[outcome_kind])


def ancova(y: np.ndarray, treat: np.ndarray, cov: np.ndarray | None) -> syninf.TTest:
    """y ~ 1 + treat (+ cov) の最小二乗。treat の係数と、その標準誤差・t 検定の p 値。"""
    cols = [np.ones_like(y), treat.astype(float)] + ([] if cov is None else [cov])
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    dof = len(y) - design.shape[1]
    sigma2 = float(((y - design @ beta) ** 2).sum() / dof)
    se = math.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
    return syninf.TTest(float(beta[1]), se, dof, float(2 * stats.t.sf(abs(beta[1] / se), dof)))


def run_trial(feats: Features, n: int, delta: float, outcome_kind: str, scores: dict, rng: np.random.Generator) -> list[dict]:
    if n > len(feats.ids):
        raise ValueError(f"n={n} が対象の被験者数 {len(feats.ids)} を超えている")
    sel = feats.take(rng.choice(len(feats.ids), size=n, replace=False))
    treat = np.arange(n) % 2
    rng.shuffle(treat)
    y = sel.y[outcome_kind].to_numpy() + delta * treat
    covariates = {"unadjusted": None, "baseline": sel.base[outcome_kind].to_numpy()}
    covariates |= {name: model.predict(sel.x) for name, model in scores.items()}
    return [{"method": name, **vars(ancova(y, treat, cov))} for name, cov in covariates.items()]


def split_hospitals(hosp: pd.Series, frac_external: float = 0.5, seed: int = 0) -> tuple[list, list]:
    """病院をランダムな順に外部へ入れていき、外部の被験者の割合が frac_external に達したら残りを対象にする。"""
    sizes = hosp.value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    ext, total = [], 0
    for h in order:
        if total / len(hosp) >= frac_external:
            break
        ext.append(h)
        total += sizes[h]
    return ext, [h for h in order if h not in ext]


DELTAS = {"mean": 5.0, "logsd": 0.2}  # 心拍数 5 bpm、心拍数の標準偏差で約 22% の変化
SCORE_KINDS = ("prog-linear", "prog-gbm", "prog-syn-var1", "prog-syn-par")


def summarize(reps: pd.DataFrame, power: float = 0.8, alpha: float = 0.05) -> pd.DataFrame:
    """条件（outcome, n, method）ごとの第 1 種の過誤率、検出力、標準誤差、調整なしとの分散比、80% の検出力に必要な人数。

    必要人数は標準誤差が 1/√n に比例するとして、n80 = n × (SE × (z_{1-α/2} + z_{power}) / δ)² で見積もる。
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    rows = []
    for (kind, n, method), g in reps.groupby(["outcome", "n", "method"], sort=False):
        null, alt = g[g["delta"] == 0], g[g["delta"] != 0]
        delta = float(alt["delta"].iloc[0]) if len(alt) else float("nan")
        se = float(g["se"].mean())
        rows.append({"outcome": kind, "n": n, "method": method, "reps": len(null),
                     "type1": float((null["p"] < alpha).mean()), "power": float((alt["p"] < alpha).mean()),
                     "se": se, "n80": n * (se * z / delta) ** 2})
    table = pd.DataFrame(rows)
    unadj = table[table["method"] == "unadjusted"].set_index(["outcome", "n"])["se"]
    table["var_ratio"] = [(r.se / unadj[(r.outcome, r.n)]) ** 2 for r in table.itertuples()]
    return table


def _hospitals(stay_ids: np.ndarray) -> pd.Series:
    patient = pd.read_csv(vitals._find("patient"), usecols=["patientunitstayid", "hospitalid"])
    return patient.set_index("patientunitstayid")["hospitalid"].reindex(stay_ids)


def experiment(ns: list[int], reps: int, seed: int = 0, frac_external: float = 0.5, par_epochs: int = 200,
               score_kinds: tuple[str, ...] = SCORE_KINDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """病院単位で外部・対象に分け、予後予測モデルを外部（またはその合成版）で学習して、対象で試行を繰り返す。"""
    pool = syninf.load_pool()
    ids = pool["seq_id"].drop_duplicates().to_numpy()
    hosp = _hospitals(ids)
    ext_h, _ = split_hospitals(hosp, frac_external, seed)
    ext_ids = ids[hosp.isin(ext_h).to_numpy()]
    ext_df, tgt_df = pool[pool["seq_id"].isin(ext_ids)], pool[~pool["seq_id"].isin(ext_ids)]
    f_ext, f_tgt = Features.of(ext_df), Features.of(tgt_df)
    synthetic = {}
    if "prog-syn-var1" in score_kinds:
        synthetic["prog-syn-var1"] = vitals.fit_generate_var1(ext_df, n_seqs=len(f_ext.ids), epochs=0, seed=seed)
    if "prog-syn-par" in score_kinds:
        synthetic["prog-syn-par"] = vitals.fit_generate_par(ext_df, n_seqs=len(f_ext.ids), epochs=par_epochs, seed=seed)

    info, rows = [], []
    rng = np.random.default_rng(seed)
    for kind in OUTCOMES:
        scores = {}
        if "prog-linear" in score_kinds:
            scores["prog-linear"] = fit_prognostic(f_ext, kind, "linear")
        if "prog-gbm" in score_kinds:
            scores["prog-gbm"] = fit_prognostic(f_ext, kind, "gbm")
        for name, syn in synthetic.items():
            scores[name] = fit_prognostic(Features.of(syn), kind, "linear")
        y = f_tgt.y[kind].to_numpy()
        info.append({"outcome": kind, "method": "baseline", "rho": float(np.corrcoef(f_tgt.base[kind], y)[0, 1])})
        info += [{"outcome": kind, "method": name, "rho": float(np.corrcoef(m.predict(f_tgt.x), y)[0, 1])} for name, m in scores.items()]
        for n in ns:
            for rep in range(reps):
                for delta in (0.0, DELTAS[kind]):
                    for r in run_trial(f_tgt, n, delta, kind, scores, rng):
                        rows.append({"outcome": kind, "n": n, "delta": delta, "rep": rep, **r})
    meta = pd.DataFrame(info).assign(
        n_external=len(f_ext.ids), n_target=len(f_tgt.ids), hospitals_external=len(ext_h), hospitals_target=hosp.nunique() - len(ext_h)
    )
    return pd.DataFrame(rows), meta
