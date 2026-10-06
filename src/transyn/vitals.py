"""eICU Demo の定期バイタル（5 分間隔の多変量時系列）での検証（issue #15）。

ICU 入室から WINDOW_HOURS 時間の心拍数・SpO2・呼吸数を 5 分刻みの固定長系列にし、生成手法を比べる。
系列は long 形式（seq_id, t, heartrate, sao2, respiration, age, died）で扱う。t は 0〜STEPS-1 の時刻番号。

評価は 3 軸:
- 忠実度: 変数ごとの分布、変数間の相関、自己相関、パワースペクトル、滞在ごとの平均・ばらつき、判別スコア
- 有用性: 合成データで死亡退院を予測するモデルを学習し、実データ B で評価する（TSTR）
- プライバシ: 最近傍距離（DCR）、距離に基づくメンバーシップ推論の AUC、学習データにごく近い系列の割合

データ分割は患者（uniquepid）単位で A（学習 50%）/ B（評価・非メンバー 25%）/ C（実データの基準 25%）。
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .models import _seed

DATA_DIR = Path("data") / "eicu_demo"
OUT = Path("outputs") / "vitals"
SPLITS = {k: DATA_DIR / f"vitals_{k}.csv.gz" for k in "ABC"}

VARS = ["heartrate", "sao2", "respiration"]
# 生理的にありえない値（0 など）は欠測として扱う
BOUNDS = {"heartrate": (20, 250), "sao2": (50, 100), "respiration": (3, 60)}
STEP_MIN = 5
WINDOW_HOURS = 12
STEPS = WINDOW_HOURS * 60 // STEP_MIN  # 144
MIN_COVERAGE = 0.8
ACF_LAGS = (1, 6, 12)  # 5 分・30 分・1 時間


def _find(name: str) -> Path:
    hits = sorted(DATA_DIR.rglob(f"{name}.csv*"))
    if not hits:
        raise FileNotFoundError(f"{name}.csv(.gz) が {DATA_DIR} にない。eICU Demo を展開すること")
    return hits[0]


def load_sequences() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(seqs, stays) を返す。seqs は long 形式の固定長系列、stays は stay, uniquepid, age, died。"""
    patient = pd.read_csv(_find("patient"))
    stays = pd.DataFrame(
        {
            "stay": patient["patientunitstayid"],
            "uniquepid": patient["uniquepid"],
            "age": pd.to_numeric(patient["age"].replace("> 89", "90"), errors="coerce"),
            "died": (patient["hospitaldischargestatus"] == "Expired").astype(int),
        }
    ).dropna(subset=["age"])

    v = pd.read_csv(_find("vitalPeriodic"), usecols=["patientunitstayid", "observationoffset", *VARS])
    v = v[(v["observationoffset"] >= 0) & (v["observationoffset"] < WINDOW_HOURS * 60)]
    for col, (lo, hi) in BOUNDS.items():
        v.loc[(v[col] < lo) | (v[col] > hi), col] = np.nan
    v["t"] = (v["observationoffset"] // STEP_MIN).astype(int)
    grid = v.groupby(["patientunitstayid", "t"])[VARS].mean()

    seqs = []
    for stay, g in grid.groupby(level=0):
        g = g.droplevel(0).reindex(range(STEPS))
        if (g.notna().mean() < MIN_COVERAGE).any():
            continue
        g = g.interpolate(limit_direction="both")
        g.insert(0, "seq_id", stay)
        seqs.append(g.rename_axis("t").reset_index())
    seqs = pd.concat(seqs, ignore_index=True)
    seqs = seqs.merge(stays[["stay", "age", "died"]], left_on="seq_id", right_on="stay").drop(columns="stay")
    seqs["age"] = seqs["age"].astype(int)
    stays = stays[stays["stay"].isin(seqs["seq_id"])]
    return seqs, stays


def prepare(seed: int = 0) -> dict[str, pd.DataFrame]:
    seqs, stays = load_sequences()
    rng = np.random.default_rng(seed)
    pids = stays["uniquepid"].unique().to_numpy(dtype=object)
    rng.shuffle(pids)
    n = len(pids)
    groups = {"A": pids[: n // 2], "B": pids[n // 2 : 3 * n // 4], "C": pids[3 * n // 4 :]}
    out = {}
    for name, members in groups.items():
        ids = stays.loc[stays["uniquepid"].isin(members), "stay"]
        out[name] = seqs[seqs["seq_id"].isin(ids)].reset_index(drop=True)
        out[name].to_csv(SPLITS[name], index=False)
    return out


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS[name])


def to_array(df: pd.DataFrame) -> np.ndarray:
    """(系列数, STEPS, 変数数) の配列にする。"""
    d = df.sort_values(["seq_id", "t"], kind="stable")
    return d[VARS].to_numpy(float).reshape(-1, STEPS, len(VARS))


def static_of(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("seq_id", sort=True)[["age", "died"]].first()


# ---------------------------------------------------------------- 生成手法


def _finalize(gen: pd.DataFrame, bounds_from: pd.DataFrame) -> pd.DataFrame:
    """学習データの範囲にクリップし、列と型を揃える。"""
    gen = gen.copy()
    for col in VARS + ["age"]:
        gen[col] = gen[col].astype(float).clip(bounds_from[col].min(), bounds_from[col].max())
    gen["age"] = gen["age"].round().astype(int)
    gen["died"] = gen["died"].astype(int)
    gen["seq_id"] = pd.factorize(gen["seq_id"])[0]
    return gen[["seq_id", "t", *VARS, "age", "died"]].sort_values(["seq_id", "t"]).reset_index(drop=True)


def fit_generate_dgan(
    train: pd.DataFrame, n_seqs: int, epochs: int, seed: int, sample_len: int = 6, batch_size: int = 1000
) -> pd.DataFrame:
    """DoppelGANger（gretel 版）。等間隔の固定長多変量時系列で、本来の対象に近い使い方。

    batch_size は学習系列数より大きいと 1 エポック 1 回の更新になる（716 系列なら既定の 1000 で 1 回）。
    """
    from gretel_synthetics.timeseries_dgan.config import DfStyle, DGANConfig
    from gretel_synthetics.timeseries_dgan.dgan import DGAN

    _seed(seed)
    frame = train[["seq_id", "t", *VARS, "age", "died"]].assign(age=train["age"].astype(float))
    model = DGAN(
        DGANConfig(
            max_sequence_len=STEPS,
            sample_len=sample_len,
            batch_size=min(batch_size, train["seq_id"].nunique()),
            epochs=epochs,
            cuda=False,
        )
    )
    started = time.time()

    def progress(info):
        if info.batch == info.total_batches - 1 and info.epoch % 100 == 0:
            print(f"epoch {info.epoch}/{info.total_epochs} ({time.time() - started:.0f}s)", flush=True)

    model.train_dataframe(
        frame,
        attribute_columns=["age", "died"],
        feature_columns=VARS,
        example_id_column="seq_id",
        time_column="t",
        discrete_columns=["died"],
        df_style=DfStyle.LONG,
        progress_callback=progress,
    )
    return _finalize(model.generate_dataframe(n_seqs), train)


def fit_generate_par(train: pd.DataFrame, n_seqs: int, epochs: int, seed: int) -> pd.DataFrame:
    """SDV の PAR。時刻番号 t を順序に、年齢・死亡退院をコンテキストにする。"""
    from sdv.metadata import Metadata
    from sdv.sequential import PARSynthesizer

    _seed(seed)
    seqs = train[["seq_id", "t", *VARS, "age", "died"]]
    metadata = Metadata.detect_from_dataframe(seqs)
    metadata.update_column("seq_id", sdtype="id")
    metadata.update_column("died", sdtype="categorical")
    metadata.set_sequence_key("seq_id")
    metadata.set_sequence_index("t")
    model = PARSynthesizer(metadata, context_columns=["age", "died"], epochs=epochs, verbose=True, enforce_rounding=False)
    model.fit(seqs)
    gen = model.sample(num_sequences=n_seqs, sequence_length=STEPS)
    # PAR は時刻番号も生成するので、生成順で 0〜STEPS-1 を振り直す
    gen["t"] = gen.groupby("seq_id").cumcount()
    return _finalize(gen[gen["t"] < STEPS], train)


def fit_generate_ctgan(train: pd.DataFrame, n_seqs: int, epochs: int, seed: int) -> pd.DataFrame:
    """CTGAN を「1 行 = 1 時点」で適用する。生成行を時刻番号ごとに 1 行ずつランダムに束ねて系列とみなす。

    時刻番号 t も列として学習させ、t ごとに n_seqs 行ずつ条件付きで生成する。行間（時点間）の依存は学習されない。
    系列単位の属性は、束ねた先頭（t=0）の行の値に揃える。
    """
    from sdv.metadata import Metadata
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer

    _seed(seed)
    rows = train[["t", *VARS, "age", "died"]]
    metadata = Metadata.detect_from_dataframe(rows)
    metadata.update_column("t", sdtype="categorical")
    metadata.update_column("died", sdtype="categorical")
    model = CTGANSynthesizer(metadata, epochs=epochs, verbose=True, enforce_rounding=False, log_frequency=False)
    model.fit(rows)
    gen = model.sample_from_conditions([Condition({"t": t}, num_rows=n_seqs) for t in range(STEPS)])
    gen["seq_id"] = gen.groupby("t").cumcount().to_numpy()
    gen["seq_id"] = gen.groupby("t")["seq_id"].transform(lambda s: np.random.permutation(s.to_numpy()))
    gen = gen.sort_values(["seq_id", "t"])
    for col in ["age", "died"]:
        gen[col] = gen.groupby("seq_id")[col].transform("first")
    return _finalize(gen, train)


def fit_generate_var1(train: pd.DataFrame, n_seqs: int, epochs: int, seed: int) -> pd.DataFrame:
    """比較用の単純な統計モデル（epochs は使わない）。

    - 滞在ごとの性質（各変数の平均・log 標準偏差、年齢）を、死亡退院の有無ごとに多変量正規分布で表して引く
    - 滞在内で標準化した変動を、変数間の VAR(1)（z_t = A z_{t-1} + e）で生成する
    学習データの系列そのものは保持しない。
    """
    _seed(seed)
    rng = np.random.default_rng(seed)
    x = to_array(train)
    static = static_of(train)
    mean, std = x.mean(1), x.std(1) + 1e-6
    z = (x - mean[:, None]) / std[:, None]
    prev, cur = z[:, :-1].reshape(-1, len(VARS)), z[:, 1:].reshape(-1, len(VARS))
    coef, *_ = np.linalg.lstsq(prev, cur, rcond=None)
    noise_cov = np.cov((cur - prev @ coef).T)
    stay = np.column_stack([mean, np.log(std), static["age"].to_numpy(float)])
    died = static["died"].to_numpy()
    rows = []
    for i in range(n_seqs):
        d = int(rng.random() < died.mean())
        group = stay[died == d]
        s = rng.multivariate_normal(group.mean(0), np.cov(group.T))
        m, sd, age = s[: len(VARS)], np.exp(s[len(VARS) : 2 * len(VARS)]), s[-1]
        zt = rng.multivariate_normal(np.zeros(len(VARS)), np.eye(len(VARS)))
        for t in range(STEPS):
            rows.append({"seq_id": i, "t": t, **dict(zip(VARS, m + sd * zt)), "age": age, "died": d})
            zt = zt @ coef + rng.multivariate_normal(np.zeros(len(VARS)), noise_cov)
    return _finalize(pd.DataFrame(rows), train)


MODELS = {
    "dgan": fit_generate_dgan,
    "dgan-b64": partial(fit_generate_dgan, batch_size=64),
    "par": fit_generate_par,
    "ctgan": fit_generate_ctgan,
    "var1": fit_generate_var1,
}


# ---------------------------------------------------------------- 評価


def _scale(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    return flat.mean(0), flat.std(0)


def _acf(x: np.ndarray, lag: int) -> np.ndarray:
    """系列ごと・変数ごとの自己相関（系列内で平均を引いて計算）を変数ごとに平均する。"""
    z = x - x.mean(1, keepdims=True)
    num = (z[:, lag:] * z[:, :-lag]).sum(1)
    den = (z**2).sum(1) + 1e-9
    return (num / den).mean(0)


def _log_psd(x: np.ndarray) -> np.ndarray:
    """系列ごとに平均を引いたパワースペクトルの、系列平均の対数（変数ごと）。"""
    z = x - x.mean(1, keepdims=True)
    p = (np.abs(np.fft.rfft(z, axis=1)) ** 2).mean(0)[1:]
    return np.log(p + 1e-9)


def summary_features(x: np.ndarray, static: pd.DataFrame) -> np.ndarray:
    """滞在ごとの要約特徴量: 変数ごとの平均・標準偏差・最小・最大・傾き・ラグ 1 自己相関と年齢。"""
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    slope = (x * t[None, :, None]).sum(1) / (t**2).sum()
    z = x - x.mean(1, keepdims=True)
    ac1 = (z[:, 1:] * z[:, :-1]).sum(1) / ((z**2).sum(1) + 1e-9)
    feats = [x.mean(1), x.std(1), x.min(1), x.max(1), slope, ac1, static["age"].to_numpy(float)[:, None]]
    return np.concatenate(feats, axis=1)


def fidelity(real: pd.DataFrame, gen: pd.DataFrame, seed: int = 0) -> dict[str, float]:
    xr, xg = to_array(real), to_array(gen)
    mu, sd = _scale(xr)
    zr, zg = (xr - mu) / sd, (xg - mu) / sd
    marg = np.mean([wasserstein_distance(zr[..., j].ravel(), zg[..., j].ravel()) for j in range(len(VARS))])
    corr = np.abs(np.corrcoef(zr.reshape(-1, len(VARS)).T) - np.corrcoef(zg.reshape(-1, len(VARS)).T))
    acf = np.mean([np.abs(_acf(zr, lag) - _acf(zg, lag)).mean() for lag in ACF_LAGS])
    psd = np.abs(_log_psd(zr) - _log_psd(zg)).mean()
    stay_mean = np.mean([wasserstein_distance(zr[..., j].mean(1), zg[..., j].mean(1)) for j in range(len(VARS))])
    stay_std = np.mean([wasserstein_distance(zr[..., j].std(1), zg[..., j].std(1)) for j in range(len(VARS))])
    return {
        "Marg": float(marg),
        "Corr": float(corr[np.triu_indices(len(VARS), 1)].mean()),
        "ACF": float(acf),
        "PSD": float(psd),
        "StayMean": float(stay_mean),
        "StayStd": float(stay_std),
        "Disc": discriminative(real, gen, seed),
        "Mortality": float(abs(static_of(real)["died"].mean() - static_of(gen)["died"].mean())),
    }


def discriminative(real: pd.DataFrame, gen: pd.DataFrame, seed: int = 0) -> float:
    """判別スコア: 要約特徴量で実データと合成データを見分ける分類器の正解率から 0.5 を引いた値（0 が理想）。

    正解率が 0.5 を下回る場合は 0 とする。合成データが学習データのコピーだと、同じ系列が両方のラベルで入り、
    交差検証の正解率が 0.5 を大きく下回るため（見分けられないことに変わりはない）。
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    fr = summary_features(to_array(real), static_of(real))
    fg = summary_features(to_array(gen), static_of(gen))
    n = min(len(fr), len(fg))
    rng = np.random.default_rng(seed)
    x = np.vstack([fr[rng.choice(len(fr), n, replace=False)], fg[rng.choice(len(fg), n, replace=False)]])
    y = np.r_[np.ones(n), np.zeros(n)]
    acc = cross_val_score(HistGradientBoostingClassifier(random_state=seed), x, y, cv=5, scoring="accuracy").mean()
    return float(max(acc - 0.5, 0.0))


def utility(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s_tr, s_te = static_of(train_df), static_of(test_df)
    if s_tr["died"].nunique() < 2:
        return float("nan")
    x_tr = summary_features(to_array(train_df), s_tr)
    x_te = summary_features(to_array(test_df), s_te)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed))
    model.fit(x_tr, s_tr["died"])
    return float(roc_auc_score(s_te["died"], model.predict_proba(x_te)[:, 1]))


def privacy(train_df: pd.DataFrame, holdout_df: pd.DataFrame, gen: pd.DataFrame) -> dict[str, float]:
    """系列そのもの（標準化して平坦化）の距離で測る。

    - DCRRatio: 合成→学習の最近傍距離の中央値 / 評価用 B→学習の最近傍距離の中央値
    - MIA: 学習（メンバー）と B（非メンバー）を、最も近い合成系列までの距離で見分けたときの AUC
    - NearCopy: 合成系列のうち、学習データとの最近傍距離が「B→学習」の距離の 1 パーセンタイルより小さいものの割合
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors

    xa, xb, xg = to_array(train_df), to_array(holdout_df), to_array(gen)
    mu, sd = _scale(xa)
    za, zb, zg = (((x - mu) / sd).reshape(len(x), -1) for x in (xa, xb, xg))
    nn_a = NearestNeighbors(n_neighbors=1).fit(za)
    d_g = nn_a.kneighbors(zg)[0][:, 0]
    d_b = nn_a.kneighbors(zb)[0][:, 0]
    rng = np.random.default_rng(0)
    members = za[rng.choice(len(za), size=min(len(za), len(zb)), replace=False)]
    nn_g = NearestNeighbors(n_neighbors=1).fit(zg)
    d_mem, d_non = nn_g.kneighbors(members)[0][:, 0], nn_g.kneighbors(zb)[0][:, 0]
    y = np.r_[np.ones(len(d_mem)), np.zeros(len(d_non))]
    return {
        "DCRRatio": float(np.median(d_g) / np.median(d_b)),
        "MIA": float(roc_auc_score(y, -np.r_[d_mem, d_non])),
        "NearCopy": float((d_g < np.quantile(d_b, 0.01)).mean()),
    }


def _shuffle_time(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """滞在ごとに時刻の順をランダムに入れ替える（値の分布と滞在ごとの平均・ばらつきは保ち、時間構造を崩す）。"""
    d = df.sort_values(["seq_id", "t"]).copy()
    perm = np.concatenate([rng.permutation(STEPS) + i * STEPS for i in range(d["seq_id"].nunique())])
    d[VARS] = d[VARS].to_numpy()[perm]
    return d


def _shuffle_stay(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """同じ時刻番号の値を滞在間で入れ替える（時点ごとの分布は保ち、滞在としてのまとまりを崩す）。"""
    d = df.sort_values(["t", "seq_id"]).copy()
    for col in VARS:
        d[col] = d.groupby("t")[col].transform(lambda s: rng.permutation(s.to_numpy()))
    return d.sort_values(["seq_id", "t"])


def evaluate_all(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b, c = (read_split(k) for k in "ABC")
    rng = np.random.default_rng(seed)

    def row(gen: pd.DataFrame) -> dict[str, float]:
        return {**fidelity(a, gen, seed), "TSTR_AUROC": utility(gen, b, seed), **privacy(a, b, gen)}

    rows = {
        "Real-C（実データ・未学習）": row(c),
        "Real-C 時刻シャッフル": row(_shuffle_time(c, rng)),
        "Real-C 滞在シャッフル": row(_shuffle_stay(c, rng)),
        "Copy-A（学習データのコピー）": row(a),
    }
    per_seed = []
    for path in sorted(OUT.glob("gen_*_s*.csv.gz")):
        name, s = path.name.removeprefix("gen_").removesuffix(".csv.gz").rsplit("_s", 1)
        per_seed.append({"model": name, "seed": int(s), **row(pd.read_csv(path))})
    by_seed = pd.DataFrame(per_seed)
    if len(by_seed):
        grouped = by_seed.drop(columns="seed").groupby("model", sort=False)
        n = grouped.size()
        for name, vals in grouped.mean().iterrows():
            rows[f"{name} (n={n[name]})"] = vals.to_dict()
    return pd.DataFrame(rows).T, by_seed
