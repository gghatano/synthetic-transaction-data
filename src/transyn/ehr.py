"""eICU Demo の EHR イベント系列での検証（issue #10）。

ICU 滞在ごとに、入室から WINDOW_HOURS 時間以内の診断・治療・投薬を時刻順に並べたイベント系列を作る。
トランザクションとの対応は 滞在＝口座、イベント＝取引、コード＝tcode（docs/survey-ehr-synthesis.md）。

評価は 3 軸:
- 忠実度: コード分布、コード 3-gram、(コード, 経過時間帯)、系列長、イベント間隔
- 有用性: 合成データで死亡退院を予測するモデルを学習し、実データ B で評価する（TSTR）
- プライバシ: 最近傍距離（DCR）と、距離に基づくメンバーシップ推論の AUC

データ分割は患者（uniquepid）単位で A（学習 50%）/ B（評価・非メンバー 25%）/ C（実データの基準 25%）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .metrics import jsd
from .models import SeqSpec

DATA_DIR = Path("data") / "eicu_demo"
OUT = Path("outputs") / "ehr"
SPLITS = {"A": DATA_DIR / "seq_A.csv.gz", "B": DATA_DIR / "seq_B.csv.gz", "C": DATA_DIR / "seq_C.csv.gz"}

WINDOW_HOURS = 48
MIN_EVENTS = 5
TOP_CODES = 80  # 頻度上位のコードだけを残し、残りは種類ごとの "other" にまとめる
ORIGIN = pd.Timestamp("2000-01-01")  # 入室時刻（オフセット 0）を置く基準日。実在の日付ではない

SPEC = SeqSpec(
    context=("age", "died"),
    discrete_context=("died",),
    value=None,
    seq_len=20,  # DGAN の固定系列長（20 件以上の滞在の先頭 20 件）
    empirical_lengths=True,
    time_unit="h",
    time_resolution="1min",
    origin=ORIGIN,
)


def _find(name: str) -> Path:
    hits = sorted(DATA_DIR.rglob(f"{name}.csv*"))
    if not hits:
        raise FileNotFoundError(f"{name}.csv(.gz) が {DATA_DIR} にない。eICU Demo を展開すること")
    return hits[0]


def _prefix(strings: pd.Series, levels: int) -> pd.Series:
    return strings.astype(str).str.split("|").str[:levels].str.join("|")


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, stays) を返す。events: stay, offset（分）, code。stays: stay, uniquepid, age, died。"""
    patient = pd.read_csv(_find("patient"))
    stays = pd.DataFrame(
        {
            "stay": patient["patientunitstayid"],
            "uniquepid": patient["uniquepid"],
            # 90 歳以上は "> 89" と匿名化されている
            "age": pd.to_numeric(patient["age"].replace("> 89", "90"), errors="coerce"),
            "died": (patient["hospitaldischargestatus"] == "Expired").astype(int),
        }
    ).dropna(subset=["age"])

    dx = pd.read_csv(_find("diagnosis"), usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
    tx = pd.read_csv(_find("treatment"), usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"])
    rx = pd.read_csv(
        _find("medication"),
        usecols=["patientunitstayid", "drugstartoffset", "drughiclseqno", "drugordercancelled"],
    )
    # 薬剤名は 4 割が欠損で表記ゆれも大きい。成分を表す HICL 番号（欠損 6%）をコードにする
    rx = rx[(rx["drugordercancelled"] != "Yes") & rx["drughiclseqno"].notna()]
    events = pd.concat(
        [
            pd.DataFrame(
                {"stay": dx["patientunitstayid"], "offset": dx["diagnosisoffset"], "code": "dx:" + _prefix(dx["diagnosisstring"], 2)}
            ),
            pd.DataFrame(
                {"stay": tx["patientunitstayid"], "offset": tx["treatmentoffset"], "code": "tx:" + _prefix(tx["treatmentstring"], 2)}
            ),
            pd.DataFrame(
                {
                    "stay": rx["patientunitstayid"],
                    "offset": rx["drugstartoffset"],
                    "code": "rx:hicl" + rx["drughiclseqno"].astype(int).astype(str),
                }
            ),
        ],
        ignore_index=True,
    )
    events = events[(events["offset"] >= 0) & (events["offset"] <= WINDOW_HOURS * 60)]
    events = events[events["stay"].isin(stays["stay"])].drop_duplicates()

    top = events["code"].value_counts().index[:TOP_CODES]
    events["code"] = events["code"].where(events["code"].isin(top), events["code"].str[:3] + "other")
    counts = events.groupby("stay").size()
    keep = counts[counts >= MIN_EVENTS].index
    events = events[events["stay"].isin(keep)]
    stays = stays[stays["stay"].isin(keep)]
    return events, stays


def to_sequences(events: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """transyn のモデルが受け取る系列形式（seq_id, datetime, tcode, age, died）にする。"""
    df = events.merge(stays[["stay", "age", "died"]], on="stay")
    df = df.sort_values(["stay", "offset", "code"], kind="stable")
    return pd.DataFrame(
        {
            "seq_id": df["stay"].to_numpy(),
            "datetime": ORIGIN + pd.to_timedelta(df["offset"].to_numpy(), unit="min"),
            "tcode": df["code"].to_numpy(),
            "age": df["age"].astype(int).to_numpy(),
            "died": df["died"].to_numpy(),
        }
    ).reset_index(drop=True)


def split(seqs: pd.DataFrame, stays: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """患者単位で A 50% / B 25% / C 25% に分ける（同じ患者の別の滞在が別の分割に入らないように）。"""
    rng = np.random.default_rng(seed)
    pids = stays["uniquepid"].unique().to_numpy(dtype=object)
    rng.shuffle(pids)
    n = len(pids)
    groups = {"A": pids[: n // 2], "B": pids[n // 2 : 3 * n // 4], "C": pids[3 * n // 4 :]}
    out = {}
    for name, members in groups.items():
        stay_ids = stays.loc[stays["uniquepid"].isin(members), "stay"]
        out[name] = seqs[seqs["seq_id"].isin(stay_ids)].reset_index(drop=True)
    return out


def prepare(seed: int = 0) -> dict[str, pd.DataFrame]:
    events, stays = load_events()
    parts = split(to_sequences(events, stays), stays, seed=seed)
    for name, df in parts.items():
        df.to_csv(SPLITS[name], index=False)
    return parts


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS[name], parse_dates=["datetime"])


# ---------------------------------------------------------------- 忠実度


def _hours(df: pd.DataFrame) -> pd.Series:
    return (df["datetime"] - ORIGIN) / pd.Timedelta(1, "h")


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["seq_id", "datetime"], kind="stable")


def code_ngrams(df: pd.DataFrame, n: int) -> pd.Series:
    d = _ordered(df)
    codes, seq = d["tcode"].to_numpy(), d["seq_id"].to_numpy()
    grams = ["|".join(codes[i : i + n]) for i in range(len(codes) - n + 1) if seq[i] == seq[i + n - 1]]
    return pd.Series(grams).value_counts()


def fidelity(real: pd.DataFrame, gen: pd.DataFrame) -> dict[str, float]:
    def gaps(df):
        return (_ordered(df).groupby("seq_id")["datetime"].diff().dropna()) / pd.Timedelta(1, "h")

    def code_time(df):
        bucket = (_hours(df) // 6).clip(upper=WINDOW_HOURS // 6 - 1).astype(int).astype(str)
        return (df["tcode"] + "@" + bucket).value_counts()

    return {
        "Code": jsd(real["tcode"].value_counts(), gen["tcode"].value_counts()),
        "Code3G": jsd(code_ngrams(real, 3), code_ngrams(gen, 3)),
        "CodeTime": jsd(code_time(real), code_time(gen)),
        "Len": wasserstein_distance(real.groupby("seq_id").size(), gen.groupby("seq_id").size()),
        "Gap": wasserstein_distance(gaps(real), gaps(gen)),
        "Mortality": abs(real.groupby("seq_id")["died"].first().mean() - gen.groupby("seq_id")["died"].first().mean()),
    }


# ---------------------------------------------------------------- 有用性・プライバシ用の特徴量


def stay_features(df: pd.DataFrame, vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """滞在ごとの特徴量（コードの出現回数、イベント数、年齢）とラベル（死亡退院）。"""
    counts = pd.crosstab(df["seq_id"], df["tcode"]).reindex(columns=vocab, fill_value=0)
    static = df.groupby("seq_id").agg(age=("age", "first"), died=("died", "first"), n=("tcode", "size"))
    static = static.loc[counts.index]
    x = np.column_stack([counts.to_numpy(dtype=float), static["n"].to_numpy(float), static["age"].to_numpy(float)])
    return x, static["died"].astype(int).to_numpy()


def utility(train_df: pd.DataFrame, test_df: pd.DataFrame, vocab: list[str], seed: int = 0) -> float:
    """train_df で死亡退院の予測モデルを学習し、test_df（実データ）の AUROC を返す。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_tr, y_tr = stay_features(train_df, vocab)
    x_te, y_te = stay_features(test_df, vocab)
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, random_state=seed))
    model.fit(np.log1p(x_tr), y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(np.log1p(x_te))[:, 1]))


def privacy(train_df: pd.DataFrame, holdout_df: pd.DataFrame, gen: pd.DataFrame, vocab: list[str]) -> dict[str, float]:
    """最近傍距離とメンバーシップ推論。

    - DCR: 合成の各滞在から最も近い学習（A）の滞在までの距離の中央値。HoldoutDCR（B → A）と比べる
    - DCRRatio: DCR / HoldoutDCR。1 を大きく下回ると、合成データが学習データに未知の実データより近い（コピーの疑い）
    - MIA: 学習（A, メンバー）と B（非メンバー）の各滞在について、最も近い合成の滞在までの距離が近いほど
      メンバーと判定したときの AUC。0.5 が理想（見分けられない）
    - ExactCopy: コードの並び（時刻は無視）が学習データのいずれかの滞在と完全に一致する合成の滞在の割合。
      学習データの系列をそのまま吐き出していないかを直接見る
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    x_a, _ = stay_features(train_df, vocab)
    x_b, _ = stay_features(holdout_df, vocab)
    x_g, _ = stay_features(gen, vocab)
    scaler = StandardScaler().fit(np.log1p(x_a))
    za, zb, zg = (scaler.transform(np.log1p(x)) for x in (x_a, x_b, x_g))

    nn_a = NearestNeighbors(n_neighbors=1).fit(za)
    dcr = float(np.median(nn_a.kneighbors(zg)[0]))
    holdout_dcr = float(np.median(nn_a.kneighbors(zb)[0]))

    rng = np.random.default_rng(0)
    members = za[rng.choice(len(za), size=min(len(za), len(zb)), replace=False)]
    nn_g = NearestNeighbors(n_neighbors=1).fit(zg)
    d_mem = nn_g.kneighbors(members)[0][:, 0]
    d_non = nn_g.kneighbors(zb)[0][:, 0]
    y = np.r_[np.ones(len(d_mem)), np.zeros(len(d_non))]
    mia = float(roc_auc_score(y, -np.r_[d_mem, d_non]))

    def code_strings(df: pd.DataFrame) -> pd.Series:
        return _ordered(df).groupby("seq_id")["tcode"].agg("|".join)

    exact = float(code_strings(gen).isin(set(code_strings(train_df))).mean())
    return {"DCR": dcr, "HoldoutDCR": holdout_dcr, "DCRRatio": dcr / holdout_dcr, "MIA": mia, "ExactCopy": exact}


def evaluate_all(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """outputs/ehr/gen_<model>_s<seed>.csv.gz をすべて評価する。基準として実データ C と、A のコピーを並べる。"""
    a, b, c = (read_split(k) for k in "ABC")
    vocab = sorted(a["tcode"].unique())

    def row(gen: pd.DataFrame) -> dict[str, float]:
        return {**fidelity(a, gen), "TSTR_AUROC": utility(gen, b, vocab, seed), **privacy(a, b, gen, vocab)}

    rng = np.random.default_rng(seed)
    rows = {
        "Real-C（実データ・未学習）": row(c),
        # 系列としての情報だけを壊した基準: C のコード／時刻を全行でランダムに入れ替える
        "Real-C コードシャッフル": row(c.assign(tcode=rng.permutation(c["tcode"].to_numpy()))),
        "Real-C 時刻シャッフル": row(c.assign(datetime=rng.permutation(c["datetime"].to_numpy()))),
        "Copy-A（学習データのコピー）": row(a),
    }
    per_seed = []
    for path in sorted(OUT.glob("gen_*_s*.csv.gz")):
        name, s = path.name.removeprefix("gen_").removesuffix(".csv.gz").rsplit("_s", 1)
        gen = pd.read_csv(path, parse_dates=["datetime"])
        per_seed.append({"model": name, "seed": int(s), **row(gen)})
    by_seed = pd.DataFrame(per_seed)
    if len(by_seed):
        grouped = by_seed.drop(columns="seed").groupby("model", sort=False)
        n = grouped.size()
        for name, vals in grouped.mean().iterrows():
            rows[f"{name} (n={n[name]})"] = vals.to_dict()
    return pd.DataFrame(rows).T, by_seed
