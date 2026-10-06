"""比較する生成手法。どれも「学習用の系列データ → 生成した系列データ」を返す。

系列データは seq_id, datetime, tcode と、SeqSpec で指定する数値列・系列単位の属性列からなる。
既定の CZECH は Czech bank の追試（#2）の形（seq_id, datetime, tcode, amount, age）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import torch

from .data import MAX_SEQ_LEN

EPOCH_ORIGIN = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class SeqSpec:
    """系列データの列構成と、モデル間で共通の扱い。

    - context: 系列単位の属性列（系列内で一定）。PAR のコンテキスト、DGAN の属性になる
    - discrete_context: context のうち離散値の列
    - value: 各イベントの数値列（なければ None）。DGAN には log1p で渡す
    - seq_len: CTGAN で 1 系列に束ねる行数、DGAN の固定系列長
    - empirical_lengths: True なら CTGAN の系列長を学習データの系列長から復元抽出する（False なら seq_len 固定）
    - time_unit / time_resolution: DGAN の時間間隔の単位と、生成した時刻を丸める刻み
    - origin: DGAN の開始時刻の基準
    """

    context: tuple[str, ...] = ("age",)
    discrete_context: tuple[str, ...] = ()
    value: str | None = "amount"
    seq_len: int = MAX_SEQ_LEN
    empirical_lengths: bool = False
    time_unit: str = "D"
    time_resolution: str = "1D"
    origin: pd.Timestamp = EPOCH_ORIGIN

    @property
    def event_columns(self) -> list[str]:
        return ["datetime", "tcode", *([self.value] if self.value else [])]

    @property
    def columns(self) -> list[str]:
        return ["seq_id", *self.event_columns, *self.context]


CZECH = SeqSpec()


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ns_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """SDV（RDT）は datetime をナノ秒単位と仮定して整数化する。pandas 3 の既定（マイクロ秒）のまま渡すと
    生成結果が 1970 年付近に化けるので、ナノ秒単位に揃えてから渡す。"""
    return df.assign(datetime=df["datetime"].astype("datetime64[ns]"))


def fit_generate_ctgan(
    train: pd.DataFrame,
    n_seqs: int,
    epochs: int,
    seed: int,
    log_frequency: bool = False,
    spec: SeqSpec = CZECH,
) -> pd.DataFrame:
    """CTGAN を「1行 = 1取引」でそのまま適用する素朴な使い方。

    CTGAN には系列や口座の概念がないので、生成した行をランダムに束ねて 1 系列とみなし、系列内を日付順に
    並べる。行間の依存は学習も生成もされない。束ねる件数は spec.seq_len 固定、または学習データの系列長から
    復元抽出する。系列単位の属性は、束ねた系列の先頭行の値に揃える。

    log_frequency は SDV の既定では True。ctgan の DataSampler.sample_original_condvec は
    「元の頻度で条件ベクトルを引く」としながら、実際には log 頻度で正規化した確率を使うため、
    True のままだと生成時のカテゴリ分布が一様分布側に平らになる（ctgan 0.12.1 で確認）。
    """
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    _seed(seed)
    rows = _ns_datetime(train[[*spec.event_columns, *spec.context]])
    metadata = Metadata.detect_from_dataframe(rows)
    metadata.update_column("tcode", sdtype="categorical")
    metadata.update_column("datetime", sdtype="datetime")
    for col in spec.discrete_context:
        metadata.update_column(col, sdtype="categorical")
    model = CTGANSynthesizer(
        metadata, epochs=epochs, verbose=True, enforce_rounding=False, log_frequency=log_frequency
    )
    model.fit(rows)
    if spec.empirical_lengths:
        lengths = np.random.choice(train.groupby("seq_id").size().to_numpy(), size=n_seqs)
        gen = model.sample(num_rows=int(lengths.sum()))
        gen["seq_id"] = np.random.permutation(np.repeat(np.arange(n_seqs), lengths))
    else:
        gen = model.sample(num_rows=n_seqs * spec.seq_len)
        gen["seq_id"] = np.random.permutation(len(gen)) // spec.seq_len
    for col in spec.context:
        gen[col] = gen.groupby("seq_id")[col].transform("first")
    return gen.sort_values(["seq_id", "datetime"], kind="stable").reset_index(drop=True)


def fit_generate_par(
    train: pd.DataFrame, n_seqs: int, epochs: int, seed: int, spec: SeqSpec = CZECH
) -> pd.DataFrame:
    """SDV の PAR（確率的自己回帰モデル）。系列キーと順序を与え、spec.context を系列単位のコンテキストにする。"""
    from sdv.metadata import Metadata
    from sdv.sequential import PARSynthesizer

    _seed(seed)
    seqs = _ns_datetime(train[spec.columns])
    metadata = Metadata.detect_from_dataframe(seqs)
    metadata.update_column("seq_id", sdtype="id")
    metadata.update_column("tcode", sdtype="categorical")
    metadata.update_column("datetime", sdtype="datetime")
    for col in spec.discrete_context:
        metadata.update_column(col, sdtype="categorical")
    metadata.set_sequence_key("seq_id")
    metadata.set_sequence_index("datetime")
    model = PARSynthesizer(
        metadata, context_columns=list(spec.context), epochs=epochs, verbose=True, enforce_rounding=False
    )
    model.fit(seqs)
    gen = model.sample(num_sequences=n_seqs)
    gen["seq_id"] = pd.factorize(gen["seq_id"])[0]
    return gen.sort_values(["seq_id", "datetime"], kind="stable").reset_index(drop=True)


def _to_dgan_frame(train: pd.DataFrame, log_td: bool = True, spec: SeqSpec = CZECH) -> pd.DataFrame:
    """DGAN 用の long 形式。時刻は「系列の開始時刻（属性）＋イベント間隔（特徴量）」で表す（論文 2 節と同じ）。

    取引間隔は 0 日が 2 割強を占め、最大 90 日と裾が長い。そのまま学習させると gretel 版 DGAN は
    全取引を同じ日に置く系列ばかり生成した（ほぼ 0 に潰れる）ため、log1p で圧縮してから渡す。

    gretel 版は固定長しか扱えないので、長さ spec.seq_len 以上の系列の先頭 spec.seq_len 件を使う
    （Czech は最大長が 80 なので「ちょうど 80 件の系列」と同じ）。
    """
    unit = pd.Timedelta(1, unit=spec.time_unit)
    ordered = train.sort_values(["seq_id", "datetime"], kind="stable")
    pos = ordered.groupby("seq_id").cumcount()
    full = ordered[(ordered.groupby("seq_id")["tcode"].transform("size") >= spec.seq_len) & (pos < spec.seq_len)]
    td = (full.groupby("seq_id")["datetime"].diff() / unit).fillna(0.0)
    if spec.time_unit == "D":
        td = np.floor(td)  # Czech の日付データでは .dt.days と同じ
    start = full.groupby("seq_id")["datetime"].transform("min")
    frame = {
        "seq_id": full["seq_id"].to_numpy(),
        "t": full.groupby("seq_id").cumcount().to_numpy(),
        **{col: full[col].astype(float).to_numpy() for col in spec.context},
        "start_day": ((start - spec.origin) / unit).astype(float).to_numpy(),
        "tcode": full["tcode"].to_numpy(),
    }
    if spec.value:
        frame["log_amount"] = np.log1p(full[spec.value]).to_numpy()
    frame["log_td"] = (np.log1p(td) if log_td else td).to_numpy()
    return pd.DataFrame(frame)


def fit_generate_dgan(
    train: pd.DataFrame,
    n_seqs: int,
    epochs: int,
    seed: int,
    log_td: bool = True,
    start: str = "attribute",
    sample_len: int = 10,
    spec: SeqSpec = CZECH,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """DoppelGANger（gretel-synthetics の PyTorch 実装）。

    論文の DG は元の TensorFlow 実装で、可変長の系列を生成している。gretel 版は固定長しか扱えないので、
    長さ spec.seq_len の系列だけで学習・生成する（論文との差分）。

    - log_td: 取引間隔を log1p で圧縮して渡すか（False だと取引間隔が 0 に潰れやすい。#2 付録 A）
    - start: 系列の開始日の与え方。"attribute" は DGAN に属性として生成させる。"empirical" は属性から外し、
      生成後に学習データの開始日から復元抽出する（論文の BF・TG と同じ扱い）
    - sample_len: 1 ステップの RNN 出力で生成する取引数（DoppelGANger の batch generation）
    - batch_size: 学習系列数より大きいと 1 エポック 1 回の更新になる。既定の 1000 は #2〜#10 の設定で、
      更新回数が足りていなかった（#17）
    """
    from gretel_synthetics.timeseries_dgan.config import DfStyle, DGANConfig
    from gretel_synthetics.timeseries_dgan.dgan import DGAN

    _seed(seed)
    frame = _to_dgan_frame(train, log_td=log_td, spec=spec)
    attributes = [*spec.context, *(["start_day"] if start == "attribute" else [])]
    features = ["tcode", *(["log_amount"] if spec.value else []), "log_td"]
    config = DGANConfig(
        max_sequence_len=spec.seq_len,
        sample_len=sample_len,
        batch_size=min(batch_size, frame["seq_id"].nunique()),
        epochs=epochs,
        cuda=False,
    )
    model = DGAN(config)
    started = time.time()

    def progress(info):
        if info.batch == info.total_batches - 1 and info.epoch % 10 == 0:
            print(f"epoch {info.epoch}/{info.total_epochs} ({time.time() - started:.0f}s)", flush=True)

    model.train_dataframe(
        frame.drop(columns=[c for c in ["start_day"] if c not in attributes]),
        attribute_columns=attributes,
        feature_columns=features,
        example_id_column="seq_id",
        time_column="t",
        discrete_columns=["tcode", *spec.discrete_context],
        df_style=DfStyle.LONG,
        progress_callback=progress,
    )
    gen = model.generate_dataframe(n_seqs)
    if start == "empirical":
        starts = frame.groupby("seq_id")["start_day"].first().to_numpy()
        sampled = np.random.choice(starts, size=gen["seq_id"].nunique())
        gen["start_day"] = gen["seq_id"].map(dict(zip(gen["seq_id"].unique(), sampled)))
    if not log_td:
        gen["log_td"] = np.log1p(gen["log_td"].astype(float).clip(lower=0))
        frame = frame.assign(log_td=np.log1p(frame["log_td"]))
    return _from_dgan_frame(gen, bounds=frame, spec=spec)


def _from_dgan_frame(gen: pd.DataFrame, bounds: pd.DataFrame | None = None, spec: SeqSpec = CZECH) -> pd.DataFrame:
    """_to_dgan_frame の逆変換。開始日に取引間隔の累積和を足して日付に戻す。

    bounds（学習データ）を渡すと、数値列を学習データの範囲にクリップする。SDV の enforce_min_max_values と
    同じ扱いで、log 空間の外れ値が expm1 で桁違いの金額になるのを防ぐ。
    """
    gen = gen.sort_values(["seq_id", "t"], kind="stable").reset_index(drop=True)
    # generate_dataframe は数値列も object 型で返すので明示的に変換する
    numeric = ["log_td", "start_day", *(["log_amount"] if spec.value else []), *spec.context]
    for col in numeric:
        if col in spec.discrete_context:
            continue
        gen[col] = gen[col].astype(float)
        if bounds is not None:
            gen[col] = gen[col].clip(bounds[col].min(), bounds[col].max())
    unit = pd.Timedelta(1, unit=spec.time_unit)
    step = pd.Timedelta(spec.time_resolution) / unit  # 丸めの刻み（time_unit 単位）。Czech は 1 日

    def snap(x: pd.Series) -> pd.Series:
        return (x / step).round() * step

    td = snap(np.expm1(gen["log_td"]).clip(lower=0))
    start = gen["start_day"]
    if spec.origin == EPOCH_ORIGIN:
        start = start.clip(lower=0)  # Czech: 期間の始まり（1993-01-01）より前には置かない
    start = snap(start)
    offset = start + td.groupby(gen["seq_id"]).cumsum() - td.groupby(gen["seq_id"]).transform("first")
    out = {
        "seq_id": pd.factorize(gen["seq_id"])[0],
        "datetime": spec.origin + pd.to_timedelta(offset * unit.total_seconds(), unit="s").dt.round(spec.time_resolution),
        "tcode": gen["tcode"].to_numpy(),
    }
    if spec.value:
        out[spec.value] = np.expm1(gen["log_amount"]).clip(lower=0).to_numpy()
    for col in spec.context:
        out[col] = gen[col].to_numpy() if col in spec.discrete_context else gen[col].round().to_numpy()
    return pd.DataFrame(out).reset_index(drop=True)


MODELS = {
    "ctgan": fit_generate_ctgan,
    "ctgan-logfreq": partial(fit_generate_ctgan, log_frequency=True),
    "par": fit_generate_par,
    "dgan": fit_generate_dgan,
    # #5 の切り分け用
    "dgan-rawtd": partial(fit_generate_dgan, log_td=False),
    "dgan-empstart": partial(fit_generate_dgan, start="empirical"),
    "dgan-sl5": partial(fit_generate_dgan, sample_len=5),
    # #17: バッチ 64 で更新回数を増やしたもの
    "dgan-b64": partial(fit_generate_dgan, batch_size=64),
    "dgan-b64-rawtd": partial(fit_generate_dgan, batch_size=64, log_td=False),
    "dgan-b64-empstart": partial(fit_generate_dgan, batch_size=64, start="empirical"),
}
