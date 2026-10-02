"""比較する生成手法。どれも「学習用の系列データ → 生成した系列データ」を返す。

生成データは data.SEQ_COLUMNS（seq_id, datetime, tcode, amount, age）の形に揃える。
"""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import pandas as pd
import torch

from .data import MAX_SEQ_LEN

EPOCH_ORIGIN = pd.Timestamp("1993-01-01")


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ns_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """SDV（RDT）は datetime をナノ秒単位と仮定して整数化する。pandas 3 の既定（マイクロ秒）のまま渡すと
    生成結果が 1970 年付近に化けるので、ナノ秒単位に揃えてから渡す。"""
    return df.assign(datetime=df["datetime"].astype("datetime64[ns]"))


def fit_generate_ctgan(
    train: pd.DataFrame, n_seqs: int, epochs: int, seed: int, log_frequency: bool = False
) -> pd.DataFrame:
    """CTGAN を「1行 = 1取引」でそのまま適用する素朴な使い方。

    CTGAN には系列や口座の概念がないので、生成した行を MAX_SEQ_LEN 件ずつランダムに束ねて
    1 系列とみなし、系列内を日付順に並べる。行間の依存は学習も生成もされない。

    log_frequency は SDV の既定では True。ctgan の DataSampler.sample_original_condvec は
    「元の頻度で条件ベクトルを引く」としながら、実際には log 頻度で正規化した確率を使うため、
    True のままだと生成時のカテゴリ分布が一様分布側に平らになる（ctgan 0.12.1 で確認）。
    """
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    _seed(seed)
    rows = _ns_datetime(train[["datetime", "tcode", "amount", "age"]])
    metadata = Metadata.detect_from_dataframe(rows)
    metadata.update_column("tcode", sdtype="categorical")
    metadata.update_column("datetime", sdtype="datetime")
    model = CTGANSynthesizer(
        metadata, epochs=epochs, verbose=True, enforce_rounding=False, log_frequency=log_frequency
    )
    model.fit(rows)
    gen = model.sample(num_rows=n_seqs * MAX_SEQ_LEN)
    gen["seq_id"] = np.random.permutation(len(gen)) // MAX_SEQ_LEN
    return gen.sort_values(["seq_id", "datetime"], kind="stable").reset_index(drop=True)


def fit_generate_par(train: pd.DataFrame, n_seqs: int, epochs: int, seed: int) -> pd.DataFrame:
    """SDV の PAR（確率的自己回帰モデル）。系列キーと順序を与え、年齢を系列単位のコンテキストにする。"""
    from sdv.metadata import Metadata
    from sdv.sequential import PARSynthesizer

    _seed(seed)
    seqs = _ns_datetime(train[["seq_id", "datetime", "tcode", "amount", "age"]])
    metadata = Metadata.detect_from_dataframe(seqs)
    metadata.update_column("seq_id", sdtype="id")
    metadata.update_column("tcode", sdtype="categorical")
    metadata.update_column("datetime", sdtype="datetime")
    metadata.set_sequence_key("seq_id")
    metadata.set_sequence_index("datetime")
    model = PARSynthesizer(metadata, context_columns=["age"], epochs=epochs, verbose=True, enforce_rounding=False)
    model.fit(seqs)
    gen = model.sample(num_sequences=n_seqs)
    gen["seq_id"] = pd.factorize(gen["seq_id"])[0]
    return gen.sort_values(["seq_id", "datetime"], kind="stable").reset_index(drop=True)


def _to_dgan_frame(train: pd.DataFrame, log_td: bool = True) -> pd.DataFrame:
    """DGAN 用の long 形式。時刻は「系列の開始日（属性）＋取引間隔（特徴量）」で表す（論文 2 節と同じ）。

    取引間隔は 0 日が 2 割強を占め、最大 90 日と裾が長い。そのまま学習させると gretel 版 DGAN は
    全取引を同じ日に置く系列ばかり生成した（ほぼ 0 に潰れる）ため、log1p で圧縮してから渡す。
    """
    full = train.groupby("seq_id").filter(lambda g: len(g) == MAX_SEQ_LEN)
    full = full.sort_values(["seq_id", "datetime"], kind="stable")
    td = full.groupby("seq_id")["datetime"].diff().dt.days.fillna(0.0)
    start = full.groupby("seq_id")["datetime"].transform("min")
    return pd.DataFrame(
        {
            "seq_id": full["seq_id"].to_numpy(),
            "t": full.groupby("seq_id").cumcount().to_numpy(),
            "age": full["age"].astype(float).to_numpy(),
            "start_day": (start - EPOCH_ORIGIN).dt.days.astype(float).to_numpy(),
            "tcode": full["tcode"].to_numpy(),
            "log_amount": np.log1p(full["amount"]).to_numpy(),
            "log_td": (np.log1p(td) if log_td else td).to_numpy(),
        }
    )


def fit_generate_dgan(
    train: pd.DataFrame,
    n_seqs: int,
    epochs: int,
    seed: int,
    log_td: bool = True,
    start: str = "attribute",
    sample_len: int = 10,
) -> pd.DataFrame:
    """DoppelGANger（gretel-synthetics の PyTorch 実装）。

    論文の DG は元の TensorFlow 実装で、可変長の系列を生成している。gretel 版は固定長しか扱えないので、
    ちょうど MAX_SEQ_LEN 件の系列だけで学習・生成する（論文との差分）。

    - log_td: 取引間隔を log1p で圧縮して渡すか（False だと取引間隔が 0 に潰れやすい。#2 付録 A）
    - start: 系列の開始日の与え方。"attribute" は DGAN に属性として生成させる。"empirical" は属性から外し、
      生成後に学習データの開始日から復元抽出する（論文の BF・TG と同じ扱い）
    - sample_len: 1 ステップの RNN 出力で生成する取引数（DoppelGANger の batch generation）
    """
    from gretel_synthetics.timeseries_dgan.config import DfStyle, DGANConfig
    from gretel_synthetics.timeseries_dgan.dgan import DGAN

    _seed(seed)
    frame = _to_dgan_frame(train, log_td=log_td)
    attributes = ["age", "start_day"] if start == "attribute" else ["age"]
    config = DGANConfig(
        max_sequence_len=MAX_SEQ_LEN,
        sample_len=sample_len,
        batch_size=min(1000, frame["seq_id"].nunique()),
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
        feature_columns=["tcode", "log_amount", "log_td"],
        example_id_column="seq_id",
        time_column="t",
        discrete_columns=["tcode"],
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
    return _from_dgan_frame(gen, bounds=frame)


def _from_dgan_frame(gen: pd.DataFrame, bounds: pd.DataFrame | None = None) -> pd.DataFrame:
    """_to_dgan_frame の逆変換。開始日に取引間隔の累積和を足して日付に戻す。

    bounds（学習データ）を渡すと、数値列を学習データの範囲にクリップする。SDV の enforce_min_max_values と
    同じ扱いで、log 空間の外れ値が expm1 で桁違いの金額になるのを防ぐ。
    """
    gen = gen.sort_values(["seq_id", "t"], kind="stable").reset_index(drop=True)
    # generate_dataframe は数値列も object 型で返すので明示的に変換する
    numeric = ["log_td", "start_day", "log_amount", "age"]
    for col in numeric:
        gen[col] = gen[col].astype(float)
        if bounds is not None:
            gen[col] = gen[col].clip(bounds[col].min(), bounds[col].max())
    td = np.expm1(gen["log_td"]).clip(lower=0).round()
    start = gen["start_day"].clip(lower=0).round()
    offset = start + td.groupby(gen["seq_id"]).cumsum() - td.groupby(gen["seq_id"]).transform("first")
    return pd.DataFrame(
        {
            "seq_id": pd.factorize(gen["seq_id"])[0],
            "datetime": EPOCH_ORIGIN + pd.to_timedelta(offset, unit="D"),
            "tcode": gen["tcode"].to_numpy(),
            "amount": np.expm1(gen["log_amount"]).clip(lower=0).to_numpy(),
            "age": gen["age"].round().to_numpy(),
        }
    ).reset_index(drop=True)


MODELS = {
    "ctgan": fit_generate_ctgan,
    "ctgan-logfreq": partial(fit_generate_ctgan, log_frequency=True),
    "par": fit_generate_par,
    "dgan": fit_generate_dgan,
    # #5 の切り分け用
    "dgan-rawtd": partial(fit_generate_dgan, log_td=False),
    "dgan-empstart": partial(fit_generate_dgan, start="empirical"),
    "dgan-sl5": partial(fit_generate_dgan, sample_len=5),
}
