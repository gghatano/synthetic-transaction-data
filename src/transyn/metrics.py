"""Banksformer 論文 Table 2 の評価指標の再実装。

論文の定義（6.1〜6.3 節）:
- Amt: 取引金額の分布の Wasserstein-1 距離
- CF: 月次キャッシュフロー（口座・月ごとの入出金の合計）の分布の Wasserstein-1 距離
- Tcode: tcode の分布の JSD
- DoM: 取引日（月内の日）の分布の JSD
- Tcode 3G: 系列内の tcode 3-gram の分布の JSD
- (Tcode, DoM): tcode と月内日の同時分布の JSD

JSD は著者コードに合わせて自然対数で計算する（scipy.special.rel_entr）。
追加で、取引間隔（日数）の Wasserstein-1 距離も出す（著者コードの nb2 で td-wasser として集計されているもの）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

from .data import signed_amount


def jsd(p_counts: pd.Series, q_counts: pd.Series) -> float:
    support = p_counts.index.union(q_counts.index)
    p = p_counts.reindex(support, fill_value=0).to_numpy(dtype=float)
    q = q_counts.reindex(support, fill_value=0).to_numpy(dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return float((rel_entr(p, m).sum() + rel_entr(q, m).sum()) / 2)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["seq_id", "datetime"], kind="stable")


def monthly_cash_flow(df: pd.DataFrame) -> pd.Series:
    d = df.assign(signed=signed_amount(df), ym=df["datetime"].dt.to_period("M"))
    return d.groupby(["seq_id", "ym"])["signed"].sum()


def tcode_ngrams(df: pd.DataFrame, n: int) -> pd.Series:
    d = _ordered(df)
    codes = d["tcode"].to_numpy()
    seq = d["seq_id"].to_numpy()
    grams = []
    for start in range(len(codes) - n + 1):
        if seq[start] == seq[start + n - 1]:
            grams.append("|".join(codes[start : start + n]))
    return pd.Series(grams).value_counts()


def inter_arrival_days(df: pd.DataFrame) -> pd.Series:
    d = _ordered(df)
    return d.groupby("seq_id")["datetime"].diff().dt.days.dropna()


def evaluate(real: pd.DataFrame, gen: pd.DataFrame) -> dict[str, float]:
    day_r, day_g = real["datetime"].dt.day, gen["datetime"].dt.day
    return {
        "Amt": wasserstein_distance(real["amount"], gen["amount"]),
        "CF": wasserstein_distance(monthly_cash_flow(real), monthly_cash_flow(gen)),
        "Tcode": jsd(real["tcode"].value_counts(), gen["tcode"].value_counts()),
        "DoM": jsd(day_r.value_counts(), day_g.value_counts()),
        "Tcode3G": jsd(tcode_ngrams(real, 3), tcode_ngrams(gen, 3)),
        "TcodeDoM": jsd(
            (real["tcode"] + "@" + day_r.astype(str)).value_counts(),
            (gen["tcode"] + "@" + day_g.astype(str)).value_counts(),
        ),
        "TD": wasserstein_distance(inter_arrival_days(real), inter_arrival_days(gen)),
    }


# 論文 Table 2（Czech）の値。比較用
PAPER_TABLE2_CZECH = {
    "BF": {"Amt": 2102, "CF": 2738, "Tcode": 0.004, "DoM": 0.011, "Tcode3G": 0.042, "TcodeDoM": 0.251},
    "DG": {"Amt": 1939, "CF": 57800, "Tcode": 0.007, "DoM": 0.090, "Tcode3G": 0.132, "TcodeDoM": 0.660},
    "TG": {"Amt": 1931, "CF": 4980, "Tcode": 0.075, "DoM": 0.059, "Tcode3G": 0.337, "TcodeDoM": 0.638},
}


def summary_stats(df: pd.DataFrame) -> dict[str, float]:
    """生成データの素性を確認するための基本統計。"""
    lengths = df.groupby("seq_id").size()
    return {
        "n_rows": len(df),
        "n_seqs": int(lengths.size),
        "mean_seq_len": float(lengths.mean()),
        "n_tcodes": int(df["tcode"].nunique()),
        "same_day_ratio": float((inter_arrival_days(df) == 0).mean()),
        "amount_mean": float(np.mean(df["amount"])),
    }
