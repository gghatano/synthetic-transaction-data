"""Czech bank データの取得と、Banksformer 論文に沿った前処理。

論文（Nickerson et al., ECML PKDD 2022）2 節の手順:
- 口座ごとに取引を日付順に並べる
- 長さ lmin=20 未満の系列は除外し、lmax=80 を超える系列は 80 件ずつの連続部分列に分割する
- tcode = type, operation, k_symbol を連結した取引コード（Czech では 16 種類）
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# 論文著者リポジトリにある前処理済み CSV（元データは PKDD'99 Discovery Challenge の Czech bank）
CZECH_URL = (
    "https://raw.githubusercontent.com/BigTuna08/Banksformer_ecml_2022/main/"
    "czech/banksformer/data/tr_by_acct_w_age.csv"
)
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "czech_tr_by_acct_w_age.csv"

MIN_SEQ_LEN = 20
MAX_SEQ_LEN = 80
TCODE_SEP = "__"

# 1 系列 = 1 行のコンテキスト。全系列で共通のカラム構成
SEQ_COLUMNS = ["seq_id", "datetime", "tcode", "amount", "age"]


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if force or not RAW_PATH.exists():
        print(f"downloading {CZECH_URL} -> {RAW_PATH}")
        urllib.request.urlretrieve(CZECH_URL, RAW_PATH)
    return RAW_PATH


def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(download(), keep_default_na=False)
    # k_symbol / operation の空欄は「値なし」というコードとして扱う（著者コードも文字列化して連結している）
    for col in ["type", "operation", "k_symbol"]:
        df[col] = df[col].replace("", "nan").astype(str)
    df["tcode"] = df["type"] + TCODE_SEP + df["operation"] + TCODE_SEP + df["k_symbol"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    # 元の並び（口座内の取引順）を保ったまま日付で安定ソートする
    df = df.sort_values(["account_id", "datetime"], kind="stable").reset_index(drop=True)
    return df[["account_id", "datetime", "tcode", "amount", "age"]]


def to_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """口座の取引列を [MIN_SEQ_LEN, MAX_SEQ_LEN] の長さの系列に分割し、seq_id を振る。"""
    pos = df.groupby("account_id").cumcount()
    chunk = pos // MAX_SEQ_LEN
    key = df["account_id"].astype(str) + "-" + chunk.astype(str)
    lengths = key.map(key.value_counts())
    out = df[lengths >= MIN_SEQ_LEN].copy()
    out["seq_id"] = pd.factorize(key[lengths >= MIN_SEQ_LEN])[0]
    return out[SEQ_COLUMNS + ["account_id"]].reset_index(drop=True)


def split_by_account(seqs: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """口座単位で半分に分ける。学習用（A）と、実データ同士の比較用（B）。"""
    rng = np.random.default_rng(seed)
    accounts = seqs["account_id"].unique()
    rng.shuffle(accounts)
    half = set(accounts[: len(accounts) // 2])
    is_a = seqs["account_id"].isin(half)
    return seqs[is_a].reset_index(drop=True), seqs[~is_a].reset_index(drop=True)


def signed_amount(df: pd.DataFrame) -> pd.Series:
    """入金は正、出金は負。月次キャッシュフローの計算に使う。"""
    sign = np.where(df["tcode"].str.startswith("DEBIT"), -1.0, 1.0)
    return df["amount"] * sign
