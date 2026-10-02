"""Banksformer 著者コード（Czech 実験）を実行し、生成データを transyn の形式で保存する。

著者リポジトリには LICENSE がないため、本リポジトリには取り込まない。実行時に固定コミットを
data/banksformer_src に clone し、ノートブックをそのまま（設定値のみ置換して）実行する。

    cd external/banksformer
    uv run python run.py --seed 0          # リポジトリ直下の data/, outputs/ を使う

前提: リポジトリ直下で `uv run transyn prepare` 済み（data/train_A.csv.gz を使う）。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

REPO_URL = "https://github.com/BigTuna08/Banksformer_ecml_2022"
COMMIT = "4e998944825f04f94f6cb3a0058b4733fd70ccf5"

ROOT = Path(__file__).resolve().parents[2]
TCODE_SEP = "__"


def src_dir() -> Path:
    return ROOT / "data" / "banksformer_src"


def work_dir() -> Path:
    return src_dir() / "czech" / "banksformer"


def checkout() -> None:
    if not src_dir().exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(src_dir())], check=True)
    subprocess.run(["git", "-C", str(src_dir()), "checkout", "-q", COMMIT], check=True)


def write_training_csv() -> None:
    """著者の前処理済み CSV のうち、学習用 A の口座だけを残す（transyn の他モデルと条件を揃える）。"""
    train_accounts = pd.read_csv(ROOT / "data" / "train_A.csv.gz", usecols=["account_id"])["account_id"].unique()
    raw = pd.read_csv(ROOT / "data" / "czech_tr_by_acct_w_age.csv")
    raw[raw["account_id"].isin(train_accounts)].to_csv(work_dir() / "data" / "tr_by_acct_w_age.csv", index=False)


def patched(nb_name: str, n_models: int, seed: int, epochs: int) -> nbformat.NotebookNode:
    """ノートブックの設定値だけを置換する。学習ロジックには手を入れない。"""
    nb = nbformat.read(work_dir() / nb_name, as_version=4)
    seed_cell = nbformat.v4.new_code_cell(
        f"import random, numpy as np, tensorflow as tf\nrandom.seed({seed}); np.random.seed({seed}); tf.random.set_seed({seed})"
    )
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # 著者は同じ設定で 3 モデル学習し、全モデルで生成している。CPU 時間の都合で数を変えられるようにする
        cell.source = cell.source.replace("for i in range(3):", f"for i in range({n_models}):")
        cell.source = cell.source.replace("EPOCHS = 80", f"EPOCHS = {epochs}")
    if nb_name.startswith("nb3"):
        nb.cells.insert(0, seed_cell)
    return nb


def execute(nb_name: str, n_models: int, seed: int, epochs: int) -> None:
    started = time.time()
    nb = patched(nb_name, n_models, seed, epochs)
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(work_dir())}}).execute()
    nbformat.write(nb, work_dir() / f"executed_{nb_name}")
    print(f"{nb_name}: {time.time() - started:.0f}s", flush=True)


def convert(seed: int) -> Path:
    """生成 CSV（著者形式）を transyn の SEQ_COLUMNS 形式に変換する。複数ある場合は最初のモデルのものを使う。"""
    generated = sorted((work_dir() / "generated_data").glob("gen_*.csv"))
    if not generated:
        sys.exit("generated_data に生成結果がない")
    # 著者の生成 CSV の列: amount, tcode_num, date_fields, days_passed, age, date, account_id, tcode
    # tcode は type__operation__k_symbol の順で、transyn の前処理と語彙が一致する
    df = pd.read_csv(generated[0], keep_default_na=False, parse_dates=["date"])
    out = pd.DataFrame(
        {
            "seq_id": df["account_id"],
            "datetime": df["date"],
            "tcode": df["tcode"],
            "amount": df["amount"],
            "age": df["age"],
        }
    )
    path = ROOT / "outputs" / f"gen_banksformer_s{seed}.csv.gz"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({generated[0].name}, {len(out)} rows)")
    return path


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-models", type=int, default=1, help="学習するモデル数（著者は 3）")
    parser.add_argument("--epochs", type=int, default=80, help="最大エポック数（著者は 80、early stopping あり）")
    parser.add_argument("--fresh", action="store_true", help="前回の中間生成物とチェックポイントを消してから実行する")
    parser.add_argument("--root", type=Path, default=ROOT, help="data/ と outputs/ があるリポジトリのルート")
    args = parser.parse_args()
    ROOT = args.root.resolve()

    checkout()
    if args.fresh:
        for d in ["stored_data", "generated_data", "checkpoints", "training_history"]:
            shutil.rmtree(work_dir() / d, ignore_errors=True)
    for d in ["stored_data", "generated_data", "generated_data/parts", "checkpoints", "training_history"]:
        (work_dir() / d).mkdir(parents=True, exist_ok=True)
    write_training_csv()
    for nb_name in ["nb1_preprocess_czech.ipynb", "nb2_encode_data.ipynb", "nb3_banksformer-v2.ipynb"]:
        execute(nb_name, args.n_models, args.seed, args.epochs)
    convert(args.seed)


if __name__ == "__main__":
    main()
