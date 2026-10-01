"""追試の実行コマンド。

    uv run transyn prepare                         # データ取得・前処理・口座単位の分割
    uv run transyn fit ctgan --epochs 50           # 学習と生成（outputs/gen_ctgan.csv.gz）
    uv run transyn evaluate                        # 指標計算（outputs/results.csv / results.md）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import data, metrics, models

OUT = Path("outputs")
TRAIN_PATH = data.DATA_DIR / "train_A.csv.gz"
HOLDOUT_PATH = data.DATA_DIR / "holdout_B.csv.gz"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["datetime"])


def prepare(args: argparse.Namespace) -> None:
    seqs = data.to_sequences(data.load_transactions())
    train, holdout = data.split_by_account(seqs, seed=args.seed)
    train.to_csv(TRAIN_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    for name, df in [("all", seqs), ("train_A", train), ("holdout_B", holdout)]:
        print(name, json.dumps(metrics.summary_stats(df)))


def fit(args: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    train = _read(TRAIN_PATH)
    if args.n_train_seqs:
        keep = train["seq_id"].drop_duplicates().sample(args.n_train_seqs, random_state=args.seed)
        train = train[train["seq_id"].isin(keep)]
    started = time.time()
    gen = models.MODELS[args.model](train, n_seqs=args.n_seqs, epochs=args.epochs, seed=args.seed)
    elapsed = time.time() - started
    gen[data.SEQ_COLUMNS].to_csv(OUT / f"gen_{args.model}.csv.gz", index=False)
    run = {
        "model": args.model,
        "epochs": args.epochs,
        "n_train_seqs": int(train["seq_id"].nunique()),
        "n_gen_seqs": args.n_seqs,
        "seed": args.seed,
        "seconds": round(elapsed),
    }
    (OUT / f"run_{args.model}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    print(json.dumps(run))


def evaluate(args: argparse.Namespace) -> None:
    train = _read(TRAIN_PATH)
    rows = {"Real-B (実データ同士)": metrics.evaluate(train, _read(HOLDOUT_PATH))}
    stats = {"Real-A (学習データ)": metrics.summary_stats(train)}
    for name in models.MODELS:
        path = OUT / f"gen_{name}.csv.gz"
        if path.exists():
            gen = _read(path)
            rows[name] = metrics.evaluate(train, gen)
            stats[name] = metrics.summary_stats(gen)
    for name, vals in metrics.PAPER_TABLE2_CZECH.items():
        rows[f"論文 {name}"] = vals

    table = pd.DataFrame(rows).T
    stat_table = pd.DataFrame(stats).T
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "results.csv")
    stat_table.to_csv(OUT / "summary_stats.csv")
    md = "## 指標（実データ A との比較）\n\n" + table.to_markdown(floatfmt=".3f")
    md += "\n\n## 生成データの基本統計\n\n" + stat_table.to_markdown(floatfmt=".3f") + "\n"
    (OUT / "results.md").write_text(md)
    print(md)


def main() -> None:
    parser = argparse.ArgumentParser(prog="transyn")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("prepare")
    p.set_defaults(func=prepare)

    p = sub.add_parser("fit")
    p.add_argument("model", choices=list(models.MODELS))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-seqs", type=int, default=5000, help="生成する系列数（論文は 5000）")
    p.add_argument("--n-train-seqs", type=int, default=None, help="学習に使う系列数（省略時は全件）")
    p.set_defaults(func=fit)

    p = sub.add_parser("evaluate")
    p.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)
