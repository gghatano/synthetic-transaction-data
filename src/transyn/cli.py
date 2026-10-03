"""追試の実行コマンド。

    uv run transyn prepare                         # データ取得・前処理・口座単位の分割
    uv run transyn --seed 1 fit ctgan --epochs 50  # 学習と生成（outputs/gen_ctgan_s1.csv.gz）
    uv run transyn evaluate                        # 指標計算（outputs/results.md ほか）

EHR（eICU Demo, issue #10）:

    uv run transyn ehr-prepare                     # data/eicu_demo/ の CSV からイベント系列を作り A/B/C に分割
    uv run transyn --seed 0 ehr-fit par --epochs 50  # 学習と生成（outputs/ehr/gen_par_s0.csv.gz）
    uv run transyn ehr-evaluate                    # 忠実度・有用性・プライバシ（outputs/ehr/results.md）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, ehr, metrics, models

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
    name = args.model + args.tag
    gen[data.SEQ_COLUMNS].to_csv(OUT / f"gen_{name}_s{args.seed}.csv.gz", index=False)
    run = {
        "model": name,
        "epochs": args.epochs,
        "n_train_seqs": int(train["seq_id"].nunique()),
        "n_gen_seqs": args.n_seqs,
        "seed": args.seed,
        "seconds": round(elapsed),
    }
    (OUT / f"run_{name}_s{args.seed}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    print(json.dumps(run))


def evaluate(args: argparse.Namespace) -> None:
    train = _read(TRAIN_PATH)
    holdout = _read(HOLDOUT_PATH)
    rng = np.random.default_rng(args.seed)
    rows = {
        "Real-B (実データ同士)": metrics.evaluate(train, holdout),
        # 指標の目安: B の一部の列だけを行間でシャッフルし、系列としての情報を壊したもの
        "Real-B tcode シャッフル": metrics.evaluate(
            train, holdout.assign(tcode=rng.permutation(holdout["tcode"].to_numpy()))
        ),
        "Real-B 日付シャッフル": metrics.evaluate(
            train, holdout.assign(datetime=rng.permutation(holdout["datetime"].to_numpy()))
        ),
    }
    stats = {"Real-A (学習データ)": metrics.summary_stats(train)}
    per_seed = []
    # outputs/ にある生成結果をすべて評価する（transyn 外で生成したものや --tag 付きの実行も含む）
    for path in sorted(OUT.glob("gen_*_s*.csv.gz")):
        name, seed = path.name.removeprefix("gen_").removesuffix(".csv.gz").rsplit("_s", 1)
        gen = _read(path)
        per_seed.append({"model": name, "seed": int(seed), **metrics.evaluate(train, gen)})
        stats[f"{name} (s{seed})"] = metrics.summary_stats(gen)

    by_seed = pd.DataFrame(per_seed)
    grouped = by_seed.drop(columns="seed").groupby("model", sort=False)
    mean, std, n = grouped.mean(), grouped.std(), grouped.size()
    for name in mean.index:
        rows[f"{name} (n={n[name]})"] = mean.loc[name].to_dict()
    for name, vals in metrics.PAPER_TABLE2_CZECH.items():
        rows[f"論文 {name}"] = vals

    table = pd.DataFrame(rows).T
    stat_table = pd.DataFrame(stats).T
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "results.csv")
    by_seed.to_csv(OUT / "results_by_seed.csv", index=False)
    std.to_csv(OUT / "results_std.csv")
    stat_table.to_csv(OUT / "summary_stats.csv")
    md = "## 指標（実データ A との比較。生成モデルはシード平均）\n\n" + table.to_markdown(floatfmt=".3f")
    md += "\n\n## シード間の標準偏差\n\n" + std.to_markdown(floatfmt=".3f")
    md += "\n\n## 生成データの基本統計\n\n" + stat_table.to_markdown(floatfmt=".3f") + "\n"
    (OUT / "results.md").write_text(md)
    print(md)


def ehr_prepare(args: argparse.Namespace) -> None:
    parts = ehr.prepare(seed=args.seed)
    for name, df in parts.items():
        lengths = df.groupby("seq_id").size()
        print(
            f"{name}: {lengths.size} 滞在, {len(df)} イベント, 系列長の中央値 {lengths.median():.0f}, "
            f"コード {df['tcode'].nunique()} 種類, 死亡退院 {df.groupby('seq_id')['died'].first().mean():.3f}"
        )


def ehr_fit(args: argparse.Namespace) -> None:
    ehr.OUT.mkdir(parents=True, exist_ok=True)
    train = ehr.read_split("A")
    n_seqs = args.n_seqs or train["seq_id"].nunique()
    started = time.time()
    gen = models.MODELS[args.model](train, n_seqs=n_seqs, epochs=args.epochs, seed=args.seed, spec=ehr.SPEC)
    name = args.model + args.tag
    gen[ehr.SPEC.columns].to_csv(ehr.OUT / f"gen_{name}_s{args.seed}.csv.gz", index=False)
    run = {"model": name, "epochs": args.epochs, "n_gen_seqs": n_seqs, "seed": args.seed, "seconds": round(time.time() - started)}
    (ehr.OUT / f"run_{name}_s{args.seed}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    print(json.dumps(run))


def ehr_evaluate(args: argparse.Namespace) -> None:
    table, by_seed = ehr.evaluate_all(seed=args.seed)
    ehr.OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(ehr.OUT / "results.csv")
    by_seed.to_csv(ehr.OUT / "results_by_seed.csv", index=False)
    md = "## eICU Demo: 忠実度・有用性・プライバシ（生成モデルはシード平均）\n\n" + table.to_markdown(floatfmt=".3f")
    if len(by_seed):
        std = by_seed.drop(columns="seed").groupby("model", sort=False).std()
        md += "\n\n## シード間の標準偏差\n\n" + std.to_markdown(floatfmt=".3f")
    (ehr.OUT / "results.md").write_text(md + "\n")
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
    p.add_argument("--tag", default="", help="出力名に付ける接尾辞（例: --tag=-e1000 → gen_dgan-e1000_s0.csv.gz）")
    p.set_defaults(func=fit)

    p = sub.add_parser("evaluate")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("ehr-prepare")
    p.set_defaults(func=ehr_prepare)

    p = sub.add_parser("ehr-fit")
    p.add_argument("model", choices=list(models.MODELS))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-seqs", type=int, default=None, help="生成する系列数（省略時は学習データと同数）")
    p.add_argument("--tag", default="", help="出力名に付ける接尾辞")
    p.set_defaults(func=ehr_fit)

    p = sub.add_parser("ehr-evaluate")
    p.set_defaults(func=ehr_evaluate)

    args = parser.parse_args()
    args.func(args)
