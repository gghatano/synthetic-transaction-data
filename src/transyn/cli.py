"""追試の実行コマンド。

    uv run transyn prepare                         # データ取得・前処理・口座単位の分割
    uv run transyn --seed 1 fit ctgan --epochs 50  # 学習と生成（outputs/gen_ctgan_s1.csv.gz）
    uv run transyn evaluate                        # 指標計算（outputs/results.md ほか）

EHR（eICU Demo, issue #10）:

    uv run transyn ehr-prepare                     # data/eicu_demo/ の CSV からイベント系列を作り A/B/C に分割
    uv run transyn --seed 0 ehr-fit par --epochs 50  # 学習と生成（outputs/ehr/gen_par_s0.csv.gz）
    uv run transyn ehr-evaluate                    # 忠実度・有用性・プライバシ（outputs/ehr/results.md）

ICU のバイタル時系列（eICU Demo, issue #15）:

    uv run transyn vitals-prepare
    uv run transyn --seed 0 vitals-fit dgan --epochs 2000
    uv run transyn vitals-evaluate                 # outputs/vitals/results.md

合成データでの検定の偽陽性（GHE issue 1）:

    uv run transyn syninf-run var1 --n 100 --m 100 --reps 500 --epochs 0
    uv run transyn syninf-summary                  # outputs/syninf/results.md
    uv run transyn syninf-tab-run ctgan --n 500 --reps 200   # 表データでの追試（Decruyenaere et al. 2024）
    uv run transyn syninf-tab-summary              # outputs/syninf_tabular/results.md

予後予測の共変量で必要な被験者数を減らせるか（GHE issue 2）:

    uv run transyn prognostic-run --ns 20 40 60 100 200 --reps 1000   # outputs/prognostic/split0/results.md

学習人数が少ないときの生成モデル（GHE issue 3）:

    uv run transyn --seed 0 smalln-run var1 --n 20
    uv run transyn smalln-evaluate                 # outputs/smalln/results.md

予測モデルの学習で合成データは実データ何人分か（GHE issue 8）:

    uv run transyn --seed 0 augment-run --ns 20 40 60 100 200 400 --reps 100   # outputs/augment/split0/results.md
    uv run transyn --seed 0 augment-regcheck --ns 20 40 60 100 200 400 --reps 100   # 正則化の対照

見本用データの忠実度の段階と公開前チェック（GHE issue 9）:

    uv run transyn showcase-evaluate               # outputs/showcase/results.md
    uv run transyn showcase-release --level L3     # outputs/showcase/release_L3/

合成データへのプライバシ攻撃（GHE issue 13）:

    uv run transyn attacks-targeted --n 20 --reps 100   # outputs/attacks/targeted_n20.csv
    uv run transyn attacks-population                    # outputs/attacks/population.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import attacks, augment, data, ehr, metrics, models, prognostic, seqgpt, showcase, smalln, syninf, syninf_tabular, vitals

# EHR では数値列のない系列向けの GPT 型モデルも使える
EHR_MODELS = {**models.MODELS, "gpt": seqgpt.fit_generate_gpt}

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
    gen = EHR_MODELS[args.model](train, n_seqs=n_seqs, epochs=args.epochs, seed=args.seed, spec=ehr.SPEC)
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


def vitals_prepare(args: argparse.Namespace) -> None:
    for name, df in vitals.prepare(seed=args.seed).items():
        s = vitals.static_of(df)
        print(f"{name}: {len(s)} 滞在, 死亡退院 {s['died'].mean():.3f}")


def vitals_fit(args: argparse.Namespace) -> None:
    vitals.OUT.mkdir(parents=True, exist_ok=True)
    train = vitals.read_split("A")
    n_seqs = args.n_seqs or train["seq_id"].nunique()
    started = time.time()
    gen = vitals.MODELS[args.model](train, n_seqs=n_seqs, epochs=args.epochs, seed=args.seed)
    name = args.model + args.tag
    gen.to_csv(vitals.OUT / f"gen_{name}_s{args.seed}.csv.gz", index=False)
    run = {"model": name, "epochs": args.epochs, "n_gen_seqs": n_seqs, "seed": args.seed, "seconds": round(time.time() - started)}
    (vitals.OUT / f"run_{name}_s{args.seed}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    print(json.dumps(run))


def vitals_evaluate(args: argparse.Namespace) -> None:
    table, by_seed = vitals.evaluate_all(seed=args.seed)
    vitals.OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(vitals.OUT / "results.csv")
    by_seed.to_csv(vitals.OUT / "results_by_seed.csv", index=False)
    md = "## eICU Demo のバイタル: 忠実度・有用性・プライバシ（生成モデルはシード平均）\n\n" + table.to_markdown(floatfmt=".3f")
    if len(by_seed):
        std = by_seed.drop(columns="seed").groupby("model", sort=False).std()
        md += "\n\n## シード間の標準偏差\n\n" + std.to_markdown(floatfmt=".3f")
    (vitals.OUT / "results.md").write_text(md + "\n")
    print(md)


def syninf_run(args: argparse.Namespace) -> None:
    syninf.run(args.model, n=args.n, m=args.m, reps=args.reps, epochs=args.epochs, start=args.start)


def syninf_summary(args: argparse.Namespace) -> None:
    table = syninf.summarize(sorted(syninf.OUT.glob("reps_*.csv")))
    table.to_csv(syninf.OUT / "results.csv", index=False)
    md = "## 第 1 種の過誤率（α=0.05、真の群間差は 0）\n\n" + table.to_markdown(index=False, floatfmt=".3f")
    (syninf.OUT / "results.md").write_text(md + "\n")
    print(md)


def syninf_tab_run(args: argparse.Namespace) -> None:
    syninf_tabular.run(args.model, n=args.n, reps=args.reps, start=args.start)


def syninf_tab_summary(args: argparse.Namespace) -> None:
    table = syninf_tabular.summarize(sorted(syninf_tabular.OUT.glob("reps_*.csv")))
    table.to_csv(syninf_tabular.OUT / "results.csv", index=False)
    md = "## 表データでの第 1 種の過誤率（M = n、真値を帰無仮説にした検定）\n\n" + table.to_markdown(index=False, floatfmt=".3f")
    (syninf_tabular.OUT / "results.md").write_text(md + "\n")
def prognostic_run(args: argparse.Namespace) -> None:
    # 病院の分け方（--seed）ごとに出力先を分ける
    out = prognostic.OUT / f"split{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    kinds = tuple(args.score_kinds or prognostic.SCORE_KINDS)
    reps, meta = prognostic.experiment(args.ns, args.reps, seed=args.seed, par_epochs=args.par_epochs, score_kinds=kinds)
    reps.to_csv(out / "reps.csv.gz", index=False)
    meta.to_csv(out / "meta.csv", index=False)
    table = prognostic.summarize(reps)
    table.to_csv(out / "results.csv", index=False)
    md = "## 予後予測と結果の相関（対象の全被験者）\n\n" + meta.to_markdown(index=False, floatfmt=".3f")
    md += "\n\n## 第 1 種の過誤率・検出力・必要人数\n\n" + table.to_markdown(index=False, floatfmt=".3f")
    (out / "results.md").write_text(md + "\n")
    print(md)


def smalln_run(args: argparse.Namespace) -> None:
    started = time.time()
    path = smalln.run(args.model, args.n, args.seed)
    print(json.dumps({"path": str(path), "seconds": round(time.time() - started)}))


def smalln_evaluate(args: argparse.Namespace) -> None:
    by_seed = smalln.evaluate()
    by_seed.to_csv(smalln.OUT / "results_by_seed.csv", index=False)
    grouped = by_seed.drop(columns="seed").groupby(["model", "n"], sort=True)
    mean, std = grouped.mean(), grouped.std()
    md = "## 学習人数ごとの忠実度・有用性・プライバシ（シード平均。忠実度は実データ C との比較）\n\n" + mean.to_markdown(floatfmt=".3f")
    md += "\n\n## シード間の標準偏差\n\n" + std.to_markdown(floatfmt=".3f")
    (smalln.OUT / "results.md").write_text(md + "\n")
    print(md)


def augment_run(args: argparse.Namespace) -> None:
    out = augment.OUT / f"split{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    reps = augment.experiment(args.ns, args.reps, seed=args.seed, par_epochs=args.par_epochs, use_par=not args.no_par)
    reps.to_csv(out / "reps.csv.gz", index=False)
    table = augment.summarize(reps)
    table.to_csv(out / "results.csv", index=False)
    md = "## AUROC の学習曲線と実データ換算の人数\n\n" + table.to_markdown(index=False, floatfmt=".3f")
    (out / "results.md").write_text(md + "\n")
    print(md)


def augment_regcheck(args: argparse.Namespace) -> None:
    out = augment.OUT / f"split{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    reps = augment.regularization_control(args.ns, args.reps, seed=args.seed)
    reps.to_csv(out / "regcheck_reps.csv.gz", index=False)
    table = reps.groupby(["task", "method", "n"], sort=False)["auroc"].mean().unstack("n")
    md = "## 実データのみ・正則化の強さ違いの AUROC\n\n" + table.to_markdown(floatfmt=".3f")
    (out / "regcheck.md").write_text(md + "\n")
    print(md)


def showcase_evaluate(args: argparse.Namespace) -> None:
    showcase.OUT.mkdir(parents=True, exist_ok=True)
    by_seed = showcase.evaluate_levels()
    by_seed.to_csv(showcase.OUT / "results_by_seed.csv", index=False)
    mean = by_seed.drop(columns="seed").groupby("level", sort=False).mean(numeric_only=True)
    md = "## 段階ごとの忠実度・関係・有用性・開示リスク（3 シード平均。忠実度は実データ C との比較）\n\n" + mean.T.to_markdown(floatfmt=".3f")
    (showcase.OUT / "results.md").write_text(md + "\n")
    print(md)


def showcase_release(args: argparse.Namespace) -> None:
    """見本用データを 1 つの段階で作り、表示・開示リスク・構造・文書化のチェック結果と一緒に書き出す。"""
    a, b = vitals.read_split("A"), vitals.read_split("B")
    out = showcase.OUT / f"release_{args.level}"
    out.mkdir(parents=True, exist_ok=True)
    gen = showcase.generate(args.level, a, n_seqs=args.n_seqs, seed=args.seed, tail_clip=args.tail_clip)
    checks = {
        "disclosure": showcase.disclosure_checks(gen, a, b),
        "structure": showcase.structure_checks(gen, a),
    }
    labelled, meta = showcase.label(gen, args.level)
    labelled.to_csv(out / "synthetic_vitals.csv.gz", index=False)
    (out / "metadata.json").write_text(json.dumps({**meta, "checks": checks}, ensure_ascii=False, indent=2, default=float))
    lines = [f"# 見本用データ（{args.level}）", "", f"- 合成データ。{meta['intended_use']}", f"- 元データ: {meta['source']}",
             f"- 作り方: {meta['generator']}", "", "## 保つ性質", *[f"- {x}" for x in meta["preserved"]], "",
             "## 保たない性質", *[f"- {x}" for x in meta["not_preserved"]], "", "## 公開前チェックの結果",
             *[f"- {k}: {v}" for part in checks.values() for k, v in part.items()]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=float))


def attacks_targeted(args: argparse.Namespace) -> None:
    attacks.OUT.mkdir(parents=True, exist_ok=True)
    df = attacks.experiment_targeted(args.n, args.reps, seed=args.seed)
    df.to_csv(attacks.OUT / f"targeted_n{args.n}.csv", index=False)
    print(df.groupby(["level", "kind"])[["auc", "eps_lb"]].agg(["mean", "max"]).to_markdown(floatfmt=".3f"))


def attacks_population(args: argparse.Namespace) -> None:
    attacks.OUT.mkdir(parents=True, exist_ok=True)
    df = attacks.experiment_population()
    df.to_csv(attacks.OUT / "population.csv", index=False)
    print(df.drop(columns="seed").groupby(["n", "level"]).mean().to_markdown(floatfmt=".3f"))


def build_site(args: argparse.Namespace) -> None:
    from . import site

    site.build(args.out)


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
    p.add_argument("model", choices=list(EHR_MODELS))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-seqs", type=int, default=None, help="生成する系列数（省略時は学習データと同数）")
    p.add_argument("--tag", default="", help="出力名に付ける接尾辞")
    p.set_defaults(func=ehr_fit)

    p = sub.add_parser("ehr-evaluate")
    p.set_defaults(func=ehr_evaluate)

    p = sub.add_parser("vitals-prepare")
    p.set_defaults(func=vitals_prepare)

    p = sub.add_parser("vitals-fit")
    p.add_argument("model", choices=list(vitals.MODELS))
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--n-seqs", type=int, default=None, help="生成する系列数（省略時は学習データと同数）")
    p.add_argument("--tag", default="", help="出力名に付ける接尾辞")
    p.set_defaults(func=vitals_fit)

    p = sub.add_parser("vitals-evaluate")
    p.set_defaults(func=vitals_evaluate)

    p = sub.add_parser("syninf-run", help="合成データで検定したときの第 1 種の過誤率の試行")
    p.add_argument("model", choices=list(syninf.MODELS))
    p.add_argument("--n", type=int, required=True, help="学習に使う実データの被験者数")
    p.add_argument("--m", type=int, required=True, help="生成する被験者数")
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--start", type=int, default=0, help="最初の試行のシード")
    p.set_defaults(func=syninf_run)

    p = sub.add_parser("syninf-summary")
    p.set_defaults(func=syninf_summary)

    p = sub.add_parser("syninf-tab-run", help="表データでの合成データの検定の追試")
    p.add_argument("model", choices=list(syninf_tabular.MODELS))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.set_defaults(func=syninf_tab_run)

    p = sub.add_parser("syninf-tab-summary")
    p.set_defaults(func=syninf_tab_summary)
    p = sub.add_parser("prognostic-run", help="予後予測の共変量による必要人数の削減の半合成実験")
    p.add_argument("--ns", type=int, nargs="+", required=True, help="対象から抜く被験者数")
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--par-epochs", type=int, default=200)
    p.add_argument("--score-kinds", nargs="+", choices=list(prognostic.SCORE_KINDS), default=None,
                   help="使う予後予測（省略時はすべて）。prog-syn-par は PAR の学習に時間がかかる")
    p.set_defaults(func=prognostic_run)

    p = sub.add_parser("smalln-run", help="学習人数を n に絞って生成モデルを学習・生成する")
    p.add_argument("model", choices=list(smalln.MODELS))
    p.add_argument("--n", type=int, required=True)
    p.set_defaults(func=smalln_run)

    p = sub.add_parser("smalln-evaluate")
    p.set_defaults(func=smalln_evaluate)

    p = sub.add_parser("augment-run", help="予測モデルの学習でのデータ拡張・合成データの学習曲線")
    p.add_argument("--ns", type=int, nargs="+", required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--par-epochs", type=int, default=200)
    p.add_argument("--no-par", action="store_true", help="外部データの PAR による合成を省く")
    p.set_defaults(func=augment_run)

    p = sub.add_parser("augment-regcheck", help="実データのみで正則化の強さを変えた対照")
    p.add_argument("--ns", type=int, nargs="+", required=True)
    p.add_argument("--reps", type=int, required=True)
    p.set_defaults(func=augment_regcheck)
    p = sub.add_parser("showcase-evaluate", help="見本用データの段階ごとの評価")
    p.set_defaults(func=showcase_evaluate)

    p = sub.add_parser("showcase-release", help="見本用データを 1 つの段階で作り、公開前チェックと一緒に書き出す")
    p.add_argument("--level", choices=list(showcase.LEVELS), required=True)
    p.add_argument("--n-seqs", type=int, default=50, help="見本として出す系列数")
    p.add_argument("--tail-clip", type=float, default=0.005, help="値を学習データのこの分位点で切り詰める（実在の極値を出さない）")
    p.set_defaults(func=showcase_release)

    p = sub.add_parser("attacks-targeted", help="狙い撃ちのメンバーシップ推論（段階ごと、標的10人）")
    p.add_argument("--n", type=int, required=True, help="学習人数")
    p.add_argument("--reps", type=int, required=True, help="標的ごとの試行数（半分は標的を入れる）")
    p.set_defaults(func=attacks_targeted)

    p = sub.add_parser("attacks-population", help="属性推論と部分系列による所属推定（段階ごと）")
    p.set_defaults(func=attacks_population)

    p = sub.add_parser("site", help="実験・調査結果のサイトを生成する（既定の出力先は _site/）")
    p.add_argument("--out", type=Path, default=Path("_site"))
    p.set_defaults(func=build_site)

    args = parser.parse_args()
    args.func(args)
