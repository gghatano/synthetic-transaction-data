# synthetic-transaction-data

トランザクションデータ・EHRイベント系列・ICUバイタル時系列の合成手法の調査と検証。

結果は https://gghatano.github.io/synthetic-transaction-data/ で読める。全体概要、レポートごとの詳細、日々の記録を分けて載せている。

## レポート

- 調査: [docs/survey-gan-transaction-synthesis.md](docs/survey-gan-transaction-synthesis.md)（issue #1）
- 追試: [docs/replication-banksformer.md](docs/replication-banksformer.md)（issue #2, #4）
- DGANの切り分け: [docs/dgan-ablation.md](docs/dgan-ablation.md)（issue #5）
- Banksformer本体の再学習: [docs/banksformer-rerun.md](docs/banksformer-rerun.md)（issue #6）。実行環境は[external/banksformer/](external/banksformer/run.py)
- EHRイベント系列の合成手法の調査: [docs/survey-ehr-synthesis.md](docs/survey-ehr-synthesis.md)（issue #10）
- EHRイベント系列での検証（eICU Demo、データ共有目的）: [docs/ehr-verification.md](docs/ehr-verification.md)（issue #10）
- ICUバイタル時系列での検証（eICU Demo）: [docs/icu-vitals-verification.md](docs/icu-vitals-verification.md)（issue #15）

## セットアップ

[uv](https://docs.astral.sh/uv/)でPython 3.12の環境を作る。PyTorchはCPU版を使う。

```bash
uv sync
```

## トランザクション（Czech bank）の追試

```bash
uv run transyn prepare                          # Czech bankデータを取得し、系列化・口座単位でA/Bに分割
uv run transyn fit dgan --epochs 400            # 学習と生成 → outputs/gen_dgan_s0.csv.gz
uv run transyn fit ctgan --epochs 50            # log_frequency=False
uv run transyn fit ctgan-logfreq --epochs 50    # SDV既定（log_frequency=True）
uv run transyn fit par --epochs 50
uv run transyn evaluate                         # 指標 → outputs/results.md
```

`uv run transyn --seed 1 fit dgan --epochs 400`のように`--seed`を付けると`outputs/gen_<model>_s<seed>.csv.gz`に保存され、`evaluate`がシード平均と標準偏差を出す。CPU 12コアでの所要時間の目安はDGAN 5分、CTGAN 12〜20分、PAR 60分。

## EHR（eICU Demo）の検証

[eICU Demo](https://physionet.org/content/eicu-crd-demo/2.0.1/)を`data/eicu_demo/`に展開してから実行する。

```bash
uv run transyn ehr-prepare                      # イベント系列を作り、患者単位でA/B/Cに分割
uv run transyn --seed 0 ehr-fit gpt --epochs 40 # 学習と生成 → outputs/ehr/gen_gpt_s0.csv.gz
uv run transyn ehr-evaluate                     # 忠実度・有用性・プライバシ → outputs/ehr/results.md
```

ICUバイタルは同じeICU Demoから作る。

```bash
uv run transyn vitals-prepare                          # 心拍数・SpO2・呼吸数の12時間の系列を作り、A/B/Cに分割
uv run transyn --seed 0 vitals-fit par --epochs 200    # 学習と生成 → outputs/vitals/gen_par_s0.csv.gz
uv run transyn vitals-evaluate                         # → outputs/vitals/results.md
```

`data/`と`outputs/`はgit管理外。

## 結果サイト

`uv run transyn site`で`_site/`にHTMLを生成する。developにpushすると、GitHub Actions（`.github/workflows/pages.yml`）が同じ生成をしてGitHub Pagesに公開する。

サイトの元になるファイルは次の3種類。

| ページ | 元のファイル | 更新のしかた |
|---|---|---|
| 全体概要 | `site/overview.md` | テーマの到達点や主な知見が変わったら手で書き直す |
| レポート | `docs/*.md` | 先頭のフロントマターから一覧を作る |
| 記録 | `journal/YYYY-MM-DD.md` | 作業した日ごとに1ファイル足す |

実験を1つ追加するときは、issueを立てて実験し、`docs/`にレポートを置く。レポートの先頭には次のフロントマターを付ける。`order`は一覧での並び順で、既存の最大値の次の番号にする。

```markdown
---
title: レポートのタイトル
kind: 実験
theme: EHR
issues: [14]
date: 2026-10-05
order: 7
status: 完了
summary: 一覧に出す1〜2文の要約
---
```

その日の作業は`journal/`に記録する。フロントマターは`date`・`title`・`issues`の3つで、本文は「やったこと」「わかったこと」「決めたこと」「次の候補」のうち書くことがある見出しだけを使う。テーマの到達点や主な知見が変わったら、`site/overview.md`も直す。
