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
- 数理モデルによるシミュレーションデータ生成の調査: [docs/survey-simulation.md](docs/survey-simulation.md)（issue #18）
- DGANを適切な更新回数で再実験した結果: [docs/dgan-rerun.md](docs/dgan-rerun.md)（issue #17）

2026-10-14以降のissueは社内のリポジトリ（GHE）で管理している。上の#番号はこの公開リポジトリのもので、番号が重なるので、社内のissueは「GHE issue 1」のように書き分ける。

- 合成データで水増しして検定すると何が起きるか: [docs/synthetic-inference.md](docs/synthetic-inference.md)（GHE issue 1）
- 外部データの予後予測を共変量にして必要な被験者数を減らす: [docs/prognostic-covariate.md](docs/prognostic-covariate.md)（GHE issue 2）
- 学習人数が少ないときの生成モデル: [docs/small-n-generators.md](docs/small-n-generators.md)（GHE issue 3）
- 予測モデルの学習で合成データは実データ何人分になるか: [docs/augmentation.md](docs/augmentation.md)（GHE issue 8）
- 見本用データの忠実度の段階と公開前チェック: [docs/showcase-levels.md](docs/showcase-levels.md)（GHE issue 9）
- 見本用データへのプライバシ攻撃: [docs/attacks.md](docs/attacks.md)（GHE issue 13）
- 見本用データの開示リスクへの対策と保証（差分プライバシー・外れた被験者の除外）: [docs/dp-showcase.md](docs/dp-showcase.md)（GHE issue 14）
- 心電図の合成データのそれっぽさと本人との紐づき: [docs/ecg-linkage.md](docs/ecg-linkage.md)（GHE issue 17）
- 心電図の合成データの本人との紐づきへの対策: [docs/ecg-defense.md](docs/ecg-defense.md)（GHE issue 18）
- 合成データの作り方（全手法の手順と式）: [docs/synthesis-methods.md](docs/synthesis-methods.md)（GHE issue 22）
- 差分プライバシーの保証の根拠と限界: [docs/dp-guarantee.md](docs/dp-guarantee.md)（GHE issue 22）

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

合成データでの推論と必要人数（GHE issue 1〜3）も、同じバイタルの系列（`vitals-prepare`の出力）を使う。

```bash
uv run transyn syninf-tab-run ctgan --n 500 --reps 200       # 表データでの追試 → outputs/syninf_tabular/
uv run transyn syninf-run var1 --n 100 --m 100 --reps 500 --epochs 0   # 合成データでの検定の偽陽性 → outputs/syninf/
uv run transyn prognostic-run --ns 20 40 60 100 200 --reps 1000        # 予後予測の共変量 → outputs/prognostic/split0/
uv run transyn --seed 0 smalln-run var1 --n 20                # 学習人数を絞った生成 → outputs/smalln/
uv run transyn smalln-evaluate
uv run transyn --seed 0 augment-run --ns 20 40 60 100 200 400 --reps 100   # 予測モデルの学習曲線 → outputs/augment/split0/
uv run transyn showcase-evaluate                                          # 見本用データの段階 → outputs/showcase/
uv run transyn showcase-release --level L3                                # 見本用データとチェック結果 → outputs/showcase/release_L3/
uv run transyn attacks-targeted --n 20 --reps 200                         # 狙い撃ちのメンバーシップ推論 → outputs/attacks/
uv run transyn dp-evaluate                                                # 差分プライバシー付きの見本の忠実度 → outputs/dpsynth/
uv run transyn dp-audit --n 20 --reps 200                                 # 差分プライバシーの監査 → outputs/dpsynth/
```

心電図（GHE issue 17・18）は、PhysioNet の [ECG-ID Database](https://physionet.org/content/ecgiddb/1.0.0/) を `data/ecgid/raw/`、[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) を `data/mitdb/raw/` に展開してから実行する（どちらも ODC-By 1.0）。WFDB形式の読み込みは numpy で実装しており、追加の依存は要らない。

```bash
uv run transyn ecg-linkage --dataset ecgid                     # それっぽさと本人との紐づき → outputs/ecglink/
uv run transyn ecg-linkage --dataset mitdb --beats 200
uv run transyn ecg-defense --dataset ecgid                     # 紐づきへの対策
uv run transyn ecg-defense --dataset mitdb --beats 200
```

テストは`uv run pytest`で実行する。

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
