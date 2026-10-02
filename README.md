# synthetic-transaction-data

トランザクションデータ合成手法の調査と追試。

- 調査: [docs/survey-gan-transaction-synthesis.md](docs/survey-gan-transaction-synthesis.md)（issue #1）
- 追試: [docs/replication-banksformer.md](docs/replication-banksformer.md)（issue #2）

## セットアップ

[uv](https://docs.astral.sh/uv/) で Python 3.12 の環境を作る。PyTorch は CPU 版を使う。

```bash
uv sync
```

## 追試の実行

```bash
uv run transyn prepare                    # Czech bank データを取得し、系列化・口座単位で A/B に分割
uv run transyn fit dgan --epochs 400      # 学習と生成 → outputs/gen_dgan.csv.gz
uv run transyn fit ctgan --epochs 50            # log_frequency=False
uv run transyn fit ctgan-logfreq --epochs 50    # SDV 既定（log_frequency=True）
uv run transyn fit par --epochs 50
uv run transyn evaluate                   # 指標 → outputs/results.md
```

`uv run transyn --seed 1 fit dgan --epochs 400` のように `--seed` を付けると `outputs/gen_<model>_s<seed>.csv.gz` に保存され、`evaluate` がシード平均と標準偏差を出す。

`data/` と `outputs/` は git 管理外。CPU 12 コアでの所要時間の目安は DGAN 5 分、CTGAN 12〜20 分、PAR 60 分。
