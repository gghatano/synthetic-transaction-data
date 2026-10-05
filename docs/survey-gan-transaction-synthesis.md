---
title: GAN系手法によるトランザクションデータ合成の事例調査
kind: 調査
theme: トランザクション
issues: [1]
date: 2026-10-01
order: 1
status: 完了
summary: CTGANを行独立に適用すると時間構造が崩れるという先行研究の指摘を整理し、系列・関係データ向け手法と比較表にまとめた。
---

# GAN系手法によるトランザクションデータ合成の事例調査

issue #1「GAN系の手法を用いたトランザクションデータの合成事例の調査」の成果物。

## スコープ

| 項目 | 決定内容 |
|---|---|
| 対象データ | トランザクションデータを種類で区別せず横断的に扱う（決済・カード、銀行入出金、小売購買） |
| 手法範囲 | GAN系（CTGAN等）を中心に、系列・関係データ向け手法を対比として含める。拡散・LLM系は名前の列挙にとどめる |
| 評価軸 | 比較表では行単位の忠実度（周辺分布・列間相関）を主軸とする |
| 成果物 | 比較表＋文献リスト。実データでの検証は別issueに切り出す |

調査は2026-10時点のWeb検索に基づく。多くは論文のアブストラクトと本文の一部しか確認していない。「要原典確認」と付けた項目は、検証フェーズで本文を読んで裏付けること。

## 要点

1. **CTGANを「1行＝1取引」でそのまま当てると、行単位の忠実度は概ね保たれるが、系列構造は保たれない。** 周辺分布・列間相関・下流分類器の性能は維持される一方、時間間隔、取引の頻度（velocity）、バースト、複数口座にまたがるパターンが大きく崩れる、と報告されている [S1]。行を独立に生成する設計なので、行をまたぐパターンは構造的に再現できない、という主張である [S1]。
2. **静的な（行単位の）指標で評価すると、系列の破綻を見逃す。** 静的指標がほぼ完全なモデルでも、タイムスタンプの重複や不自然な間隔を生む。静的指標と時系列を考慮した指標とで、モデルの順位が大きく入れ替わる [S2]。行単位の忠実度だけを評価軸にするとCTGANと系列手法の差が見えない、というのが本調査で最も重要な示唆である（「検証issueへの申し送り」参照）。
3. **トランザクション向けのGAN系手法は、CTGANを拡張するか系列モデルと組み合わせる形で発展している。** BankGANはCTGANの生成器・識別器構成を引き継ぎ、系列の条件ベクトルとBanksformerの日付生成機構を加えた [S6]。TRGANは時間間隔を条件に組み込んだGANである [S7]。小売ではRNNで顧客状態を持ち、GANでバスケットを生成する構成がある [S10]。
4. **系列の再現では、Transformer・自己回帰系がGAN系より優位という報告が多い。** Banksformerは日付に関するパターンでDoppelGANger・TimeGANを上回った [S5]。ただしTRGANはBanksformerを上回ったと主張している [S7]（要原典確認）。手法間の優劣は、データと指標の選び方に強く依存する。
5. **金融合成データの文献はGAN系が多数を占めるが、評価方法が統一されておらず、特にプライバシ評価が手薄** という系統的レビューの指摘がある [S3]。

## 手法の比較表

「行単位忠実度」と「系列・関係の再現」の列は、各論文の報告内容の要約である。本調査で再現実験はしていない。

### A. 行独立の表形式生成（GAN系中心）

| 手法 | 系統 | 扱える構造 | トランザクションへの適用事例 | 行単位忠実度 | 系列・関係の再現 | 実装 |
|---|---|---|---|---|---|---|
| CTGAN [M1] | 条件付きGAN | 単一表・行独立 | カード不正検知のオーバーサンプリング [S8]、銀行取引のベンチマーク [S4]、不正パターン保持のベンチマーク [S1] | 高いと報告。周辺分布・相関の順位は保たれるが、強い連続相関の大きさはやや縮む [S8] | 不可（構造上）。時間間隔・velocityが大きく劣化 [S1] | SDV / `ctgan` |
| TVAE [M1] | VAE | 単一表・行独立 | [S1][S4] | 高い（データ複製には有利との評価 [S4]） | 不可 | SDV |
| CopulaGAN / GaussianCopula | GAN / 統計 | 単一表・行独立 | [S1][S7] でベースライン | 中〜高 | 不可 | SDV |
| CTAB-GAN+ [M2] | 条件付きGAN | 単一表・行独立 | 汎用表データ（金融特化の事例は今回未確認） | 高いと報告 | 不可 | 公開実装あり |

### B. 系列・時系列向け（GAN系）

| 手法 | 系統 | 扱える構造 | トランザクションへの適用事例 | 行単位忠実度 | 系列・関係の再現 | 実装 |
|---|---|---|---|---|---|---|
| DoppelGANger (DGAN) [M3] | GAN（メタデータと系列を分離して生成） | 主体属性＋固定長系列 | 銀行取引のベンチマーク [S4]、Banksformerのベースライン [S5] | 中 | 系列は扱える。ただし日付由来の特徴（曜日・月内日など）の関係は弱い [S5] | `gretel-synthetics`等 |
| TimeGAN [M4] | GAN＋教師あり損失 | 等間隔の時系列 | Banksformerのベースライン [S5] | 中 | 等間隔前提のため、不規則間隔の取引は苦手 [S5] | 公開実装あり |
| BankGAN [S6] | CTGAN拡張＋系列条件ベクトル＋日付機構 | 口座×取引系列（不規則間隔） | Czech bankデータ [S6] | CTGAN相当 | 周期的取引（給与など）の再現でRNN・Transformer系を上回ったと報告 [S6] | 要確認 |
| TRGAN [S7] | 条件付きGAN＋Supervisor | 顧客×取引（MCC・金額・時間間隔） | 銀行取引 [S7] | 高いと報告 | 時間間隔の再現でBanksformer・CTGAN・CopulaGANを上回ったと報告 [S7]（要原典確認） | 要確認 |
| 小売バスケットGAN（Doanら）[S10] | RNN（顧客埋め込み）＋条件付きGAN | 顧客×バスケット系列 | 小売購買履歴 [S10] | 商品種別・ブランド・価格の頻度が類似 | 主要な系列パターンを再現と報告 | 要確認 |
| 在庫制約付き購買GAN [S11] | GAN＋ハイパーグラフ埋め込み | 購買トランザクション＋SKU在庫制約 | 大規模小売 [S11] | 改善を報告 | 在庫制約を考慮 | 要確認 |

### C. 対比: 系列・関係データ向けの非GAN手法

| 手法 | 系統 | 扱える構造 | トランザクションへの適用事例 | 位置づけ |
|---|---|---|---|---|
| PAR / CPAR [M5] | 確率的自己回帰（NN） | 主体属性＋可変長系列（multi-sequence） | SDVのSequentialモデル | SDVで手軽に使える系列手法。CTGANとの対比ベースラインに使える |
| Banksformer [S5] | decoder-only Transformer＋日付機構 | 口座×取引系列（不規則間隔） | Czech bank、UK合成データ [S5] | 不規則間隔・日付パターンの再現が強み。BankGAN・TRGANの比較対象 |
| TabFormer / TabGPT [M6] | Transformer（GPT型） | 取引系列 | IBM合成カード取引データ [M6] | 生成と表現学習（不正検知）の両方に使える |
| REaLTabFormer [M7] | GPT-2＋Seq2Seq | 親子テーブル（顧客→取引） | 汎用の関係データ | 関係データ（1対多）を直接扱える |
| HMA（SDV）[M8] | 統計（階層モデル） | マルチテーブル | 汎用 | 関係データの古典的ベースライン |
| FinDiff [M9] | 拡散モデル | 単一表（金融） | 銀行データのベンチマーク [S4] | 拡散系の参考。本調査の主対象外 |
| TabularARGN | 自己回帰 | 表（系列対応あり） | 不正パターンのベンチマーク [S1] | 行独立型より劣化は小さいが、依然大きい（Amazon FDBで17.2倍）[S1] |
| エージェントベース（PaySim, AMLSim等） | シミュレーション | 取引ネットワーク | 不正・AML研究 | 学習型ではない。比較の外側に置く |

## CTGAN適用時の論点（調査結果に基づく整理）

| 観点 | CTGAN（1行＝1取引）で起きること | 根拠 |
|---|---|---|
| 列ごとの周辺分布 | 概ね保たれる | [S1][S8] |
| 同じ行の中の列間相関 | 順位は保たれ、強い連続相関の大きさはやや縮む | [S8] |
| 下流分類器の性能（TSTR） | 保たれるとの報告がある | [S1][S8] |
| 顧客ごとの件数・取引の頻度 | 崩れる | [S1] |
| 時間間隔・バースト | 大きく崩れる。行独立なので構造上再現できない | [S1][S2] |
| 複数口座にまたがるパターン | 再現できない | [S1] |
| 取引ネットワークのグラフ構造 | どの手法も十分に再現できない | [S4] |

回避策として、CTGAN本体を拡張する方向（BankGAN, TRGAN）と、系列モデルに置き換える方向（PAR, Banksformer, TabFormer）が並行して存在する。ラグ特徴量の追加や顧客単位への集約といった前処理だけで凌ぐアプローチの効果は、今回の調査ではまとまった報告を見つけられなかった。

## 検証issueへの申し送り

1. **評価軸。** 今回の比較表は行単位の忠実度を主軸にした。しかし [S1][S2] が示すとおり、この軸だけではCTGANと系列手法の差が出ない。検証では最低限、系列の指標（顧客ごとの件数分布、時間間隔の分布）を加えることを推奨する。
2. **比較候補の手法。** CTGAN（ベースライン）、PAR（SDVで同じ環境で動く系列手法）、DoppelGANger。余力があればBanksformerかTabFormerを加える。
3. **データセット候補（ライセンスは要確認）。**
   - Czech bank（PKDD'99 / Berka）: 実データの銀行取引。Banksformer・BankGANと同じデータなので、既報と比較できる [S5][S6]
   - IBM TabFormerの合成カード取引データ [M6]
   - IEEE-CIS Fraud Detection: [S1] と比較できる
   - 小売: Online Retail II（UCI）、Instacart
4. **原典を読んで裏付ける項目。** TRGANの比較条件とデータ [S7]、BankGANの定量結果 [S6]、[S8] の数値がどの論文によるものか。

## 文献リスト

### 基盤手法

- [M1] Xu, Skoularidou, Cuesta-Infante, Veeramachaneni. *Modeling Tabular Data using Conditional GAN* (CTGAN / TVAE). NeurIPS 2019. https://arxiv.org/abs/1907.00503
- [M2] Zhao et al. *CTAB-GAN+: Enhancing Tabular Data Synthesis*. 2022. https://arxiv.org/abs/2204.00401
- [M3] Lin, Jain, Wang, Fanti, Sekar. *Using GANs for Sharing Networked Time Series Data* (DoppelGANger). IMC 2020. https://arxiv.org/abs/1909.13403
- [M4] Yoon, Jarrett, van der Schaar. *Time-series Generative Adversarial Networks* (TimeGAN). NeurIPS 2019.
- [M5] Zhang, Patki, Veeramachaneni. *Sequential Models in the Synthetic Data Vault* (PAR / CPAR). 2022. https://arxiv.org/abs/2207.14406
- [M6] Padhi et al. *Tabular Transformers for Modeling Multivariate Time Series* (TabFormer). ICASSP 2021. https://arxiv.org/abs/2011.01843 / https://github.com/IBM/TabFormer
- [M7] Solatorio, Dupriez. *REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers*. 2023. https://arxiv.org/abs/2302.02041
- [M8] Patki, Wedge, Veeramachaneni. *The Synthetic Data Vault*. IEEE DSAA 2016.
- [M9] Sattarov, Schreyer, Borth. *FinDiff: Diffusion Models for Financial Tabular Data Generation*. ICAIF 2023. https://arxiv.org/abs/2309.01472

### トランザクション合成の事例・ベンチマーク

- [S1] Sajja. *Synthetic Tabular Generators Fail to Preserve Behavioral Fraud Patterns: A Benchmark on Temporal, Velocity, and Multi-Account Signals*. 2026（DMLR投稿）. https://arxiv.org/abs/2604.13125 — CTGAN, TVAE, GaussianCopula, TabularARGNをIEEE-CISとAmazon Fraudで比較
- [S2] Kwon et al. *Seq2Synth: Benchmarking Temporal Fidelity in Synthetic Sequential Tabular Data*. CIKM 2026. https://arxiv.org/abs/2607.15606
- [S3] Meldrum, Suleiman, Rabhi, Alibasa. *New Money: A Systematic Review of Synthetic Data Generation for Finance*. 2025. https://arxiv.org/abs/2510.26076 — 2018年以降の72研究のレビュー
- [S4] Karst et al. *Generative AI for Banks: Benchmarks and Algorithms for Synthetic Financial Transaction Data*. WITS 2024. https://arxiv.org/abs/2412.14730 — CTGAN, DGAN, WGAN, FinDiff, TVAEを比較
- [S5] Nickerson et al. *Banksformer: A Deep Generative Model for Synthetic Transaction Sequences*. ECML PKDD 2022. https://doi.org/10.1007/978-3-031-26422-1_8 / https://github.com/BigTuna08/Banksformer_ecml_2022
- [S6] Mehri et al. *BankGAN: A Generative Model for Synthetic Financial Transactions*. Canadian AI 2024. https://assets.pubpub.org/vgjxzekz/Mehri-71716794117674.pdf（修士論文版: Mehri, *Generating Bank Transaction Sequences with Tabular GAN models*, Memorial University, 2024）
- [S7] Zakharov, Stavinova, Lysenko. *TRGAN: A Time-Dependent Generative Adversarial Network for Synthetic Transactional Data Generation*. ICSeB 2023. https://doi.org/10.1145/3641067.3641076
- [S8] CTGANによるカード不正データ合成・オーバーサンプリングの事例群（例: PeerJ Computer Science 2023, https://peerj.com/articles/cs-1634.pdf）。検索結果の要約に頼った部分があり、数値の出典は要原典確認
- [S9] Potluru et al. *Synthetic Data Applications in Finance*. 2024. https://arxiv.org/abs/2401.00081 — 金融における合成データの用途サーベイ
- [S10] Doan, Veira, Ray, Keng. *Generating Realistic Sequences of Customer-level Transactions for Retail Datasets*. ICDM Workshops 2018. https://arxiv.org/abs/1901.05577
- [S11] Tkachuk, Łukasik, Wróblewska. *Consumer Transactions Simulation Through Generative Adversarial Networks Under Stock Constraints in Large-Scale Retail*. Electronics 2025. https://doi.org/10.3390/electronics14020284
- [S12] Xia, Wang, Mabry, Cheng. *Advancing Retail Data Science: Comprehensive Evaluation of Synthetic Data*. 2024. https://arxiv.org/abs/2406.13130
