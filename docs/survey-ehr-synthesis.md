---
title: EHRイベント系列の合成手法の調査
kind: 調査
theme: EHR
issues: [10]
date: 2026-10-03
order: 5
status: 完了
summary: EHRのイベント系列はトランザクションとほぼ同じ構造で、手法も行単位GAN → 系列GAN → 自己回帰Transformerと同じ流れ。データ共有にはプライバシ評価が必須。
---

# EHRイベント系列の合成手法の調査（データ共有目的）

issue #10の調査パート。トランザクション合成（#1〜#6）の知見を電子カルテ（EHR）のイベント系列に広げ、データ共有を目的とした検証の方針を決める。

調査は2026-10時点のWeb検索に基づく。多くは論文のアブストラクトと一部の本文しか確認していない。「要原典確認」の項目は検証フェーズで裏付ける。

## 要点

1. **EHRのイベント系列はトランザクションとほぼ同じ構造を持つ。** 患者＝口座、受診・イベント＝取引、診断・処方コード＝tcode、検査値＝金額、不規則な受診間隔＝取引間隔、と対応する。違いは、1回の受診に複数のコードが入る「入れ子構造」（小売のバスケットに近い）と、コードの種類が数千〜数万と桁違いに多いこと。
2. **手法の系統はトランザクションと同じ流れをたどっている。** 行（患者）単位のGAN（medGAN, CorGAN）→ 系列を扱うGAN（SynTEG, EHR-M-GAN, EHR-Safe）→ 自己回帰Transformer（HALO, PromptEHR, SynEHRgy）→ 拡散モデル。トランザクションでCTGAN → DoppelGANger → Banksformerと進んだのと同じ構図。
3. **最近の比較ではTransformer系が忠実度で優位、GAN系は一長一短。** HALOはGAN系のSynTEGより系列の忠実度が高いと報告されている [E6][E11]。一方、MIMICでの比較レビューではGAN系（MedGAN, CorGAN）は忠実度・有用性で健闘し、プライバシではルールベースが強いとされる [E10]。
4. **データ共有が目的なら、プライバシ評価が中心になる。** EHR分野では、メンバーシップ推論（その患者が学習に使われたかを当てる攻撃）、属性推論、再識別リスク、最近傍距離が標準的に評価されている [E5][E6][E9]。トランザクションの追試では後回しにした軸だが、ここでは必須。
5. **公開データは小さい。** 誰でも使えるEHRはMIMIC-IV Demo（100人）とeICU Demo（約2,500 ICU滞在）程度で、本格的なMIMIC-IV / eICUはPhysioNetでの資格認定（研修の受講と申請）が要る。検証の規模はデータの入手方法で決まる。

## トランザクションとの対応

| トランザクション（#1〜#6） | EHRイベント系列 | 補足 |
|---|---|---|
| 口座 | 患者 | |
| 取引 | 受診（visit）またはイベント | EHRは1受診に複数コードが入る入れ子構造 |
| tcode（16種類） | 診断（ICD）・処方・処置コード | 数千〜数万種類。ロングテールの再現が難所 [E11] |
| 金額 | 検査値・バイタル | 欠測が多く、測定の有無自体が情報を持つ [E5] |
| 取引間隔 | 受診間隔・イベント間隔 | 不規則間隔の扱いは共通の論点 |
| 年齢などの属性 | 年齢・性別・入院時情報 | |
| CTGAN（行独立） | medGAN, CorGAN（患者単位の集約ベクトル） | 時間構造を捨てる点が同じ |
| DoppelGANger | SynTEG, EHR-M-GAN, EHR-Safe | 系列を扱うGAN |
| Banksformer, TabGPT | HALO, PromptEHR, SynEHRgy | 自己回帰Transformer |

#2〜#6で作った評価の仕組み（取引間隔・密度の指標、シャッフル基準、複数シード）は、この対応でそのまま使える。

## 手法の比較表

| 手法 | 系統 | 扱う構造 | 主な評価データ | プライバシ評価 | 実装 |
|---|---|---|---|---|---|
| medGAN [E1] | GAN＋オートエンコーダ | 患者単位の集約コードベクトル（時間なし） | 独自EHR | メンバーシップ推論・属性推論 | 公開実装あり（mp2893/medgan） |
| CorGAN [E2] | GAN（CNN） | 患者単位の集約コードベクトル | MIMIC-III | メンバーシップ推論（要原典確認） | 公開実装あり |
| SynTEG [E3] | GAN＋Transformer | 受診の系列（コード＋受診間隔） | 独自EHR | 要原典確認 | 要確認 |
| EHR-M-GAN [E4] | GAN（二重VAE＋結合RNN） | 連続値と離散値が混在する時系列 | ICUデータベース3つ（約14万人） | 要原典確認 | 要確認 |
| EHR-Safe [E5] | エンコーダ・デコーダ＋WGAN-GP | 静的・時系列の数値と離散値、測定時刻、欠測パターン、可変長 | MIMIC-III（約2万人）、eICU（約20万人） | メンバーシップ推論0.49〜0.50（理想0.5）、再識別、属性推論 | 要確認 |
| HALO [E6] | 階層的自己回帰Transformer | 受診の系列×各受診のコード集合（高次元） | 大規模な外来・入院データ（要原典確認） | メンバーシップ推論・属性推論・最近傍 | [HALO_Inpatient](https://github.com/btheodorou99/HALO_Inpatient) |
| PromptEHR [E7] | 言語モデル（プロンプト条件付き） | 受診の系列 | MIMIC-III | 要原典確認 | 公開実装あり（PyTrial） |
| SynEHRgy [E8] | decoder-only Transformer | コード・検査値・時刻が混在する構造化EHR | MIMIC | メンバーシップ推論・属性推論 | 要確認 |
| EHR時系列の拡散モデル [E9] | 拡散 | EHRの時系列 | ICUデータ | プライバシの評価を重視 | 要確認 |

### 比較のための基盤

| 名前 | 内容 | 位置づけ |
|---|---|---|
| SynthEHRella [E10] | MIMIC-III/IVの表現型データで7手法（GAN系・ルールベース等）を比較したレビュー（42研究）とPythonパッケージ | 忠実度・有用性・プライバシ・計算コストを同じ枠組みで比較できる |
| PyHealthベースのベンチマーク [E11] | MedGAN, CorGAN, PromptEHR, HALO, GPT-2を縦断的なICD診断コードで比較する枠組み | 既存手法はロングテール（まれなコード）の再現が弱いと指摘 |

## プライバシの評価方法（データ共有目的で必須）

| 攻撃・指標 | 何を測るか | 理想値の目安 |
|---|---|---|
| メンバーシップ推論 | ある患者の記録が学習に使われたかを、合成データから当てられるか | 正解率0.5（当て推量と同じ） |
| 属性推論 | 一部の属性を知る攻撃者が、残りの属性（診断など）を推定できるか | 実データを使わない場合と差がない |
| 再識別 | 合成レコードから実在の患者を特定できるか | 低いほど良い |
| 最近傍距離（DCR） | 合成レコードと最も近い学習レコードとの距離。コピーの検出 | 学習データ同士・ホールドアウトとの距離と同程度 |

多くの研究は、合成データ上のk近傍で学習用と評価用の患者を見分けられるかで、メンバーシップ推論を評価している [E8]。

## 公開データと利用条件

| データ | 規模 | 内容 | 入手条件 |
|---|---|---|---|
| MIMIC-IV Clinical Database Demo v2.2 [D1] | 100人 | MIMIC-IVの全26テーブル（診断・処方・検査・ICU等）。自由記述は除く | 誰でも可。ODbL v1.0。15 MB |
| eICU Collaborative Research Database Demo v2.0.1 [D2] | 約2,500 ICU滞在、20病院 | 診断（時刻付きの問題リスト）・治療・投薬・検査・バイタル等 | 誰でも可。ODbL v1.0。130 MB |
| MIMIC-IV / eICU本体 | 数万〜20万人規模 | 同上の全体 | PhysioNetでの資格認定（研修の受講と利用申請）。本人の手続きが必要 |

ODbLはデータベースの派生物を公開する場合に同じライセンスでの公開（share-alike）と出典表示を求める。デモデータから作った合成データを共有する場合の扱いは、利用条件を確認する必要がある。

## 検証の方針（案）

1. **データ**: まず誰でも使えるeICU Demoで検証の仕組みを作る。ICU滞在内の診断・治療・投薬・検査をイベント系列（滞在＝系列、オフセット＝時刻）として扱う。規模が小さいため、メンバーシップ推論のリスクは本番データより高く出る可能性がある点に注意する。本番規模の検証にはMIMIC-IV / eICU本体の資格認定が要る。
2. **手法**: トランザクションでの比較と揃え、行（患者）単位のGAN（CTGAN相当、またはmedGAN）、系列GAN（DGAN）、自己回帰Transformer（HALO、またはBanksformer系）から2〜3手法。
3. **評価**: #2〜#6の忠実度の指標（コード分布・間隔・密度・n-gram・シャッフル基準）に加えて、有用性（合成で学習し実データで評価）と、プライバシ（メンバーシップ推論・最近傍距離・属性推論）。

## 文献リスト

- [E1] Choi et al. *Generating Multi-label Discrete Patient Records using Generative Adversarial Networks* (medGAN). MLHC 2017. https://arxiv.org/abs/1703.06490
- [E2] Torfi, Fox. *CorGAN: Correlation-Capturing Convolutional Generative Adversarial Networks for Generating Synthetic Healthcare Records*. FLAIRS 2020. https://arxiv.org/abs/2001.09346
- [E3] Zhang et al. *SynTEG: a framework for temporal structured electronic health data simulation*. JAMIA 2021.（要原典確認）
- [E4] Li et al. *Generating synthetic mixed-type longitudinal electronic health records for artificial intelligent applications* (EHR-M-GAN). npj Digital Medicine 2023. https://www.nature.com/articles/s41746-023-00834-7
- [E5] Yoon et al. *EHR-Safe: generating high-fidelity and privacy-preserving synthetic electronic health records*. npj Digital Medicine 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10421926/
- [E6] Theodorou, Xiao, Sun. *Synthesize high-dimensional longitudinal electronic health records via hierarchical autoregressive language model* (HALO). Nature Communications 14, 5305 (2023). https://www.nature.com/articles/s41467-023-41093-0
- [E7] Wang, Sun. *PromptEHR: Conditional Electronic Healthcare Records Generation with Prompt Learning*. EMNLP 2022.（要原典確認）
- [E8] *SynEHRgy: Synthesizing Mixed-Type Structured Electronic Health Records using Decoder-Only Transformers*. 2024. https://arxiv.org/abs/2411.13428
- [E9] *Reliable generation of privacy-preserving synthetic electronic health record time series via diffusion models*. JAMIA 31(11), 2024. https://academic.oup.com/jamia/article/31/11/2529/7747780
- [E10] Chen, Wu, Shi, Cho, Mukherjee. *Generating synthetic electronic health record data: a methodological scoping review with benchmarking on phenotype data and open-source software* (SynthEHRella). JAMIA 2025. https://arxiv.org/abs/2411.04281 / https://github.com/chenxran/synthEHRella
- [E11] Jiang, Gao, Rasmussen, Xie, Sun. *Accelerating Reproducible Research in Synthetic EHR Generation*. 2026. https://arxiv.org/abs/2606.06990
- [E12] *Synthesizing Multimodal Electronic Health Records via Predictive Diffusion Models*. 2024. https://arxiv.org/abs/2406.13942
- [E13] *SynEHR: Joint Modeling Inter-visit Temporal Evolution and Intra-visit Clinical Structure for Longitudinal EHR Synthesis*. 2026. https://arxiv.org/abs/2608.21673
- [D1] Johnson et al. *MIMIC-IV Clinical Database Demo* (v2.2). PhysioNet, 2023. https://physionet.org/content/mimic-iv-demo/2.2/
- [D2] *eICU Collaborative Research Database Demo* (v2.0.1). PhysioNet, 2021. https://physionet.org/content/eicu-crd-demo/2.0.1/
