---
title: 数理モデルによるシミュレーションデータ生成の調査
kind: 調査
theme: シミュレーション
issues: [18]
date: 2026-10-07
order: 8
status: 完了
summary: 取引・EHR・生体信号のそれぞれに、数理モデルやシミュレータで合成データを作る系統がある。学習型の生成モデルと比べて、ラベルや因果を正確に持てて個人の記録を写さない一方、作り込んだ仕組みの外の多様性は出せず、実データとの較正が品質を決める。
---

# 数理モデルによるシミュレーションデータ生成の調査

issue #18の成果物。これまで比べてきた生成モデル（GAN、自己回帰モデルなど）は、実データから分布を学習して合成データを作る。ここでは別のアプローチとして、数理モデルやシミュレータで合成データを作る方法を、これまで扱った3つのテーマ（取引、EHR、生体信号・バイタル）ごとに調べた。

調査は2026-10時点のWeb検索に基づき、多くは論文やプロジェクトページの概要しか確認していない。「要原典確認」の項目は、検証に進むときに裏付ける。

## 要点

1. **3つのテーマすべてに、数理モデルで合成データを作る系統がある。** 取引ではエージェントベースのシミュレータ（PaySim、AMLSim、AMLworld）、EHRでは診療の流れを状態遷移で書くSynthea、生体信号では心電図の力学系モデル（ECGSYN）、循環・呼吸の生理エンジン（Pulse、BioGears）、血糖・インスリンのモデル（UVA/Padovaシミュレータ）がある。
2. **学習型の生成モデルにない強みは、正解ラベルと因果の構造を持てること。** AMLworldは資金洗浄の取引に正確なラベルを付けられ、UVA/Padovaシミュレータはインスリン投与の効果を前もって試せる。FDAはUVA/Padovaシミュレータを、閉ループ制御の前臨床試験で動物実験の代わりに使うことを認めた。
3. **個人の記録を写さないので、プライバシのリスクの性質が違う。** 学習型モデルではEHRの検証（#10）で学習データの系列の完全なコピーが出たが、シミュレータは個人の記録を入力にしない。ただし、実データに合わせてパラメータを較正すれば、較正に使った統計量（頻度、平均、遷移確率など）は合成データに反映される。
4. **弱点は、作り込んだ仕組みの外の多様性を出せないこと。** Syntheaの検証では、人口構成や医療サービスの提供確率はよく合ったが、診療後の転帰のばらつきは再現できなかった。PaySimも、新しい不正の手口は作り込まない限り出てこない。
5. **品質を決めるのは実データとの較正と検証。** PaySimとAMLworldは実データに合わせてパラメータを調整し、生理エンジンPulseはMIMIC-IIIの敗血症患者の経過と突き合わせて食い違いを洗い出す検証（因果的反証）が行われた。
6. **数理モデルと学習型モデルを組み合わせる研究が増えている。** 既知の微分方程式を潜在空間に組み込んだ生成モデル（GOKU-net）は、少ない学習データで生理的なパラメータを推定できる。ICUバイタルの検証（#15）で単純な統計モデル（VAR(1)）が深層モデルを上回ったことも、データが少ないときは構造を仮定したモデルが有利という同じ方向の結果と言える。

## 学習型の生成モデルとの比較

| 観点 | 学習型の生成モデル（GAN、自己回帰など） | 数理モデル・シミュレータ |
|---|---|---|
| 作り方 | 実データの分布を学習する | 仕組み（行動規則、状態遷移、微分方程式）を書き、パラメータを決める |
| 実データの必要量 | 多い。少ないと学習できない（#15） | 較正に使う統計量があればよい。実データなしで作るものもある（AMLSim） |
| 忠実度 | 学習データの分布には近づくが、崩壊や見落としが起きる | 作り込んだ仕組みの範囲では整合的。仕組みの外の多様性は出ない |
| ラベル・因果 | 学習データにある関係しか持てず、弱い関係は引き継げないこともある（#10、#15） | 正確なラベルと、介入（投薬、制御）への応答を持てる |
| プライバシ | 学習データの記録を写すことがある（#10の完全一致） | 個人の記録は写さない。較正に使った統計量は反映される |
| 検証の方法 | 実データとの分布の比較、TSTR、メンバーシップ推論 | 実データとの突き合わせ（臨床指標、経過）、反証テスト |
| 向いている用途 | 実データの代わりとして分析・学習に使う | アルゴリズムの試験、まれな事象や介入の検討、教育 |

## テーマごとの系統

### 取引

| 名前 | 方式 | 較正・検証 | 特徴 |
|---|---|---|---|
| PaySim [T1] | エージェントベース。モバイル送金の利用者の行動を規則で表す | アフリカの国のモバイル送金サービスの実データのサンプルからパラメータを推定 | 不正取引のラベル付きデータを作れる。新しい手口は作り込まないと出ない |
| AMLSim [T2] | マルチエージェント。既知の資金洗浄の型（typology）を取引ネットワークに埋め込む | 参照データなしで動く | 資金洗浄の検出に特化。IBMが公開 |
| AMLworld [T3] | マルチエージェントの仮想世界。犯罪者の収入から資金洗浄の3段階（placement、layering、integration）と8つの型を生成 | 実取引に近づけるよう較正 | 資金洗浄の取引に完全な正解ラベルを付けられる。HI-Smallは約51.5万口座・約500万取引・約10日分で、資金洗浄は約1,000件に1件 |
| AMLgentex [T4] | 資金洗浄の研究用のデータ生成・評価の枠組み | 要原典確認 | 2025年の新しい取り組み |

### EHR

| 名前 | 方式 | 較正・検証 | 特徴 |
|---|---|---|---|
| Synthea [H1] | 疾患・診療の流れを状態遷移（モジュール）で書き、人口統計に沿って仮想患者の生涯の記録を作る | 公開統計に基づいてモジュールを作成。臨床品質指標で検証された [H2] | 実データを使わずに大規模なEHRを作れる。人口構成や医療サービスの提供確率はよく合うが、診療後の転帰のばらつきは再現できない（大腸がん検診の実施率は合成63%、マサチューセッツ州の公表値77.3%） |

### 生体信号・バイタル

| 名前 | 方式 | 較正・検証 | 特徴 |
|---|---|---|---|
| ECGSYN [P1] | 3つの連立常微分方程式で心電図波形を作る力学系モデル | 心拍数の平均・ばらつき、PQRST波形の形、心拍間隔のパワースペクトルを指定 | 心拍ごとの形と間隔の揺らぎ、呼吸性不整脈などを再現。信号処理手法の評価に使われる |
| Pulse Physiology Engine [P2] | 循環・呼吸などを集中定数モデル（電気回路の類比）で表す全身の生理シミュレータ。BioGearsから分岐 | 心拍数・呼吸数・SpO2などの臨床指標で検証。MIMIC-IIIの敗血症患者11,677人の経過と突き合わせた因果的反証で食い違いを洗い出した [P3] | 年齢・性別・バイタルの初期値と病態（敗血症、COPD、ARDSなど）を与え、輸液や昇圧薬などの介入を加えながら経過を作れる |
| BioGears [P4] | 全身の生理エンジン。Pulseの元 | 要原典確認 | 医療訓練向け |
| UVA/Padova T1Dシミュレータ [P5] | 消化管・糖動態・インスリン動態のサブモデルからなる糖代謝モデル | 臨床試験の血糖推移と一致することを確認。2008年にFDAが閉ループ制御の前臨床試験での利用を認めた | 仮想患者（成人・青年・小児各100人から始まり、現在は6,000人超のデジタルツイン）で、インスリン療法を前もって試せる |

### 数理モデルと学習型モデルの組み合わせ

| 名前 | 方式 | 特徴 |
|---|---|---|
| GOKU-net [X1] | 変分オートエンコーダの潜在空間に、既知の常微分方程式（循環系モデルなど）を組み込む | 観測されない生理パラメータを推定でき、少ない学習データでも外挿が良い |
| シミュレータの較正・反証 [P3] | 実データの経過と突き合わせ、シミュレータの食い違いを統計的に検出する | 合成データの品質を、分布の一致ではなく仮説の反証として検証する |

## これまでの検証とのつながり

- ICUバイタル（#15）では、滞在ごとの性質を正規分布で引き、時間変化をVAR(1)で作る単純な統計モデルが、深層モデルより多くの指標で良かった。これはごく小さな数理モデルを実データで較正したものと見なせる。生理エンジンを使えば、心拍数・呼吸数・SpO2の間の生理的な関係を仕組みとして持たせられる。
- EHR（#10）では、学習型モデルが学習データの系列を写した。シミュレータは個人の記録を入力にしないので、このリスクがない。共有したいのが「現実らしい構造」であって個々の記録ではない場合、シミュレータは有力な選択肢になる。
- 一方、どのシミュレータも、作り込んだ仕組みの外の多様性（まれな経過、新しい手口、施設ごとの癖）は出せない。学習型モデルで実データの多様性を補う組み合わせが考えられる。

## 検証に進む場合の候補

| 候補 | 内容 | 比べる相手 |
|---|---|---|
| 生理エンジンでバイタルを作る | Pulseで、年齢・病態・初期バイタルを実データから引いて経過を作り、#15と同じ指標で評価する | #15のVAR(1)、PAR、DGAN |
| 取引のエージェントベースシミュレータ | PaySimやAMLworldの考え方で、Czechの口座の取引を規則で作り、#2の指標で評価する | #2のCTGAN、PAR、DGAN、Banksformer |
| Syntheaの記録をeICU Demoと比べる | ICUに関係するモジュールの出力を、#10のイベント系列の指標で比べる | #10のGPT型など |
| 心電図の力学系モデル | ECGSYNで作った波形を、MIMIC-IV-ECG Demo（誰でも入手可）と比べる | 学習型の心電図生成モデル |

## 文献リスト

### 取引

- [T1] Lopez-Rojas, Elmir, Axelsson. *PaySim: A financial mobile money simulator for fraud detection*. EMSS 2016. https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf
- [T2] IBM AMLSim（資金洗浄のマルチエージェントシミュレータ）。要原典確認
- [T3] Altman et al. *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models* (AMLworld). NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2306.16424
- [T4] *AMLgentex: Mobilizing Data-Driven Research to Combat Money Laundering*. 2025. https://arxiv.org/abs/2506.13989

### EHR

- [H1] Walonoski et al. *Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record*. JAMIA 25(3), 2018. https://academic.oup.com/jamia/article/25/3/230/4098271
- [H2] Chen et al. *The validity of synthetic clinical data: a validation study of a leading synthetic data generator (Synthea) using clinical quality measures*. BMC Medical Informatics and Decision Making 19, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6416981/

### 生体信号・バイタル

- [P1] McSharry, Clifford, Tarassenko, Smith. *A dynamical model for generating synthetic electrocardiogram signals* (ECGSYN). IEEE Transactions on Biomedical Engineering 50(3), 2003. https://physionet.org/content/ecgsyn/1.0.0/
- [P2] Kitware. *Pulse Physiology Engine*. https://pulse.kitware.com/
- [P3] Cornish et al. *Causal Falsification of Digital Twins*. 2023. https://arxiv.org/abs/2301.07210
- [P4] ARA. *BioGears*. https://www.ara.com/biogears/
- [P5] Man et al. *The UVA/PADOVA Type 1 Diabetes Simulator: New Features*. Journal of Diabetes Science and Technology, 2014。臨床試験の血糖推移との一致は *The University of Virginia/Padova Type 1 Diabetes Simulator Matches the Glucose Traces of a Clinical Trial*（https://pubmed.ncbi.nlm.nih.gov/24571584/ ）。FDAの受け入れはKovatchev et al.（Journal of Diabetes Science and Technology, 2009）が報告している（要原典確認）

### 組み合わせ

- [X1] Linial, Ravid, Eytan, Shalit. *Generative ODE Modeling with Known Unknowns* (GOKU-net). ACM CHIL 2021. https://arxiv.org/abs/2003.10775
