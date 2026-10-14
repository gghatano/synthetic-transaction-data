import numpy as np
import pytest

from transyn import syninf_tabular as tab


def test_疾患データは決まった列と分布を持つ():
    df = tab.sample_disease(5000, seed=0)
    assert list(df.columns) == ["age", "stage", "biomarker", "therapy", "death"]
    assert df["age"].mean() == pytest.approx(50, abs=0.6)
    assert df["therapy"].mean() == pytest.approx(0.5, abs=0.03)
    assert set(df["stage"]) <= {0, 1, 2, 3}
    assert tab.sample_disease(10, seed=1).equals(tab.sample_disease(10, seed=1))


def test_ロジスティック回帰は真の係数を推定し標準誤差を返す():
    df = tab.sample_disease(40000, seed=2)
    est = tab.estimate(df)
    assert est["death_therapy"].value == pytest.approx(tab.TRUTH["death_therapy"], abs=0.08)
    assert 0 < est["death_therapy"].se < 0.1
    assert est["age_mean"].value == pytest.approx(50, abs=0.2)


def test_パラメトリックな合成器は構造を保つ():
    df = tab.sample_disease(20000, seed=3)
    gen = tab.fit_generate_parametric(df, m=20000, seed=0)
    assert list(gen.columns) == list(df.columns)
    assert tab.estimate(gen)["death_therapy"].value == pytest.approx(tab.TRUTH["death_therapy"], abs=0.12)


def test_コピーする生成器なら合成と実データの推定が一致する():
    row = tab.run_rep(n=200, model_fn=lambda df, m, seed: df.copy(), seed=0)
    assert row["death_therapy_syn"] == pytest.approx(row["death_therapy_real"])
    assert row["death_therapy_reject_naive"] == row["death_therapy_reject_real"]
    # 補正で標準誤差を √2 倍にすると、棄却は増えない
    assert row["death_therapy_reject_corrected"] <= row["death_therapy_reject_naive"]
