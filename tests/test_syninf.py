import math

import numpy as np
import pytest

from conftest import make_vitals
from transyn import syninf


def test_補正係数はMとnの比から決まる():
    assert syninf.correction_factor(m=100, n=100) == pytest.approx(math.sqrt(2))
    assert syninf.correction_factor(m=500, n=100) == pytest.approx(math.sqrt(6))


def test_群ラベルは人数がそろい被験者内で一定(toy_vitals):
    df = syninf.assign_groups(toy_vitals, np.random.default_rng(0))
    per_seq = df.groupby("seq_id")["died"].agg(["nunique", "first"])
    assert (per_seq["nunique"] == 1).all()
    counts = per_seq["first"].value_counts()
    assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1
    assert len(df) == len(toy_vitals)


def test_被験者ごとの平均は滞在数と同じ長さ(toy_vitals):
    means = syninf.subject_means(toy_vitals)
    assert len(means) == toy_vitals["seq_id"].nunique()
    assert set(means.columns) == {"value", "group"}


def test_t検定は明らかな差を有意にする():
    df = make_vitals(60, shift_group=30.0)
    assert syninf.ttest_subjects(df).p < 0.01


def test_補正係数を掛けるとp値が大きくなる():
    df = make_vitals(60, shift_group=5.0)
    assert syninf.ttest_subjects(df, se_factor=2.0).p > syninf.ttest_subjects(df).p


def test_記録単位の検定は自由度が記録数に基づく(toy_vitals):
    res = syninf.ttest_records(toy_vitals)
    assert 0.0 <= res.p <= 1.0
    assert res.dof == len(toy_vitals) - 2


def test_学習データをそのまま返す生成器なら合成と実データの検定が一致する():
    pool = make_vitals(60)

    def copy_model(train, n_seqs, epochs, seed):
        return train.copy()

    row = syninf.run_rep(pool, n=40, m=40, model_fn=copy_model, seed=3, epochs=0)
    assert row["p_syn_naive"] == pytest.approx(row["p_real_subject"])
    assert row["diff_syn"] == pytest.approx(row["diff_real"])
    assert row["sd_ratio"] == pytest.approx(1.0)


def test_人数が少なすぎるとエラー():
    pool = make_vitals(10)
    with pytest.raises(ValueError):
        syninf.run_rep(pool, n=2, m=10, model_fn=lambda *a, **k: None, seed=0, epochs=0)
    with pytest.raises(ValueError):
        syninf.run_rep(pool, n=20, m=10, model_fn=lambda *a, **k: None, seed=0, epochs=0)
