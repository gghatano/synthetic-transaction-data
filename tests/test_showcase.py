import numpy as np
import pandas as pd
import pytest

from conftest import make_vitals
from transyn import showcase, vitals


@pytest.fixture
def train():
    # テスト用の系列は SpO2 が 100% を超えることがあるので、実データと同じく 100% 以下にする
    df = make_vitals(60, seed=0)
    return df.assign(sao2=df["sao2"].clip(upper=100.0))


@pytest.mark.parametrize("level", showcase.LEVELS)
def test_どの段階も同じ形の系列を作る(train, level):
    gen = showcase.generate(level, train, n_seqs=30, seed=0)
    assert list(gen.columns) == ["seq_id", "t", *vitals.VARS, "age", "died"]
    assert gen["seq_id"].nunique() == 30
    assert (gen.groupby("seq_id").size() == vitals.STEPS).all()
    assert np.isfinite(gen[vitals.VARS].to_numpy()).all()


def test_L0は学習データを使わない(train):
    a = showcase.generate("L0", train, n_seqs=20, seed=0)
    b = showcase.generate("L0", train.assign(heartrate=train["heartrate"] + 50), n_seqs=20, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_L1は時点間の相関を持たない(train):
    gen = showcase.generate("L1", train, n_seqs=200, seed=0)
    z = vitals.to_array(gen)[..., 0]
    z = z - z.mean(1, keepdims=True)
    lag1 = (z[:, 1:] * z[:, :-1]).sum() / (z**2).sum()
    assert abs(lag1) < 0.05


def test_L2は被験者ごとの違いを持たずL3は持つ(train):
    l2 = vitals.to_array(showcase.generate("L2", train, n_seqs=200, seed=0))[..., 0].mean(1)
    l3 = vitals.to_array(showcase.generate("L3", train, n_seqs=200, seed=0))[..., 0].mean(1)
    real = vitals.to_array(train)[..., 0].mean(1)
    assert l2.std() < 0.5 * real.std()
    assert l3.std() > 0.7 * real.std()


def test_合成である旨の表示と実在しないID(train):
    gen = showcase.generate("L3", train, n_seqs=10, seed=0)
    labelled, meta = showcase.label(gen, level="L3")
    assert labelled["seq_id"].str.startswith("SYN-").all()
    assert meta["synthetic"] is True and meta["level"] == "L3"
    assert set(meta) >= {"preserved", "not_preserved", "generator", "intended_use"}


def test_開示チェックは実データの区間の完全一致を見つける(train):
    gen = showcase.generate("L3", train, n_seqs=10, seed=0)
    leaked = gen.copy()
    src = train[train["seq_id"] == 0].sort_values("t")
    mask = (leaked["seq_id"] == leaked["seq_id"].iloc[0]) & leaked["t"].between(30, 49)
    leaked.loc[mask, vitals.VARS] = src[src["t"].between(30, 49)][vitals.VARS].to_numpy()
    assert showcase.exact_segment_matches(gen, train, length=12) == 0
    assert showcase.exact_segment_matches(leaked, train, length=12) >= 1


def test_開示チェックは実データの極値の再現と少数の群を数える(train):
    gen = train.copy()
    r = showcase.disclosure_checks(gen, train, holdout=train, min_group=10)
    assert r["extreme_values_reproduced"] > 0
    small = train.assign(died=(train["seq_id"] < 3).astype(int))
    assert showcase.disclosure_checks(small, small, holdout=small, min_group=10)["small_groups"] == ["died=1"]


def test_構造チェックは列と時点数と範囲を確かめる(train):
    gen = showcase.generate("L3", train, n_seqs=10, seed=0)
    ok = showcase.structure_checks(gen, train)
    assert all(ok.values())
    broken = gen[gen["t"] < 100]
    assert not showcase.structure_checks(broken, train)["steps"]
