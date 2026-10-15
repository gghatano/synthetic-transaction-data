import numpy as np
import pytest

from test_ecglink import toy_beats
from transyn import ecg, ecgdefense, ecglink


@pytest.fixture
def setup():
    bs = toy_beats(n_persons=24, seed=7)
    mem, non = ecglink.split_persons(bs, seed=0)
    return bs, mem, non, bs.select(person=mem, part="gen")


def _person_means(syn):
    return np.array([syn.beats[syn.person == p].mean(0) for p in syn.persons])


def test_k人の平均はk1で本人ごとkが全員なら全員同じ形(setup):
    bs, mem, _, train = setup
    k1 = ecgdefense.generate("E3-kanon", train, len(mem), 200, seed=0, k=1)
    kall = ecgdefense.generate("E3-kanon", train, len(mem), 200, seed=0, k=len(mem))
    assert _person_means(kall).std(0).mean() < 0.3 * _person_means(k1).std(0).mean()
    assert ecglink.linkage_attack(k1, bs.select(part="enroll"), mem)["source_top1"] > 0.8


def test_雑音を足すと拍が元から離れる(setup):
    _, mem, _, train = setup
    a = ecgdefense.generate("E3-noise", train, len(mem), 50, seed=0, sigma=0.0)
    b = ecgdefense.generate("E3-noise", train, len(mem), 50, seed=0, sigma=0.3)
    assert b.beats.std() > a.beats.std()


def test_除外は外れ具合の上位の人を学習から除く(setup):
    _, _, _, train = setup
    kept = ecgdefense.trim_persons(train, 0.25)
    score = ecglink.person_outlier_scores(train)
    dropped = set(train.persons) - set(kept.persons)
    assert len(dropped) == round(0.25 * len(train.persons))
    assert score[list(dropped)].min() >= score.drop(list(dropped)).max()


def test_差分プライバシー版は形がそろいεが小さいほど雑音が大きい(setup):
    _, mem, _, train = setup
    s_inf = ecgdefense.generate("E2-dp", train, len(mem), 30, seed=0, epsilon=float("inf"))
    assert s_inf.beats.shape[1] == ecg.BEAT_LEN and len(s_inf.persons) == len(mem)
    lo = ecgdefense.dp_noise_scales(1.0, 1e-5)
    hi = ecgdefense.dp_noise_scales(8.0, 1e-5)
    assert all(lo[k] > hi[k] > 0 for k in lo)


def test_対策の実験は設定ごとに紐づきとそれっぽさを返す():
    bs = toy_beats(n_persons=16, seed=8)
    configs = [("kanon-k2", "E3-kanon", {"k": 2}), ("dp-inf", "E2-dp", {"epsilon": float("inf")})]
    rows = ecgdefense.experiment(bs, seeds=(0,), configs=configs, beats_per_person=20)
    assert {r["config"] for r in rows} == {"kanon-k2", "dp-inf"}
    assert all({"auc", "mmd", "disc"} <= set(r) for r in rows)
