import math

import numpy as np
import pytest

from conftest import make_vitals
from transyn import dpsynth, vitals


@pytest.fixture
def train():
    df = make_vitals(200, seed=0, shift_group=15.0)
    return df.assign(sao2=df["sao2"].clip(upper=100.0))


def test_εからρへの換算はzCDPの式を満たす():
    rho = dpsynth.rho_from_eps(1.0, 1e-5)
    assert rho + 2 * math.sqrt(rho * math.log(1e5)) == pytest.approx(1.0)
    assert dpsynth.rho_from_eps(float("inf"), 1e-5) == float("inf")


def test_雑音の大きさはεが小さいほど大きい():
    s1 = dpsynth.noise_scales(1.0, 1e-5)
    s8 = dpsynth.noise_scales(8.0, 1e-5)
    assert all(s1[k] > s8[k] > 0 for k in s1)
    assert all(v == 0 for v in dpsynth.noise_scales(float("inf"), 1e-5).values())


@pytest.mark.parametrize("level", ["L3", "L4"])
def test_DPの見本は形がそろい公開の範囲に収まる(train, level):
    gen = dpsynth.generate_dp(level, train, n_seqs=50, seed=0, epsilon=1.0)
    assert gen["seq_id"].nunique() == 50
    assert (gen.groupby("seq_id").size() == vitals.STEPS).all()
    for v in vitals.VARS:
        lo, hi = vitals.BOUNDS[v]
        assert gen[v].between(lo, hi).all()


def test_雑音なしなら被験者ごとの平均の分布を再現する(train):
    gen = dpsynth.generate_dp("L3", train, n_seqs=2000, seed=0, epsilon=float("inf"))
    real = train.groupby("seq_id")["heartrate"].mean()
    syn = gen.groupby("seq_id")["heartrate"].mean()
    assert syn.mean() == pytest.approx(real.mean(), abs=1.5)
    assert syn.std() == pytest.approx(real.std(), rel=0.15)


def test_L4は死亡退院との関係を持ちL3は持たない(train):
    def diff(df):
        m = df.groupby("seq_id").agg(hr=("heartrate", "mean"), d=("died", "first"))
        return m.loc[m.d == 1, "hr"].mean() - m.loc[m.d == 0, "hr"].mean()

    l4 = dpsynth.generate_dp("L4", train, n_seqs=2000, seed=0, epsilon=float("inf"))
    l3 = dpsynth.generate_dp("L3", train, n_seqs=2000, seed=0, epsilon=float("inf"))
    assert diff(l4) > 8.0
    assert abs(diff(l3)) < 3.0


def test_監査は標的ごとにεと攻撃の結果を返す(train):
    rows = dpsynth.audit(train, epsilons=(1.0, float("inf")), n=20, k=1, reps=20, seed=0)
    assert {r["epsilon"] for r in rows} == {1.0, float("inf")}
    assert all({"auc", "eps_lb", "kind"} <= set(r) for r in rows)
