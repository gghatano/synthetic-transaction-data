import numpy as np
import pytest

from conftest import make_vitals
from transyn import smalln


def test_学習人数ぶんの被験者を再現可能に選ぶ():
    a = make_vitals(50)
    ids1 = smalln.choose_ids(a, n=20, seed=0)
    ids2 = smalln.choose_ids(a, n=20, seed=0)
    assert len(ids1) == 20 and list(ids1) == list(ids2)
    assert set(ids1) <= set(a["seq_id"])
    with pytest.raises(ValueError):
        smalln.choose_ids(a, n=60, seed=0)


def test_DGANのエポック数は更新回数をそろえる():
    assert smalln.epochs_for_updates(n=716, batch=64, updates=24000) == 2000
    assert smalln.epochs_for_updates(n=20, batch=64, updates=24000) == 24000
    assert smalln.epochs_for_updates(n=100, batch=64, updates=24000) == 12000


def test_学習データのコピーなら外れた被験者の近接比は0():
    pool = make_vitals(120, seed=1)
    train = pool[pool["seq_id"] < 60]
    holdout = pool[pool["seq_id"] >= 60]
    ratio = smalln.outlier_proximity(train, holdout, gen=train.copy(), reference=pool)
    assert ratio == pytest.approx(0.0)


def test_無関係な実データなら外れた被験者の近接比は0より大きい():
    pool = make_vitals(180, seed=2)
    train = pool[pool["seq_id"] < 60]
    holdout = pool[(pool["seq_id"] >= 60) & (pool["seq_id"] < 120)]
    other = pool[pool["seq_id"] >= 120]
    ratio = smalln.outlier_proximity(train, holdout, gen=other, reference=pool)
    assert np.isfinite(ratio) and ratio > 0.3
