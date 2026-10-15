import numpy as np
import pytest

from conftest import make_vitals
from transyn import augment, vitals

SPO2 = vitals.VARS.index("sao2")
HR = vitals.VARS.index("heartrate")


def _flat(n=1):
    x = np.zeros((n, vitals.STEPS, len(vitals.VARS)))
    x[..., HR], x[..., SPO2] = 80.0, 97.0
    return x


def test_低酸素のラベルは後半の30分以上の低下で決まる():
    x = _flat(3)
    x[0, 80:86, SPO2] = 90.0  # 後半に 6 点（30 分）
    x[1, 80:85, SPO2] = 90.0  # 後半に 5 点
    x[2, 10:30, SPO2] = 85.0  # 前半だけ
    assert augment.LABELS["desat"](x).tolist() == [1, 0, 0]


def test_頻脈のラベルは後半の30分以上の頻脈で決まる():
    x = _flat(2)
    x[0, 100:106, HR] = 120.0
    x[1, 100:105, HR] = 120.0
    assert augment.LABELS["tachy"](x).tolist() == [1, 0]


def test_単純な変換の拡張は件数をk倍にしラベルを引き継ぐ():
    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).normal(80, 5, (10, augment.SPLIT, 3))
    y = np.arange(10) % 2
    xa, ya = augment.jitter(x, y, k=4, rng=rng)
    assert xa.shape == (40, augment.SPLIT, 3)
    assert ya.tolist() == np.tile(y, 4).tolist()
    assert not np.allclose(xa[:10], x)
    assert np.abs(xa[:10] - x).mean() < 2.0


def test_実データ換算の人数は対数で補間する():
    ns, auc = [20, 40, 80], [0.6, 0.7, 0.8]
    assert augment.real_equivalent_n(0.75, ns, auc) == pytest.approx(np.sqrt(40 * 80))
    assert augment.real_equivalent_n(0.7, ns, auc) == pytest.approx(40)
    assert np.isnan(augment.real_equivalent_n(0.85, ns, auc))
    assert np.isnan(augment.real_equivalent_n(0.55, ns, auc))


def test_1回の試行は全手法と全課題のAUROCを返す():
    pool = make_vitals(160, seed=3)
    data = augment.Data.of(pool)
    test_idx = np.arange(60)
    train_idx = np.arange(60, 160)
    ext_pool = augment.Data.of(make_vitals(80, seed=4))
    syn_ext = {"var1-ext": augment.Data.of(make_vitals(80, seed=5))}
    rows = augment.run_rep(pool, data, train_idx, test_idx, n=40, ext=ext_pool, syn_ext=syn_ext,
                           rng=np.random.default_rng(0), seed=0)
    methods = {r["method"] for r in rows}
    assert methods == {"real", "jitter", "var1-same", "var1-ext", "ext-real"}
    assert {r["task"] for r in rows} == set(augment.LABELS)
    assert all(np.isnan(r["auroc"]) or 0 <= r["auroc"] <= 1 for r in rows)
