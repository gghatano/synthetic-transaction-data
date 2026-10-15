import numpy as np
import pytest

from conftest import make_vitals
from transyn import attacks, showcase, vitals


@pytest.fixture
def pool():
    df = make_vitals(120, seed=0)
    return df.assign(sao2=df["sao2"].clip(upper=100.0))


def test_εの下限は区別できなければ0で完全に分かれれば正():
    rng = np.random.default_rng(0)
    same = attacks.eps_lower_bound(rng.normal(size=200), rng.normal(size=200))
    assert same == pytest.approx(0.0, abs=0.3)
    split = attacks.eps_lower_bound(np.full(200, 1.0), np.full(200, 0.0))
    assert split > 2.0 and np.isfinite(split)


def test_標的は平均的な被験者と外れた被験者に分かれる(pool):
    avg, out = attacks.choose_targets(pool, k=3)
    assert len(avg) == 3 and len(out) == 3 and not set(avg) & set(out)
    score = attacks.outlier_scores(pool)
    assert score[out].min() > score[avg].max()


def test_学習データを返す生成器なら狙い撃ちは当たりL0なら当たらない(pool):
    _, out = attacks.choose_targets(pool, k=1)

    def copy(level, train, n_seqs, seed):
        return train.copy()

    hit = attacks.targeted_mia(pool, out[0], level="L3", n=20, reps=40, seed=0, generate=copy)
    miss = attacks.targeted_mia(pool, out[0], level="L0", n=20, reps=40, seed=0)
    assert hit["auc"] > 0.9
    assert abs(miss["auc"] - 0.5) < 0.25


def test_属性推論は学習した人としていない人の差を返す(pool):
    train, holdout = pool[pool["seq_id"] < 60], pool[pool["seq_id"] >= 60]
    gen = showcase.generate("L4", train, n_seqs=100, seed=0)
    r = attacks.attribute_inference(gen, train, holdout, seed=0)
    assert set(r) == {"auroc_members", "auroc_nonmembers", "advantage"}
    assert r["advantage"] == pytest.approx(r["auroc_members"] - r["auroc_nonmembers"])


def test_部分系列の照合は写した区間を距離0で見つける(pool):
    train = pool[pool["seq_id"] < 60]
    gen = showcase.generate("L3", train, n_seqs=20, seed=0)
    target = train[train["seq_id"] == 5].sort_values("t")
    leaked = gen.copy()
    rows = (leaked["seq_id"] == leaked["seq_id"].iloc[0]) & leaked["t"].between(40, 51)
    leaked.loc[rows, vitals.VARS] = target[target["t"].between(40, 51)][vitals.VARS].to_numpy()
    window = target[target["t"].between(40, 51)][vitals.VARS].to_numpy()
    assert attacks.nearest_window_distance(window, leaked) == pytest.approx(0.0, abs=1e-9)
    assert attacks.nearest_window_distance(window, gen) > 0
