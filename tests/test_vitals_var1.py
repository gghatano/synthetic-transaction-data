import numpy as np
import pandas as pd

from conftest import make_vitals
from transyn import vitals


def test_死亡退院が1人しかいなくてもVAR1は生成できる():
    df = make_vitals(20)
    df["died"] = (df["seq_id"] == 0).astype(int)
    gen = vitals.fit_generate_var1(df, n_seqs=30, epochs=0, seed=0)
    assert gen["seq_id"].nunique() == 30
    assert np.isfinite(gen[vitals.VARS].to_numpy()).all()


def test_群が十分に大きいときは群ごとの共分散を使う(monkeypatch):
    """群の人数が多い場合は従来どおり np.cov(group.T) を使う（公開リポジトリの #15 の結果を変えない）。"""
    df = make_vitals(60)
    seen = []
    orig = np.cov

    def spy(m, *a, **k):
        seen.append(np.asarray(m).shape)
        return orig(m, *a, **k)

    monkeypatch.setattr(vitals.np, "cov", spy)
    vitals.fit_generate_var1(df, n_seqs=5, epochs=0, seed=0)
    # 雑音の共分散 1 回 + 群ごとの滞在の性質（7 次元 × 30 人）の共分散を生成ごとに計算
    assert (7, 30) in seen
