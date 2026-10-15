import numpy as np
import pandas as pd
import pytest

from transyn import vitals


def make_vitals(n_seqs: int, seed: int = 0, shift_group: float = 0.0) -> pd.DataFrame:
    """vitals と同じ long 形式の小さなデータ。滞在ごとの平均（被験者差）と AR(1) の変動を持つ。

    shift_group を与えると、died=1 の滞在の心拍数をその分だけ上げる。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_seqs):
        died = i % 2
        base = np.array([85.0, 96.0, 18.0]) + rng.normal(0, [12, 2, 4]) + np.array([shift_group, 0, 0]) * died
        z = np.zeros(3)
        for t in range(vitals.STEPS):
            z = 0.9 * z + rng.normal(0, [2, 0.5, 1])
            rows.append({"seq_id": i, "t": t, **dict(zip(vitals.VARS, base + z)), "age": 60 + i % 20, "died": died})
    return pd.DataFrame(rows)


@pytest.fixture
def toy_vitals() -> pd.DataFrame:
    return make_vitals(40)
