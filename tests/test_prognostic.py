import numpy as np
import pandas as pd
import pytest

from conftest import make_vitals
from transyn import prognostic, vitals


def test_ベースラインと結果は系列の前半と後半から作る():
    x = np.zeros((3, vitals.STEPS, len(vitals.VARS)))
    x[:, prognostic.SPLIT :, 0] = 10.0
    assert prognostic.outcome(x, "mean") == pytest.approx(np.full(3, 10.0))
    assert prognostic.baseline_mean(x) == pytest.approx(np.zeros(3))


def test_ANCOVAは効果を推定し調整なしより標準誤差が小さい():
    rng = np.random.default_rng(0)
    n = 400
    cov = rng.normal(0, 10, n)
    treat = np.arange(n) % 2
    y = 2 * cov + 3 * treat + rng.normal(0, 1, n)
    adj = prognostic.ancova(y, treat, cov)
    unadj = prognostic.ancova(y, treat, None)
    assert adj.diff == pytest.approx(3, abs=0.3)
    assert adj.se < unadj.se / 5


def test_効果がなければp値は範囲内で効果が大きければ有意():
    pool = make_vitals(120)
    feats = prognostic.Features.of(pool)
    model = prognostic.fit_prognostic(feats, "mean", kind="linear")
    rng = np.random.default_rng(1)
    null = prognostic.run_trial(feats, n=60, delta=0.0, outcome_kind="mean", scores={"prog": model}, rng=rng)
    assert all(0 <= r["p"] <= 1 for r in null)
    big = prognostic.run_trial(feats, n=60, delta=30.0, outcome_kind="mean", scores={"prog": model}, rng=rng)
    assert all(r["p"] < 0.01 for r in big)
    assert {r["method"] for r in big} == {"unadjusted", "baseline", "prog"}


def test_病院の分け方は重ならず全病院を覆う():
    hosp = pd.Series(np.repeat(np.arange(10), np.arange(1, 11) * 5), name="hospitalid")
    ext, tgt = prognostic.split_hospitals(hosp, frac_external=0.5, seed=0)
    assert set(ext).isdisjoint(tgt)
    assert set(ext) | set(tgt) == set(range(10))
    share = hosp.isin(ext).mean()
    assert 0.3 < share < 0.7


def test_予後予測は持続する被験者差から後半を予測する():
    feats = prognostic.Features.of(make_vitals(200, seed=2))
    model = prognostic.fit_prognostic(feats, "mean", kind="linear")
    pred = model.predict(feats.x)
    assert np.corrcoef(pred, feats.y["mean"])[0, 1] > 0.8


def test_人数が実データより多いとエラー():
    feats = prognostic.Features.of(make_vitals(10))
    with pytest.raises(ValueError):
        prognostic.run_trial(feats, n=20, delta=0.0, outcome_kind="mean", scores={}, rng=np.random.default_rng(0))


def test_集計は第1種の過誤率と分散比と必要人数を出す():
    rows = []
    for rep in range(100):
        for method, se in [("unadjusted", 2.0), ("baseline", 1.0)]:
            rows.append({"outcome": "mean", "n": 100, "delta": 0.0, "rep": rep, "method": method, "se": se, "p": 0.5 if rep >= 5 else 0.01})
            rows.append({"outcome": "mean", "n": 100, "delta": 5.0, "rep": rep, "method": method, "se": se, "p": 0.01 if rep < 80 else 0.5})
    table = prognostic.summarize(pd.DataFrame(rows))
    base = table[table["method"] == "baseline"].iloc[0]
    assert base["type1"] == pytest.approx(0.05)
    assert base["power"] == pytest.approx(0.80)
    assert base["var_ratio"] == pytest.approx(0.25)
    # n80 = n × (se × (z_0.975 + z_0.8) / δ)²
    assert base["n80"] == pytest.approx(100 * (1.0 * 2.8016 / 5.0) ** 2, rel=1e-3)
