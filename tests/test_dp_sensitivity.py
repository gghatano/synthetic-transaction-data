"""差分プライバシーの感度の上界が実装で成り立つことを確かめる（GHE issue 22）。

被験者 1 人を別の人（極端な値・一定の値・まれな形の人を含む）に入れ替えたとき、雑音を加える前の各総和の
変化の L2 ノルム（行列はフロベニウスノルム）が、決めた感度を超えないこと。
"""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_vitals
from test_ecglink import toy_beats
from transyn import dpsynth, ecg, ecgdefense, ecglink, vitals


def _replace_subject(df: pd.DataFrame, victim, replacement: pd.DataFrame) -> pd.DataFrame:
    rep = replacement.assign(seq_id=victim)
    return pd.concat([df[df["seq_id"] != victim], rep], ignore_index=True)


def _weird_subjects(rng):
    """入れ替え先の被験者。生理的にありえる範囲の端・一定の値・激しく揺れる値・極端な年齢と死亡退院。"""
    t = np.arange(vitals.STEPS)
    cases = []
    for hr, sp, rr, age, died in [
        (np.full(vitals.STEPS, 250.0), np.full(vitals.STEPS, 50.0), np.full(vitals.STEPS, 60.0), 90, 1),
        (np.full(vitals.STEPS, 20.0), np.full(vitals.STEPS, 100.0), np.full(vitals.STEPS, 3.0), 18, 0),
        (np.where(t % 2, 250.0, 20.0), np.where(t % 2, 100.0, 50.0), np.where(t % 2, 60.0, 3.0), 50, 1),
        (rng.uniform(20, 250, vitals.STEPS), rng.uniform(50, 100, vitals.STEPS), rng.uniform(3, 60, vitals.STEPS), 120, 1),
    ]:
        cases.append(pd.DataFrame({"seq_id": 0, "t": t, "heartrate": hr, "sao2": sp, "respiration": rr, "age": age, "died": died}))
    return cases


def _norm(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))


def test_バイタルのDPの総和は1人を入れ替えても感度の上界に収まる():
    rng = np.random.default_rng(0)
    base = make_vitals(30, seed=1)
    s0 = dpsynth.query_sums(base)
    worst = {k: 0.0 for k in dpsynth.SENSITIVITY}
    for victim in (0, 7, 29):
        for rep in _weird_subjects(rng):
            s1 = dpsynth.query_sums(_replace_subject(base, victim, rep))
            for k in dpsynth.SENSITIVITY:
                worst[k] = max(worst[k], _norm(s0[k], s1[k]))
    for k, d in dpsynth.SENSITIVITY.items():
        assert worst[k] <= d + 1e-9, (k, worst[k], d)


def test_バイタルのDPは雑音の大きさがρの配分と一致する():
    eps, delta = 2.0, 1e-5
    s = dpsynth.noise_scales(eps, delta)
    rho = sum(dpsynth.SENSITIVITY[k] ** 2 / (2 * s[k] ** 2) for k in s)
    assert rho == pytest.approx(dpsynth.rho_from_eps(eps, delta))
    assert rho + 2 * math.sqrt(rho * math.log(1 / delta)) == pytest.approx(eps)


def _weird_persons(rng):
    """入れ替え先の心電図の被験者。振幅の大きい拍・平らな拍・雑音だけの拍・心拍間隔の端。"""
    n = 30
    t = np.linspace(0, 1, ecg.BEAT_LEN)
    return [
        (np.tile(20 * np.exp(-((t - 0.36) ** 2) / 0.001), (n, 1)), np.full(n, 0.2)),
        (np.zeros((n, ecg.BEAT_LEN)), np.full(n, 5.0)),
        (rng.normal(0, 10, (n, ecg.BEAT_LEN)), rng.uniform(0.1, 3.0, n)),
    ]


def _replace_person(bs, victim, beats, rr):
    keep = bs.person != victim
    return ecglink.BeatSet(np.vstack([bs.beats[keep], beats]), np.r_[bs.person[keep], [victim] * len(beats)],
                           np.r_[bs.part[keep], ["gen"] * len(beats)], np.r_[bs.rr[keep], rr],
                           np.r_[bs.symbol[keep], ["N"] * len(beats)])


def test_心電図のDPの総和は1人を入れ替えても感度の上界に収まる():
    rng = np.random.default_rng(0)
    base = toy_beats(n_persons=12, seed=2).select(part="gen")
    s0 = ecgdefense.dp_sums(base)
    worst = {k: 0.0 for k in ecgdefense.DP_SENSITIVITY}
    for victim in base.persons[[0, 5, 11]]:
        for beats, rr in _weird_persons(rng):
            s1 = ecgdefense.dp_sums(_replace_person(base, victim, beats, rr))
            for k in ecgdefense.DP_SENSITIVITY:
                worst[k] = max(worst[k], _norm(s0[k], s1[k]))
    for k, d in ecgdefense.DP_SENSITIVITY.items():
        assert worst[k] <= d + 1e-9, (k, worst[k], d)


def test_DPの生成は雑音を加えた総和だけを使う(monkeypatch):
    """生成の関数が、雑音を加える前の個々の被験者のデータを、総和の計算以外で読まないこと。
    総和を固定値に差し替えると、学習データを変えても生成結果が変わらない。"""
    a = make_vitals(30, seed=1)
    b = make_vitals(30, seed=9)
    fixed = dpsynth.query_sums(a)
    monkeypatch.setattr(dpsynth, "query_sums", lambda train: fixed)
    ga = dpsynth.generate_dp("L4", a, n_seqs=10, seed=0, epsilon=1.0)
    gb = dpsynth.generate_dp("L4", b, n_seqs=10, seed=0, epsilon=1.0)
    pd.testing.assert_frame_equal(ga, gb)


def test_心電図のDPの生成は雑音を加えた総和だけを使う(monkeypatch):
    a = toy_beats(n_persons=12, seed=2).select(part="gen")
    b = toy_beats(n_persons=12, seed=3).select(part="gen")
    fixed = ecgdefense.dp_sums(a)
    monkeypatch.setattr(ecgdefense, "dp_sums", lambda train: fixed)
    ga = ecgdefense.generate("E2-dp", a, 6, 10, seed=0, epsilon=1.0)
    gb = ecgdefense.generate("E2-dp", b, 6, 10, seed=0, epsilon=1.0)
    np.testing.assert_allclose(ga.beats, gb.beats)
    np.testing.assert_allclose(ga.rr, gb.rr)
