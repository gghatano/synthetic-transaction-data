import numpy as np
import pytest

from transyn import ecg, ecglink


def toy_beats(n_persons=24, beats=30, seed=0):
    """人ごとに山の位置と高さが違う拍。生成用（gen）と登録用（enroll）を同じ人から作る。"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, ecg.BEAT_LEN)
    rows, persons, parts, rr = [], [], [], []
    for p in range(n_persons):
        c, h, w = rng.uniform(0.2, 0.8), rng.uniform(0.5, 1.5), rng.uniform(0.03, 0.08)
        tmpl = h * np.exp(-((t - c) ** 2) / (2 * w**2)) + 0.3 * np.exp(-((t - 0.85) ** 2) / 0.005)
        for part in ("gen", "enroll"):
            for _ in range(beats):
                rows.append(tmpl + rng.normal(0, 0.05, ecg.BEAT_LEN))
                persons.append(f"p{p:02d}")
                parts.append(part)
                rr.append(rng.normal(0.8 + 0.01 * p, 0.03))
    return ecglink.BeatSet(np.array(rows), np.array(persons), np.array(parts), np.array(rr), np.full(len(rows), "N"))


@pytest.fixture
def bs():
    return toy_beats()


def test_メンバーと非メンバーを半分ずつに分ける(bs):
    mem, non = ecglink.split_persons(bs, seed=0)
    assert len(mem) == len(non) == 12 and not set(mem) & set(non)


def test_メンバーの拍を写すと本人に紐づき集団の平均だけなら紐づかない(bs):
    mem, non = ecglink.split_persons(bs, seed=0)
    train = bs.select(person=mem, part="gen")
    enroll = bs.select(part="enroll")
    copy = ecglink.generate("copy", train, n_persons=len(mem), beats_per_person=30, seed=0)
    e1 = ecglink.generate("E1", train, n_persons=len(mem), beats_per_person=30, seed=0)
    assert ecglink.linkage_attack(copy, enroll, mem)["auc"] > 0.95
    assert abs(ecglink.linkage_attack(e1, enroll, mem)["auc"] - 0.5) < 0.25


def test_本人ごとに作ると識別器が元の本人を当てる(bs):
    mem, _ = ecglink.split_persons(bs, seed=0)
    train = bs.select(person=mem, part="gen")
    e3 = ecglink.generate("E3", train, n_persons=len(mem), beats_per_person=30, seed=0)
    r = ecglink.linkage_attack(e3, bs.select(part="enroll"), mem)
    assert r["source_top1"] > 0.8


def test_それっぽさは実データ同士で近く雑音では遠い(bs):
    mem, non = ecglink.split_persons(bs, seed=0)
    ref = bs.select(person=non, part="gen")
    same = bs.select(person=non, part="enroll")
    rng = np.random.default_rng(0)
    noise = ecglink.BeatSet(rng.normal(0, 0.5, same.beats.shape), same.person, same.part, same.rr, same.symbol)
    near = ecglink.realism(same, ref)
    far = ecglink.realism(noise, ref)
    assert near["mmd"] < far["mmd"] and near["disc"] < far["disc"]
    assert far["disc"] > 0.3


def _toy_tree(tmp_path, persons=3, recs=2, fs=500):
    from test_ecg import _write_record

    for p in range(persons):
        d = tmp_path / f"Person_{p + 1:02d}"
        d.mkdir()
        for r in range(recs):
            sig, _ = ecg.ecgsyn(duration=20, fs=fs, hr_mean=60 + 10 * p, seed=10 * p + r)
            _write_record(d, f"rec_{r + 1}", np.column_stack([sig, sig]), fs, 16, comments=[f"Age: {30 + p}", "Sex: male"])
    return tmp_path


def test_ECGIDは1回目の記録を生成用に残りを登録用にする(tmp_path):
    bs = ecglink.load_ecgid(_toy_tree(tmp_path))
    assert set(bs.persons) == {"Person_01", "Person_02", "Person_03"}
    assert set(bs.part) == {"gen", "enroll"}
    assert bs.beats.shape[1] == ecg.BEAT_LEN
    assert (bs.rr > 0.3).all() and (bs.rr < 2.0).all()


def test_MITBIHは注釈の拍を前半と後半に分ける(tmp_path):
    from test_ecg import _write_record

    fs = 360
    sig, peaks = ecg.ecgsyn(duration=60, fs=fs, hr_mean=70, seed=0)
    _write_record(tmp_path, "100", np.column_stack([sig, sig]), fs, 212, baseline=1024)
    words, prev = [], 0
    for i, s in enumerate(peaks):
        words.append(((5 if i % 10 == 0 else 1) << 10) | (int(s) - prev))
        prev = int(s)
    words.append(0)
    (tmp_path / "100.atr").write_bytes(np.array(words, dtype="<u2").tobytes())
    (tmp_path / "RECORDS").write_text("100\n")
    bs = ecglink.load_mitdb(tmp_path)
    assert set(bs.part) == {"gen", "enroll"}
    assert set(bs.symbol) <= {"N", "V"} and (bs.symbol == "V").any()


def test_判別スコアは別の人の実データ同士なら小さい():
    """被験者単位で交差検証するので、人の違いではなく合成かどうかを見分ける。"""
    bs = toy_beats(n_persons=40, seed=3)
    mem, non = ecglink.split_persons(bs, seed=0)
    r = ecglink.realism(bs.select(person=mem, part="gen"), bs.select(person=non, part="gen"))
    assert r["disc"] < 0.25


def test_実験は分け方と作り方ごとに紐づきとそれっぽさを返す():
    bs = toy_beats(n_persons=20, seed=4)
    rows, per_person = ecglink.experiment(bs, seeds=(0, 1), kinds=("copy", "E1"), beats_per_person=20)
    assert {(r["seed"], r["kind"]) for r in rows} == {(s, k) for s in (0, 1) for k in ("copy", "E1")}
    assert all({"auc", "mmd", "disc", "id_acc_real"} <= set(r) for r in rows)
    assert {"person", "member", "score", "outlier", "kind", "seed"} <= set(per_person.columns)
