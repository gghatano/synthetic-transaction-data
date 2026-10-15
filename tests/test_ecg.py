import numpy as np
import pytest

from transyn import ecg


def _write_record(tmp_path, name, signals, fs, fmt, gain=200.0, baseline=0, comments=()):
    """テスト用に WFDB 形式の記録（.hea と .dat）を書く。signals は (サンプル数, 信号数) の mV。"""
    digital = np.round(signals * gain).astype(int) + baseline
    n, k = digital.shape
    if fmt == 16:
        raw = digital.astype("<i2").tobytes()
    else:  # 212: 2 サンプル（12 ビット）を 3 バイトに詰める
        flat = (digital.reshape(-1) & 0xFFF).astype(np.uint16)
        if len(flat) % 2:
            flat = np.r_[flat, 0]
        a, b = flat[0::2], flat[1::2]
        raw = np.column_stack([a & 0xFF, ((b >> 8) << 4) | (a >> 8), b & 0xFF]).astype(np.uint8).tobytes()
    (tmp_path / f"{name}.dat").write_bytes(raw)
    lines = [f"{name} {k} {fs} {n}"]
    lines += [f"{name}.dat {fmt} {gain:g}/mV 12 {baseline} 0 0 0 sig{j}" for j in range(k)]
    lines += [f"# {c}" for c in comments]
    (tmp_path / f"{name}.hea").write_text("\n".join(lines) + "\n")
    return tmp_path / name


@pytest.mark.parametrize("fmt", [16, 212])
def test_WFDBの信号を書いた値どおりに読める(tmp_path, fmt):
    rng = np.random.default_rng(0)
    sig = np.round(rng.normal(0, 1, (1001, 2)) * 200) / 200
    rec = ecg.read_record(_write_record(tmp_path, "r", sig, 360, fmt, baseline=1024 if fmt == 212 else 0, comments=["Age: 25"]))
    assert rec.fs == 360
    assert rec.signal.shape == (1001, 2)
    np.testing.assert_allclose(rec.signal, sig, atol=1e-9)
    assert rec.comments == ["Age: 25"]


def test_注釈の拍の位置と種類を読める(tmp_path):
    samples, codes = [100, 460, 830, 1200], [1, 1, 5, 1]  # N, N, V, N
    words, prev = [], 0
    for s, c in zip(samples, codes):
        words.append((c << 10) | (s - prev))
        prev = s
    words.append(0)
    (tmp_path / "r.atr").write_bytes(np.array(words, dtype="<u2").tobytes())
    ann = ecg.read_annotations(tmp_path / "r", "atr")
    assert ann["sample"].tolist() == samples
    assert ann["symbol"].tolist() == ["N", "N", "V", "N"]


def test_R波の検出は合成心電図のR波をほぼ正しく見つける():
    fs = 250
    signal, peaks = ecg.ecgsyn(duration=30, fs=fs, hr_mean=70, seed=0)
    found = ecg.detect_r_peaks(signal, fs)
    matched = sum(np.min(np.abs(found - p)) <= 0.05 * fs for p in peaks)
    assert matched / len(peaks) > 0.95
    assert len(found) <= len(peaks) * 1.05


def test_拍の切り出しは決まった長さで端の拍を除く():
    fs = 250
    signal, peaks = ecg.ecgsyn(duration=10, fs=fs, hr_mean=60, seed=1)
    beats, kept = ecg.extract_beats(signal, np.r_[5, peaks, len(signal) - 3], fs)
    assert beats.shape[1] == ecg.BEAT_LEN
    assert 5 not in kept and len(signal) - 3 not in kept
    assert len(beats) == len(kept)
