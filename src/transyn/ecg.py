"""心電図の読み込み・R 波の検出・拍の切り出し・力学系モデル（GHE issue 17）。

依存を増やさないため、WFDB 形式（PhysioNet の信号・注釈）の読み込みと R 波の検出を numpy・scipy で実装する。
対応する信号の形式は 16（16 ビット整数）と 212（12 ビットの 2 サンプルを 3 バイトに詰める）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

PRE, POST = 0.25, 0.45  # R 波の前後（秒）
BEAT_LEN = 128  # 拍を固定長に揃えるサンプル数

ANN_SYMBOLS = {
    1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E", 11: "j", 12: "/", 13: "Q",
    14: "~", 16: "|", 18: "s", 19: "T", 20: "*", 21: "D", 22: '"', 23: "=", 24: "p", 25: "B", 26: "^", 27: "t",
    28: "+", 29: "u", 30: "?", 31: "!", 32: "[", 33: "]", 34: "e", 35: "n", 36: "@", 37: "x", 38: "f", 39: "(",
    40: ")", 41: "r",
}


@dataclass
class Record:
    name: str
    fs: float
    signal: np.ndarray  # (サンプル数, 信号数)、物理量（mV）
    sig_names: list[str]
    comments: list[str] = field(default_factory=list)


def _decode_212(raw: np.ndarray, n_values: int) -> np.ndarray:
    raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3).astype(np.int32)
    s0 = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    s1 = raw[:, 2] | ((raw[:, 1] & 0xF0) << 4)
    out = np.column_stack([s0, s1]).reshape(-1)[:n_values]
    return np.where(out > 2047, out - 4096, out)


def read_record(path: str | Path) -> Record:
    """WFDB の記録（path は拡張子なし）を読む。すべての信号が同じ .dat に同じ形式で入っている場合に対応する。"""
    path = Path(path)
    lines = path.with_suffix(".hea").read_text().splitlines()
    comments = [l.lstrip("#").strip() for l in lines if l.startswith("#")]
    body = [l for l in lines if l.strip() and not l.startswith("#")]
    head = body[0].split()
    n_sig, fs = int(head[1]), float(head[2].split("/")[0])
    n_samp = int(head[3]) if len(head) > 3 else None
    gains, baselines, names, fmt, dat = [], [], [], None, None
    for line in body[1 : 1 + n_sig]:
        parts = line.split()
        dat, fmt = parts[0], int(parts[1].split("x")[0].split(":")[0].split("+")[0])
        gain_spec = parts[2] if len(parts) > 2 else "200"
        adczero = int(parts[4]) if len(parts) > 4 else 0
        g = gain_spec.split("/")[0]
        if "(" in g:
            g, b = g.rstrip(")").split("(")
            baselines.append(int(b))
        else:
            baselines.append(adczero)
        gains.append(float(g) or 200.0)
        names.append(" ".join(parts[8:]) if len(parts) > 8 else f"sig{len(names)}")
    raw = np.fromfile(path.parent / dat, dtype=np.uint8)
    if fmt == 16:
        digital = raw[: len(raw) // 2 * 2].view("<i2").astype(np.int32)
    elif fmt == 212:
        digital = _decode_212(raw, n_samp * n_sig if n_samp else len(raw) * 2 // 3)
    else:
        raise ValueError(f"形式 {fmt} には対応していない")
    n = len(digital) // n_sig if n_samp is None else n_samp
    digital = digital[: n * n_sig].reshape(n, n_sig)
    phys = (digital - np.array(baselines)) / np.array(gains)
    return Record(path.name, fs, phys, names, comments)


def read_annotations(path: str | Path, ext: str = "atr") -> pd.DataFrame:
    """MIT 形式の注釈を読み、拍などの位置（sample）と記号（symbol）を返す。"""
    words = np.fromfile(Path(f"{path}.{ext}"), dtype="<u2").astype(np.int64)
    t, i, rows = 0, 0, []
    while i < len(words):
        w = words[i]
        code, interval = w >> 10, w & 0x3FF
        if code == 0 and interval == 0:
            break
        if code == 59:  # SKIP: 続く 4 バイトが間隔（上位 16 ビットが先）
            t += int((words[i + 1] << 16) | words[i + 2])
            i += 3
            continue
        if code in (60, 61, 62):  # NUM, SUB, CHN（時刻は進まない）
            i += 1
            continue
        if code == 63:  # AUX: interval バイトの補助情報
            i += 1 + (int(interval) + 1) // 2
            continue
        t += int(interval)
        rows.append((t, ANN_SYMBOLS.get(int(code), "?")))
        i += 1
    return pd.DataFrame(rows, columns=["sample", "symbol"])


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    b, a = sps.butter(order, [lo, hi], btype="band", fs=fs)
    return sps.filtfilt(b, a, x)


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins 法の流れ（帯域通過・微分・二乗・移動積分）で候補を見つけ、元の波形の絶対値の最大に合わせ直す。"""
    f = bandpass(x, fs, 5, 20)
    e = np.convolve(np.diff(f, prepend=f[0]) ** 2, np.ones(int(0.15 * fs)) / int(0.15 * fs), mode="same")
    cand, _ = sps.find_peaks(e, distance=int(0.3 * fs), height=0.25 * np.percentile(e, 99))
    clean = bandpass(x, fs, 0.5, 40)
    w = int(0.1 * fs)
    peaks = [max(0, c - w) + int(np.argmax(np.abs(clean[max(0, c - w) : c + w]))) for c in cand]
    return np.unique(np.array(peaks, dtype=int))


def extract_beats(x: np.ndarray, peaks: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """0.5〜40 Hz に絞った波形から、R 波の前 PRE 秒〜後 POST 秒を切り出し、BEAT_LEN 点に揃える。
    窓の中央値を引いて基線をそろえる。記録の端にかかる拍は除く。"""
    clean = bandpass(x, fs, 0.5, 40)
    pre, post = int(round(PRE * fs)), int(round(POST * fs))
    grid = np.linspace(0, pre + post - 1, BEAT_LEN)
    beats, kept = [], []
    for p in peaks:
        if p - pre < 0 or p + post > len(clean):
            continue
        seg = clean[p - pre : p + post]
        beats.append(np.interp(grid, np.arange(len(seg)), seg) - np.median(seg))
        kept.append(p)
    return np.array(beats).reshape(-1, BEAT_LEN), np.array(kept, dtype=int)


# ---------------------------------------------------------------- 力学系モデル（ECGSYN）

# McSharry et al. 2003, IEEE TBME 50(3) の P・Q・R・S・T 波の既定のパラメータ
ECGSYN_THETA = np.array([-math.pi / 3, -math.pi / 12, 0.0, math.pi / 12, math.pi / 2])
ECGSYN_A = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
ECGSYN_B = np.array([0.25, 0.1, 0.1, 0.1, 0.4])


def ecgsyn(duration: float, fs: float, hr_mean: float = 60.0, hr_std: float = 1.0, seed: int = 0,
           a: np.ndarray | None = None, b: np.ndarray | None = None, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """ECGSYN の 3 変数の常微分方程式を、内部で fs の 4 倍の刻みのオイラー法で解き、z を心電図（mV に縮尺）として返す。
    心拍間隔は平均 60/hr_mean 秒、拍ごとに標準偏差 hr_std/hr_mean × 平均 のゆらぎ。R 波（θ = 0）の位置も返す。"""
    rng = np.random.default_rng(seed)
    a = ECGSYN_A if a is None else a
    b = ECGSYN_B if b is None else b
    rr_mean = 60.0 / hr_mean
    n_beats = int(duration / rr_mean) + 3
    rr = np.clip(rr_mean + rng.normal(0, hr_std / hr_mean * rr_mean, n_beats), 0.3, 2.0)
    beat_start = np.r_[0, np.cumsum(rr)]
    sub = 4
    dt = 1.0 / (fs * sub)
    n = int(duration * fs * sub)
    x, y, z = -1.0, 0.0, 0.0
    zs, thetas = np.empty(n), np.empty(n)
    k = 0
    bscale = b * math.sqrt(hr_mean / 60.0)
    for i in range(n):
        t = i * dt
        while k + 1 < len(beat_start) and t >= beat_start[k + 1]:
            k += 1
        omega = 2 * math.pi / rr[k]
        alpha = 1 - math.sqrt(x * x + y * y)
        th = math.atan2(y, x)
        d = (th - ECGSYN_THETA + math.pi) % (2 * math.pi) - math.pi
        dz = -np.sum(a * d * np.exp(-(d**2) / (2 * bscale**2))) - z
        x, y, z = x + dt * (alpha * x - omega * y), y + dt * (alpha * y + omega * x), z + dt * dz
        zs[i], thetas[i] = z, th
    sig = zs[::sub]
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-12) * 1.6 - 0.4
    if noise:
        sig = sig + rng.normal(0, noise, len(sig))
    th = thetas[::sub]
    peaks = np.where((th[:-1] < 0) & (th[1:] >= 0))[0] + 1
    return sig, peaks
