"""心電図の合成データのそれっぽさと本人との紐づき（GHE issue 17）。

拍（R 波の前 0.25 秒〜後 0.45 秒を固定長に揃えたもの）の単位で合成し、次を測る。

- 紐づき（linkage_attack）: 攻撃者は、合成に使っていない別の記録（登録用、enroll）で全員の心電図を登録した
  個人識別器（主成分＋収縮つき線形判別分析）を持つ。合成の被験者ごとに、各拍の識別器の確率を平均し、ある人と
  判定される強さの最大値で、その人が合成に使われた（メンバー）かを当てる
- それっぽさ（realism）: 合成に使っていない実データの拍との、主成分の空間での最大平均差（MMD）・判別スコア・
  被験者ごとの平均の波形の分布の距離・心拍間隔の分布の距離

合成の作り方（generate）:
- copy: メンバーの拍をそのまま写す（最悪の場合）
- E0: 心電図の力学系モデル（ECGSYN）。データから使うのは、拍の振幅の大きさ（集団の中央値）だけ
- E1: 集団の平均の波形 ＋ 被験者内のゆらぎ（全員が同じ形）
- E2: 被験者ごとの平均の波形を集団の多変量正規分布から引き、被験者内のゆらぎを加える
- E3: 本人ごとに、その人の平均の波形とゆらぎから作る（個人の再現型）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from . import ecg

OUT = Path("outputs") / "ecglink"
KINDS = ("copy", "E0", "E1", "E2", "E3")


@dataclass
class BeatSet:
    beats: np.ndarray       # (拍数, BEAT_LEN)
    person: np.ndarray      # 拍ごとの被験者
    part: np.ndarray        # "gen"（合成に使える側）/ "enroll"（攻撃者の登録用）/ "syn"
    rr: np.ndarray          # 拍ごとの直前の心拍間隔（秒）
    symbol: np.ndarray      # 拍の種類（N, V など）
    source: dict | None = field(default=None)  # 合成の被験者 → 元にした実在の被験者（E3・copy のみ）

    def select(self, person=None, part=None, symbol=None) -> "BeatSet":
        m = np.ones(len(self.beats), bool)
        if person is not None:
            m &= np.isin(self.person, list(person))
        if part is not None:
            m &= self.part == part
        if symbol is not None:
            m &= np.isin(self.symbol, list(symbol) if not isinstance(symbol, str) else [symbol])
        return BeatSet(self.beats[m], self.person[m], self.part[m], self.rr[m], self.symbol[m], self.source)

    @property
    def persons(self) -> np.ndarray:
        return np.array(sorted(set(self.person)))


def split_persons(bs: BeatSet, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """被験者を、合成に使う人（メンバー）と使わない人（非メンバー）に半分ずつ分ける。"""
    persons = bs.persons.copy()
    np.random.default_rng(seed).shuffle(persons)
    half = len(persons) // 2
    return np.sort(persons[:half]), np.sort(persons[half:])


# ---------------------------------------------------------------- 合成


@dataclass
class _Fit:
    pca: object
    persons: np.ndarray
    means: np.ndarray        # 被験者ごとの平均の係数
    within: np.ndarray       # 被験者内の共分散（全員でまとめたもの）
    own: dict                # 被験者ごとの共分散（全体の共分散と半々に混ぜたもの）
    between: np.ndarray      # 被験者ごとの平均の共分散（収縮推定）
    rr: pd.DataFrame         # 被験者ごとの心拍間隔の平均・標準偏差


def fit(train: BeatSet, k: int = 20) -> _Fit:
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(k, len(train.beats) - 1), random_state=0).fit(train.beats)
    coef = pca.transform(train.beats)
    persons = train.persons
    means = np.array([coef[train.person == p].mean(0) for p in persons])
    resid = coef - means[np.searchsorted(persons, train.person)]
    within = np.cov(resid.T)
    own = {p: 0.5 * np.cov(resid[train.person == p].T) + 0.5 * within for p in persons}
    between = LedoitWolf().fit(means).covariance_ if len(persons) > 2 else np.cov(means.T)
    rr = pd.DataFrame({"rr": train.rr, "p": train.person}).groupby("p")["rr"].agg(["mean", "std"]).reindex(persons)
    return _Fit(pca, persons, means, within, own, between, rr)


def _ecgsyn_beats(n_persons: int, beats_per_person: int, amp: float, rng: np.random.Generator) -> tuple[list, list]:
    fs = 250
    beats, rrs = [], []
    for _ in range(n_persons):
        hr = float(np.clip(rng.normal(70, 8), 45, 110))
        a = ecg.ECGSYN_A * (1 + rng.normal(0, 0.1, 5))
        sig, peaks = ecg.ecgsyn(duration=(beats_per_person + 3) * 60 / hr, fs=fs, hr_mean=hr, seed=int(rng.integers(1 << 31)), a=a)
        b, kept = ecg.extract_beats(sig, peaks, fs)
        b = b[:beats_per_person]
        scale = amp / (np.ptp(b, axis=1).mean() + 1e-9)
        beats.append(b * scale)
        rrs.append(np.r_[np.diff(kept) / fs, 60 / hr][: len(b)])
    return beats, rrs


def generate(kind: str, train: BeatSet, n_persons: int, beats_per_person: int, seed: int, fit_: _Fit | None = None) -> BeatSet:
    rng = np.random.default_rng(seed)
    ids = [f"S{j:03d}" for j in range(n_persons)]
    beats, rrs, source = [], [], {}
    if kind == "copy":
        for j, p in zip(range(n_persons), np.resize(train.persons, n_persons)):
            idx = np.where(train.person == p)[0]
            idx = rng.choice(idx, size=min(beats_per_person, len(idx)), replace=False)
            beats.append(train.beats[idx])
            rrs.append(train.rr[idx])
            source[ids[j]] = p
    elif kind == "E0":
        amp = float(np.median(np.ptp(train.beats, axis=1)))
        beats, rrs = _ecgsyn_beats(n_persons, beats_per_person, amp, rng)
    else:
        f = fit_ or fit(train)
        k = f.means.shape[1]
        grand = f.means.mean(0)
        rr_mu, rr_between, rr_within = f.rr["mean"].mean(), f.rr["mean"].std(), f.rr["std"].mean()
        for j in range(n_persons):
            if kind == "E1":
                m, cov, rr_m = grand, f.within, rr_mu
            elif kind == "E2":
                m, cov, rr_m = rng.multivariate_normal(grand, f.between), f.within, rng.normal(rr_mu, rr_between)
            elif kind == "E3":
                p = f.persons[j % len(f.persons)]
                i = int(np.searchsorted(f.persons, p))
                m, cov, rr_m = f.means[i], f.own[p], f.rr["mean"].iloc[i]
                source[ids[j]] = p
            else:
                raise ValueError(kind)
            coef = rng.multivariate_normal(m, cov, beats_per_person) if k > 1 else m + rng.normal(0, 1, (beats_per_person, 1))
            beats.append(f.pca.inverse_transform(coef))
            rrs.append(np.clip(rng.normal(rr_m, rr_within, beats_per_person), 0.3, 2.0))
    person = np.concatenate([[ids[j]] * len(b) for j, b in enumerate(beats)])
    beats_arr = np.vstack(beats)
    return BeatSet(beats_arr, person, np.full(len(person), "syn"), np.concatenate(rrs), np.full(len(person), "N"),
                   source or None)


# ---------------------------------------------------------------- 紐づきの攻撃


def _identifier(enroll: BeatSet, k: int = 30):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline

    return make_pipeline(PCA(n_components=min(k, len(enroll.beats) - 1), random_state=0),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(enroll.beats, enroll.person)


def _person_scores(model, bs: BeatSet) -> tuple[np.ndarray, np.ndarray]:
    """被験者ごとに、各拍の識別器の確率を平均したもの（被験者 × 登録された人）。"""
    proba = model.predict_proba(bs.beats)
    persons = bs.persons
    return persons, np.array([proba[bs.person == p].mean(0) for p in persons])


def identification_accuracy(test: BeatSet, enroll: BeatSet) -> float:
    """実データ同士の個人識別の正解率（被験者単位）。識別ができることの確認に使う。"""
    model = _identifier(enroll)
    persons, s = _person_scores(model, test)
    return float(np.mean(model.classes_[s.argmax(1)] == persons))


def linkage_attack(syn: BeatSet, enroll: BeatSet, members, model=None) -> dict:
    from sklearn.metrics import roc_auc_score

    from .attacks import eps_lower_bound

    model = model or _identifier(enroll)
    syn_ids, s = _person_scores(model, syn)
    score = s.max(0)  # 登録された人ごとに、最もその人らしい合成の被験者の確率
    labels = np.isin(model.classes_, list(members)).astype(int)
    top = model.classes_[s.argmax(1)]
    out = {
        "auc": float(roc_auc_score(labels, score)),
        "eps_lb": eps_lower_bound(score[labels == 1], score[labels == 0]),
        "member_top1_share": float(np.isin(top, list(members)).mean()),
    }
    if syn.source:
        out["source_top1"] = float(np.mean([top[i] == syn.source[sid] for i, sid in enumerate(syn_ids)]))
    return out


# ---------------------------------------------------------------- それっぽさ


def _mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, m: int = 800) -> float:
    x = x[rng.choice(len(x), min(m, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(m, len(y)), replace=False)]
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    h = np.median(d2[d2 > 0])
    k = np.exp(-d2 / h)
    nx = len(x)
    kxx, kyy, kxy = k[:nx, :nx], k[nx:, nx:], k[:nx, nx:]
    v = kxx.sum() / nx**2 + kyy.sum() / len(y) ** 2 - 2 * kxy.mean()
    return float(np.sqrt(max(v, 0.0)))


def realism(syn: BeatSet, ref: BeatSet, k: int = 10, seed: int = 0) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(seed)
    pca = PCA(n_components=k, random_state=0).fit(ref.beats)
    fs, fr = pca.transform(syn.beats), pca.transform(ref.beats)
    n = min(len(fs), len(fr), 2000)
    x = np.vstack([fs[rng.choice(len(fs), n, replace=False)], fr[rng.choice(len(fr), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    acc = cross_val_score(HistGradientBoostingClassifier(random_state=seed), x, y, cv=5).mean()
    ms = np.array([fs[syn.person == p].mean(0) for p in syn.persons])
    mr = np.array([fr[ref.person == p].mean(0) for p in ref.persons])
    return {
        "mmd": _mmd(fs, fr, rng),
        "disc": float(max(acc - 0.5, 0.0)),
        "person_w1": float(np.mean([wasserstein_distance(ms[:, j], mr[:, j]) for j in range(3)])),
        "rr_w1": float(wasserstein_distance(syn.rr, ref.rr)),
    }


# ---------------------------------------------------------------- データの読み込み

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ")  # MIT-BIH の拍の注釈


def _beatset_rows(beats, kept, fs, person, part, symbols=None):
    rr = np.diff(kept, prepend=kept[0] - int(fs)) / fs if len(kept) else np.array([])
    sym = np.full(len(kept), "N") if symbols is None else symbols
    return beats, np.full(len(kept), person), np.full(len(kept), part), rr, sym


def _concat(rows) -> BeatSet:
    rows = [r for r in rows if len(r[0])]
    return BeatSet(*(np.concatenate([r[i] for r in rows]) if i else np.vstack([r[0] for r in rows]) for i in range(5)))


def load_ecgid(root: str | Path, signal_index: int = 1) -> BeatSet:
    """ECG-ID Database（Person_XX/rec_N）を読む。1 回目の記録を生成用（gen）、2 回目以降を登録用（enroll）にする。
    信号は 2 本目（フィルタ済みの I 誘導）を使う。R 波は detect_r_peaks で検出する。"""
    root = Path(root)
    rows = []
    for hea in sorted(root.rglob("rec_*.hea")):
        rec = ecg.read_record(hea.with_suffix(""))
        x = rec.signal[:, min(signal_index, rec.signal.shape[1] - 1)]
        peaks = ecg.detect_r_peaks(x, rec.fs)
        beats, kept = ecg.extract_beats(x, peaks, rec.fs)
        part = "gen" if hea.stem == "rec_1" else "enroll"
        b, p, pt, rr, sym = _beatset_rows(beats, kept, rec.fs, hea.parent.name, part)
        ok = (rr > 0.3) & (rr < 2.0)
        rows.append((b[ok], p[ok], pt[ok], rr[ok], sym[ok]))
    return _concat(rows)


def load_mitdb(root: str | Path, lead: int = 0) -> BeatSet:
    """MIT-BIH Arrhythmia Database を読む。注釈の拍の位置を R 波とし、記録の前半を生成用（gen）、後半を登録用（enroll）
    にする。被験者は記録ごと（201 と 202 は同じ人だが、ここでは別に扱う）。"""
    root = Path(root)
    names = (root / "RECORDS").read_text().split()
    rows = []
    for name in names:
        rec = ecg.read_record(root / name)
        ann = ecg.read_annotations(root / name, "atr")
        ann = ann[ann["symbol"].isin(BEAT_SYMBOLS)]
        x = rec.signal[:, lead]
        beats, kept = ecg.extract_beats(x, ann["sample"].to_numpy(), rec.fs)
        sym = ann.set_index("sample").loc[kept, "symbol"].to_numpy()
        half = len(x) // 2
        for part, sel in (("gen", kept < half), ("enroll", kept >= half)):
            b, p, pt, rr, s = _beatset_rows(beats[sel], kept[sel], rec.fs, name, part, sym[sel])
            ok = (rr > 0.2) & (rr < 2.5)
            rows.append((b[ok], p[ok], pt[ok], rr[ok], s[ok]))
    return _concat(rows)
