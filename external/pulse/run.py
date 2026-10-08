"""Pulse Physiology Engine で仮想患者のバイタルを作る（issue #21）。

ICUバイタルの検証（#15、src/transyn/vitals.py）と同じ形式（seq_id, t, heartrate, sao2, respiration, age, died）で、
入室後12時間分を5分刻みで出す。outputs/vitals/gen_pulse_s<seed>.csv.gz に保存する。

仮想患者の作り方（個人の記録は写さない）:
- 学習用Aの滞在から、死亡退院の割合、死亡退院ごとの性別の割合と、性別×死亡退院ごとの
  (年齢, 身長, 体重, 入室後1時間の心拍数・呼吸数の平均) の多変量正規分布を推定し、そこから引く
- Pulseの有効範囲に切り詰める: 年齢18〜65歳、心拍数の基準値50〜110、呼吸数の基準値8〜20、身長137〜213cm、BMI 16〜29.9
- 呼吸器の診断（呼吸不全・肺炎・ARDS）がある割合を死亡退院ごとに求め、その確率で両肺の肺炎（重症度0.05〜0.3）を与える
  Pulse 4.4の敗血症（Sepsis）は設定項目はあるが生理モデルに組み込まれていないため使わない

実行（Pulseの専用環境で。先に external/pulse/build.sh でビルドしておく）:

    cd data/pulse/build/install/bin
    PYTHONPATH=../python:. ../../../venv/bin/python ../../../../../external/pulse/run.py --n 120 --workers 12
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EICU = ROOT / "data" / "eicu_demo" / "eicu-collaborative-research-database-demo-2.0.1"
STEP_S, STEPS = 300, 144
LIMITS = {"age": (18, 65), "height": (137, 213), "weight": (40, 200), "hr": (50, 110), "rr": (8, 20)}


def fit_population(seed: int) -> list[dict]:
    """学習用Aの集計値から仮想患者の設定を引く。個々の滞在の値はそのまま使わない。"""
    a = pd.read_csv(ROOT / "data" / "eicu_demo" / "vitals_A.csv.gz")
    first = a[a["t"] < 12].groupby("seq_id").agg(hr=("heartrate", "mean"), rr=("respiration", "mean"))
    st = a.groupby("seq_id")[["age", "died"]].first().join(first)
    pt = pd.read_csv(EICU / "patient.csv.gz").set_index("patientunitstayid")
    st = st.join(pt[["gender", "admissionheight", "admissionweight"]].rename(columns={"admissionheight": "height", "admissionweight": "weight"}))
    st = st[st["gender"].isin(["Male", "Female"])].dropna()
    dx = pd.read_csv(EICU / "diagnosis.csv.gz", usecols=["patientunitstayid", "diagnosisstring"])
    resp = dx[dx["diagnosisstring"].str.contains("respiratory failure|pneumonia|ARDS", case=False)]["patientunitstayid"]
    st["resp"] = st.index.isin(resp)
    return [{"stats": st}]


def sample_patients(n: int, seed: int) -> list[dict]:
    st = fit_population(seed)[0]["stats"]
    rng = np.random.default_rng(seed)
    cols = ["age", "height", "weight", "hr", "rr"]
    p_died = st["died"].mean()
    out = []
    for i in range(n):
        died = int(rng.random() < p_died)
        by_died = st[st["died"] == died]
        sex = rng.choice(by_died["gender"].to_numpy())  # 死亡退院ごとの性別の割合で引く
        group = by_died[by_died["gender"] == sex][cols].to_numpy(float)
        x = rng.multivariate_normal(group.mean(0), np.cov(group.T))
        rec = dict(zip(cols, x))
        for k, (lo, hi) in LIMITS.items():
            rec[k] = float(np.clip(rec[k], lo, hi))
        # PulseはBMI 30を超える患者を受け付けない（肥満は別の病態として与える仕様）。BMIが16〜29.9に入るよう体重を切り詰める
        h2 = (rec["height"] / 100) ** 2
        rec["weight"] = float(np.clip(rec["weight"], 16 * h2, 29.9 * h2))
        resp = rng.random() < by_died["resp"].mean()
        out.append({"seq_id": i, "sex": sex, "died": died, "pneumonia": float(rng.uniform(0.05, 0.3)) if resp else 0.0, **rec})
    return out


def simulate(p: dict) -> list[dict]:
    from pulse.cdm.engine import SEDataRequest, SEDataRequestManager
    from pulse.cdm.patient import SEPatientConfiguration, eSex
    from pulse.cdm.physiology import eLungCompartment
    from pulse.cdm.scalars import FrequencyUnit, LengthUnit, MassUnit, TimeUnit
    from pulse.engine.PulseEngine import PulseEngine

    reqs = [
        SEDataRequest.create_physiology_request("HeartRate", unit=FrequencyUnit.Per_min),
        SEDataRequest.create_physiology_request("OxygenSaturation"),
        SEDataRequest.create_physiology_request("RespirationRate", unit=FrequencyUnit.Per_min),
    ]
    engine = PulseEngine()
    engine.log_to_console(False)
    engine.set_log_filename(f"./test_results/pulse_vitals/patient_{p['seq_id']}.log")
    pc = SEPatientConfiguration()
    pat = pc.get_patient()
    pat.set_name(f"virtual_{p['seq_id']}")
    pat.set_sex(eSex.Male if p["sex"] == "Male" else eSex.Female)
    pat.get_age().set_value(p["age"], TimeUnit.yr)
    pat.get_height().set_value(p["height"], LengthUnit.cm)
    pat.get_weight().set_value(p["weight"], MassUnit.kg)
    pat.get_heart_rate_baseline().set_value(p["hr"], FrequencyUnit.Per_min)
    pat.get_respiration_rate_baseline().set_value(p["rr"], FrequencyUnit.Per_min)
    if p["pneumonia"] > 0:
        pn = pc.get_conditions().get_pneumonia()
        for lung in (eLungCompartment.LeftLung, eLungCompartment.RightLung):
            pn.get_severity(lung).set_value(p["pneumonia"])
    pc.set_data_root_dir("./")
    started = time.time()
    if not engine.initialize_engine(pc, SEDataRequestManager(reqs)):
        print(f"patient {p['seq_id']}: initialization failed", flush=True)
        return []
    rows = []
    for t in range(STEPS):
        engine.advance_time_s(STEP_S)
        r = engine.pull_data()
        rows.append({"seq_id": p["seq_id"], "t": t, "heartrate": r[1], "sao2": r[2] * 100, "respiration": r[3],
                     "age": round(p["age"]), "died": p["died"]})
    print(f"patient {p['seq_id']}: done in {time.time() - started:.0f}s (pneumonia {p['pneumonia']:.2f})", flush=True)
    return rows


def main() -> None:
    global STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS, help="試運転用。既定は12時間分（144点）")
    args = parser.parse_args()
    STEPS = args.steps
    patients = sample_patients(args.n, args.seed)
    pd.DataFrame(patients).to_csv(ROOT / "outputs" / "vitals" / f"pulse_patients_s{args.seed}.csv", index=False)
    with mp.Pool(args.workers) as pool:
        results = pool.map(simulate, patients, chunksize=1)
    gen = pd.DataFrame([row for rows in results for row in rows])
    path = ROOT / "outputs" / "vitals" / f"gen_pulse_s{args.seed}.csv.gz"
    if STEPS != 144:
        path = path.with_name(f"pulse_trial_s{args.seed}.csv.gz")
    gen.to_csv(path, index=False)
    print(f"wrote {path}: {gen['seq_id'].nunique()} patients", file=sys.stderr)


if __name__ == "__main__":
    main()
