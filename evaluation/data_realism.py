"""
HemoSmart - Synthetic Patient Data vs Real Public Data
----------------------------------------------------------
Is data/patient_data.csv realistic? Compares the generator's pre-operative
hemoglobin (by sex) and platelets with real adults from NHANES 2021-23
(US CDC, public domain), and the blood-group shares with Agrawal et al.
2014 (10,000 Indian donors).

NHANES is a US general population; HemoSmart's target is Indian surgical
patients. Some differences are intended: lower Hb and more anaemia
(NFHS-5, ISOS), and an 8% thrombocytopenic subgroup a hospital sees and a
household survey doesn't. NHANES numbers use the MEC exam weights
(WTMEC2YR), so they estimate the US adult population, not the sample.
NHANES has no INR, so INR isn't checked here.

Compared: `hemoglobin_preop` (before the peri-operative drop; the model's
`hemoglobin` is the value at the transfusion decision, lower by design)
and `platelets`.

Run (from the repo root):
    python -m evaluation.data_realism

Downloads CBC_L.xpt and DEMO_L.xpt to data/external/ if missing.
Writes evaluation/results/data_realism.md and .json.
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from data.demand_process import GROUP_SHARE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
EXTERNAL = os.path.join(REPO_ROOT, "data", "external")
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{}.xpt"
WHO_ANAEMIA = {"M": 13.0, "F": 12.0}  # g/dL, WHO 2024 (non-pregnant women)
PERCENTILES = (5, 50, 95)
# Agrawal et al., Asian J Transfus Sci 2014;8(2):121-5, all-India, 10,000 donors
AGRAWAL_2014 = {"O": 37.12, "B": 32.26, "A": 22.88, "AB": 7.74, "Rh-": 5.39}


def _download(name: str) -> str:
    path = os.path.join(EXTERNAL, f"{name}.xpt")
    if not os.path.exists(path):
        os.makedirs(EXTERNAL, exist_ok=True)
        url = NHANES_URL.format(name)
        try:
            urllib.request.urlretrieve(url, path)
        except (urllib.error.URLError, OSError) as exc:
            if os.path.exists(path):
                os.remove(path)
            raise SystemExit(f"Could not download NHANES {name} ({exc}). Download {url} "
                             f"by hand and save it as {path}.")
    return path


def load_nhanes() -> pd.DataFrame:
    """Adults (18+) with Hb, platelets and an exam weight: sex, age, hb, platelets (/uL), weight."""
    cbc = pd.read_sas(_download("CBC_L"), format="xport")[["SEQN", "LBXHGB", "LBXPLTSI"]]
    demo = pd.read_sas(_download("DEMO_L"), format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR"]]
    df = cbc.merge(demo, on="SEQN").dropna()
    df = df[(df.RIDAGEYR >= 18) & (df.WTMEC2YR > 0)]
    return pd.DataFrame({
        "sex": df.RIAGENDR.map({1.0: "M", 2.0: "F"}),
        "age": df.RIDAGEYR,
        "hb": df.LBXHGB,
        "platelets": df.LBXPLTSI * 1000,  # 1000 cells/uL -> /uL, as in patient_data.csv
        "weight": df.WTMEC2YR,
    }).reset_index(drop=True)


def weighted_quantile(x, w, q) -> float:
    order = np.argsort(x)
    x, cw = np.asarray(x)[order], np.cumsum(np.asarray(w)[order])
    return float(np.interp(q / 100 * cw[-1], cw, x))


def weighted_ks(synthetic, real, weights) -> float:
    """Two-sample KS distance between the synthetic ECDF and the weighted real ECDF."""
    grid = np.union1d(synthetic, real)
    order = np.argsort(real)
    real_cdf = np.cumsum(np.asarray(weights)[order]) / np.sum(weights)
    f_real = np.concatenate([[0.0], real_cdf])[np.searchsorted(np.asarray(real)[order], grid, side="right")]
    f_syn = np.searchsorted(np.sort(synthetic), grid, side="right") / len(synthetic)
    return float(np.max(np.abs(f_syn - f_real)))


def describe(x, w=None, cutoff=None) -> dict:
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    mean = float(np.average(x, weights=w))
    out = {"n": int(len(x)), "mean": round(mean, 2),
           "sd": round(float(np.sqrt(np.average((x - mean) ** 2, weights=w))), 2),
           **{f"p{q}": round(weighted_quantile(x, w, q), 2) for q in PERCENTILES}}
    if cutoff is not None:
        out["below_cutoff"] = round(float(np.sum(w * (x < cutoff)) / np.sum(w)), 3)
    return out


def blood_groups() -> dict:
    ours = {abo: 0.0 for abo in ("O", "B", "A", "AB")}
    for group, share in GROUP_SHARE.items():
        ours[group[:-1]] += share * 100
    ours["Rh-"] = sum(s for g, s in GROUP_SHARE.items() if g.endswith("-")) * 100
    return {k: {"hemosmart": round(ours[k], 2), "agrawal_2014": AGRAWAL_2014[k],
                "diff": round(ours[k] - AGRAWAL_2014[k], 2)} for k in AGRAWAL_2014}


def run() -> dict:
    syn = pd.read_csv(os.path.join(REPO_ROOT, "data", "patient_data.csv"))
    real = load_nhanes()
    hb = {}
    for sex in ("M", "F"):
        s, r = syn[syn.sex == sex].hemoglobin_preop, real[real.sex == sex]
        hb[sex] = {"synthetic": describe(s, cutoff=WHO_ANAEMIA[sex]),
                   "nhanes": describe(r.hb, r.weight, cutoff=WHO_ANAEMIA[sex]),
                   "ks": round(weighted_ks(s.values, r.hb.values, r.weight.values), 3)}
    platelets = {
        "synthetic": describe(syn.platelets, cutoff=150_000),
        "nhanes": describe(real.platelets, real.weight, cutoff=150_000),
        "ks": round(weighted_ks(syn.platelets.values, real.platelets.values, real.weight.values), 3),
    }
    anaemic = (syn.hemoglobin_preop < syn.sex.map(WHO_ANAEMIA)).mean()
    return {"hemoglobin_preop": hb, "platelets": platelets, "blood_groups": blood_groups(),
            "synthetic_preop_anaemia": round(float(anaemic), 3),
            "synthetic_sex_share_female": round(float((syn.sex == "F").mean()), 3),
            "nhanes_n": len(real)}


def _row(label, d, unit=1):
    return (f"| {label} | {d['n']} | {d['mean'] / unit:.2f} | {d['sd'] / unit:.2f} | "
            + " | ".join(f"{d[f'p{q}'] / unit:.2f}" for q in PERCENTILES)
            + f" | {d['below_cutoff'] * 100:.1f}% |")


def to_markdown(result: dict, meta: dict) -> str:
    header = ["| Source | n | Mean | SD | 5th | 50th | 95th | Below cut-off |", "|---|---|---|---|---|---|---|---|"]
    lines = [
        "# HemoSmart — Synthetic Patient Data vs Real Data",
        "",
        f"Generated by `python -m evaluation.data_realism` on {meta['generated_at']}. Synthetic: "
        "`data/patient_data.csv` (5,000 patients). Real: NHANES 2021–23 CBC_L + DEMO_L, adults 18+ "
        f"({result['nhanes_n']:,} people), **weighted with the MEC exam weights (WTMEC2YR)**. "
        "Sources and generator parameters: [data/DATA_CARD.md](../../data/DATA_CARD.md).",
        "",
        "**Read the differences with the target in mind.** NHANES is the US general population. "
        "HemoSmart simulates Indian surgical patients, so lower hemoglobin and more anaemia are intended "
        "(NFHS-5: anaemia in 25% of men, 57% of women; ISOS: 30% pre-operative anaemia), as is a "
        "thrombocytopenic tail. A small KS distance would be the wrong goal here.",
        "",
        "## Pre-operative hemoglobin (g/dL), by sex",
        "",
        "Compared column: `hemoglobin_preop`. The model's `hemoglobin` is the value at the transfusion "
        "decision, after a surgery-specific peri-operative drop, so it is lower again by design. "
        "Cut-off: WHO 2024 anaemia, men < 13, non-pregnant women < 12 (pregnant NHANES women are not excluded here).",
        "",
    ]
    for sex, name in (("M", "Men"), ("F", "Women")):
        h = result["hemoglobin_preop"][sex]
        lines += [f"### {name} (cut-off {WHO_ANAEMIA[sex]:.0f})", "", *header,
                  _row("HemoSmart synthetic", h["synthetic"]), _row("NHANES 2021–23 (weighted)", h["nhanes"]),
                  "", f"KS distance: {h['ks']:.3f}.", ""]
    p = result["platelets"]
    lines += [
        f"Synthetic pre-operative anaemia overall: {result['synthetic_preop_anaemia'] * 100:.1f}% "
        f"({result['synthetic_sex_share_female'] * 100:.0f}% women). Targets: ISOS 30.1% (LMIC 32.5%); "
        "NFHS-5 men 25.0%, women 57.0%.",
        "",
        "## Platelets (×10³/µL)",
        "",
        "Cut-off: 150 ×10³/µL (thrombocytopenia).",
        "",
        *header, _row("HemoSmart synthetic", p["synthetic"], 1000), _row("NHANES 2021–23 (weighted)", p["nhanes"], 1000),
        "",
        f"KS distance: {p['ks']:.3f}. The synthetic draw is N(250k, 70k) clipped to 50–450k, plus an 8% "
        "thrombocytopenic subgroup (10–120k) so the platelet criterion can be learned; that share has no "
        "published source. Not modelled: women's higher counts (NHANES ~+30k).",
        "",
        "## Blood groups (%) vs Agrawal et al. 2014 (all-India, 10,000 donors)",
        "",
        "From `GROUP_SHARE` in `data/demand_process.py` (patient demand mix).",
        "",
        "| Group | HemoSmart | Agrawal 2014 | Difference (points) |",
        "|---|---|---|---|",
        *(f"| {k} | {v['hemosmart']:.2f} | {v['agrawal_2014']:.2f} | {v['diff']:+.2f} |"
          for k, v in result["blood_groups"].items()),
        "",
        "## Not checked here",
        "",
        "- **INR:** NHANES has no coagulation tests. The generator's ~4% above 1.5 is an estimate "
        "(no clean published pre-operative figure found).",
        "- **Transfusion rates by surgery:** no open patient-level dataset; they are calibrated to published "
        "rates (TRICS III, NSQIP, Indian MSBOS audit, level-I trauma) — see the data card.",
        "- **Age:** uniform 18–89 in the generator; not compared.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "nhanes_cycle": "2021-2023 (L)"}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "data_realism.json"), "w") as f:
        json.dump({"meta": meta, **result}, f, indent=2)
    md = to_markdown(result, meta)
    with open(os.path.join(RESULTS_DIR, "data_realism.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
