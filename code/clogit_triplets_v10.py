#!/usr/bin/env python3
"""
Conditional-logit on 14 triplets (1 case : 2 controls) using strata_vhl_10.csv.

Predictors & labels:
    treat_cat           – Any ocular treatment (ordinal 0/1/2)
    RCC_0_yes           – Renal-cell carcinoma present
    Pancreas_Cyst_0_yes – Pancreatic cysts present
    HighRCH             – High RCH burden (RCH_at_baseline = 1)
    Spinal_HB_yes_no    – Spinal hemangioblastoma present
    CNS_HB_0_yes        – CNS hemangioblastoma present
"""

import pandas as pd, numpy as np, argparse
from statsmodels.discrete.conditional_models import ConditionalLogit

def cat3(x):
    if pd.isna(x):        return np.nan
    if x == 0:            return 0
    elif x <= 2:          return 1
    return 2

PREDICTORS = {
    "treat_cat"           : "Any ocular treatment (ordinal 0/1/2)",
    "RCC_0_yes"           : "Renal-cell carcinoma present",
    "Pancreas_Cyst_0_yes" : "Pancreatic cysts present",
    "HighRCH"             : "High RCH burden (≥ 3 lesions)",
    "Spinal_HB_yes_no"    : "Spinal hemangioblastoma present",
    "CNS_HB_0_yes"        : "CNS hemangioblastoma present",
}

def load_clean(fp):
    df = pd.read_csv(fp)
    for c in df.columns:
        if c.lower() in {"match_id","case","subject_id"}: continue
        df[c] = pd.to_numeric(df[c].replace({"Na":np.nan,"NA":np.nan,"na":np.nan}),
                              errors="coerce")
    df["treat_cat"] = df["Combined_treatments"].apply(cat3)
    df["HighRCH"]   = df["RCH_at_baseline"]   # 0/1 already
    return df

def triplets_only(df):
    return df.groupby("match_id", group_keys=False)\
             .filter(lambda g: len(g)==3 and g["case"].sum()==1)

def fit_clogit(df, xcol):
    d = df[["case","match_id",xcol]].dropna()
    # Need at least one stratum with 0 & 1
    if not (d.groupby("match_id")[xcol].nunique() > 1).any():
        return None
    m = ConditionalLogit(d["case"], d[[xcol]], groups=d["match_id"]).fit(disp=False)
    b, se = m.params[xcol], m.bse[xcol]
    return np.exp(b), np.exp(b-1.96*se), np.exp(b+1.96*se), m.pvalues[xcol], d["match_id"].nunique()

def main(fp_in, fp_out):
    df   = load_clean(fp_in)
    trip = triplets_only(df)
    rows = []
    for col, label in PREDICTORS.items():
        res = fit_clogit(trip, col)
        if res:
            OR, lo, hi, p, n = res
            rows.append([label, OR, lo, hi, p, n])
    tbl = pd.DataFrame(rows, columns=["Predictor","OR","CI_low","CI_high","p","Triplets"])\
            .round({"OR":2,"CI_low":2,"CI_high":2,"p":4})
    tbl.to_csv(fp_out, index=False)
    print(tbl.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  default="data/strata_vhl_10.csv")
    ap.add_argument("--out", default="results/triplet_results.csv")
    args = ap.parse_args()
    main(args.in, args.out)
Add conditional-logit script for 14 triplets (clogit_triplets_v10.py)
