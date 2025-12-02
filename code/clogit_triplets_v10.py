#!/usr/bin/env python3
  """
  Univariate conditional logistic regression using strata_vhl.csv

  Outcome   : case (1 = retinal detachment, 0 = control)
  Stratum   : match_id
  Treatment : Combined_treatments  →  treat_cat (0 / 1 / 2)

  Predictor columns handled here
  ------------------------------------------------
  Family, Pheo, CNS_HB, RCC, Renal_Cysts, NETs_pancreas,
  Pancreas_CA, Pancreas_Cyst, Spinal_HB, HighRCH (built
  from RCH_at_baseline), and treat_cat.
  """

  import pandas as pd, numpy as np, argparse
  from statsmodels.discrete.conditional_models import ConditionalLogit

  # ---------- helper functions -------------------------------------------------
  def cat3(x):
      """Collapse number of ocular treatments into 0 / 1 / 2."""
      if pd.isna(x):   return np.nan
      if x == 0:       return 0
      elif x <= 2:     return 1       # 1 or 2 treatments
      return 2                         # 3 or more

  def load_clean(path):
      """Read CSV, convert Na strings, build treat_cat & HighRCH."""
      df = pd.read_csv(path)
      for c in df.columns:
          if c.lower() in {"match_id","case","subject_id"}:
              continue
          df[c] = pd.to_numeric(
              df[c].replace({"Na":np.nan,"NA":np.nan,"na":np.nan}),
              errors="coerce"
          )

      # ordinal treatment variable
      df["treat_cat"] = df["Combined_treatments"].apply(cat3)

      # RCH ≥3 lesions dummy
      df["HighRCH"]   = df["RCH_at_baseline"]   # already 0 / 1

      return df

  def keep_strata(df):
      """Return only strata with exactly 3 rows and 1 case."""
      return df.groupby("match_id", group_keys=False)\
               .filter(lambda g: (len(g)==3) and (g["case"].sum()==1))

  def fit_clogit(df, pred, label):
      """Fit conditional logit for one predictor; return dict of results."""
      d = df[["case","match_id",pred]].dropna()

      # must vary within at least one stratum
      if not (d.groupby("match_id")[pred].nunique() > 1).any():
          return None

      m = ConditionalLogit(d["case"], d[[pred]], groups=d["match_id"]).fit(disp=False)
      b, se = m.params[pred], m.bse[pred]
      return {
          "Predictor": label,
          "OR"       : np.exp(b),
          "CI_low"   : np.exp(b - 1.96*se),
          "CI_high"  : np.exp(b + 1.96*se),
          "p"        : m.pvalues[pred],
          "Strata"   : d["match_id"].nunique()
      }

  # ---------- main script ------------------------------------------------------
  PREDICTORS = {
      "treat_cat"      : "Any ocular treatment (ordinal 0/1/2)",
      "RCC"            : "Renal-cell carcinoma present",
      "Pancreas_Cyst"  : "Pancreatic cysts present",
      "HighRCH"        : "High RCH burden (≥ 3 lesions)",
      "Spinal_HB"      : "Spinal hemangioblastoma present",
      "CNS_HB"         : "CNS hemangioblastoma present",
      "Family"         : "Positive family history",
      "Pheo"           : "Pheochromocytoma present",
      "Renal_Cysts"    : "Renal cysts present",
      "NETs_pancreas"  : "Pancreatic NETs present",
      "Pancreas_CA"    : "Pancreatic carcinoma present"
  }

  def main(fp_in, fp_out):
      df     = load_clean(fp_in)
      strata = keep_strata(df)

      rows = []
      for col, label in PREDICTORS.items():
          res = fit_clogit(strata, col, label)
          if res: rows.append(res)

      out = (pd.DataFrame(rows)
               .round({"OR":2,"CI_low":2,"CI_high":2,"p":4})
               .sort_values("p"))

      out.to_csv(fp_out, index=False)
      print(out.to_string(index=False))

  if __name__ == "__main__":
      ap = argparse.ArgumentParser()
      ap.add_argument("--in",  default="data/strata_vhl.csv")
      ap.add_argument("--out", default="results/clogit_results.csv")
      args = ap.parse_args()
      main(getattr(args, "in"), args.out)
