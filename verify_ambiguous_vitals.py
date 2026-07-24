# -*- coding: utf-8 -*-
"""
Verification script for 3 vital-sign features whose binning thresholds could
not be confirmed from ranges_with_links.xlsx:

  - VITALS:10155611 (VITAL_10155611_MEAN) -- "oxygen flow rate"
      LSTM thresholds assume a 0-1 FRACTION scale: [0.22, 0.50]... wait, no,
      actually LSTM has [1, 7] (looks like it might be L/min, not %).
      BERT thresholds assume a PERCENTAGE scale (21-100): [1, 24, 32, 46]
  - VITALS:10155613 (VITAL_10155613_MEAN) -- "fraction of inspired oxygen" (FiO2)
      LSTM thresholds assume a 0-1 FRACTION scale: [0.22, 0.50]
      BERT thresholds assume a PERCENTAGE scale (21-100): [22, 36, 61]
  - VITALS:10541434 (VITAL_10541434_MEAN) -- labeled "head circumference
      (babies and children)" in the reference spreadsheet, but this is an
      adult T2D cohort, so this needs a sanity check on whether the label is
      even correct.

Run this on your machine (wherever deid_vital.csv and deid_DEM.csv live) and
send me the printed output -- no need to send any patient-level data, just
these aggregate summaries.
"""

import pandas as pd
import numpy as np
import os

CODES_TO_CHECK = {
    "VITALS:10155611": "oxygen flow rate (VITAL_10155611_MEAN)",
    "VITALS:10155613": "fraction of inspired oxygen / FiO2 (VITAL_10155613_MEAN)",
    "VITALS:10541434": "head circumference?? (VITAL_10541434_MEAN)",
}

PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

# Candidate locations to search automatically, relative to wherever this
# script is run from (script_dir/, script_dir/data/, script_dir/../data/,
# plus the old hardcoded project path in case it still exists elsewhere).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIRS = [
    SCRIPT_DIR,
    os.path.join(SCRIPT_DIR, "data"),
    os.path.join(SCRIPT_DIR, "0data"),
    os.path.join(SCRIPT_DIR, ".."),
    os.path.join(SCRIPT_DIR, "..", "data"),
    os.path.join(SCRIPT_DIR, "..", "0data"),
    os.path.join(SCRIPT_DIR, "..", "diabetesRiskPrediction", "data"),
    os.path.join(SCRIPT_DIR, "..", "..", "diabetesRiskPrediction", "data"),
    r"C:\Users\universidad\clases\iit\TFM\MODELO-LSTM\diabetesRiskPrediction\data",
]


def find_file(filename):
    """Search known candidate folders for filename; if not found, ask for the path."""
    for d in CANDIDATE_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            print(f"Found {filename} at: {os.path.abspath(candidate)}")
            return candidate
    # Not found automatically -- ask the user directly.
    while True:
        typed = input(f"\nCould not find '{filename}' automatically.\n"
                       f"Please paste its full path (or press Enter to skip): ").strip().strip('"')
        if typed == "":
            return None
        if os.path.isfile(typed):
            return typed
        print(f"  '{typed}' does not exist, try again.")


def describe_values(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    print(f"  Non-null readings: {len(s)}")
    if len(s) == 0:
        print("  (no data found for this code)")
        return
    print(f"  Min: {s.min():.4f}   Max: {s.max():.4f}   Mean: {s.mean():.4f}   Median: {s.median():.4f}")
    print("  Percentiles:")
    for p in PERCENTILES:
        print(f"    {p:>3}th: {np.percentile(s, p):.4f}")
    # A quick text histogram to eyeball the shape/scale of the distribution
    counts, bin_edges = np.histogram(s, bins=10)
    print("  Rough histogram:")
    for c, lo, hi in zip(counts, bin_edges[:-1], bin_edges[1:]):
        bar = "#" * int(60 * c / counts.max()) if counts.max() > 0 else ""
        print(f"    [{lo:8.3f}, {hi:8.3f}) {c:>8}  {bar}")


def main():
    print("Looking for deid_vital.csv...")
    vitals_path = find_file("deid_vital.csv")
    if vitals_path is None:
        print("Cannot continue without deid_vital.csv. Exiting.")
        return

    codes_of_interest = set(CODES_TO_CHECK.keys())

    # Read the file in chunks and only keep rows for the 3 codes we care about.
    # This avoids ever holding the (potentially huge) full vitals file in RAM --
    # only the small filtered subset accumulates across chunks.
    print("Scanning vitals data in chunks (this only keeps the 3 codes we need)...")
    matches = []
    rows_scanned = 0
    CHUNK_SIZE = 500_000
    try:
        reader = pd.read_csv(
            vitals_path,
            usecols=["PATIENT_ID", "VITAL_CODE", "MEASUREMENT"],
            chunksize=CHUNK_SIZE,
        )
    except ValueError:
        # In case column names differ slightly, fall back to reading all columns
        # (still chunked) and select what we need after the fact.
        reader = pd.read_csv(vitals_path, chunksize=CHUNK_SIZE)

    for chunk in reader:
        rows_scanned += len(chunk)
        subset = chunk[chunk["VITAL_CODE"].isin(codes_of_interest)]
        if not subset.empty:
            matches.append(subset[["PATIENT_ID", "VITAL_CODE", "MEASUREMENT"]].copy())
        print(f"  ...scanned {rows_scanned:,} rows so far", end="\r")

    print(f"\nFinished scanning {rows_scanned:,} rows.")
    df_vitals = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
        columns=["PATIENT_ID", "VITAL_CODE", "MEASUREMENT"]
    )
    print(f"Kept {len(df_vitals)} matching rows for the 3 codes of interest.\n")

    # Try to also load demographics, for the head-circumference age sanity check.
    # This file is normally much smaller, but we still only load the 2 columns we need.
    demo_df = None
    demo_path = find_file("deid_DEM.csv")
    if demo_path is not None:
        try:
            try:
                demo_df = pd.read_csv(demo_path, usecols=lambda c: c in ("PATIENT_ID", "AGE_AT_END"))
            except ValueError:
                demo_df = pd.read_csv(demo_path)
            if "PATIENT_ID" not in demo_df.columns:
                for alt in ["patient_id", "PID", "Patient_ID"]:
                    if alt in demo_df.columns:
                        demo_df = demo_df.rename(columns={alt: "PATIENT_ID"})
                        break
        except Exception as e:
            print(f"(Could not load demographics for the age sanity check: {e})\n")

    for code, label in CODES_TO_CHECK.items():
        print("=" * 70)
        print(f"CODE: {code}  ->  {label}")
        print("=" * 70)
        subset = df_vitals[df_vitals["VITAL_CODE"] == code]
        if subset.empty:
            print(f"  No rows found with VITAL_CODE == '{code}' anywhere in the {rows_scanned:,} rows scanned.")
            print(f"  (This exact code string may not exist in your data -- double-check the spelling.)")
            continue

        describe_values(subset["MEASUREMENT"])

        # Extra sanity check for the head-circumference candidate: what ages
        # are the patients who have this reading? If this cohort is adults
        # (not babies/children), the spreadsheet label for this code is
        # probably wrong / refers to something else.
        if code == "VITALS:10541434" and demo_df is not None and "AGE_AT_END" in demo_df.columns:
            patient_ids = subset["PATIENT_ID"].unique()
            ages = demo_df[demo_df["PATIENT_ID"].isin(patient_ids)]["AGE_AT_END"]
            ages = pd.to_numeric(ages, errors="coerce").dropna()
            if len(ages) > 0:
                print("\n  Age (AGE_AT_END) of patients who have this vital recorded:")
                print(f"    Min: {ages.min():.1f}  Max: {ages.max():.1f}  Median: {ages.median():.1f}")
                print(f"    % of patients under 18: {(ages < 18).mean() * 100:.1f}%")
            else:
                print("\n  (Could not match any patient ages for this code.)")
        print()

    print("Done. Please paste the full output back so we can pick the right thresholds/units.")


if __name__ == "__main__":
    main()