# -*- coding: utf-8 -*-
"""
Diagnosis (DX) Preprocessing Script
=====================================================================
Adds diagnosis-code features to the pipeline, matching what BERT uses
(individual DX codes, no ICD9/ICD10 harmonization) so the two models see
the same predictive information (see LSTM_methodology_decisions_log.md).

What this script does:
1. Loads the labs+vitals pickle from step 2 (already has one column per
   established visit date, per patient).
2. PASS 1 (chunked, memory-safe): scans deid_visit_dx.csv and counts how
   often each DX code appears among patients already in our cohort,
   EXCLUDING T2D/prediabetes-related codes (to prevent label leakage).
   Prints a coverage curve so TOP_N can be tuned with real numbers.
3. Selects the TOP_N most frequent (non-excluded) DX codes.
4. PASS 2 (chunked again): re-scans the file, keeping only rows for those
   TOP_N codes, and builds a CUMULATIVE binary flag per patient per visit:
   flag = 1 if that DX code was ever recorded on or before the visit date,
   else 0. Cumulative because a diagnosis is an ongoing clinical fact (once
   diagnosed, still relevant at every later visit) -- unlike a point-in-time
   lab/vital reading. This also lets BERT and the LSTM be compared fairly:
   BERT can "remember" an earlier diagnosis via attention over the full
   sequence, so making the LSTM's DX flags cumulative gives it a comparable
   ability to use diagnosis history at every future visit, not just the
   one visit closest to when it happened.
5. Merges the DX flags into the labs+vitals matrices and saves the result.

Run this AFTER step 2 (vitals) and BEFORE step 3 (demographics).
"""

import pandas as pd
import numpy as np
import pickle
import os
import tqdm

#  0) Shared utility import 
from streaming_pickle_utils import save_streamed_patient_dict, load_streamed_patient_dict

#  1) Paths 
input_labs_vitals_path = r"preprocessing\output_pickles\2_lab_vitals.pkl"
dx_csv_path = r"C:\Users\universidad\clases\iit\TFM\MODELO-LSTM\diabetesRiskPrediction\data\deid_visit_dx.csv"
output_dir = r"preprocessing\output_pickles"
output_path = os.path.join(output_dir, "2b_lab_vitals_dx.pkl")

TOP_N = 500  # how many most-frequent (non-excluded) DX codes to use as features;
             # adjust after looking at the coverage curve this script prints

#  2) Codes excluded from candidate DX features (label leakage prevention) 
# Diagnosis codes for (pre)diabetes must NEVER be used as predictors -- they
# are the outcome itself (or a close proxy for it). This list is a best
# effort based on standard ICD9/ICD10 diabetes code ranges -- REVIEW THIS
# WITH YOUR TUTOR before trusting it fully; missing an exclusion here is a
# silent, serious leakage bug.
EXCLUDED_DX_PREFIXES = (
    "ICD10CM:E08", "ICD10CM:E09", "ICD10CM:E10", "ICD10CM:E11", "ICD10CM:E13",  # ICD10 diabetes mellitus (all types)
    "ICD9CM:250.",   # ICD9 diabetes mellitus (all subtypes, 250.00-250.93)
    "ICD10CM:R73",   # ICD10 hyperglycemia / prediabetes
    "ICD9CM:790.2",  # ICD9 abnormal glucose
    "ICD10CM:O24",   # ICD10 diabetes in pregnancy (gestational)
    "ICD9CM:648.0", "ICD9CM:648.8",  # ICD9 diabetes/abnormal glucose tolerance complicating pregnancy
    # "ICD10CM:Z13.1",  # OPTIONAL: encounter for diabetes screening -- not a diagnosis,
    #                    # uncomment to exclude if you want to be extra conservative
)


def is_excluded(dx_code):
    return isinstance(dx_code, str) and dx_code.startswith(EXCLUDED_DX_PREFIXES)


#  3) Load the cohort (patients + their established visit dates) 
print("Loading labs+vitals data (for cohort + visit dates)...")
patient_data = load_streamed_patient_dict(input_labs_vitals_path, desc="Loading labs+vitals")
print(f"Loaded {len(patient_data)} patients.")

valid_patient_ids = set(patient_data.keys())
# Visit dates per patient, as Timestamps, sorted -- reused in both passes.
patient_visit_dates = {
    pid: sorted(pd.to_datetime(df.columns)) for pid, df in patient_data.items()
}

CHUNK_SIZE = 500_000

#  4) PASS 1: count DX code frequency (chunked, low memory) 
print("\n PASS 1: Counting DX code frequency (this may take a while for a large file) ")
code_counts = {}
rows_scanned = 0
pbar1 = tqdm.tqdm(desc="Pass 1/2: counting DX frequency", unit=" rows", unit_scale=True, colour='cyan')
for chunk in pd.read_csv(dx_csv_path, usecols=["PATIENT_ID", "DX"], chunksize=CHUNK_SIZE):
    rows_scanned += len(chunk)
    pbar1.update(len(chunk))
    chunk = chunk[chunk["PATIENT_ID"].isin(valid_patient_ids)]
    chunk = chunk[~chunk["DX"].apply(is_excluded)]
    counts = chunk["DX"].value_counts()
    for code, c in counts.items():
        code_counts[code] = code_counts.get(code, 0) + int(c)
pbar1.close()

print(f"Finished scanning {rows_scanned:,} rows. Found {len(code_counts)} distinct (non-excluded) DX codes.")

# Coverage curve: what fraction of (non-excluded) DX events would be captured
# by the top-K codes, for a few candidate K values -- to help pick TOP_N.
sorted_codes = sorted(code_counts.items(), key=lambda kv: kv[1], reverse=True)
total_events = sum(code_counts.values())
print("\nCoverage curve (top-K codes vs. % of all non-excluded DX events captured):")
cumulative = 0
next_checkpoint_idx = 0
checkpoints = [10, 20, 50, 75, 100, 150, 200, 300, 500]
for i, (code, c) in enumerate(sorted_codes, start=1):
    cumulative += c
    if next_checkpoint_idx < len(checkpoints) and i == checkpoints[next_checkpoint_idx]:
        print(f"  Top {i:>4}: {cumulative / total_events * 100:5.1f}% of events covered")
        next_checkpoint_idx += 1
    if next_checkpoint_idx >= len(checkpoints):
        break

top_codes = [code for code, _ in sorted_codes[:TOP_N]]
print(f"\nUsing TOP_N = {TOP_N} codes, covering "
      f"{sum(c for _, c in sorted_codes[:TOP_N]) / total_events * 100:.1f}% of non-excluded DX events.")
top_codes_set = set(top_codes)

#  5) PASS 2: build cumulative binary flags per patient per visit 
print("\n PASS 2: Building cumulative DX flags per visit ")
# dx_events[pid] = list of (date, code) for the TOP_N codes only
dx_events = {pid: [] for pid in valid_patient_ids}
rows_scanned = 0
pbar2 = tqdm.tqdm(desc="Pass 2/2: scanning for top-N codes", unit=" rows", unit_scale=True, colour='cyan')
for chunk in pd.read_csv(dx_csv_path, usecols=["PATIENT_ID", "DX", "Shifted_date"], chunksize=CHUNK_SIZE):
    rows_scanned += len(chunk)
    pbar2.update(len(chunk))
    chunk = chunk[chunk["PATIENT_ID"].isin(valid_patient_ids) & chunk["DX"].isin(top_codes_set)]
    if not chunk.empty:
        chunk["Shifted_date"] = pd.to_datetime(chunk["Shifted_date"])
        for pid, code, date in zip(chunk["PATIENT_ID"], chunk["DX"], chunk["Shifted_date"]):
            dx_events[pid].append((date, code))
pbar2.close()
print(f"Finished scanning {rows_scanned:,} rows for the {TOP_N} selected codes.")

# Build a (DX code) x (visit date) binary matrix per patient, cumulative.
print("\nBuilding per-patient DX matrices...")
dx_row_names = [f"DX_{code}" for code in top_codes]
n_codes = len(top_codes)
patient_dx_matrices = {}
for pid in tqdm.tqdm(list(valid_patient_ids), desc="Building DX matrices", colour='yellow'):
    visit_dates = patient_visit_dates[pid]
    events = dx_events.get(pid, [])
    # earliest date the patient ever had each code recorded (or NaT if never)
    first_seen = {}
    for date, code in events:
        if code not in first_seen or date < first_seen[code]:
            first_seen[code] = date

    if not visit_dates:
        patient_dx_matrices[pid] = pd.DataFrame(
            np.zeros((n_codes, 0)), index=dx_row_names, columns=[]
        )
        continue

    # Vectorized instead of a nested Python loop over (codes x visits):
    # one array of "first seen" dates per code (NaT where never seen), one
    # array of visit dates, then a single broadcasted comparison gives the
    # whole binary matrix at once. NaT comparisons are always False, so a
    # code that was never seen correctly ends up all-zero without a special case.
    first_seen_arr = np.array(
        [first_seen.get(code, np.datetime64('NaT')) for code in top_codes],
        dtype='datetime64[ns]'
    )
    visit_dates_arr = np.array(visit_dates, dtype='datetime64[ns]')
    flags = (visit_dates_arr[np.newaxis, :] >= first_seen_arr[:, np.newaxis]).astype('float64')

    patient_dx_matrices[pid] = pd.DataFrame(flags, index=dx_row_names, columns=visit_dates)

#  6) Merge DX flags into the labs+vitals matrices 
print("\nMerging DX flags into labs+vitals matrices...")
for pid, df_dx in tqdm.tqdm(patient_dx_matrices.items(), desc="Merging", colour='green'):
    patient_data[pid] = pd.concat([patient_data[pid], df_dx], axis=0)

# Free intermediate structures we no longer need -- their content now lives
# inside patient_data via the concat above, so keeping them around just wastes
# memory right before the memory-heaviest step (saving).
del patient_dx_matrices, dx_events, code_counts
import gc
gc.collect()

#  7) Save -- INCREMENTALLY, one patient at a time (see streaming_pickle_utils.py) 
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving to: {output_path} (incrementally, one patient at a time)")
save_streamed_patient_dict(patient_data, output_path, desc="Saving")
print(" Saved successfully.")