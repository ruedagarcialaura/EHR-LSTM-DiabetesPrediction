# -*- coding: utf-8 -*-
"""
Categorization + Z-Score Normalization Script (Categorized/Binned branch)
===========================================================================
MEMORY-EFFICIENT REWRITE (full cohort, no subsampling needed).

The original version of this script processed the full patient population by
exploding every patient's DataFrame into one pandas.Series per feature (via
df.iterrows()), storing all of those in a dict, then deep-copying that whole
structure, and finally building a plain Python list of every single value per
feature to compute mean/std. With ~190 features x ~289k patients that created
tens of millions of small Python/pandas objects and duplicated the dataset in
memory -- this is what froze the machine, not the number of patients.

This version avoids all three problems:
1. Each patient stays as ONE DataFrame throughout (never split into
   per-feature Series), transformed in place.
2. No deepcopy -- we mutate `patient_data` directly, one patient at a time.
3. Z-score statistics (mean/std) are computed with running sum / sum-of-squares
   accumulators (a single pass over the data), never materializing a full
   Python list of every value.

Two passes over the data are still needed (you can't z-score before you know
the global mean/std), but each pass only ever holds ONE patient's DataFrame
in memory at a time plus a handful of running totals per feature -- not the
whole dataset duplicated.
"""

import re
import sys
import pickle
import numpy as np
import pandas as pd
import tqdm
from streaming_pickle_utils import save_streamed_patient_dict, load_streamed_patient_dict

print(" Categorization + Z-Score Normalization Script (memory-efficient) ")

#  1) Paths 
input_path = r"preprocessing\output_pickles\4_patients_filtered_unnormalized_75.pkl"
output_path_categorized = r"preprocessing\output_pickles\5_patients_filtered_all_categorized_norm_75.pkl"

#  2) Load Data 
try:
    patient_data = load_streamed_patient_dict(input_path, desc="Loading unnormalized data")
    print(f" Loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f" Could not load input: {e}")
    sys.exit(1)

#  3) Helpers 
def feature_name(idx):
    """Canonical feature name whether the row index is a tuple (NAME, CODE) or a plain string."""
    return idx[0] if isinstance(idx, tuple) and len(idx) > 0 else idx

_num_regex = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

def clean_number_like(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = _num_regex.search(s)
    return float(m.group(0)) if m else np.nan

#  4) CONFIGURATION 
features_to_skip = {
    'Time_Delta',
    'FLAG_Hypertensive_Crisis', 'FLAG_High_BUN', 'FLAG_Severe_Hyperkalemia', 'FLAG_Hypoxia_Emergency'
}

BINS_THRESHOLDS = {
    'ALT(SGPT)': [10, 49, 98, 245, 735],
    'AST(SGOT)': [10, 34, 68, 170, 510],
    'BUN': [6, 20, 50, 100, 250],
    'CHOLESTEROL': [200, 240],
    'HDL': [45, 80, 100],
    'HGB A1C': [5.7, 6.5],
    'POTASSIUM': [3.5, 5.1, 6.5],
    'TRIGLYCERIDE': [150, 200, 500],
    'VITAL_10541041_MEAN': [18.5, 25, 30, 35, 40],
    'VITAL_10541455_MEAN': [36.1, 37.3, 38.1, 39.1, 41.1],
    'VITAL_10541467_MEAN': [60, 100],
    'VITAL_10541503_MEAN': [12, 20],
    'VITAL_266705352_MEAN': [90, 92, 94, 97, 100],
    'VITAL_283305634_MEAN': [97.0, 99.1, 100.6, 102.4, 105.8],
    'VITAL_68924855_MEAN': [120, 130, 140, 180],

    # new categorization thresholds added for 7 more features (when >75% missingness eliminated only up to these variables are needed)
    'LDL-POCT': [100, 130, 160, 190],
    'VITAL_10541029_MEAN': [120, 130, 140, 181],
    'VITAL_34506073_MEAN': [12, 21, 31],
    'VITAL_68924858_MEAN': [89, 90, 95],

    # rest of the variables
    'ALB CONC': [3.5, 5.1],
    'FERRITIN': [20, 250],
    'INR': [0.8, 1.2, 2.0, 3.1],
    'PROTHROMBIN TIME': [11, 13.5],  # confirmed correct against ranges_with_links.xlsx (not [11.0, 13.6])
    'UR CREATININE': [20, 321],
    'UR TOTAL PROTEIN': [150, 3000],
    'm_Bilirubin.direct': [0.3],
    'VITAL_10155324_MEAN': [1, 4, 7],
    # NOTE: VITAL_10155611_MEAN ("oxygen flow rate") and VITAL_10155613_MEAN
    # ("FiO2") -- verified against raw data (actual ranges 0-492 and 0-969) --
    # do NOT match any plausible unit for what the reference spreadsheet
    # claims they are. These thresholds are copied AS-IS from the (frozen,
    # already-pretraining) BERT pipeline for parity, even though they push
    # almost all real values into a single top bin. Known limitation, to be
    # documented in the thesis rather than "fixed" unilaterally on one side.
    'VITAL_10155611_MEAN': [1, 24, 32, 46],
    'VITAL_10155613_MEAN': [22, 36, 61],
    # NOTE: VITAL_10541434_MEAN is mislabeled "head circumference" in
    # ranges_with_links.xlsx -- verified against raw data, it's actually SpO2
    # (99% of values in 92-103%). BERT already bins it with [32, 51], which
    # (given real values are ~90-114) pushes nearly everything into the top
    # bin. Kept identical to BERT for parity rather than "corrected" to
    # proper SpO2 thresholds, since BERT is frozen (mid-pretraining).
    'VITAL_10541434_MEAN': [32, 51],
    'VITAL_10541511_MEAN': [60, 101, 131],
    'VITAL_10541524_MEAN': [36.1, 37.3, 38.0],
    'VITAL_10541596_MEAN': [9, 13],
    'VITAL_14049161_MEAN': [40, 60]
}
# NOTE: weight (VITAL_283303272_MEAN), height (VITAL_279181488_MEAN), and Body
# Surface Area (VITAL_14049215_MEAN) are DELIBERATELY absent from this dict.
# BERT has no bin threshold for any of the three, so under the parity rule
# below they get dropped -- do NOT add them back here "just to bin them",
# that would silently undo the parity decision (this happened once already).

# Demographic columns that must never be dropped -- both models use them
# (LSTM as its demographics branch, BERT to compute the per-event dynamic AGE
# embedding). This matches training_newstructure_masked.py's DEMO_COLUMNS.
# Demographic columns that must never be dropped -- both models use them.
# BERT's architecture (confirmed: BertConfig/BEHRTForSequenceClassification,
# 4 embedding types -- word, position, token_type/segment, age) uses ONLY
# age, never gender/race/ethnicity in any form. So for parity, AGE_AT_END is
# the only demographic column now (script 3 no longer even computes the
# others -- this set just documents/enforces that).
DEMO_COLUMNS = {'AGE_AT_END'}

# BERT's pipeline only ever tokenizes labs/vitals that have a defined bin
# threshold -- anything else is silently dropped from its sequence. For a
# fair comparison, the LSTM should not have access to information the BERT
# never gets either. Rather than maintaining a manual list, this is enforced
# generically below: during Pass 1, any feature that is neither in
# BINS_THRESHOLDS (kept, binned) nor in features_to_skip (kept, demographics)
# is dropped. This automatically excludes things like weight/height/BSA and
# less common labs (e.g. LDL Calculated, Urine Ketones) that BERT never sees,
# without needing to keep a separate list in sync by hand.

#  5) PASS 1: clean + bin each patient in place, accumulate z-score stats 
print("\n PASS 1: Cleaning, binning, and accumulating z-score statistics ")

# Classify every feature ONCE (from the canonical feature set, which is
# uniform across patients after step 4's global >75%-missingness filter and
# step 2b's fixed top-500 DX columns) instead of re-checking set membership
# for every row of every patient (521 rows x 92k patients = ~48M redundant
# checks otherwise). This turns three O(rows x patients) Python loops into
# three small O(~20) lists reused across all patients.
sample_df = next(iter(patient_data.values()))
all_feature_names = [feature_name(idx) for idx in sample_df.index]

bin_names = [n for n in all_feature_names if n in BINS_THRESHOLDS]
passthrough_names = [n for n in all_feature_names if n in features_to_skip or str(n).startswith('DX_')]
demo_names = [n for n in all_feature_names if n in DEMO_COLUMNS]
keep_names = set(bin_names) | set(passthrough_names) | set(demo_names)
drop_names = [n for n in all_feature_names if n not in keep_names]
zscore_names = bin_names + demo_names  # binned features + demographics get z-scored; passthrough (DX/flags) doesn't

print(f" Feature classification (from {len(all_feature_names)} total features): "
      f"{len(bin_names)} binned, {len(demo_names)} demographic, {len(passthrough_names)} passthrough (DX/flags), "
      f"{len(drop_names)} to drop (not in BERT's vocabulary).")

running_sum = {}    # idx -> running sum of known values
running_sumsq = {}  # idx -> running sum of squares of known values
running_count = {}  # idx -> running count of known values

for pid in tqdm.tqdm(list(patient_data.keys()), desc="Pass 1/2: clean + bin", colour='cyan'):
    df = patient_data[pid]

    # Vectorized numeric cleaning across the WHOLE patient DataFrame at once
    # (instead of exploding it into one Series per row/feature).
    df = df.apply(lambda col: col.map(clean_number_like))

    # Drop features BERT never sees, for a fair comparison -- a single
    # vectorized call using the precomputed list, instead of rebuilding the
    # drop decision from scratch for every row of every patient.
    if drop_names:
        df = df.drop(index=drop_names, errors='ignore')

    # Apply clinical binning -- only touches the ~20 rows that need it.
    for name in bin_names:
        if name in df.index:
            thresholds = BINS_THRESHOLDS[name]
            df.loc[name] = df.loc[name].apply(lambda x: np.digitize(x, thresholds) if pd.notna(x) else np.nan)

    # Overwrite in place -- the old raw DataFrame is dropped/garbage-collected here,
    # so we never hold both the raw and cleaned versions at once.
    patient_data[pid] = df

    # Accumulate running stats -- only over the ~20 features that get z-scored.
    for name in zscore_names:
        if name not in df.index:
            continue
        vals = df.loc[name].to_numpy(dtype='float64')
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        running_sum[name] = running_sum.get(name, 0.0) + vals.sum()
        running_sumsq[name] = running_sumsq.get(name, 0.0) + np.square(vals).sum()
        running_count[name] = running_count.get(name, 0) + vals.size

print(f"Accumulated statistics for {len(running_count)} features.")
if drop_names:
    print(f"\nDropped {len(drop_names)} feature(s) not present in BERT's vocabulary (no bin threshold there):")
    for name in sorted(drop_names):
        print(f"  - {name}")

#  6) Finalize mean/std per feature 
stats = {}
for idx, count in running_count.items():
    mean = running_sum[idx] / count
    var = max(running_sumsq[idx] / count - mean ** 2, 0.0)  # E[x^2] - E[x]^2, clipped at 0 for float safety
    std = var ** 0.5
    stats[idx] = {'mean': mean, 'std': 1.0 if (std < 1e-6 or np.isnan(std)) else std}

#  7) PASS 2: apply z-score in place and finalize row order 
print("\n PASS 2: Applying z-score normalization ")

for pid in tqdm.tqdm(list(patient_data.keys()), desc="Pass 2/2: z-score", colour='green'):
    df = patient_data[pid]
    for idx, s in stats.items():
        if idx in df.index:
            df.loc[idx] = (df.loc[idx] - s['mean']) / s['std']
    patient_data[pid] = df.sort_index(key=lambda ix: ix.map(str))

#  8) Save 
print(f"\nSaving to: {output_path_categorized}")
save_streamed_patient_dict(patient_data, output_path_categorized, desc="Saving")
print(" Saved successfully.")
