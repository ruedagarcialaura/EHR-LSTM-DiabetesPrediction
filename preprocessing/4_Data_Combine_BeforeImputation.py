# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:49:39 2025
Revised on Feb 1 2026

This script combines demographic data with lab/vital data, calculates feature
missingness, removes unreliable features, and then normalizes the remaining data.
(Time Delta calculation has been removed)
@author: inanc
@changes by: Laura Rueda
"""
"""
# Step-by-Step Explanation

**1. Setup & Imports (Lines 1-24)**
- Defines script metadata and imports libraries (pandas, numpy, pickle, sklearn)
- Sets the base directory where input/output files are stored

**2. Load Input Files (Lines 26-40)**
- Loads two pickle files:
    - `patients_demograph.pkl`: Demographic info (age, gender, etc.) indexed by patient ID
    - `lab_vitals.pkl`: Lab and vital measurements for each patient across multiple visits

**3. Combine Data (Lines 42-55)**
- Merges demographic data with lab/vital data for each patient
- Replicates demographic features across all visits (since demographics don't change per visit)
- Creates `patient_data_combined`: dictionary where each patient ID maps to a combined DataFrame

**4. Calculate Missingness (Lines 57-71)**
- Concatenates all patient data into one large matrix
- Calculates what percentage of values are missing for each feature
- Displays a report showing features ranked by missingness

**5. Remove High-Missingness Features (Lines 73-89)**
- Identifies features with >95% missing values
- Removes those features from all patients' DataFrames
- Stores cleaned data in `patients_filtered`

**6. Save Unnormalized Data (Lines 91-99)**
- Saves the filtered (but not yet normalized) data to a pickle file for reference

**7. Normalize Features (Lines 101-130)**
- Uses MinMaxScaler to scale all features to [0, 1] range
- Creates unique MultiIndex columns (patient_id, visit_date) to prevent data leakage
- Applies global normalization across all patients simultaneously
- Splits normalized data back into individual patient DataFrames

**8. Save Final Output (Lines 132-140)**
- Saves the fully processed, normalized data to `patients_before_imp.pkl`
"""



import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import tqdm
from streaming_pickle_utils import save_streamed_patient_dict, load_streamed_patient_dict

#  1. Define Directories & Load Data 
# All inputs and outputs are in this specific folder
base_dir = r"preprocessing\output_pickles"

file_path_demographics = os.path.join(base_dir, "3_patients_demograph.pkl")
file_path_lab_vitals = os.path.join(base_dir, "2b_lab_vitals_dx.pkl")  # now includes DX flags (see 2b_Data_Preprocessing_DX.py)

# Ensure the directory exists for saving later
os.makedirs(base_dir, exist_ok=True)

try:
    with open(file_path_demographics, "rb") as file:
        demographics_df = pickle.load(file)
    print(" Demographics file loaded successfully.")

    # Uses the streaming loader (auto-detects old- or new-style pickle) --
    # this file now includes DX flags and can be large enough that a plain
    # single pickle.load() risks a MemoryError, same as the one hit when
    # 2b_Data_Preprocessing_DX.py originally tried to save it in one shot.
    patient_matrices = load_streamed_patient_dict(file_path_lab_vitals, desc="Loading labs+vitals+DX")
    print(" Lab/Vitals file loaded successfully.")

except FileNotFoundError as e:
    print(f" Error: File not found. Please check the path: {e.filename}")
    sys.exit()
except Exception as e:
    print(f" An error occurred while loading files: {e}")
    sys.exit()

#  1b. Flatten label row indices (fixes a mixed-index bug) 
# Labs (from step 1) have a 2-level MultiIndex row label: (LAB_NAME, LAB_CODE).
# Vitals/DX/demographics (steps 2/2b/3) have plain single-level string labels.
# pd.concat-ing the two doesn't keep a proper MultiIndex -- the result is a
# mixed index of tuples and strings, which breaks `.loc[idx]` later (a tuple
# label like ('TROPONIN I', 'LOINC:10839-9') gets misread as "row=TROPONIN I,
# column=LOINC:10839-9" instead of one label, since the index isn't formally
# a pd.MultiIndex).
#
# Dropping LAB_CODE naively (keeping LAB_NAME only) is NOT safe on its own:
# step 1's manual naming corrections deliberately unify multiple LAB_CODEs
# under one LAB_NAME (e.g. two historical codes for the same test), so a
# single patient can legitimately have two rows with the same LAB_NAME. That
# needs a real MERGE (not just relabeling), or you'd end up with duplicate
# index labels -- which is exactly what caused the next crash (`.loc[idx]`
# returning a 2-row DataFrame instead of one Series). Merging with the mean
# of whatever's non-null per visit combines the two historical codes'
# readings for the same test into one row, the same way multiple vitals
# readings already get mean-aggregated within a visit window.
n_merged = 0
for pid in tqdm.tqdm(list(patient_matrices.keys()), desc="Flattening + merging lab index", colour='cyan'):
    df = patient_matrices[pid]
    new_index = [idx[0] if isinstance(idx, tuple) else idx for idx in df.index]
    if len(new_index) != len(set(new_index)):
        n_merged += 1
        df.index = new_index
        df = df.groupby(level=0).mean()  # merges duplicate-named rows, skipping NaN
    else:
        df.index = new_index
    patient_matrices[pid] = df

print(f" Flattened lab row index for all patients ({n_merged} patients had duplicate "
      f"LAB_NAMEs from multiple historical LAB_CODEs, now merged by mean).")

#  2. Combine Demographic and Lab/Vital Data 
# Mutates patient_matrices IN PLACE instead of building a separate
# patient_data_combined dict, for the same reason as the filtering step below
# -- avoids two full copies of the (now DX-inflated) dataset being resident
# in memory at once.
demographics_df.index = demographics_df.index.astype(int)  # Ensure index is integer for matching

patients_to_drop = []
for pid, lab_vital_df in tqdm.tqdm(patient_matrices.items(), desc="Combining patient data", colour='blue'):
    # Check if the patient exists in the demographic data
    if pid not in demographics_df.index:
        patients_to_drop.append(pid)
        continue

    # Get the demographic data for the current patient (it's a Series)
    demo_row = demographics_df.loc[pid]

    # Create a new DataFrame from the demographic data, replicating it for each visit
    demo_as_df = pd.DataFrame(
        {visit_col: demo_row for visit_col in lab_vital_df.columns},
        index=demo_row.index
    )

    # Combine the lab/vital data with the new demographic DataFrame, in place
    patient_matrices[pid] = pd.concat([lab_vital_df, demo_as_df])

# Drop patients with no demographic match (can't do this while iterating above)
for pid in patients_to_drop:
    del patient_matrices[pid]

patient_data_combined = patient_matrices  # same dict, just renamed for clarity below

print(f" Combined demographic data for {len(patient_data_combined)} patients.")

#  3. Calculate Feature Missingness (memory-efficient, streaming) 
# NOTE: the original approach concatenated every patient's DataFrame side-by-side
# into ONE giant DataFrame (columns = total visits across the whole cohort --
# potentially millions of columns). With the full patient population that risks
# freezing the machine the same way the normalization scripts did. Instead, we
# accumulate NaN counts and total counts per feature incrementally, one patient
# at a time, never holding more than one patient's data in memory at once.
if not patient_data_combined:
    print(" No patient data to process after combining. Exiting.")
    sys.exit()

nan_counts = {}
total_counts = {}
for pid, df in tqdm.tqdm(patient_data_combined.items(), desc="Calculating missingness", colour='yellow'):
    # Vectorized: one call gives NaN counts for every row at once, instead of
    # calling df.loc[idx] once per row (571 rows x 92k patients = ~53M slow
    # indexed lookups -- this was the actual cause of the ~1h runtime here).
    row_nan_counts = df.isna().sum(axis=1)
    n_cols = df.shape[1]
    for idx, nan_count in row_nan_counts.items():
        nan_counts[idx] = nan_counts.get(idx, 0) + int(nan_count)
        total_counts[idx] = total_counts.get(idx, 0) + n_cols

missing_percentage = {idx: (nan_counts[idx] / total_counts[idx]) * 100 for idx in nan_counts}

# Create and display the missingness report
missing_report_df = pd.DataFrame({
    'Feature': list(missing_percentage.keys()),
    'Missing_Percentage': list(missing_percentage.values())
}).sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)

print("\n Feature Missingness Report ")
print(missing_report_df.to_string())
print("-\n")

#  4. Eliminate Features with >75% Missingness 
# Identify features to remove
features_to_remove = missing_report_df[missing_report_df['Missing_Percentage'] > 75]['Feature'].tolist()

if features_to_remove:
    print(f" Identifying {len(features_to_remove)} features to remove due to >75% missingness.")

    # Mutate patient_data_combined IN PLACE instead of building a separate
    # patients_filtered dict -- with DX now adding ~500 rows per patient,
    # having both the original and filtered versions fully resident in
    # memory at once risks another MemoryError, same shape of problem as
    # the one hit when saving 2b_Data_Preprocessing_DX.py's output.
    for pid in tqdm.tqdm(list(patient_data_combined.keys()), desc="Filtering features", colour='magenta'):
        patient_data_combined[pid] = patient_data_combined[pid].drop(labels=features_to_remove, errors='ignore')
    patients_filtered = patient_data_combined

    print(f" Successfully removed high-missingness features from all {len(patients_filtered)} patients.")
else:
    print(" No features exceeded the 75% missingness threshold. No features were removed.")
    patients_filtered = patient_data_combined


#  5. Save Unnormalized Data 
unnormalized_output_path = os.path.join(base_dir, "4_patients_filtered_unnormalized_75.pkl")
try:
    save_streamed_patient_dict(patients_filtered, unnormalized_output_path, desc="Saving unnormalized data")
    print(f" Unnormalized data successfully saved to:\n{unnormalized_output_path}")
except Exception as e:
    print(f" Error saving the unnormalized file: {e}")