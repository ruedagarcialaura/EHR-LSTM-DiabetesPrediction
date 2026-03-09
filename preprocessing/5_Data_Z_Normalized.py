# -*- coding: utf-8 -*-
"""
====================================================
OVERVIEW:
This script performs comprehensive preprocessing of clinical patient data with two distinct
normalization approaches: Clinical Normalization (with domain-specific adjustments) and 
Standard Normalization (pure Z-score without clinical transformations).
STEP-BY-STEP EXECUTION FLOW:
STEP 1: INITIALIZATION & IMPORTS
--
- Imports required libraries: re, sys, pickle, numpy, pandas, collections
- Defines input/output file paths for patient data storage
- Sets up logging/print statements for process tracking
STEP 2: DATA LOADING
--
- Loads pickled patient data from input_parquets directory
- Contains unnormalized clinical measurements for all patients
- Handles exceptions if file not found or corrupted
- Reports total number of patients loaded
STEP 3: HELPER FUNCTIONS DEFINITION

- _norm_label(): Standardizes label formatting (tuple or string conversion with stripping)
- clean_number_like(): Extracts numerical values from mixed-type strings using regex
    * Uses regex to find floating-point numbers (including scientific notation)
    * Returns NaN if no valid number found
- Ensures data uniformity before processing
STEP 4: DATA SANITIZATION & FEATURE GROUPING

- Processes first 50% of patient data (subsampling for efficiency)
- Converts all feature values to clean floats using clean_number_like()
- Creates base_patient_maps with cleaned numerical Series
- Identifies all unique feature keys across patients
- Groups features by canonical name (tuple[0]) in feature_groups dictionary
- Creates two independent copies for parallel processing:
    * clinical_maps: for clinical transformation pipeline
    * standard_maps: for standard Z-score only pipeline
STEP 5: CONFIGURATION SETUP

Defines clinical adjustment rules:
- LOG_TRANSFORM_VARS: Features requiring log1p transformation (e.g., liver enzymes)
- HARD_CAP_VARS: Features with absolute maximum limits (age, vital signs, BMI)
- CAP_AND_FLAG_VARS: Features requiring value capping AND binary flag creation for extremes
- LOW_FLAG_VARS: Features requiring flags for values below threshold (hypoxia)
- features_to_skip: Categorical/temporal features excluded from normalization
STEP 6: CLINICAL TRANSFORMATIONS (PHASE 1)
--
For each feature in clinical_maps:
- Applies strategy-specific transformations:
    * Log transformation: log1p(max(x, 0)) for skewed distributions
    * Hard capping: clips values to defined upper limits
    * Cap & flag: both clips values AND creates binary flag for threshold breaches
    * Low flagging: creates flags for dangerously low values
- Stores new binary flag features (e.g., FLAG_Hypertensive_Crisis)
- Injects flags into clinical_maps as new feature columns
STEP 7: Z-SCORE NORMALIZATION (PHASE 2 & 3)

Applies apply_zscore() function to both datasets:
- CLINICAL PATH: Normalizes already-transformed features
- STANDARD PATH: Normalizes original features (no prior transformations)
Process for each path:
1. Calculates global statistics (mean, std) across all patients per feature
2. Handles edge case: std < 1e-6 → sets std=1.0 (prevents division issues)
3. Applies Z-score formula: (value - mean) / std to all feature values
4. Skips features in features_to_skip list (categorical/flags)
STEP 8: FINALIZATION & STORAGE
-
- Converts both clinical_maps and standard_maps back to DataFrame format
- Sorts features alphabetically by string representation
- Saves two pickle files:
    * patients_filtered_clinical_norm.pkl: with clinical preprocessing + Z-scores
    * patients_filtered_standard_norm.pkl: with Z-scores only (no clinical adjustments)
- Outputs confirmation messages with file paths
OUTPUT FILES:
==============
1. Clinical Normalized: Patient data with log transforms, capping, flags, then Z-score
2. Standard Normalized: Patient data with only Z-score normalization
     Both are indexed by [patient_id], contain features as columns, values are normalized floats
"""




"""
Final Clinical Data Preprocessing Script (Updated)
========================================
1. Clinical Normalization: Includes Log-transforms, Capping, and Flagging.
2. Standard Normalization: Pure Z-score normalization without clinical adjustments.
"""

""""""

import re
import sys
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import tqdm

print(" Clinical Preprocessing and Standard Normalization Script ")

#  1) Paths 
input_path = r"preprocessing\output_pickles\4_patients_filtered_unnormalized_100.pkl"
output_path_clinical = r"preprocessing\output_pickles\5_patients_filtered_clinical_norm_100.pkl"
output_path_standard = r"preprocessing\output_pickles\5_patients_filtered_standard_norm_100.pkl"
output_path_categorized = r"preprocessing\output_pickles\5_patients_filtered_categorized_norm_75.pkl" # new script

#  2) Load Data 
try:
    with open(input_path, "rb") as f:
        patient_data = pickle.load(f)
    print(f" Loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f" Could not load input: {e}")
    sys.exit(1)

#  3) Helpers 
def _norm_label(x):
    if isinstance(x, tuple):
        return tuple(str(p).strip() for p in x)
    return str(x).strip()

_num_regex = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

def clean_number_like(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = _num_regex.search(s)
    return float(m.group(0)) if m else np.nan

#  4) Sanitize and Group Features 
print("Sanitizing data...")
# We create a base map of cleaned numerical values first
base_patient_maps = {}
all_keys = set()
#patient_data = dict(patient_data)
patient_data_subsample = dict(list(patient_data.items())[:len(patient_data)//2])  # One-half of original data

for pid, df in tqdm.tqdm(patient_data_subsample.items(), desc="Processing Patients"):
    # Convert every row to clean floats immediately
    cleaned_rows = {}
    for idx, row in df.iterrows():
        cleaned_rows[_norm_label(idx)] = pd.to_numeric(pd.Series(row).map(clean_number_like), errors='coerce')
    base_patient_maps[pid] = cleaned_rows
    all_keys.update(cleaned_rows.keys())
print(f"Identified {len(all_keys)} unique features across patients.")

feature_groups = defaultdict(list)
for key in tqdm.tqdm(all_keys, desc="Grouping features"):
    canonical_name = key[0] if isinstance(key, tuple) and len(key) > 0 else key
    feature_groups[canonical_name].append(key)
print(f"Grouped into {len(feature_groups)} canonical features.")

# We create two copies: one for Clinical logic, one for Standard logic
clinical_maps = copy.deepcopy(base_patient_maps)
print("clinical maps copy created.")
#standard_maps = copy.deepcopy(base_patient_maps)
#print("standard maps copy created.") 
#categorized_maps = copy.deepcopy(base_patient_maps)
#print("categorized maps copy created.")

#  5) CONFIGURATION 
LOG_TRANSFORM_VARS = {'ALT(SGPT)', 'AST(SGOT)', 'CHOLESTEROL', 'LDL-POCT', 'HDL', 'TRIGLYCERIDE'}
HARD_CAP_VARS = {
    'AGE_AT_END': 104.0, 'VITAL_10541041_MEAN': 70.0, 'BMI': 70.0,
    'VITAL_266705352_MEAN': 100.0, 'VITAL_283305634_MEAN': 106.0, 'VITAL_10541455_MEAN': 41.1
}
CAP_AND_FLAG_VARS = {
    'VITAL_68924855_MEAN': {'cap': 180.0, 'flag_thresh': 180.0, 'flag_name': 'FLAG_Hypertensive_Crisis'},
    'BUN': {'cap': 250.0, 'flag_thresh': 100.0, 'flag_name': 'FLAG_High_BUN'},
    'POTASSIUM': {'cap': 10.0, 'flag_thresh': 6.5, 'flag_name': 'FLAG_Severe_Hyperkalemia'}
}
LOW_FLAG_VARS = {'VITAL_266705352_MEAN': {'flag_thresh': 90.0, 'flag_name': 'FLAG_Hypoxia_Emergency'}}

features_to_skip = {
    'GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS', 'RACE_B', 'RACE_H', 'RACE_NA', 'RACE_NI', 'RACE_W',
    'ETHNICITY_N', 'ETHNICITY_NI', 'ETHNICITY_Y', 'Time_Delta',
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
    'VITAL_68924855_MEAN': [120, 130, 140, 180]
}

categorized_features_names = set(BINS_THRESHOLDS.keys())

#  6) ROUTE A: Clinical Transformations 
print("\n PHASE 1: Applying Clinical Transformations ")
new_flag_features = defaultdict(dict)

for name, variants in feature_groups.items():
    if name in features_to_skip: continue
    
    strategy = "standard"
    if any(v in name for v in LOG_TRANSFORM_VARS): strategy = "log"
    elif name in HARD_CAP_VARS: strategy = "hard_cap"
    elif name in CAP_AND_FLAG_VARS: strategy = "cap_flag"
    check_low_flag = name in LOW_FLAG_VARS

    for pid, data_map in clinical_maps.items():
        for key in variants:
            if key in data_map:
                vals = data_map[key]
                if check_low_flag:
                    cfg = LOW_FLAG_VARS[name]
                    new_flag_features[pid][cfg['flag_name']] = (vals < cfg['flag_thresh']).astype(float)
                
                if strategy == "log": vals = np.log1p(vals.clip(lower=0))
                elif strategy == "hard_cap": vals = vals.clip(upper=HARD_CAP_VARS[name])
                elif strategy == "cap_flag":
                    cfg = CAP_AND_FLAG_VARS[name]
                    new_flag_features[pid][cfg['flag_name']] = (vals > cfg['flag_thresh']).astype(float)
                    vals = vals.clip(upper=cfg['cap'])
                
                data_map[key] = vals
print("Clinical transformations applied.")

# Inject Clinical Flags
for pid, flags in new_flag_features.items():
    for flag_name, flag_series in flags.items():
        clinical_maps[pid][(flag_name,)] = flag_series
print("Clinical flags added.")


'''print("\n PHASE 1.5: Applying Clinical Categorization (Binnings) ")
for name, variants in tqdm.tqdm(feature_groups.items(), desc="Processing feature groups"):
    if name in BINS_THRESHOLDS:
        thresholds = BINS_THRESHOLDS[name]
        for pid, data_map in tqdm.tqdm(categorized_maps.items(), desc="Categorizing data"):
            for key in variants:
                if key in data_map:
                    data_map[key] = data_map[key].apply(
                        lambda x: np.digitize(x, thresholds) if pd.notna(x) else np.nan
                    )
print("Clinical categorization applied.")'''

#  7) PHASE 2 & 3: Z-Score  

def apply_zscore(target_maps, skip_list):
    # Calculate global stats
    stats = {}
    current_keys = set().union(*(m.keys() for m in target_maps.values()))
    
    for key_tuple in tqdm.tqdm(current_keys, desc="Processing variables for z-score", colour = 'pink'):
        name = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
        if name in skip_list: continue
        
        all_vals = []
        for pid in target_maps:
            if key_tuple in target_maps[pid]:
                all_vals.extend(target_maps[pid][key_tuple].dropna().tolist())
        
        if all_vals:
            s = pd.Series(all_vals)
            std = s.std()
            stats[key_tuple] = {'mean': s.mean(), 'std': 1.0 if (pd.isna(std) or std < 1e-6) else std}
    
    # Apply normalization
    for pid in tqdm.tqdm(target_maps.keys(), desc="Normalizing patients with z-score", colour = 'green'):
        for key_tuple, s_val in stats.items():
            if key_tuple in target_maps[pid]:
                target_maps[pid][key_tuple] = (target_maps[pid][key_tuple] - s_val['mean']) / s_val['std']

print("Calculating and applying Clinical Z-Scores...")
apply_zscore(clinical_maps, features_to_skip)

#print("Calculating and applying Standard Z-Scores (No Capping/Log)...")
#apply_zscore(standard_maps, features_to_skip)

#print("Calculating and applying Z-scores TO ALL VARIABLES (INCLUDING CATEGORIZED ONES) except for the skip list (GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS',...)")
#apply_zscore(categorized_maps, features_to_skip)


'''# 9) Visualize results for one patient
#1st patient
first_pid = next(iter(categorized_maps))
#2nd patient
second_pid = next(iter(list(categorized_maps.keys())[1:]))

patient_df = pd.DataFrame.from_dict(categorized_maps[first_pid], orient='index')
patient_df = patient_df.sort_index(key=lambda idx: idx.map(str))
num_visits_to_show = min(3, patient_df.shape[1])
preview_table = patient_df.iloc[:50, :num_visits_to_show]

print(f" TABLE PREVIEW FOR PATIENT (ID: {first_pid}) ")
print(preview_table.to_string())'''

#  8) Save Both 
def finalize_and_save(maps, path):
    final_data = {}
    for pid, data_map in tqdm.tqdm(maps.items(), desc="Finalizing and saving data", colour='cyan'):
        df = pd.DataFrame.from_dict(data_map, orient='index')
        df = df.sort_index(key=lambda idx: idx.map(str))
        final_data[pid] = df
    with open(path, "wb") as f:
        pickle.dump(final_data, f)
    print(f" Saved to: '{path}'")

finalize_and_save(clinical_maps, output_path_clinical)
#finalize_and_save(standard_maps, output_path_standard)
#finalize_and_save(categorized_maps, output_path_categorized)