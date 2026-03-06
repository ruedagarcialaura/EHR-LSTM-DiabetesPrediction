# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:49:39 2025
Revised on Feb 1 2026

This script combines demographic data with lab/vital data, calculates feature
missingness, removes unreliable features, and then normalizes the remaining data.
(Time Delta calculation has been removed)
@author: inanc
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

#  1. Define Directories & Load Data 
# All inputs and outputs are in this specific folder
base_dir = r"preprocessing\output_pickles"

file_path_demographics = os.path.join(base_dir, "3_patients_demograph.pkl")
file_path_lab_vitals = os.path.join(base_dir, "2_lab_vitals.pkl")

# Ensure the directory exists for saving later
os.makedirs(base_dir, exist_ok=True)

try:
    with open(file_path_demographics, "rb") as file:
        demographics_df = pickle.load(file)
    print(" Demographics file loaded successfully.")

    with open(file_path_lab_vitals, "rb") as file:
        patient_matrices = pickle.load(file)
    print(" Lab/Vitals file loaded successfully.")

except FileNotFoundError as e:
    print(f" Error: File not found. Please check the path: {e.filename}")
    sys.exit()
except Exception as e:
    print(f" An error occurred while loading files: {e}")
    sys.exit()

#  2. Combine Demographic and Lab/Vital Data 
patient_data_combined = {}
demographics_df.index = demographics_df.index.astype(int)  # Ensure index is integer for matching

# Iterate through each patient's lab/vital data
for pid, lab_vital_df in tqdm.tqdm(patient_matrices.items(), desc="Combining patient data", colour='blue'):
    # Check if the patient exists in the demographic data
    if pid not in demographics_df.index:
        continue

    # Get the demographic data for the current patient (it's a Series)
    demo_row = demographics_df.loc[pid]

    # Create a new DataFrame from the demographic data, replicating it for each visit
    demo_as_df = pd.DataFrame(
        {visit_col: demo_row for visit_col in lab_vital_df.columns},
        index=demo_row.index
    )

    # Combine the lab/vital data with the new demographic DataFrame
    combined_df = pd.concat([lab_vital_df, demo_as_df])
    patient_data_combined[pid] = combined_df

print(f" Combined demographic data for {len(patient_data_combined)} patients.")

#  3. Calculate Feature Missingness 
# Concatenate all patient data into one large DataFrame for global analysis
if not patient_data_combined:
    print(" No patient data to process after combining. Exiting.")
    sys.exit()

all_patient_data_df = pd.concat(list(patient_data_combined.values()), axis=1)

# Calculate the missing percentage for each feature (row)
nan_counts = all_patient_data_df.isna().sum(axis=1)
total_visits = all_patient_data_df.shape[1]
missing_percentage = (nan_counts / total_visits) * 100

# Create and display the missingness report
missing_report_df = pd.DataFrame({
    'Feature': missing_percentage.index,
    'Missing_Percentage': missing_percentage.values
}).sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)

print("\n Feature Missingness Report ")
print(missing_report_df.to_string())
print("-\n")

#  4. Eliminate Features with >75% Missingness 
# Identify features to remove
features_to_remove = missing_report_df[missing_report_df['Missing_Percentage'] > 75]['Feature'].tolist()

if features_to_remove:
    print(f" Identifying {len(features_to_remove)} features to remove due to >75% missingness.")

    # Create a new dictionary for the filtered data
    patients_filtered = {}
    for pid, data in tqdm.tqdm(patient_data_combined.items(), desc="Filtering features", colour='magenta'):
        # Drop the identified rows from each patient's DataFrame
        filtered_data = data.drop(labels=features_to_remove, errors='ignore')
        patients_filtered[pid] = filtered_data

    print(f" Successfully removed high-missingness features from all {len(patients_filtered)} patients.")
else:
    print(" No features exceeded the 75% missingness threshold. No features were removed.")
    patients_filtered = patient_data_combined


#  5. Save Unnormalized Data 
unnormalized_output_path = os.path.join(base_dir, "patients_filtered_unnormalized.pkl")
try:
    with open(unnormalized_output_path, "wb") as f:
        pickle.dump(patients_filtered, f)
    print(f" Unnormalized data successfully saved to:\n{unnormalized_output_path}")
except Exception as e:
    print(f" Error saving the unnormalized file: {e}")


