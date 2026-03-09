"""
1. **Data Loading**: Reads laboratory test data from a CSV file containing
    patient test results with SUNQUEST codes.
2. **Code Standardization**: Converts SUNQUEST (SUNQ) laboratory codes to
    standardized LOINC (Logical Observation Identifiers Names and Codes) codes
    using a mapping CSV file, improving data interoperability.
3. **Test Name Standardization**: For each unique LAB_CODE, standardizes the
    LAB_NAME by selecting the most frequently occurring name (mode) to handle
    naming inconsistencies in the source data.
4. **Manual Corrections**: Applies manual name corrections for specific LOINC
    codes, particularly for insulin-related tests, to ensure consistency in
    test nomenclature.
5. **Data Cleaning**: Removes unnecessary columns (UNITS, AIM_GROUP), filters
    out lab codes with entirely missing RESULT values, and converts RESULT
    values to numeric format for analysis.
6. **Visit Grouping**: Organizes patient test results into distinct visits by
    identifying gaps > 30 days between consecutive test dates. Assigns tests
    to visits based on the visit start date (earliest test in the visit).
7. **Pivot Table Creation**: Transforms data for each patient into a pivot
    table with (LAB_NAME, LAB_CODE) pairs as rows and visit dates as columns,
    with mean aggregation for multiple tests per visit.
8. **Multi-Visit Filtering**: Retains only patients with 2 or more visits,
    as these are required for longitudinal analysis.
9. **Index Standardization**: Reindexes all patient matrices to a common,
    comprehensive MultiIndex containing all unique lab tests. Missing values
    are filled with NaN, ensuring uniform row structure across all patients.
10. **Visualization**: Generates a bar chart showing the distribution of
    patients across lab codes to assess data coverage before analysis.
11. **Persistence**: Saves the final reindexed patient matrices dictionary
    to a pickle file for downstream processing and analysis.
Returns:
    Output is persisted to disk as a pickle file containing the
    reindexed_patients_matrices dictionary.
"""


"""Data Preprocessing Pipeline for Laboratory Test Results
This module performs comprehensive preprocessing of laboratory test data, including:
1. Loading laboratory test data from CSV files
2. Converting SUNQUEST codes to LOINC standardized codes
3. Standardizing laboratory test names per test code
4. Applying manual naming corrections for specific tests
5. Data cleaning and validation
6. Organizing test results by patient visits (grouping by 30-day gaps)
7. Creating pivot tables with visits as columns and tests as rows
8. Filtering patients to retain only those with 2+ visits
9. Reindexing all patient matrices to ensure consistent test coverage
10. Visualizing test availability across the patient population
11. Persisting processed data as pickle files
Output:
    - reindexed_patients_matrices: Dictionary mapping patient IDs to DataFrames
      where rows are (LAB_NAME, LAB_CODE) pairs and columns are visit dates,
      with all patients having identical row indices for downstream analysis.
"""


"""
Created on Mon Mar 10 12:31:28 2025

@author: inanc
@changes by: Laura Rueda
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


#%% --- 1. Data Loading and Initial Setup ---
file_path = r"C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\data\deid_Lab_out.csv"
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
print("Original DataFrame head:")
print(df.head())

#%% --- 2. Replace SUNQ with LOINC codes ---
map_path = r'C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\data\dpi_lab_map.csv'
lab_map = pd.read_csv(map_path)
sunq_to_loinc = dict(zip(lab_map['SUNQUEST_CODE'], lab_map['LOINC_CODE']))

# Replace SUNQ codes with LOINC codes in the main dataframe
df['LAB_CODE'] = df['LAB_CODE'].apply(lambda code: sunq_to_loinc.get(code, code))
#
df = df[~df['LAB_CODE'].str.startswith("SUNQ")]
print("\nAfter replacing SUNQ codes:")
print(df.head())

#%% --- 3. Standardize LAB_NAME for each LAB_CODE ---
df['LAB_NAME'] = df.groupby('LAB_CODE')['LAB_NAME'].transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
unique_labs = df[['LAB_CODE', 'LAB_NAME']].drop_duplicates()
print("\nUpdated unique LAB_CODE and LAB_NAME pairs:")
print(unique_labs.head())

#%% --- 4. Apply Manual Naming for specific LAB_CODEs ---
manual_lab_changes = {
    "LOINC:13927-9": "m_Pancreatic islet cell Ab [Titer] in Serum",
    "LOINC:1558-6": "m_Fasting glucose [Mass/volume] in Serum or Plasma",
    "LOINC:18262-6": "m_Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay",
    "LOINC:1968-7": "m_Bilirubin.direct [Mass/volume] in Serum or Plasma",
    "LOINC:20448-7": "m_Insulin [Units/volume] in Serum or Plasma",
    "LOINC:27379-7": "m_Insulin [Units/volume] in Serum or Plasma --1 hour post XXX challenge",
    "LOINC:27826-7": "m_Insulin [Units/volume] in Serum or Plasma --2 hours post XXX challenge",
    "LOINC:6901-3": "m_Insulin Free [Units/volume] in Serum or Plasma",
    "LOINC:27330-0": "Insulin 1.5 Hr post Unsp challenge Qn",
    "LOINC:27827-5": "Insulin [Units/volume] in Serum or Plasma --30 minutes post XXX challenge",
}
df['LAB_NAME'] = df.apply(lambda row: manual_lab_changes.get(row['LAB_CODE'], row['LAB_NAME']), axis=1)
unique_labs_df_2 = df[['LAB_CODE', 'LAB_NAME']].drop_duplicates().sort_values(by='LAB_CODE').reset_index(drop=True)
print("\nFinal unique lab codes and names:")
print(unique_labs_df_2.head())

#%% --- 5. Data Cleaning ---

# removes unnecessary columns
df.drop(columns=['UNITS', 'AIM_GROUP'], inplace=True)
# remove LAB_CODEs with all missing RESULT values
missing_lab_codes = df.groupby('LAB_CODE')['RESULT'].apply(lambda x: x.isna().all()).loc[lambda x: x].index
# ~df means keep rows where LAB_CODE is not in missing_lab_codes
df = df[~df['LAB_CODE'].isin(missing_lab_codes)]
print(f"\nLAB_CODES removed due to all NaN values: {list(missing_lab_codes)}")
#convert RESULT to numeric, coercing errors to NaN
df['RESULT'] = pd.to_numeric(df['RESULT'], errors='coerce')


#%% --- 6. Group data into visits for each patient ---
patients_matrices = {}
df['Shifted_date'] = pd.to_datetime(df['Shifted_date'])
gap_threshold = 30

for pid, patient_df in df.groupby("PATIENT_ID"):
    patient_df = patient_df.sort_values('Shifted_date').copy()
    patient_df['gap_days'] = patient_df['Shifted_date'].diff().dt.days.fillna(0)
    patient_df['visit_group'] = (patient_df['gap_days'] > gap_threshold).cumsum()
    patient_df['Visit_Start'] = patient_df.groupby('visit_group')['Shifted_date'].transform('min')
    
    df_pivot = patient_df.pivot_table(index=['LAB_NAME', 'LAB_CODE'],
                                      columns='Visit_Start',
                                      values='RESULT',
                                      aggfunc='mean')
    df_pivot.attrs["PATIENT_ID"] = pid
    patients_matrices[pid] = df_pivot

print(f"\nCreated matrices for {len(patients_matrices)} patients.")

#%% --- 7. Filter out patients with fewer than 2 visits ---
original_count = len(patients_matrices)
filtered_patients_matrices = {
    pid: df for pid, df in patients_matrices.items() if df.shape[1] >= 2
}
filtered_count = len(filtered_patients_matrices)
eliminated_count = original_count - filtered_count
print(f"Filtering complete! Retained {filtered_count} patients with at least 2 visits.")
print(f"{eliminated_count} patients were eliminated.")


#%% --- 8. CORRECTION: Ensure all patients have all unique vitals ---
# Create a comprehensive MultiIndex of all unique lab codes and names
all_vitals_index = pd.MultiIndex.from_frame(unique_labs_df_2[['LAB_NAME', 'LAB_CODE']])

# Reindex each patient's DataFrame to include all unique vitals.
# This ensures every patient matrix has the same set of rows, filling missing ones with NaN.
reindexed_patients_matrices = {}
for pid, df_pivot in filtered_patients_matrices.items():
    # Reindex the patient's DataFrame against the master list of all vitals
    df_reindexed = df_pivot.reindex(all_vitals_index)
    
    # Preserve the patient ID as metadata
    df_reindexed.attrs["PATIENT_ID"] = pid
    
    # Store the reindexed DataFrame in the final dictionary
    reindexed_patients_matrices[pid] = df_reindexed

print("\nReindexing complete. All patient matrices now have the same vitals.")


#%% --- 9. Example Output & Verification ---
# You can now use 'reindexed_patients_matrices' for any subsequent steps.
# The shape of each patient's matrix will now have the same number of rows.
patient_id_to_check = 4 # Use a patient ID that exists in your filtered set
if patient_id_to_check in reindexed_patients_matrices:
    print(f"\nExample Matrix for Patient ID: {patient_id_to_check}")
    print(f"Shape of matrix: {reindexed_patients_matrices[patient_id_to_check].shape}")
    print("Head of the patient's reindexed matrix:")
    print(reindexed_patients_matrices[patient_id_to_check].head())
else:
    print(f"\nPatient {patient_id_to_check} not found in the filtered dataset.")


#%% --- 10. Visualization of patient counts per lab code (using the original filtered data) ---
lab_code_counts = {}
for pid, df_pivot in filtered_patients_matrices.items():
    patient_lab_codes = df_pivot.index.get_level_values('LAB_CODE').unique()
    for lab_code in patient_lab_codes:
        if lab_code not in lab_code_counts:
            lab_code_counts[lab_code] = 0
        if df_pivot.xs(lab_code, level='LAB_CODE', drop_level=False).notna().any().any():
            lab_code_counts[lab_code] += 1

lab_code_counts_df = pd.DataFrame.from_dict(lab_code_counts, orient='index', columns=['Patient Count'])
lab_code_counts_df.sort_values(by='Patient Count', ascending=False, inplace=True)

plt.figure(figsize=(15, 7))
plt.bar(lab_code_counts_df.index.astype(str), lab_code_counts_df['Patient Count'])
plt.xlabel('Lab Code')
plt.ylabel('Number of Patients')
plt.title('Number of Patients with at least one valid value for each Lab Code (Before Reindexing)')
plt.xticks(rotation=90)
plt.tight_layout() # Adjust layout to make room for rotated x-axis labels
plt.show()


#%% --- 11. Save the final, corrected dictionary to a specific directory ---

# Define the specific directory
output_dir = r"C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\preprocessing\output_pickles"

# Create the directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define the full file path
file_name = "patients_lab.pkl"
full_output_path = os.path.join(output_dir, file_name)

# Save the dictionary
with open(full_output_path, "wb") as file:
    pickle.dump(reindexed_patients_matrices, file)

print(f"\nDictionary saved successfully to:\n{full_output_path}")