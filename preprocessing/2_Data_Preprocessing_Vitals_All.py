"""
WORKFLOW OVERVIEW:
==================
1. DATA LOADING & INITIALIZATION:
    - Loads vital signs data from 'deid_vital.csv'
    - Converts 'Shifted_date' column to datetime format
    - Sorts data by PATIENT_ID and date for temporal consistency
    - Loads pre-existing base patient data from 'patients_lab.pkl' (pickle format)
    - This base data contains structured patient visit information
2. VITAL CODES DISCOVERY:
    - Dynamically discovers all unique vital codes from the loaded data
    - Creates a mapping from vital codes to standardized row names (e.g., "VITAL_HR")
    - This approach makes the script adaptable to different data sources
3. FEATURE ENGINEERING & AGGREGATION:
    - For each patient and their visits in the base data:
      * Extracts vital measurements within a ±15 day window around each visit date
      * Calculates mean statistics for all vital codes in that window
      * Creates new feature rows with names following pattern: "VITAL_CODE_MEAN"
    - Aggregates all computed features and appends them to the patient's visit matrix
    - Currently configured to calculate only MEAN (can be extended to include other stats)
    - Handles missing vitals by inserting NaN values
4. DATA FILTERING (CURRENTLY DISABLED):
    - Commented-out code section that:
      * Calculates missingness percentage for each vital feature
      * Identifies and removes features with >50% missing data
      * Applies filtering across all patient records
    - Can be uncommented to enable automatic feature removal based on data quality threshold
5. OUTPUT & PERSISTENCE:
    - Creates output directory if it doesn't exist
    - Saves augmented patient data (with vital features) to 'lab_vitals.pkl'
    - Preserves all data including NaN values for downstream processing
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import timedelta




#%% Load your vitals data
file_path_vitals = r"C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\data\deid_vital.csv"
df_vitals = pd.read_csv(file_path_vitals)
#df_vitals = pd.read_csv(file_path_vitals, nrows=len(pd.read_csv(file_path_vitals)) // 10)  # Load only 10% of the data for testing
print(" Vitals data successfully loaded from 'deid_vital.csv'.")

# Ensure 'Shifted_date' is in datetime format and sort the DataFrame
df_vitals['Shifted_date'] = pd.to_datetime(df_vitals['Shifted_date'])
df_vitals = df_vitals.sort_values(by=['PATIENT_ID', 'Shifted_date'])
print(" 'Shifted_date' column converted to datetime and data sorted.")

# Load the base patient data
file_path_base_data = r"C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\preprocessing\output_pickles\patients_lab.pkl"
try:
    with open(file_path_base_data, "rb") as file:
        base_patient_data = pickle.load(file)
    print(" Base patient data successfully loaded from 'patients_lab.pkl'.")
except FileNotFoundError:
    print(f" Error: File not found at {file_path_base_data}")
    sys.exit() # Use sys.exit() to correctly terminate the script
except Exception as e:
    print(f" An error occurred loading the pickle file: {e}")
    sys.exit() # Corrected from exit()

#%% Dynamically discover all unique vitals from the data file
unique_vitals = df_vitals['VITAL_CODE'].unique()
vital_row_map = {code: f"VITAL_{code.split(':')[-1]}" for code in unique_vitals}
print(f" Discovered {len(vital_row_map)} unique vital signs from the data file.")

#  CHANGE: Only calculate the mean 
stats_to_calc = ["mean"]
patient_with_vitals = {}

# Main processing loop
for pid, patient_vitals_df in df_vitals.groupby("PATIENT_ID"):
    if pid not in base_patient_data:
        continue
    processed_visit_matrix = base_patient_data[pid].copy()
    patient_visit_dates = pd.to_datetime(processed_visit_matrix.columns)
    temp_new_vital_rows = {}
    for visit_date in patient_visit_dates:
        start_date = visit_date - pd.Timedelta(days=15)
        end_date = visit_date + pd.Timedelta(days=15)
        mask = patient_vitals_df["Shifted_date"].between(start_date, end_date)
        window_vitals = patient_vitals_df.loc[mask]

        
        # Now only calculates the mean
        summary_stats = window_vitals.groupby("VITAL_CODE")["MEASUREMENT"].agg(stats_to_calc) if not window_vitals.empty else pd.DataFrame()
        
        for vital_code, base_row_name in vital_row_map.items():
            # This loop will now only run once (for "mean")
            for stat_name in stats_to_calc:
                full_row_name = f"{base_row_name}_{stat_name.upper()}"
                if full_row_name not in temp_new_vital_rows:
                    temp_new_vital_rows[full_row_name] = []
                if vital_code in summary_stats.index:
                    stat_value = summary_stats.loc[vital_code, stat_name]
                    temp_new_vital_rows[full_row_name].append(stat_value)
                else:
                    temp_new_vital_rows[full_row_name].append(np.nan)
    if temp_new_vital_rows:
        new_rows_df = pd.DataFrame(temp_new_vital_rows, index=processed_visit_matrix.columns).T
        processed_visit_matrix = pd.concat([processed_visit_matrix, new_rows_df])
    patient_with_vitals[pid] = processed_visit_matrix
print(" Vital sign feature engineering complete (Mean only).")

#%%  Step 1: Calculate and display missingness for each vital statistic 
# missing_data_df = pd.DataFrame()
# if patient_with_vitals:
#     new_vital_features = [f"{name}_{stat.upper()}" for name in vital_row_map.values() for stat in stats_to_calc]
#     all_visits_df = pd.concat(list(patient_with_vitals.values()), axis=1)
#     existing_new_features = [feat for feat in new_vital_features if feat in all_visits_df.index]
#     vitals_only_df = all_visits_df.loc[existing_new_features]
#     nan_counts = vitals_only_df.isna().sum(axis=1)
#     total_counts = vitals_only_df.shape[1]
#     missing_percentage = (nan_counts / total_counts) * 100
#     missing_data_df = pd.DataFrame({
#         'Vital Statistic': missing_percentage.index,
#         'Missing Percentage': missing_percentage.values
#     }).sort_values(by='Missing Percentage', ascending=False).reset_index(drop=True)
#     print("\n Missing Data Report by Vital Statistic (Mean only) ")
#     print(missing_data_df.to_string())
#     print("-\n")

# #%%  Step 2: Identify and remove vital features with >50% missingness 
# if not missing_data_df.empty:
#     vitals_to_remove = missing_data_df[missing_data_df['Missing Percentage'] > 50]['Vital Statistic'].tolist()
    
#     if vitals_to_remove:
#         print(f"🗑️ Identifying {len(vitals_to_remove)} vital features to remove due to >50% missingness.")
        
#         patients_filtered = {}
#         for pid, data in patient_with_vitals.items():
#             filtered_data = data.drop(labels=vitals_to_remove, errors='ignore')
#             patients_filtered[pid] = filtered_data
            
#         print(f" Successfully removed high-missingness features from all {len(patients_filtered)} patients.")
#     else:
#         print(" No vital features exceeded the 50% missingness threshold. No features were removed.")
#         patients_filtered = patient_with_vitals
# else:
#     print("⚠️ No vitals data to process for filtering.")
#     patients_filtered = patient_with_vitals


#%% Define output directory
output_dir = r"preprocessing\output_pickles"
os.makedirs(output_dir, exist_ok=True)
print(f" Output will be saved to: {output_dir}")

# Save the augmented data (with NaNs, without any filtering)
output_path = os.path.join(output_dir, "lab_vitals.pkl")
try:
    with open(output_path, "wb") as f:
        #  FIX: Save patient_with_vitals instead of the non-existent patients_filtered 
        pickle.dump(patient_with_vitals, f)
    print(f" Final augmented data saved successfully to {output_path}")
except Exception as e:
    print(f" Error saving the file: {e}")