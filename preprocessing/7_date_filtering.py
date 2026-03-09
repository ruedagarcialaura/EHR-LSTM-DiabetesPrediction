# -*- coding: utf-8 -*-
"""
Module: date_filtering.py
Purpose:
    Analyzes patient visit data to identify patterns of regular follow-ups.
    Categorizes patients based on visit regularity (3 months, 6 months, 1 year, 2 years, 3 years),
    overlays Type 2 Diabetes (T2D) diagnosis data, and saves filtered data for each regular 
    category into separate pickle files.
Step-by-Step Workflow:
    1. Data Loading:
        - Loads imputed patient visit matrices from pickle file
        - Loads T2D diagnosis data from CSV file to identify diabetic patients
        - Handles file not found errors gracefully with fallback empty sets
    2. Define Visit Interval Targets:
        - Sets tolerance threshold (25%) for visit interval regularity
        - Defines 5 standard interval categories: 3M, 6M, 1Y, 2Y, 3Y
        - Calculates acceptable windows around each interval with tolerance bounds
    3. Analyze Patient Visit Regularity (Dynamic Programming):
        - For each patient with 2+ visits:
            a. Extracts visit dates from dataframe columns
            b. Sorts visits chronologically while preserving original column indices
            c. For each interval category:
                - Uses DP to find longest sequence of visits matching that interval pattern
                - Requires minimum 3 consecutive regular visits for categorization
                - Tracks both total and T2D patient counts per category
                - Stores filtered dataframes (only regular visit columns) for saving
    4. Calculate Summary Statistics:
        - Counts analyzed patients and identifies irregular patients (not fitting any pattern)
        - Separates T2D patients into regular vs. irregular categories
        - Prints comprehensive results showing distribution across all categories
    5. Save Filtered Data:
        - Creates output directory for pickle files if not exists
        - Saves separate pickle file for each interval category containing:
            - Dictionary mapping patient IDs to filtered DataFrames (regular visits only)
        - Provides feedback on success/failure of file operations
    6. Visualize Results:
        - Creates side-by-side bar chart comparing total vs. T2D patient distribution
        - Displays counts for all visit pattern categories
        - Saves plot as high-resolution PNG image to output directory
        - Displays interactive plot to user
Output Files:
    - regular_patients_3_months.pkl
    - regular_patients_6_months.pkl
    - regular_patients_1_year.pkl
    - regular_patients_2_years.pkl
    - regular_patients_3_years.pkl
    - visit_patterns_distribution.png
Key Parameters:
    - TOLERANCE: 0.25 (25% acceptable deviation from target intervals)
    - Minimum visits for regular pattern: 3 consecutive visits
    - Output format: Dictionary[patient_id -> DataFrame with regular visit columns]
Note:
    A single patient may appear in multiple interval categories if their visit pattern
    satisfies multiple regularity conditions.
"""


"""
Created on Tue Jul  1 13:40:00 2025

@author: inanc
@changes by: Laura Rueda

This script analyzes patient visit data to identify patterns of regular follow-ups.
It categorizes patients based on visit regularity, overlays Type 2 Diabetes (T2D) diagnosis data,
and saves the filtered data for each regular category into separate pickle files.
"""





import pickle
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

# %% Load Imputed Patient Visit Data
file_path_clinical = r"preprocessing\output_pickles\5_patients_filtered_clinical_norm_100.pkl"
file_path_categorized = r"preprocessing\output_pickles\5_patients_filtered_all_categorized_norm_100.pkl"
file_path_standard= r"preprocessing\output_pickles\5_patients_filtered_standard_norm_100.pkl"

output_dir_clinical = r"preprocessing\output_pickles\7_filtered_patient_groups_clinical_NoImputed_100"
output_dir_categorized = r"preprocessing\output_pickles\7_filtered_patient_groups_categorized_NoImputed_100"
output_dir_standard = r"preprocessing\output_pickles\7_filtered_patient_groups_standard_NoImputed_100"


try:
    with open(file_path_clinical, "rb") as file:
        imputed_patients_matrices_all = pickle.load(file)
        print("File successfully loaded.")
except FileNotFoundError:
    print(f"Error: File not found at the specified path.\nPlease check the path: {file_path_clinical}")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")

# %% Load T2D Diagnosis Data
file_path2 = r"C:\Users\universidad\clases\iit\TFM\diabetesRiskPrediction\data\EARLIEST_DX_deid.csv"
try:
    df2 = pd.read_csv(file_path2)
    t2d_pids = set(df2['PATIENT_ID'].unique())
    print(f"T2D diagnosis data loaded for {len(t2d_pids)} patients.")
except FileNotFoundError:
    print(f"Error: T2D diagnosis file not found at {file_path2}. T2D data will not be included.")
    t2d_pids = set()  # Create an empty set if file is not found
except Exception as e:
    print(f"An error occurred while loading the T2D diagnosis file: {e}")
    t2d_pids = set()

# %% Analyze Regular Visit Intervals
print("\nAnalyzing patient visit regularity...")

TOLERANCE = 0.25
intervals = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "2 Years": 730,
    "3 Years": 1095,
}

interval_windows = {
    name: (days * (1 - TOLERANCE), days * (1 + TOLERANCE))
    for name, days in intervals.items()
}

# Dictionaries to store results
patient_counts = {name: 0 for name in intervals.keys()}
t2d_patient_counts = {name: 0 for name in intervals.keys()}
# This dictionary will hold the filtered dataframes for saving
filtered_data_for_saving = {name: {} for name in intervals.keys()}
analyzed_pids = []

# Iterate through each patient in the dataset
for pid, patient_df in tqdm.tqdm(imputed_patients_matrices_all.items(), desc="Analyzing Patients", colour='cyan'):
    if patient_df.shape[1] < 2:
        continue

    # Store original column references along with dates
    visit_info = []
    for i, col in enumerate(patient_df.columns):
        try:
            if isinstance(col, pd.Timestamp):
                visit_info.append({'date': col, 'original_index': i})
            elif isinstance(col, tuple) and len(col) > 0 and isinstance(col[0], pd.Timestamp):
                visit_info.append({'date': col[0], 'original_index': i})
            else:
                visit_info.append({'date': pd.to_datetime(col), 'original_index': i})
        except (ValueError, TypeError):
            continue
    
    # Sort by date while keeping track of original positions
    visit_info.sort(key=lambda x: x['date'])
    
    if len(visit_info) < 2:
        continue
    
    analyzed_pids.append(pid)
    is_t2d_patient = pid in t2d_pids
    
    # Check each interval for every patient
    for name, (lower_bound, upper_bound) in interval_windows.items():
        dp = [1] * len(visit_info)
        predecessor = [-1] * len(visit_info)  # To reconstruct the path

        for i in range(1, len(visit_info)):
            for j in range(i):
                diff_days = (visit_info[i]['date'] - visit_info[j]['date']).days
                if lower_bound <= diff_days <= upper_bound:
                    if 1 + dp[j] > dp[i]:
                        dp[i] = 1 + dp[j]
                        predecessor[i] = j
        
        max_len = max(dp)
        if max_len >= 3:
            patient_counts[name] += 1
            if is_t2d_patient:
                t2d_patient_counts[name] += 1
            
            # Reconstruct the sequence of visits
            path_indices = []
            end_index = dp.index(max_len)
            curr = end_index
            while curr != -1:
                path_indices.append(visit_info[curr]['original_index'])
                curr = predecessor[curr]
            path_indices.reverse()
            
            # Create the filtered dataframe and store it
            filtered_df = patient_df.iloc[:, path_indices]
            filtered_data_for_saving[name][pid] = filtered_df

# %% Calculate Irregular Visits and Print Summary

analyzed_patient_count = len(analyzed_pids)
all_regular_pids = set(pid for category_pids in filtered_data_for_saving.values() for pid in category_pids)
irregular_count = analyzed_patient_count - len(all_regular_pids)
patient_counts['Irregular'] = irregular_count

t2d_analyzed_pids = t2d_pids.intersection(set(analyzed_pids))
t2d_regular_pids = {pid for pid in all_regular_pids if pid in t2d_pids}
t2d_irregular_count = len(t2d_analyzed_pids) - len(t2d_regular_pids)
t2d_patient_counts['Irregular'] = t2d_irregular_count

print("\n Visit Pattern Analysis Results ")
print(f"Total patients with at least 2 visits: {analyzed_patient_count}")
print("NOTE: A patient can be counted in multiple regular categories.")
for name, count in patient_counts.items():
    t2d_count = t2d_patient_counts.get(name, 0)
    print(f"Category '{name}': {count} patients total ({t2d_count} with T2D)")

# %%  Save Filtered Data to Pickle Files
if not os.path.exists(output_dir_clinical):
    os.makedirs(output_dir_clinical)
    print(f"\nCreated directory: {output_dir_clinical}")
else:
    print(f"\nOutput directory already exists: {output_dir_clinical}")


print("\nSaving filtered patient data to pickle files...")
for name, data_dict in tqdm.tqdm(filtered_data_for_saving.items(), desc="Saving Pickle Files", colour='green'):
    if not data_dict:  # Skip empty categories
        print(f"  - Skipping '{name}' (no patients in this category).")
        continue
    
    # Sanitize the name for the filename
    filename_cat = name.replace(" ", "_").lower()
    output_path = os.path.join(output_dir_clinical, f"regular_patients_{filename_cat}.pkl")
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"  - Successfully saved {len(data_dict)} patients to {output_path}")
    except Exception as e:
        print(f"  - FAILED to save file for '{name}'. Error: {e}")


'''# %% Plot the Results
labels = list(patient_counts.keys())
total_counts = list(patient_counts.values())
t2d_counts = list(t2d_patient_counts.values())

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, total_counts, width, label='Total Patients', color='skyblue', edgecolor='black')
rects2 = ax.bar(x + width/2, t2d_counts, width, label='T2D Patients', color='coral', edgecolor='black')

ax.set_ylabel('Number of Patients', fontsize=14)
ax.set_title('Distribution of Patient Visit Patterns (Total vs. T2D)', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()'''

'''# Save the plot to the same directory
plot_output_path = os.path.join(output_dir_categorized, "visit_patterns_distribution.png")
try:
    plt.savefig(plot_output_path, dpi=300)
    print(f"\nPlot saved successfully to: {plot_output_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
    
plt.show()'''

