# -*- coding: utf-8 -*-
"""
Final Corrected Script: This version uses a more robust method (.iloc)
to access data rows, resolving issues with mixed-type indexes.
"""
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import sys

print(" Final Plotting Script ")

#  1. Define Paths 
input_path = r"C:\Users\inanc\OneDrive - Illinois Institute of Technology\Desktop\Research\Sequential\Filter_and_Label\Vitals\Imputation_with_Vitals\filtered_patient_groups_labelchanged\Preprocessing\Data\patients_filtered_unnormalized.pkl"
output_dir = r"C:\Users\inanc\OneDrive - Illinois Institute of Technology\Desktop\Research\Sequential\Filter_and_Label\Vitals\Imputation_with_Vitals\filtered_patient_groups_labelchanged\Preprocessing\Data\feature_plots_unnormalized"

#  2. Load Data 
try:
    with open(input_path, "rb") as f:
        patient_data = pickle.load(f)
    print(f" Successfully loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f" Could not load the file: {e}")
    sys.exit()

#  3. Generate Plots 
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved in: '{output_dir}'")

# Get a sorted, unique list of all features
all_features = set(idx for df in patient_data.values() for idx in df.index)
sorted_features = sorted(list(all_features), key=str)
print(f"Found {len(sorted_features)} unique features to plot.")

for feature in sorted_features:
    plt.figure(figsize=(15, 7))
    ax = plt.gca()
    
    data_plotted = False
    # Loop through each patient
    for pid, df in patient_data.items():
        # Check if the feature exists in this patient's DataFrame
        if feature in df.index:
            try:
                #  ROBUST ACCESS METHOD 
                # 1. Get the integer position of the feature in the index
                row_position = df.index.get_loc(feature)
                # 2. Access the data using the reliable integer position
                data_series = df.iloc[row_position]
                
                # Check if there is any valid data to plot
                if not data_series.isnull().all():
                    ax.plot(df.columns, data_series, marker='o', linestyle='-', alpha=0.3, markersize=3)
                    data_plotted = True
            except Exception as e:
                # This safety catch can help diagnose any other unexpected issues
                print(f"Could not process feature '{feature}' for patient {pid}. Error: {e}")

    feature_title = str(feature)
    
    # Only save the plot if data was actually added to it
    if data_plotted:
        ax.set_title(f"Feature: {feature_title} (All Patients - Raw Values)", fontsize=16, weight='bold')
        ax.set_xlabel("Visit Date", fontsize=12)
        ax.set_ylabel("Raw Value", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        safe_feature_name = "".join(c for c in feature_title if c.isalnum() or c in (' ', '_')).rstrip()
        plot_filename = f"{safe_feature_name}.png"
        full_plot_path = os.path.join(output_dir, plot_filename)
        
        plt.tight_layout()
        plt.savefig(full_plot_path)
        print(f" Generated plot for feature: {feature_title}")
    else:
        print(f"⚠️ SKIPPED plot for feature: '{feature_title}' (No valid data found).")

    plt.close()

print("\n Script Finished ")