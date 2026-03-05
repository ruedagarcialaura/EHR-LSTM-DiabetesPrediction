# -*- coding: utf-8 -*-
"""
This script generates a histogram for EVERY feature to visualize its distribution.
"""
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

print(" Full Feature Histogram Generation Script ")

#  1. Define Paths 
input_path = r"C:\Users\inanc\OneDrive - Illinois Institute of Technology\Desktop\Research\Sequential\Filter_and_Label\Vitals\Imputation_with_Vitals\filtered_patient_groups_labelchanged\Preprocessing\Data\patients_filtered_unnormalized.pkl"
output_dir = r"C:\Users\inanc\OneDrive - Illinois Institute of Technology\Desktop\Research\Sequential\Filter_and_Label\Vitals\Imputation_with_Vitals\filtered_patient_groups_labelchanged\Preprocessing\Data\histograms"

#  2. Load Data 
try:
    with open(input_path, "rb") as f:
        patient_data = pickle.load(f)
    print(f" Successfully loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f"❌ Could not load the file: {e}")
    sys.exit()

#  3. Generate and Save Histograms for ALL features 
os.makedirs(output_dir, exist_ok=True)
print(f"Histograms will be saved in: '{output_dir}'")

#  CHANGE: Automatically get ALL features from the data 
all_features = set(idx for df in patient_data.values() for idx in df.index)
sorted_features = sorted(list(all_features), key=str)
print(f"Found {len(sorted_features)} features. Generating a histogram for each...")

# Loop through every feature found in the data
for feature in sorted_features:
    feature_title = str(feature)
    print(f"Generating histogram for: {feature_title}")
    
    # Gather all values for the feature
    all_values = []
    for df in patient_data.values():
        if feature in df.index:
            # Use the robust access method
            row_position = df.index.get_loc(feature)
            data_series = df.iloc[row_position]
            all_values.extend(data_series.values)

    y_series = pd.to_numeric(pd.Series(all_values), errors='coerce').dropna()

    if y_series.empty:
        print(f"   ⚠️ SKIPPED: No data found for '{feature_title}'.")
        continue

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_series, bins=100, edgecolor='k', alpha=0.7)
    plt.title(f"Distribution of {feature_title}", fontsize=16)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot
    safe_feature_name = "".join(c for c in feature_title if c.isalnum() or c in (' ', '_')).rstrip()
    plot_filename = f"histogram_{safe_feature_name}.png"
    full_plot_path = os.path.join(output_dir, plot_filename)
    
    plt.tight_layout()
    plt.savefig(full_plot_path)
    plt.close()

print("\n Script Finished ")