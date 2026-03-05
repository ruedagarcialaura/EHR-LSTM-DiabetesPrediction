# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:36:44 2025

@author: inanc
"""

# -*- coding: utf-8 -*-
"""
This script generates normalized histograms for EVERY feature,
eliminating values with a Z-score higher than 3, and saves them.
"""
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import zscore

print(" Normalized Feature Histogram Generation Script (Z-score > 3 eliminated) ")

#  1. Define Paths 
input_path = r"preprocessing\output_pickles\patients_filtered_unnormalized.pkl"
output_dir = r"preprocessing\output_pickles\z_score_histograms" 

#  2. Load Data 
try:
    with open(input_path, "rb") as f:
        patient_data = pickle.load(f)
    print(f" Successfully loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f" Could not load the file: {e}")
    sys.exit()

#  3. Generate and Save Normalized Histograms for ALL features 
os.makedirs(output_dir, exist_ok=True)
print(f"Normalized histograms will be saved in: '{output_dir}'")

#  CHANGE: Automatically get ALL features from the data 
all_features = set(idx for df in patient_data.values() for idx in df.index)
sorted_features = sorted(list(all_features), key=str)
print(f"Found {len(sorted_features)} features. Generating a normalized histogram for each...")

# Loop through every feature found in the data
for feature in sorted_features:
    feature_title = str(feature)
    print(f"Generating normalized histogram for: {feature_title}")
    
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
        print(f"    ⚠️ SKIPPED: No data found for '{feature_title}'.")
        continue

    #  Z-Score Normalization and Outlier Elimination 
    # Calculate Z-scores for the *entire* series first
    z_scores = np.abs(zscore(y_series)) 
    
    # Filter out values where the absolute Z-score is greater than 3
    # Apply the filter to the original series, not the z_scores directly for plotting
    filtered_y_series = y_series[z_scores <= 3] 

    if filtered_y_series.empty:
        print(f"    ⚠️ SKIPPED: All data points for '{feature_title}' were outliers (Z-score > 3).")
        continue

    # Create the histogram for the filtered data
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_y_series, bins=100, edgecolor='k', alpha=0.7) # Plotting filtered data
    plt.title(f"Normalized Distribution of {feature_title} (Z-score < 3)", fontsize=16) # Updated title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot to the new output directory
    safe_feature_name = "".join(c for c in feature_title if c.isalnum() or c in (' ', '_')).rstrip()
    plot_filename = f"normalized_histogram_{safe_feature_name}.png" 
    full_plot_path = os.path.join(output_dir, plot_filename)
    
    plt.tight_layout()
    plt.savefig(full_plot_path)
    plt.close()

print("\n Script Finished ")