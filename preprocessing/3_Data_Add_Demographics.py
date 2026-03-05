# -*- coding: utf-8 -*-
"""
Data Preprocessing and Demographics Feature Engineering Module
This script processes demographic data from a CSV file and prepares it for machine learning
by performing data cleaning, categorical encoding, and feature engineering.
WORKFLOW STEPS:
===============
1. DATA LOADING
    - Reads demographic data from CSV file (deid_DEM.csv)
    - Handles FileNotFoundError gracefully with user feedback
    - Displays initial data structure for verification
2. INITIAL DATA CLEANING
    - Removes duplicate patient records, keeping the first occurrence
    - Corrects data entry errors by converting negative ages to absolute values
3. CATEGORICAL COLUMN CLEANING
    - Processes columns containing "ID:Value" format strings (GENDER, RACE, ETHNICITY)
    - Extracts the value portion after the colon separator
    - Removes leading/trailing whitespace from cleaned values
    - Gracefully handles non-string or malformed data
4. ONE-HOT ENCODING
    - Converts categorical variables into numeric binary features using scikit-learn's OneHotEncoder
    - Creates separate binary columns for each category (e.g., GENDER_Male, GENDER_Female)
    - Preserves original DataFrame index for proper alignment during join operation
    - Configures encoder to handle unknown categories in future predictions
5. DATAFRAME FINALIZATION
    - Removes original categorical columns (GENDER, RACE, ETHNICITY)
    - Drops unnecessary columns (ZIP_CODE, AIM_GROUP, index column)
    - Merges one-hot encoded features with remaining demographic features
    - Sets PATIENT_ID as the index for easier patient-level data retrieval
6. DATA PERSISTENCE
    - Saves processed DataFrame as pickle file (.pkl format)
    - Creates output directory structure if it doesn't exist
    - Provides success/error feedback for file operations
OUTPUT:
=======
- File: patients_demograph.pkl
- Location: preprocessing/output_pickles/
- Format: Serialized pandas DataFrame with engineered features and PATIENT_ID as index



Created on Tue Jun 3 02:47:35 2025
Revised on Thu Jul 17 04:10:00 2025

@author: inanc
"""

import pickle
import pandas as pd
# Import the OneHotEncoder from scikit-learn
from sklearn.preprocessing import OneHotEncoder

#  1. Load Data 
# This block now simply reports an error if the file isn't found.
try:
    path = r'C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\data\deid_DEM.csv'
    df_demog = pd.read_csv(path)
    print(" Data loaded successfully. Original head:")
    print(df_demog.head())
except FileNotFoundError:
    print(f" Error: The file was not found at {path}. Please check the file path.")
    # Exit the script if the data can't be loaded
    exit()

#  2. Initial Cleaning 
# Remove duplicate patients, keeping the first record
df_demog = df_demog.drop_duplicates(subset='PATIENT_ID', keep='first')

# Correct potential data entry errors where age is negative
df_demog['AGE_AT_END'] = df_demog['AGE_AT_END'].abs()

#  3. Robustly Clean Categorical Columns 
# Define a function to safely clean columns with "ID:Value" format
def clean_demographic_column(series):
    """
    Splits strings by ':' and returns the last element.
    Handles non-string or malformed values gracefully.
    """
    return series.astype(str).str.split(':').str[-1].str.strip()

# List of columns to clean and encode
categorical_cols = ['GENDER', 'RACE', 'ETHNICITY']
for col in categorical_cols:
    if col in df_demog.columns:
        df_demog[col] = clean_demographic_column(df_demog[col])

#  4. One-Hot Encode Using Scikit-learn 
# Separate the categorical data for encoding
categorical_data = df_demog[categorical_cols]

# Initialize the encoder
# sparse_output=False returns a regular numpy array, not a sparse matrix
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder to the data and transform it
encoded_data = encoder.fit_transform(categorical_data)

# Create a new DataFrame with the encoded data
# get_feature_names_out creates the new column names (e.g., 'GENDER_Male')
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df_demog.index # Preserve the original index for joining
)

#  5. Finalize DataFrame 
# Drop the original categorical columns and unnecessary columns from the main DataFrame
cols_to_drop = categorical_cols + ['ZIP_CODE', 'AIM_GROUP', 'Unnamed: 0']
df_demog.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Join the original data with the new one-hot encoded DataFrame
df_final = df_demog.join(encoded_df)

# Set PATIENT_ID as the index
df_final.set_index('PATIENT_ID', inplace=True)

print("\n Final processed data head (using OneHotEncoder):")
print(df_final.head())


#  6. Save the Processed DataFrame to the specific directory 
import os

# Define the target directory
output_dir = r"preprocessing\output_pickles"

# Create the directory if it does not exist (safety check)
os.makedirs(output_dir, exist_ok=True)

# Define the full file path
file_name = "patients_demograph.pkl"
full_output_path = os.path.join(output_dir, file_name)

try:
    with open(full_output_path, "wb") as file:
        pickle.dump(df_final, file)
    print(f"\n DataFrame successfully saved to:\n{full_output_path}")
except Exception as e:
    print(f"\n Error saving the file: {e}")