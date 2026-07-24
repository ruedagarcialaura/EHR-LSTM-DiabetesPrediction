# -*- coding: utf-8 -*-
"""
Data Preprocessing and Demographics Feature Engineering Module
This script processes demographic data from a CSV file and prepares it for machine learning.

PARITY UPDATE: BERT's architecture (confirmed against BertConfig/
BEHRTForSequenceClassification in MLM.ipynb) has exactly 4 embedding types:
word (clinical code), position, token_type (visit/segment), and age. It does
NOT use gender/race/ethnicity in any form -- not as tokens, not as a separate
embedding channel. For a fair comparison, the LSTM's demographic branch is
now restricted to AGE_AT_END only -- GENDER/RACE/ETHNICITY are no longer
computed at all (previously one-hot encoded here, then silently dropped
later by the parity filter in step 5 -- simpler to just not compute them).

WORKFLOW STEPS:
===============
1. DATA LOADING
    - Reads demographic data from CSV file (deid_DEM.csv)
    - Handles FileNotFoundError gracefully with user feedback
2. INITIAL DATA CLEANING
    - Removes duplicate patient records, keeping the first occurrence
    - Corrects data entry errors by converting negative ages to absolute values
3. DATAFRAME FINALIZATION
    - Keeps only PATIENT_ID and AGE_AT_END (parity with BERT, see note above)
    - Sets PATIENT_ID as the index for easier patient-level data retrieval
4. DATA PERSISTENCE
    - Saves processed DataFrame as pickle file (.pkl format)
    - Creates output directory structure if it doesn't exist
    - Provides success/error feedback for file operations
OUTPUT:
=======
- File: 3_patients_demograph.pkl
- Location: preprocessing/output_pickles/
- Format: Serialized pandas DataFrame with AGE_AT_END and PATIENT_ID as index

Created on Tue Jun 3 02:47:35 2025
Revised on Thu Jul 17 04:10:00 2025
Revised again for BERT parity (AGE_AT_END only)

@author: inanc
@changes by: Laura Rueda
"""

import pickle
import pandas as pd
import os

#  1. Load Data 
try:
    path = r'C:\Users\universidad\clases\iit\TFM\MODELO-LSTM\diabetesRiskPrediction\data\deid_DEM.csv'
    df_demog = pd.read_csv(path)
    print(" Data loaded successfully. Original head:")
    print(df_demog.head())
except FileNotFoundError:
    print(f" Error: The file was not found at {path}. Please check the file path.")
    exit()

#  2. Initial Cleaning 
# Remove duplicate patients, keeping the first record
df_demog = df_demog.drop_duplicates(subset='PATIENT_ID', keep='first')

# Correct potential data entry errors where age is negative
df_demog['AGE_AT_END'] = df_demog['AGE_AT_END'].abs()

#  3. Finalize DataFrame -- AGE_AT_END only (BERT parity, see module docstring) 
df_final = df_demog[['PATIENT_ID', 'AGE_AT_END']].copy()
df_final.set_index('PATIENT_ID', inplace=True)

print("\n Final processed data head (AGE_AT_END only, for BERT parity):")
print(df_final.head())

#  4. Save the Processed DataFrame to the specific directory 
output_dir = r"preprocessing\output_pickles"
os.makedirs(output_dir, exist_ok=True)

file_name = "3_patients_demograph.pkl"
full_output_path = os.path.join(output_dir, file_name)

try:
    with open(full_output_path, "wb") as file:
        pickle.dump(df_final, file)
    print(f"\n DataFrame successfully saved to:\n{full_output_path}")
except Exception as e:
    print(f"\n Error saving the file: {e}")