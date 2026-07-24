import pandas as pd
import re

# 1. Load the CSV (update the file path accordingly)
print("Loading the CSV file...")
df_proc = pd.read_csv(r'C:\Users\universidad\clases\iit\TFM\MODELO-LSTM\diabetesRiskPrediction\data\deid_proc_out.csv')

# Change this if your column has a different name (e.g., 'CPT4', 'CODE', etc.)
column_name = 'PROC_CODE' 

print(f"\nAnalyzing column: {column_name}")

# 2. Define a function to classify each code's format
def classify_format(raw_code):
    # Convert to string in case of nulls or pure numbers, and strip whitespace
    code = str(raw_code).strip().upper()
    

    # STEP A: Remove prefixes like "CPT4:", "CPT:", "HCPCS:" if present
    # If there is a colon (:), we keep only the part on the right
    if ':' in code:
        clean_code = code.split(':')[-1]
    else:
        clean_code = code
        
    # STEP B: Check the pattern using Regular Expressions
    # ^\d{5}$ means: starts (^), has exactly 5 digits (\d{5}), and ends ($)
    if re.match(r'^\d{5}$', clean_code):
        return 'Only 5 Digits (Standard CPT)'
        
    # ^[A-Z]\d{4}$ means: starts with 1 uppercase letter, followed by 4 digits
    elif re.match(r'^[A-Z]\d{4}$', clean_code):
        return 'Letter + 4 Digits (HCPCS/Other)'
        
    # If it contains letters but doesn't follow the HCPCS pattern above
    elif re.search(r'[A-Z]', clean_code):
        return 'Contains letters (Irregular format)'
        
    # If it is only numbers but NOT exactly 5 digits long
    elif clean_code.isdigit():
        return f'Only numbers, but {len(clean_code)} digits'
        
    else:
        return 'Unknown / Null'

# 3. Apply the function to the entire column to create a new "FORMAT_TYPE" column
df_proc['FORMAT_TYPE'] = df_proc[column_name].apply(classify_format)

# 4. Display the statistical summary
print("\n--- SUMMARY OF FORMATS FOUND ---")
summary = df_proc['FORMAT_TYPE'].value_counts()
print(summary)


# =====================================================================
# 5. DETAILED PRINTOUTS (Irregular, CPT, and HCPCS)
# =====================================================================

columns_to_show = ['PATIENT_ID', 'PROC_CODE', 'PROC_NAME']

# A) Filter and show the exactly 7 irregular cases
df_irregular = df_proc[df_proc['FORMAT_TYPE'] == 'Contains letters (Irregular format)']
print("\n--- DETAILS OF THE 7 IRREGULAR CODES ---")
print(df_irregular[columns_to_show])


# B) Filter and show standard CPT codes
df_cpt = df_proc[df_proc['FORMAT_TYPE'] == 'Only 5 Digits (Standard CPT)']
print("\n--- EXAMPLES OF STANDARD CPT CODES (5 Digits) ---")
# .drop_duplicates ensures we see 5 different procedures, not the same one 5 times
print(df_cpt[columns_to_show].drop_duplicates(subset=[column_name]).head(5))


# C) Filter and show standard HCPCS codes
df_hcpcs = df_proc[df_proc['FORMAT_TYPE'] == 'Letter + 4 Digits (HCPCS/Other)']
print("\n--- EXAMPLES OF HCPCS CODES (Letter + 4 Digits) ---")
print(df_hcpcs[columns_to_show].drop_duplicates(subset=[column_name]).head(5))


# =====================================================================
# 6. DETAILED PRINTOUTS (total unique codes and top 20 frequent codes)
# =====================================================================

print("\n--- CODE EXPLORATION ---")
# 1. Total unique codes
print(f"Total UNIQUE procedure codes: {df_proc['PROC_CODE'].nunique():,}")
print(f"Total UNIQUE procedure names: {df_proc['PROC_NAME'].nunique():,}")

# 2. Top 20 most frequent codes
print("\n--- TOP 20 MOST FREQUENT PROCEDURES ---")
top_20 = df_proc.groupby(['PROC_CODE', 'PROC_NAME']).size().reset_index(name='COUNT').sort_values('COUNT', ascending=False).head(20)
top_20.index = range(1, 21)
print(top_20.to_string())