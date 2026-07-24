import pandas as pd

# 1. Load data
print("Loading procedures CSV...")
df_proc = pd.read_csv('data/deid_proc_out.csv')

# 2. Define Category Dictionary (Your equivalent to BINS)
# Here you need to decide (or search online/in papers) which CPT/HCPCS codes go into which category.
# THIS IS AN EXAMPLE, you must adapt it:
PROCEDURE_CATEGORIES = {
    'PROC_GLUCOSE_TESTS': ['CPT:82962', 'CPT:82947', 'CPT:83036'], # Glucose and HbA1c codes
    'PROC_CBC_BLOOD': ['CPT:85025', 'HCPCS:G0306'], # Complete Blood Count (CBC)
    'PROC_METABOLIC_PANEL': ['CPT:80048', 'CPT:80053'], # Metabolic panels
    # ... add relevant categories for diabetes
}

# 3. Create a new column for the "Super Category"
def categorize_procedure(code):
    for category_name, code_list in PROCEDURE_CATEGORIES.items():
        if code in code_list:
            return category_name
    return 'OTHER' # Or ignore the ones you don't care about

df_proc['PROC_CATEGORY'] = df_proc['PROC_CODE'].apply(categorize_procedure)

# Filter out the ones we don't care about (optional, to clean up noise)
df_proc_filtered = df_proc[df_proc['PROC_CATEGORY'] != 'OTHER']

# 4. Convert to static matrix (Patients x Categories)
# We want to know whether a patient had a procedure from that category or not
print("Pivoting table to create static variables...")

# We use pivot_table or crosstab
# This counts how many times each procedure category was performed per patient
patient_proc_counts = pd.crosstab(df_proc_filtered['PATIENT_ID'], df_proc_filtered['PROC_CATEGORY'])

# Since we want binary variables (0 = Not done, 1 = Done at least once):
patient_proc_binary = (patient_proc_counts > 0).astype(int)

# Add prefix to make it clear they are demographic/static variables
patient_proc_binary.columns = ['DEMO_' + col for col in patient_proc_binary.columns]

patient_proc_binary.reset_index(inplace=True)

print("Sample of the new static columns:")
print(patient_proc_binary.head())

# 5. Save to use in the neural network
patient_proc_binary.to_csv('preprocessing/patient_procedures_static.csv', index=False)