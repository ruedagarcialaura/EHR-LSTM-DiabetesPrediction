import re
import sys
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import tqdm

print(" Categorization + Z-Score Normalization Script ")

#  1) Paths 
input_path = r"preprocessing\output_pickles\4_patients_filtered_unnormalized_100.pkl"
output_path_categorized = r"preprocessing\output_pickles\5_patients_filtered_all_categorized_norm_100.pkl"

#  2) Load Data 
try:
    with open(input_path, "rb") as f:
        patient_data = pickle.load(f)
    print(f" Loaded data for {len(patient_data)} patients.")
except Exception as e:
    print(f" Could not load input: {e}")
    sys.exit(1)

#  3) Helpers 
def _norm_label(x):
    if isinstance(x, tuple):
        return tuple(str(p).strip() for p in x)
    return str(x).strip()

_num_regex = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')

def clean_number_like(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    m = _num_regex.search(s)
    return float(m.group(0)) if m else np.nan

#  4) Sanitize and Group Features 
print("Sanitizing data...")
# We create a base map of cleaned numerical values first
base_patient_maps = {}
all_keys = set()
#patient_data = dict(patient_data)
patient_data_subsample = dict(list(patient_data.items())[:len(patient_data)//2])  # One-half of original data

for pid, df in tqdm.tqdm(patient_data_subsample.items(), desc="Processing Patients"):
    # Convert every row to clean floats immediately
    cleaned_rows = {}
    for idx, row in df.iterrows():
        cleaned_rows[_norm_label(idx)] = pd.to_numeric(pd.Series(row).map(clean_number_like), errors='coerce')
    base_patient_maps[pid] = cleaned_rows
    all_keys.update(cleaned_rows.keys())
print(f"Identified {len(all_keys)} unique features across patients.")

feature_groups = defaultdict(list)
for key in tqdm.tqdm(all_keys, desc="Grouping features"):
    canonical_name = key[0] if isinstance(key, tuple) and len(key) > 0 else key
    feature_groups[canonical_name].append(key)
print(f"Grouped into {len(feature_groups)} canonical features.")

# We create one copy for categorization.  
categorized_maps = copy.deepcopy(base_patient_maps)
print("categorized maps copy created.")

#  5) CONFIGURATION 
features_to_skip = {
    'GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS', 'RACE_B', 'RACE_H', 'RACE_NA', 'RACE_NI', 'RACE_W',
    'ETHNICITY_N', 'ETHNICITY_NI', 'ETHNICITY_Y', 'Time_Delta',
    'FLAG_Hypertensive_Crisis', 'FLAG_High_BUN', 'FLAG_Severe_Hyperkalemia', 'FLAG_Hypoxia_Emergency'
}

BINS_THRESHOLDS = {
    'ALT(SGPT)': [10, 49, 98, 245, 735],
    'AST(SGOT)': [10, 34, 68, 170, 510],
    'BUN': [6, 20, 50, 100, 250],
    'CHOLESTEROL': [200, 240],
    'HDL': [45, 80, 100],
    'HGB A1C': [5.7, 6.5],
    'POTASSIUM': [3.5, 5.1, 6.5],
    'TRIGLYCERIDE': [150, 200, 500],
    'VITAL_10541041_MEAN': [18.5, 25, 30, 35, 40],
    'VITAL_10541455_MEAN': [36.1, 37.3, 38.1, 39.1, 41.1],
    'VITAL_10541467_MEAN': [60, 100],
    'VITAL_10541503_MEAN': [12, 20],
    'VITAL_266705352_MEAN': [90, 92, 94, 97, 100],
    'VITAL_283305634_MEAN': [97.0, 99.1, 100.6, 102.4, 105.8],
    'VITAL_68924855_MEAN': [120, 130, 140, 180],

    # new categorization thresholds added for 7 more features (when >75% missingness eliminated only up to these variables are needed)
    'LDL-POCT': [100, 130, 160, 190],
    'VITAL_10541029_MEAN': [90, 120, 140, 160],
    'VITAL_14049215_MEAN': [1.6, 1.9],
    'VITAL_279181488_MEAN': [150, 180],
    'VITAL_283303272_MEAN': [50, 86, 101],
    'VITAL_34506073_MEAN': [12, 20],
    'VITAL_68924858_MEAN': [90, 95],

    # rest of the variables
    'ALB CONC': [3.4, 5.4],
    'FERRITIN': [20, 250],
    'INR': [0.8, 1.1, 3.0],
    'PROTHROMBIN TIME': [11, 13.5],
    'UR CREATININE': [500, 2000],
    'UR TOTAL PROTEIN': [150, 500],
    'm_Bilirubin.direct': [0.3],
    'VITAL_10155324_MEAN': [1, 4, 7],
    'VITAL_10155611_MEAN': [1, 7],
    'VITAL_10155613_MEAN': [0.22, 0.50],
    'VITAL_10541434_MEAN': [52, 58],
    'VITAL_10541511_MEAN': [90, 95],
    'VITAL_10541524_MEAN': [35.0, 37.5],
    'VITAL_10541596_MEAN': [9, 13],
    'VITAL_14049161_MEAN': [40, 60]
}

categorized_features_names = set(BINS_THRESHOLDS.keys())


print("\n PHASE 1: Applying Clinical Categorization (Binnings) ")
for name, variants in tqdm.tqdm(feature_groups.items(), desc="Processing feature groups"):
    if name in BINS_THRESHOLDS:
        thresholds = BINS_THRESHOLDS[name]
        for pid, data_map in tqdm.tqdm(categorized_maps.items(), desc="Categorizing data"):
            for key in variants:
                if key in data_map:
                    data_map[key] = data_map[key].apply(
                        lambda x: np.digitize(x, thresholds) if pd.notna(x) else np.nan
                    )
print("Clinical categorization applied.")

#  7) PHASE 2 & 3: Z-Score  

def apply_zscore(target_maps, skip_list):
    # Calculate global stats
    stats = {}
    current_keys = set().union(*(m.keys() for m in target_maps.values()))
    
    for key_tuple in tqdm.tqdm(current_keys, desc="Processing variables for z-score", colour = 'pink'):
        name = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple
        if name in skip_list: continue
        
        all_vals = []
        for pid in target_maps:
            if key_tuple in target_maps[pid]:
                all_vals.extend(target_maps[pid][key_tuple].dropna().tolist())
        
        if all_vals:
            s = pd.Series(all_vals)
            std = s.std()
            stats[key_tuple] = {'mean': s.mean(), 'std': 1.0 if (pd.isna(std) or std < 1e-6) else std}
    
    # Apply normalization
    for pid in tqdm.tqdm(target_maps.keys(), desc="Normalizing patients with z-score", colour = 'green'):
        for key_tuple, s_val in stats.items():
            if key_tuple in target_maps[pid]:
                target_maps[pid][key_tuple] = (target_maps[pid][key_tuple] - s_val['mean']) / s_val['std']


print("Calculating and applying Z-scores TO ALL VARIABLES (INCLUDING CATEGORIZED ONES) except for the skip list (GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS',...)")
apply_zscore(categorized_maps, features_to_skip)

# 9) Visualize results for one patient

#1st patient
first_pid = next(iter(categorized_maps))
#2nd patient
second_pid = next(iter(list(categorized_maps.keys())[1:]))

patient_df = pd.DataFrame.from_dict(categorized_maps[first_pid], orient='index')
patient_df = patient_df.sort_index(key=lambda idx: idx.map(str))
num_visits_to_show = min(3, patient_df.shape[1])
preview_table = patient_df.iloc[:50, :num_visits_to_show]

print(f" TABLE PREVIEW FOR PATIENT (ID: {first_pid}) ")
print(preview_table.to_string())

#  8) Save Both 
def finalize_and_save(maps, path):
    final_data = {}
    for pid, data_map in tqdm.tqdm(maps.items(), desc="Finalizing and saving data", colour='cyan'):
        df = pd.DataFrame.from_dict(data_map, orient='index')
        df = df.sort_index(key=lambda idx: idx.map(str))
        final_data[pid] = df
    with open(path, "wb") as f:
        pickle.dump(final_data, f)
    print(f" Saved to: '{path}'")

finalize_and_save(categorized_maps, output_path_categorized)