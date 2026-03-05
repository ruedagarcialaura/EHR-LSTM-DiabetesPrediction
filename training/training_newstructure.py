# -*- coding: utf-8 -*-
"""
Patient-Level Type 2 Diabetes (T2D) Risk Prediction Using Hybrid LSTM-DNN Model
This module implements an end-to-end machine learning pipeline for predicting T2D risk 
at the patient level using a hybrid architecture combining LSTM for sequential visit data 
and DNN for demographic features.
PIPELINE OVERVIEW:
==================
1. CONFIGURATION & DATA LOADING
    - Loads patient visit sequences from pickle file (imputed_patients_matrices_all)
    - Loads T2D diagnosis dates from EARLIEST_DX_deid.csv
    - Loads demographic information from deid_DEM.csv
    - Loads external test set from matched_test.csv
    - Configures hyperparameters (LSTM hidden size, batch size, learning rate, etc.)
2. DATA VALIDATION & EXPLORATION
    - Compares T2D patient counts from different sources (AIM_GROUP vs EARLIEST_DX)
    - Analyzes visit count distribution for T2D patients
    - Inspects individual patient timelines and diagnosis dates
3. FEATURE ENGINEERING & PREPROCESSING
    - Dynamically detects sequential and demographic feature dimensions
    - Extracts sequential features (medical visit measurements) from patient DataFrames
    - Extracts demographic features (age, ethnicity, gender, race) from cleaned data
    - Handles NaN values by forward-filling with 0.0
    - Creates patient-level labels based on diagnosis dates
    - Only includes patients with >=2 visits before diagnosis for positive class
4. CLASS BALANCING
    - Applies RandomUnderSampler to achieve ~1:1 T2D to Non-T2D ratio
    - Preserves balance across train/val/test splits using stratified splitting
5. TRAIN-VALIDATION-TEST SPLIT
    - Uses StratifiedShuffleSplit (test_size=0.3, random_state=42)
    - Splits remaining data 50/50 for validation and test sets
    - Generates LaTeX-formatted statistics for dataset composition
6. MODEL ARCHITECTURE
    - LSTM Encoder: Processes sequential visit data (2 layers, 128 hidden units)
    - Demographics Encoder: Processes demographic features (32 hidden units)
    - Classifier: Combines both embeddings to predict T2D risk (64 hidden units)
    - Uses packed sequences to handle variable visit counts
    - Output: Binary classification (T2D vs Non-T2D) using sigmoid activation
7. TRAINING
    - Loss: BCEWithLogitsLoss with positive class weighting
    - Optimizer: Adam with learning rate 1e-4 and weight decay 1e-5
    - Early stopping: Stops after 15 epochs without validation loss improvement
    - Gradient clipping: Max norm of 1.0 to prevent exploding gradients
8. EVALUATION
    - Computes: Accuracy, Precision, Recall, Macro F1-Score, ROC-AUC
    - Generates confusion matrix visualization
    - Evaluates on three datasets:
      a) Balanced test set (used for hyperparameter tuning)
      b) Original unbalanced dataset (for real-world performance estimation)
      c) External test set (held-out validation cohort)
KEY FEATURES:
=============
- Dynamic feature dimension detection
- Handles variable-length visit sequences
- Balances class imbalance via undersampling
- Generates reproducible results (fixed random states)
- Comprehensive logging and visualization
- LaTeX output for documentation
- Evaluates on multiple test scenarios
IMPORTANT NOTES:
================
- NaN values are filled with 0.0 (critical to prevent training instability)
- Diagnosis dates must be >= 2 visits before them to be included in positive class
- Sequential features are transposed (visits × features) for LSTM input
- Stratified splitting preserves label proportions across splits
- External test patient IDs must be in the matched_test.csv file
"""



"""
Created on Thu Jul 24 09:39:15 2025

@author: inanc
@description: Patient-level T2D risk prediction using a hybrid LSTM-DNN model.
"""




import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os

# %%  Configuration & Hyperparameters 

class Config:
    # Data Paths
    patient_data_path_clinical = r"preprocessing\output_pickles\7_filtered_patient_groups_clinical\regular_patients_1_year.pkl"
    patient_data_path_standard = r"preprocessing\output_pickles\7_filtered_patient_groups_standard\regular_patients_1_year.pkl"
    patient_data_path_categorized = r"preprocessing\output_pickles\7_filtered_patient_groups_categorized\regular_patients_1_year.pkl"
    earliest_dx_path = r"C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\data\EARLIEST_DX_deid.csv"
    demographics_path = r"C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\data\deid_DEM.csv"
    external_test_path = r"C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\matched_test.csv"

    #  CAMBIO AQUÍ: Definir las columnas demográficas exactas 
    DEMO_COLUMNS = [
        'AGE_AT_END', 'ETHNICITY_N', 'ETHNICITY_NI', 'ETHNICITY_Y',
        'GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS', 'RACE_B',
        'RACE_H', 'RACE_NA', 'RACE_NI', 'RACE_W'
    ]
    
    # Estos valores se actualizarán automáticamente ahora
    SEQ_INPUT_SIZE = None 
    DEMO_INPUT_SIZE = None

    # Model Hyperparameters
    #SEQ_INPUT_SIZE = 22
    #DEMO_INPUT_SIZE = 13
    LSTM_HIDDEN_SIZE = 128
    DEMO_HIDDEN_SIZE = 32
    COMBINED_HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5

    # Training Hyperparameters
    BATCH_SIZE = 256 #it means number of patients per batch, not visits
    EPOCHS = 1000 #superior limit 
    LEARNING_RATE = 1e-4 
    WEIGHT_DECAY = 1e-5 
    PATIENCE = 15 # Early stopping patience
    MIN_DELTA = 1e-4

    # Other settings
    TEST_SPLIT_SIZE = 0.3
    RANDOM_STATE = 42 # For reproducibility

config = Config()


# %% 🧬 Load Data and Preprocess Labels
try:
    with open(config.patient_data_path_clinical, "rb") as file:
        imputed_patients_matrices_all = pickle.load(file)
    df2 = pd.read_csv(config.earliest_dx_path)
    print(" Data files successfully loaded.")
except Exception as e:
    print(f" Error loading data files: {e}")

df2 = df2.drop(columns=['DX'], errors='ignore')
df2['EARLIEST_DX'] = pd.to_datetime(df2['EARLIEST_DX'], errors='coerce')
df2_earliest = df2.sort_values('EARLIEST_DX').drop_duplicates(subset='PATIENT_ID', keep='first')
imputed_patients = set(imputed_patients_matrices_all.keys())
df2_filtered = df2_earliest[df2_earliest['PATIENT_ID'].isin(imputed_patients)]
dx_date_dict = df2_filtered.set_index('PATIENT_ID')['EARLIEST_DX'].dt.date.to_dict()



# %% 📊  NEW: Compare T2D Counts from Different Sources 
print("\n Comparing T2D Patient Counts ")
try:
    # Load the demographics data from the CSV file
    demo_df = pd.read_csv(config.demographics_path) # <<< UPDATED LOADING METHOD
    
    # Ensure PATIENT_ID is the correct column name
    if 'PATIENT_ID' not in demo_df.columns:
        id_col_options = ['patient_id', 'PID', 'Patient_ID']
        id_col = next((col for col in id_col_options if col in demo_df.columns), None)
        if id_col:
            demo_df.rename(columns={id_col: 'PATIENT_ID'}, inplace=True)
        else:
            raise KeyError("'PATIENT_ID' column not found in demographics file.")
            
    # Filter demographics for patients in the current dataset
    demo_df_filtered = demo_df[demo_df['PATIENT_ID'].isin(imputed_patients)]
    
    # Count T2D patients based on AIM_GROUP
    # IMPORTANT: Verify 'Type 2 Diabetes' is the exact string in your AIM_GROUP column
    # This will correctly find the T2D patients
    t2d_from_demo = demo_df_filtered[demo_df_filtered['AIM_GROUP'] == '2_Type2']
    t2d_from_demo_count = len(t2d_from_demo['PATIENT_ID'].unique())
    
    # Get the count from the EARLIEST_DX file for comparison
    t2d_from_dx_count = len(dx_date_dict)
    
    print(f"Patients in regular_patients_1_year.pkl: {len(imputed_patients)}")
    print(f"Count from AIM_GROUP: {t2d_from_demo_count} T2D patients.")
    print(f"Count from EARLIEST_DX: {t2d_from_dx_count} T2D patients.")

except FileNotFoundError:
    print(f" Demographics file not found at: {config.demographics_path}")
except KeyError as e:
    print(f" A required column was not found in the demographics file: {e}")
except Exception as e:
    print(f" An error occurred during the comparison: {e}")

# %% 🩺  NEW: Analyze T2D Patient Visit Counts 
print("\n Analyzing Visit Counts for T2D Patients ")
# Get the set of T2D patient IDs from the reliable dx_date_dict
t2d_patient_ids = set(dx_date_dict.keys())

visit_counts = []
for pid, df in imputed_patients_matrices_all.items():
    if pid in t2d_patient_ids:
        # The number of visits is the number of columns in the DataFrame
        visit_counts.append(df.shape[1])

if visit_counts:
    # Calculate and print statistics
    visit_counts_series = pd.Series(visit_counts)
    print(f"Total T2D patients analyzed: {len(visit_counts)}")
    print("Statistics for number of visits per T2D patient:")
    print(visit_counts_series.describe())

    # Plot a histogram to visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(visit_counts, bins=range(min(visit_counts), max(visit_counts) + 2), edgecolor='black', align='left')
    plt.title('Distribution of Visit Counts for T2D Patients')
    plt.xlabel('Number of Visits')
    plt.ylabel('Number of Patients')
    plt.xticks(np.arange(min(visit_counts), max(visit_counts) + 1, step=1))
    plt.grid(axis='y', alpha=0.75)
    plt.show()
else:
    print("No T2D patients found in the dataset to analyze.")
# %% 📋 Prepare Visit-Level Sequences and Labels for Prediction
print("Extracting features dynamically and cleaning NaNs...")

#  DETECCIÓN DINÁMICA DE TAMAÑOS 
first_pid = next(iter(imputed_patients_matrices_all))
all_cols = imputed_patients_matrices_all[first_pid].index
seq_indices = [idx for idx in all_cols if idx not in config.DEMO_COLUMNS]

config.SEQ_INPUT_SIZE = len(seq_indices)
config.DEMO_INPUT_SIZE = len(config.DEMO_COLUMNS)

print(f"📊 Red configurada: {config.SEQ_INPUT_SIZE} variables secuenciales, {config.DEMO_INPUT_SIZE} demográficas.")

X_sequential, X_demographics, y_patient_labels = [], [], []
pids_per_sample = [] 
eliminated_t2d_patients = []

for pid, df_patient in imputed_patients_matrices_all.items():
    # 1. Extraer y limpiar de NaNs (rellenamos con 0.0)
    # Rellenamos NaNs para evitar que el modelo explote (Train Loss: nan)
    df_cleaned = df_patient.fillna(0.0)
    
    demo_features = df_cleaned.loc[config.DEMO_COLUMNS].iloc[:, 0].values
    seq_features = df_cleaned.loc[seq_indices].T.values 
    
    visit_dates = [pd.to_datetime(col).date() for col in df_patient.columns]
    diagnosis_date = dx_date_dict.get(pid)

    first_dx_idx = -1
    if diagnosis_date:
        for i, v_date in enumerate(visit_dates):
            if v_date >= diagnosis_date:
                first_dx_idx = i
                break
    
    if first_dx_idx != -1:
        if first_dx_idx >= 2:
            X_sequential.append(seq_features[:first_dx_idx, :])
            X_demographics.append(demo_features)
            y_patient_labels.append(1)
            pids_per_sample.append(pid)
        else:
            eliminated_t2d_patients.append(pid)
    else:
        if df_cleaned.shape[1] >= 2:
            X_sequential.append(seq_features[:-1, :])
            X_demographics.append(demo_features)
            y_patient_labels.append(0)
            pids_per_sample.append(pid)

print(f" Visit-level sequences prepared. Total training samples created: {len(y_patient_labels)}.")
# %% ⚖️ Undersampling to a 2:1 Ratio (Healthy:T2D)
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print("Original dataset composition:")
original_counts = Counter(y_patient_labels)
print(pd.Series(y_patient_labels).value_counts())

# The sampling_strategy of 0.5 means the number of samples in the minority class (1)
# will be 50% of the number of samples in the majority class (0).
# This creates the desired 2:1 ratio.
rus = RandomUnderSampler(sampling_strategy=1, random_state=config.RANDOM_STATE)

# We provide the feature data and labels to the resampler
X_demo_np = np.array(X_demographics)
y_labels_np = np.array(y_patient_labels)

# Resample the indices to keep the corresponding sequential data
all_indices = np.arange(len(y_labels_np)).reshape(-1, 1)
resampled_indices_flat, y_balanced_np = rus.fit_resample(all_indices, y_labels_np)
resampled_indices = resampled_indices_flat.flatten()

# Reconstruct the full dataset using the balanced indices
X_seq_balanced = [X_sequential[i] for i in resampled_indices]
X_demo_balanced = [X_demographics[i] for i in resampled_indices]
y_balanced = y_balanced_np.tolist()

print(f"\n Data re-sampled with RandomUnderSampler. Original samples: {len(y_patient_labels)} -> New samples: {len(y_balanced)}")
print("New dataset composition (Healthy:T2D ratio should be approx. 2:1):")
print(pd.Series(y_balanced).value_counts())
# %% 🔎  NEW: Inspect a Single Patient's Timeline 
print("\n Single Patient Timeline Inspection ")

# <<< CHANGE THIS VALUE to the Patient ID you want to inspect
patient_id_to_check = 512

# Check if the patient exists in the dataset
if patient_id_to_check in imputed_patients_matrices_all:
    
    # 1. Get the T2D diagnosis date from the dictionary we already created
    diagnosis_date = dx_date_dict.get(patient_id_to_check)
    
    # 2. Get all visit dates from the patient's DataFrame columns
    patient_df = imputed_patients_matrices_all[patient_id_to_check]
    visit_dates = [pd.to_datetime(col).date() for col in patient_df.columns]
    
    # 3. Print the results
    print(f"Showing details for PATIENT_ID: {patient_id_to_check}")
    if diagnosis_date:
        print(f"  - T2D Diagnosis Date: {diagnosis_date}")
    else:
        print("  - T2D Diagnosis Date: Patient is not in the T2D cohort.")
        
    print("  - Recorded Visit Dates:")
    for i, visit_date in enumerate(visit_dates, 1):
        print(f"    - Visit {i}: {visit_date}")
        
else:
    print(f" Error: Patient with ID {patient_id_to_check} not found in the dataset.")

# %% 🔪 Train-Validation-Test Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
train_idx, temp_idx = next(sss.split(X_demo_balanced, y_balanced)) # Stratify on demo data
val_idx = temp_idx[:len(temp_idx) // 2]
test_idx = temp_idx[len(temp_idx) // 2:]

def get_data(idxs):
    X_s = [X_seq_balanced[i] for i in idxs]
    X_d = [X_demo_balanced[i] for i in idxs]
    y = [y_balanced[i] for i in idxs]
    return X_s, X_d, y

X_train_seq, X_train_demo, y_train = get_data(train_idx)
X_val_seq, X_val_demo, y_val = get_data(val_idx)
X_test_seq, X_test_demo, y_test = get_data(test_idx)

#%%
# %% 📈 Split Statistics + LaTeX Section
from collections import Counter
import os

def count_pos_neg(labels):
    c = Counter(labels)
    return int(c.get(1, 0)), int(c.get(0, 0))  # (pos, neg)

n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
pos_tr, neg_tr = count_pos_neg(y_train)
pos_va, neg_va = count_pos_neg(y_val)
pos_te, neg_te = count_pos_neg(y_test)

print("\n=== Split Sizes (patients) ===")
print(f"Train: {n_train}  (T2D: {pos_tr}, Non-T2D: {neg_tr})")
print(f"Val:   {n_val}    (T2D: {pos_va}, Non-T2D: {neg_va})")
print(f"Test:  {n_test}   (T2D: {pos_te}, Non-T2D: {neg_te})")

# Optional: overall post-undersampling composition
all_balanced_labels = y_balanced
pos_all, neg_all = count_pos_neg(all_balanced_labels)
print("\nOverall (balanced set used for splitting):")
print(f"Total: {len(all_balanced_labels)}  (T2D: {pos_all}, Non-T2D: {neg_all})")

#  LaTeX subsection text 
latex_section = rf"""
\subsection{{Cohort Splits}}
We report the number of patients used for training, validation, and testing after class balancing and patient-level splitting. The training set contains {n_train} patients (T2D: {pos_tr}, Non-T2D: {neg_tr}), the validation set contains {n_val} patients (T2D: {pos_va}, Non-T2D: {neg_va}), and the test set contains {n_test} patients (T2D: {pos_te}, Non-T2D: {neg_te}). In total, the balanced cohort used for splitting includes {len(all_balanced_labels)} patients (T2D: {pos_all}, Non-T2D: {neg_all}). Stratified splitting was applied to preserve label proportions across splits.
"""

print("\n=== LaTeX subsection (copy to Overleaf) ===")
print(latex_section)

# Save to a .tex snippet (adjust path if you want it next to your script)
out_path = os.path.join(os.getcwd(), "split_stats_section.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(latex_section)
print(f"Saved LaTeX subsection to: {out_path}")

# %% 📦 PyTorch Dataset and Dataloader
class PatientDataset(Dataset):
    def __init__(self, seq_data, demo_data, labels):
        self.seq_data = seq_data
        self.demo_data = demo_data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.seq_data[idx], self.demo_data[idx], self.labels[idx]

def collate_fn(batch):
    seq_batch, demo_batch, labels_batch = zip(*batch)
    
    # Handle sequential data
    seq_tensors = [torch.tensor(x, dtype=torch.float32) for x in seq_batch]
    lengths = torch.tensor([len(x) for x in seq_tensors])
    seq_padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0.0)
    
    # Handle demographic and label data
    demo_tensor = torch.tensor(np.array(demo_batch), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_batch, dtype=torch.float32)
    
    return seq_padded, demo_tensor, labels_tensor, lengths

# %% 🤖 Define Hybrid LSTM-DNN Model
class PatientRiskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Visit Sequence Encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=config.SEQ_INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0
        )
        # 2. Demographics Encoder (DNN)
        self.demo_encoder = nn.Sequential(
            nn.Linear(config.DEMO_INPUT_SIZE, config.DEMO_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        # 3. Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE + config.DEMO_HIDDEN_SIZE, config.COMBINED_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.COMBINED_HIDDEN_SIZE, 1)
        )

    def forward(self, x_seq, x_demo, lengths):
        # Process sequence with LSTM
        packed_x = pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        # Use the hidden state from the last layer as the sequence representation
        seq_embedding = h_n[-1]
        
        # Process demographics with DNN
        demo_embedding = self.demo_encoder(x_demo)
        
        # Combine and classify
        combined_embedding = torch.cat((seq_embedding, demo_embedding), dim=1)
        output = self.classifier(combined_embedding)
        return output.squeeze(-1)

# Data Loaders
train_loader = DataLoader(PatientDataset(X_train_seq, X_train_demo, y_train), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(PatientDataset(X_val_seq, X_val_demo, y_val), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(PatientDataset(X_test_seq, X_test_demo, y_test), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
unbalanced_dataset = PatientDataset(X_sequential, X_demographics, y_patient_labels)
unbalanced_loader = DataLoader(unbalanced_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

# Training Function
def train_model(model, train_loader, val_loader, config):
    # Calculate positive class weight for the training set to handle any remaining imbalance
    train_labels = np.array(train_loader.dataset.labels)
    pos_weight = torch.tensor(np.sum(train_labels == 0) / np.sum(train_labels == 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_loss, patience_counter = float('inf'), 0
    print(f"⚖️ Using positive class weight: {pos_weight.item():.2f}")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for seq_batch, demo_batch, y_batch, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(seq_batch, demo_batch, lengths)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_batch, demo_batch, y_batch, lengths in val_loader:
                logits = model(seq_batch, demo_batch, lengths)
                val_loss += criterion(logits, y_batch).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1:03}/{config.EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > config.MIN_DELTA:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"⏳ Early stopping triggered after {epoch+1} epochs.")
                break

    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    return model

# %% 📊 Evaluation Function

def evaluate(model, data_loader, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for seq_batch, demo_batch, y_batch, lengths in data_loader:
            logits = model(seq_batch, demo_batch, lengths)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            
            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) 
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"\nAccuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | Macro F1 Score: {f1:.4f}") 
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-T2D", "T2D"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Patient-Level)")
    #plt.show()

# Run Model Training and Evaluation
model = PatientRiskModel(config)
trained_model = train_model(model, train_loader, val_loader, config)

print("\n  Evaluation on Test Set ")
evaluate(trained_model, test_loader)

print("\n Final Evaluation on Original Unbalanced Data ")
evaluate(trained_model, unbalanced_loader)

# %% Final Evaluation on External Test Set 
print("\n Final Evaluation on External Test Set ")
try:
    external_test_df = pd.read_csv(config.external_test_path)
    external_test_ids = set(external_test_df['PATIENT_ID'].unique())
    print(f"Loaded {len(external_test_ids)} unique patient IDs from the external test file.")
    external_test_indices = [i for i, pid in enumerate(pids_per_sample) if pid in external_test_ids]
    
    if external_test_indices:
        X_ext_seq = [X_sequential[i] for i in external_test_indices]
        X_ext_demo = [X_demographics[i] for i in external_test_indices]
        y_ext_labels = [y_patient_labels[i] for i in external_test_indices]

        external_dataset = PatientDataset(X_ext_seq, X_ext_demo, y_ext_labels)
        external_loader = DataLoader(external_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
        
        print(f"\n Evaluating performance on {len(y_ext_labels)} samples from the external test set ")
        evaluate(trained_model, external_loader)
    else:
        print("No valid samples found for the external test set to evaluate.")

except Exception as e:
    print(f" An error occurred during external evaluation: {e}")