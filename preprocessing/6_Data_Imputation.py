# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:09:01 2025
Sequential Autoencoder Imputation Script (Z-Score Optimized)
============================================================
This script trains an LSTM Autoencoder to impute missing medical data.
Performance is validated on raw model outputs, while the final saved 
dataset preserves all original measurements.
"""


"""
# Step-by-Step Explanation

## **Overview**
This script uses an LSTM Autoencoder to impute missing medical data while preserving original values where they exist.

## Steps

### **1. Imports (Lines 1-17)**
Loads required libraries: NumPy, Pandas, PyTorch, scikit-learn for machine learning operations.

### **2. Load Data (Lines 22-30)**
Reads z-score normalized patient data with NaN values from a pickle file.

### **3. Data Preparation (Lines 35-44)**
- Converts patient dataframes into sequences
- Creates masks to track which values are missing (NaN)
- Fills NaN with zeros for model input

### **4. Stratified Train/Val/Test Split (Lines 46-53)**
- Uses K-Means clustering to ensure balanced splits
- Divides data: 70% train, 15% validation, 15% test
- Maintains similar data distributions across splits

### **5. PyTorch Dataset Setup (Lines 58-68)**
- Creates a custom Dataset class for sequences and masks
- Implements padding collate function to handle variable-length sequences
- Loads data with batch size of 64

### **6. LSTM Autoencoder Model (Lines 73-97)**
- **Encoder**: Bidirectional LSTM compresses input to hidden state
- **Decoder**: LSTM reconstructs full sequence from hidden state
- **Loss Function**: Masked MSE (only penalizes known values)

### **7. Training Loop (Lines 102-135)**
- Trains for up to 1000 epochs with early stopping
- Monitors validation loss and stops if no improvement after 10 epochs
- Prints loss metrics each epoch

### **8. Imputation & Validation (Lines 140-171)**
- Runs model on all patients
- Validates reconstruction accuracy on known values (RMSE)
- **Hybridizes**: keeps original values where they exist, uses predictions for missing data

### **9. Save Results (Lines 176-186)**
- Exports final imputed dataset as pickle file

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import copy
import os
import sys
import matplotlib.pyplot as plt
import tqdm

# =============================================================================
# 1. Load Z-Score Normalized Data
# =============================================================================
input_dir = r"preprocessing\output_pickles"
file_path_znormalized_standard = os.path.join(input_dir, "5_patients_filtered_standard_norm_75.pkl")
file_path_znormalized_categories = os.path.join(input_dir, "5_patients_filtered_categorized_norm_75.pkl")
file_path_znormalized_clinical = os.path.join(input_dir, "5_patients_filtered_clinical_norm_75.pkl")


try:
    with open(file_path_znormalized_standard, "rb") as file:
        normalized_matrices_with_nans = pickle.load(file)
    print(f" Z-score normalized data loaded successfully from '{file_path_znormalized_standard}'.")
except FileNotFoundError:
    print(f" Error: Input file not found at {file_path_znormalized_standard}")
    sys.exit()


# 1.2 Imputation on Already categorized Features

categorized_features_list = [
    'ALT(SGPT)', 'AST(SGOT)', 'BUN', 'CHOLESTEROL', 'HDL', 'HGB A1C', 
    'POTASSIUM', 'TRIGLYCERIDE', 'VITAL_10541041_MEAN', 'VITAL_10541455_MEAN', 
    'VITAL_283305634_MEAN', 'VITAL_10541467_MEAN', 'VITAL_10541503_MEAN', 
    'VITAL_266705352_MEAN', 'VITAL_68924855_MEAN'
]

# =============================================================================
# 2. Prepare Data & 3. Stratified Splitting
# =============================================================================
def prepare_data(normalized_matrices):
    sequences, masks, patient_ids = [], [], []
    for pid, df in tqdm.tqdm(normalized_matrices.items(), desc="Preparing data", colour='magenta'):
        sequence_data = df.values.T
        if sequence_data.shape[0] > 0:
            mask = (~np.isnan(sequence_data)).astype(float)
            sequence_filled = np.nan_to_num(sequence_data, nan=0.0)
            sequences.append(sequence_filled)
            masks.append(mask)
            patient_ids.append(pid)
    if not sequences: raise ValueError("No valid sequences found.")
    return sequences, masks, patient_ids

normalized_matrices_with_nans_subsample = dict(list(normalized_matrices_with_nans.items())[:len(normalized_matrices_with_nans)//100])
sequences, masks, pids = prepare_data(normalized_matrices_with_nans_subsample)

# Cluster patients to ensure balanced Train/Val/Test splits
sequence_means = np.array([np.mean(seq, axis=0) for seq in sequences])
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(sequence_means)

indices = np.arange(len(sequences))
train_indices, temp_indices = train_test_split(indices, test_size=0.30, stratify=cluster_labels, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.50, stratify=cluster_labels[temp_indices], random_state=42)

X_train, mask_train = [sequences[i] for i in train_indices], [masks[i] for i in train_indices]
X_val, mask_val = [sequences[i] for i in val_indices], [masks[i] for i in val_indices]

print(f"Data prepared and split into Train/Val: {len(X_train)}/{len(X_val)} patients.")

# =============================================================================
# 4. PyTorch Dataset and DataLoader
# =============================================================================
class SequenceDataset(Dataset):
    def __init__(self, sequences, masks):
        self.sequences, self.masks = sequences, masks
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.masks[idx], dtype=torch.float32)

def collate_fn(batch):
    sequences, masks = zip(*batch)
    return pad_sequence(sequences, batch_first=True, padding_value=0.0), pad_sequence(masks, batch_first=True, padding_value=0.0)

batch_size = 64
train_loader = DataLoader(SequenceDataset(X_train, mask_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(SequenceDataset(X_val, mask_val), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# =============================================================================
# 5. Sequential Autoencoder Model
# =============================================================================
class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super(LSTM_Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        # Average the bidirectional states
        hidden = hidden.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size).mean(1)
        cell = cell.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size).mean(1)

        seq_len = x.shape[1]
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoder_output)

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
    def forward(self, outputs, targets, mask):
        squared_error = (outputs - targets) ** 2
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum() + 1e-8)
        return loss

# =============================================================================
# 6. Training Loop
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience, self.min_delta, self.best_loss, self.counter = patience, min_delta, float("inf"), 0
        self.best_model_state = None
    def check_early_stop(self, loss, model):
        if self.best_loss - loss > self.min_delta:
            self.best_loss, self.counter, self.best_model_state = loss, 0, copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience
    def restore_best_weights(self, model):
        if self.best_model_state: model.load_state_dict(self.best_model_state)

input_dim = sequences[0].shape[1]
model = LSTM_Autoencoder(input_dim, hidden_dim=64, n_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MaskedMSELoss()
early_stopper = EarlyStopping(patience=10, min_delta=1e-5)

print("\n Starting Training ")
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, mask in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, mask in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs, mask)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    if early_stopper.check_early_stop(avg_val_loss, model):
        print("Early stopping triggered.")
        early_stopper.restore_best_weights(model)
        break




# 7.2 Imputation on Already categorized features
def run_imputation_and_validation(normalized_matrices, model, categorized_features_list):
    model.eval()
    imputed_dict = {}
    actual_performance_errors = []

    total_cells = 0
    total_nans_before = 0

    for pid, df in tqdm.tqdm(normalized_matrices.items(), desc="Processing patients", colour='cyan'):
        original_seq_with_nans = df.values.T
        if original_seq_with_nans.shape[0] == 0:
            imputed_dict[pid] = df
            continue

        # TRACKING MISSINGNESS
        total_cells += original_seq_with_nans.size
        total_nans_before += np.isnan(original_seq_with_nans).sum()
            
        # Prepare Data
        original_mask = (~np.isnan(original_seq_with_nans)).astype(float)
        seq_filled = np.nan_to_num(original_seq_with_nans, nan=0.0)
        seq_tensor = torch.tensor(seq_filled, dtype=torch.float32).unsqueeze(0)

        # Get Raw Prediction for validation
        with torch.no_grad():
            raw_reconstruction = model(seq_tensor).squeeze(0).numpy()

        # VALIDATION: Compare raw prediction vs original values only where data existed
        known_indices = (original_mask == 1)
        if np.any(known_indices):
            mse = np.mean((original_seq_with_nans[known_indices] - raw_reconstruction[known_indices])**2)
            actual_performance_errors.append(np.sqrt(mse))

        # HYBRIDIZATION: Use original values where known, reconstructed where missing
        final_imputed_seq = np.where(known_indices, original_seq_with_nans, raw_reconstruction)
        
        df_imputed = pd.DataFrame(final_imputed_seq.T, index=df.index, columns=df.columns)

        for i, feature in enumerate(df_imputed.index):
            feat_name = feature[0] if isinstance(feature, tuple) else str(feature)
            if feat_name in categorized_features_list:
                df_imputed.iloc[i] = df_imputed.iloc[i].round(0)

        imputed_dict[pid] = df_imputed

    # Calculate global missingness
    nan_percentage = (total_nans_before / total_cells) * 100
    print(f"\n DATASET MISSINGNESS: {nan_percentage:.2f}% of all values were NaNs (now imputed).")
    print(f" MODEL RMSE: {np.mean(actual_performance_errors):.4f}")
        
    print(f"\n REAL Model Performance (RMSE on known values): {np.mean(actual_performance_errors):.4f}")
    return imputed_dict, nan_percentage

# 7.3 Missingness Visualization

def plot_missingness(nan_percentage):
    labels = ['Missing (Imputed)', 'Available Data']
    sizes = [nan_percentage, 100 - nan_percentage]
    colors = ['#ff6666', '#66b3ff']
    
    plt.figure(figsize=(7, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.1, 0))
    plt.title("Percentage of Values Imputed by LSTM Autoencoder")
    plt.show()

imputed_patients_matrices, nan_perc = run_imputation_and_validation(normalized_matrices_with_nans, model, categorized_features_list)
plot_missingness(nan_perc)  
# =============================================================================
#%% 8. Save the Final Imputed Data
# =============================================================================
final_output_dir = r"preprocessing\output_pickles"
os.makedirs(final_output_dir, exist_ok=True)
final_output_path_standard = os.path.join(final_output_dir, "6_imputed_normalized_data_standard_75.pkl")
final_output_path_categorized = os.path.join(final_output_dir, "6_imputed_normalized_data_categorized_75.pkl")
final_output_path_clinical = os.path.join(final_output_dir, "6_imputed_normalized_data_clinical_75.pkl")


try:
    with open(final_output_path_standard, "wb") as f:
        pickle.dump(imputed_patients_matrices, f)
    print(f"\n Final imputed standard data saved successfully to '{final_output_path_standard}'")
except Exception as e:
    print(f"\n Error saving file: {e}")
'''
# =============================================================================
# 9) Visualize results for one patient 
# =============================================================================

# Take first patient for visualization
first_pid = next(iter(imputed_patients_matrices))
df_imputed = imputed_patients_matrices[first_pid]
df_original = normalized_matrices_with_nans[first_pid]

# Original NaNs mask
was_imputed = df_original.isna()

print(f"\n" + "="*80)
print(f" Imputation Check for Patient ID: {first_pid} ")
print(f" Values with '*' were originally NaN and are now filled by the model")
print("="*80)


n_rows = 60
n_cols = 3
preview = df_imputed.iloc[:n_rows, :n_cols].copy().astype(object)
mask_preview = was_imputed.iloc[:n_rows, :n_cols]

# Loop through the preview and add asterisks to imputed values
for r in tqdm.tqdm(range(preview.shape[0]), desc="Formatting preview", colour='yellow'):
    for c in range(preview.shape[1]):
        val = preview.iat[r, c]
        if pd.isna(val): # Manejo de seguridad por si quedara algún NaN
            preview.iloc[r, c] = "NaN"
        elif mask_preview.iat[r, c]:
            # Ahora Pandas permite esto porque el dtype es 'object'
            preview.iloc[r, c] = f"{val:.2f}*"
        else:
            preview.iloc[r, c] = f"{val:.2f}"

print(preview.to_string())'''



