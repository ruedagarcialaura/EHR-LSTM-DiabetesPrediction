# -*- coding: utf-8 -*-
"""
HYBRID LSTM-DNN MODEL WITH MASKED TRAINING FOR PATIENT RISK PREDICTION
This module implements a deep learning model that combines LSTM and DNN architectures
to predict patient diagnosis risk (Type 2 Diabetes) using sequential clinical visits
and demographic information.
WORKFLOW OVERVIEW:
==================
1. CONFIGURATION & SETUP
    - Defines file paths for patient data, diagnosis dates, and external test sets
    - Sets hyperparameters for model architecture (LSTM hidden size, dropout rates, etc.)
    - Configures training parameters (batch size, learning rate, early stopping patience)
2. DATA LOADING & PREPROCESSING
    - Loads pickled patient matrices and diagnosis date CSV
    - Filters patients based on earliest diagnosis date
    - Extracts diagnosis dates indexed by patient ID
3. FEATURE ENGINEERING & SEQUENCING
    - Separates demographic features (13 columns: age, ethnicity, gender, race)
    - Treats all remaining features as sequential visit data
    - Creates masked sequences: concatenates actual values with validity masks (NaN indicators)
    - Extracts first diagnosis date for each patient to create target labels
4. LABEL CREATION
    - Positive label (1): Patients with ≥2 pre-diagnosis visits
    - Negative label (0): Patients without diagnosis date with ≥2 visits
5. CLASS BALANCING & TRAIN/VAL/TEST SPLIT
    - Applies random undersampling to balance class distribution (1:1 ratio)
    - Performs stratified shuffle split: 70% train, 15% validation, 15% test
    - Maintains class distribution across splits
6. PYTORCH DATA PIPELINE
    - Custom PatientDataset class wraps sequences, demographics, and labels
    - Collate function handles variable-length sequences:
      * Converts data to tensors
      * Pads sequences to batch max length
      * Preserves original sequence lengths for packed LSTM
7. MODEL ARCHITECTURE (PatientRiskModel)
    - LSTM Branch: Processes sequential visit data (2 layers, 128 hidden units)
      * Input: masked features (original + validity mask)
      * Output: final hidden state embedding
    - Demographics Branch: Encodes demographic features (32 hidden units)
      * Input: 13 demographic features
      * Output: dense embedding
    - Fusion: Concatenates both embeddings
    - Classification Head: 2-layer DNN predicting binary risk (64 → 1 hidden unit)
8. TRAINING PROCEDURE
    - Uses BCEWithLogitsLoss with class balancing weights
    - Adam optimizer with weight decay regularization
    - Gradient clipping (max norm = 1.0) for stability
    - Early stopping based on validation loss (patience=15, min_delta=0.0001)
    - Saves best model state during training
9. EVALUATION METRICS
    - Computes: Accuracy, AUC-ROC, Precision, Recall, Macro F1-Score
    - Generates confusion matrices for both balanced and unbalanced test sets
    - Visualizes results with matplotlib
10. TESTING
     - Evaluates on held-out test set (balanced)
     - Evaluates on full unbalanced dataset
     - Evaluates on external test set (if CSV provided)
KEY FEATURES:
=============
- Handles missing data with masking mechanism (concatenates NaN flags)
- Accounts for imbalanced classes via undersampling and weighted loss
- Packed LSTM processing for efficient variable-length sequence handling
- Temporal integrity: excludes pre-diagnosis visits from input
- Stratified sampling to maintain class balance across splits
INPUT REQUIREMENTS:
===================
- patient_data_path: Pickled dict of patient DataFrames (columns=visit dates, rows=features)
- earliest_dx_path: CSV with columns [PATIENT_ID, EARLIEST_DX, ...]
- external_test_path: CSV with external test patient IDs
OUTPUT:
=======
- Trained binary classification model (diagnosis risk prediction)
- Performance metrics and confusion matrices for all dataset splits
- Model predictions with probability scores
"""



"""
Created on Thu Jul 24 09:39:15 2025
@author: inanc
@description: Hybrid LSTM-DNN with Masked Training. 
Demographics defined by titles; all other rows treated as sequential.
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

# %%  Configuration & Hyperparameters 

class Config:
    patient_data_path = r"preprocessing\output_pickles\filtered_patient_groups\regular_patients_1_year.pkl"
    earliest_dx_path = r"C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\data\EARLIEST_DX_deid.csv"
    external_test_path = r"C:\Users\universidad\clases\iit\TFM\code_emirhan_in_order_feb2026\matched_test.csv"
    
    #  DEFINE DEMO COLUMNS ONLY 
    DEMO_COLUMNS = [
        'AGE_AT_END', 'ETHNICITY_N', 'ETHNICITY_NI', 'ETHNICITY_Y',
        'GENDER_F', 'GENDER_M', 'GENDER_NI', 'RACE_AS', 'RACE_B',
        'RACE_H', 'RACE_NA', 'RACE_NI', 'RACE_W'
    ]
    
    # Architecture
    LSTM_HIDDEN_SIZE = 128
    DEMO_HIDDEN_SIZE = 32
    COMBINED_HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5

    # Training
    BATCH_SIZE = 128
    EPOCHS = 1000
    LEARNING_RATE = 0.5*1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    MIN_DELTA = 1e-4
    TEST_SPLIT_SIZE = 0.3
    RANDOM_STATE = 42

config = Config()

# %% 🧬 Load Data
try:
    with open(config.patient_data_path, "rb") as file:
        imputed_patients_matrices_all = pickle.load(file)
    df2 = pd.read_csv(config.earliest_dx_path)
    print(" Data files successfully loaded.")
except Exception as e:
    print(f" Error loading data files: {e}")

# Process Diagnosis Dates
df2 = df2.drop(columns=['DX'], errors='ignore')
df2['EARLIEST_DX'] = pd.to_datetime(df2['EARLIEST_DX'], errors='coerce')
df2_earliest = df2.sort_values('EARLIEST_DX').drop_duplicates(subset='PATIENT_ID', keep='first')
imputed_patients = set(imputed_patients_matrices_all.keys())
df2_filtered = df2_earliest[df2_earliest['PATIENT_ID'].isin(imputed_patients)]
dx_date_dict = df2_filtered.set_index('PATIENT_ID')['EARLIEST_DX'].dt.date.to_dict()

# %% 📋 Prepare Visit-Level Sequences with Dynamic Extraction
print("Extracting demographics by title. Remaining rows are sequential...")

X_sequential, X_demographics, y_patient_labels = [], [], []
pids_per_sample = []

first_pid = next(iter(imputed_patients_matrices_all))
all_indices = imputed_patients_matrices_all[first_pid].index
seq_indices = [idx for idx in all_indices if idx not in config.DEMO_COLUMNS]

config.ORIGINAL_SEQ_SIZE = len(seq_indices)
config.SEQ_INPUT_SIZE = config.ORIGINAL_SEQ_SIZE * 2 
config.DEMO_INPUT_SIZE = len(config.DEMO_COLUMNS)

print(f"📊 Features detected: {len(seq_indices)} sequential, {len(config.DEMO_COLUMNS)} demographic.")

for pid, df_patient in imputed_patients_matrices_all.items():
    demo_features = df_patient.loc[config.DEMO_COLUMNS].iloc[:, 0].values
    seq_raw = df_patient.loc[seq_indices].T.values 
    
    mask = (~np.isnan(seq_raw)).astype(np.float32)
    seq_vals = np.nan_to_num(seq_raw, nan=0.0)
    masked_features = np.concatenate([seq_vals, mask], axis=1) 

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
            X_sequential.append(masked_features[:first_dx_idx, :])
            X_demographics.append(demo_features)
            y_patient_labels.append(1)
            pids_per_sample.append(pid)
    else:
        if df_patient.shape[1] >= 2:
            X_sequential.append(masked_features[:-1, :])
            X_demographics.append(demo_features)
            y_patient_labels.append(0)
            pids_per_sample.append(pid)

# %% ⚖️ Undersampling and Splitting
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=config.RANDOM_STATE)
all_indices = np.arange(len(y_patient_labels)).reshape(-1, 1)
resampled_indices_flat, y_balanced_raw = rus.fit_resample(all_indices, y_patient_labels)
resampled_indices = resampled_indices_flat.flatten()

y_balanced = y_balanced_raw if isinstance(y_balanced_raw, list) else y_balanced_raw.tolist()
X_seq_balanced = [X_sequential[i] for i in resampled_indices]
X_demo_balanced = [X_demographics[i] for i in resampled_indices]

sss = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_STATE)
train_idx, temp_idx = next(sss.split(X_demo_balanced, y_balanced))
val_idx = temp_idx[:len(temp_idx) // 2]
test_idx = temp_idx[len(temp_idx) // 2:]

def get_data(idxs):
    return [X_seq_balanced[i] for i in idxs], [X_demo_balanced[i] for i in idxs], [y_balanced[i] for i in idxs]

X_train_seq, X_train_demo, y_train = get_data(train_idx)
X_val_seq, X_val_demo, y_val = get_data(val_idx)
X_test_seq, X_test_demo, y_test = get_data(test_idx)

# %% 📦 PyTorch Dataset and Dataloader
class PatientDataset(Dataset):
    def __init__(self, seq_data, demo_data, labels):
        self.seq_data = seq_data
        self.demo_data = demo_data
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.seq_data[idx], self.demo_data[idx], self.labels[idx]

def collate_fn(batch):
    seq_batch, demo_batch, labels_batch = zip(*batch)
    seq_tensors = [torch.tensor(x, dtype=torch.float32) for x in seq_batch]
    lengths = torch.tensor([len(x) for x in seq_tensors])
    seq_padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0.0)
    demo_tensor = torch.tensor(np.array(demo_batch), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_batch, dtype=torch.float32)
    return seq_padded, demo_tensor, labels_tensor, lengths

train_loader = DataLoader(PatientDataset(X_train_seq, X_train_demo, y_train), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(PatientDataset(X_val_seq, X_val_demo, y_val), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(PatientDataset(X_test_seq, X_test_demo, y_test), batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

# %% 🤖 Model Definition
class PatientRiskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.SEQ_INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0
        )
        self.demo_encoder = nn.Sequential(
            nn.Linear(config.DEMO_INPUT_SIZE, config.DEMO_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE + config.DEMO_HIDDEN_SIZE, config.COMBINED_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.COMBINED_HIDDEN_SIZE, 1)
        )

    def forward(self, x_seq, x_demo, lengths):
        packed_x = pack_padded_sequence(x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        seq_embedding = h_n[-1]
        demo_embedding = self.demo_encoder(x_demo)
        combined_embedding = torch.cat((seq_embedding, demo_embedding), dim=1)
        return self.classifier(combined_embedding).squeeze(-1)

# %% 🚀 Training and Evaluation Functions
def train_model(model, train_loader, val_loader, config):
    train_labels = np.array(train_loader.dataset.labels)
    pos_weight = torch.tensor(np.sum(train_labels == 0) / np.sum(train_labels == 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    best_val_loss, patience_counter = float('inf'), 0
    best_model_state = None

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for seq_batch, demo_batch, y_batch, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(seq_batch, demo_batch, lengths)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_batch, demo_batch, y_batch, lengths in val_loader:
                logits = model(seq_batch, demo_batch, lengths)
                val_loss += criterion(logits, y_batch).item()
        
        val_loss /= len(val_loader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if best_val_loss - val_loss > config.MIN_DELTA:
            best_val_loss, patience_counter = val_loss, 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"⏳ Early stopping at epoch {epoch+1}")
                break

    if best_model_state: model.load_state_dict(best_model_state)
    return model

def evaluate(model, data_loader, title="Evaluation"):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for seq_batch, demo_batch, y_batch, lengths in data_loader:
            logits = model(seq_batch, demo_batch, lengths)
            y_true.extend(y_batch.tolist())
            y_prob.extend(torch.sigmoid(logits).tolist())

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    
    #  Calculation of Metrics 
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"\n {title} ")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | Macro F1 Score: {f1_macro:.4f}")
    
    #  Confusion Matrix 
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-T2D", "T2D"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"{title}: Confusion Matrix")
    plt.show()

# %% ▶️ Run
model = PatientRiskModel(config)
trained_model = train_model(model, train_loader, val_loader, config)

evaluate(trained_model, test_loader, "Test Set Results")

# Full Unbalanced evaluation
full_unbalanced_loader = DataLoader(PatientDataset(X_sequential, X_demographics, y_patient_labels), 
                                    batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
evaluate(trained_model, full_unbalanced_loader, "Unbalanced Data Results")

# %% 🧪 External Test Set
print("\n 🧪 Final Evaluation on External Test Set ")
try:
    ext_df = pd.read_csv(config.external_test_path)
    ext_ids = set(ext_df['PATIENT_ID'].unique())
    ext_idx = [i for i, pid in enumerate(pids_per_sample) if pid in ext_ids]
    
    if ext_idx:
        X_e_s = [X_sequential[i] for i in ext_idx]
        X_e_d = [X_demographics[i] for i in ext_idx]
        y_e = [y_patient_labels[i] for i in ext_idx]
        evaluate(trained_model, DataLoader(PatientDataset(X_e_s, X_e_d, y_e), batch_size=config.BATCH_SIZE, collate_fn=collate_fn), "External Test")
    else:
        print("No external samples found.")
except Exception as e:
    print(f" External evaluation error: {e}")