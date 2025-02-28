import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define constants
SEQ_LEN = 48      # 4 hours of historical data (4 * 60 / 5 minutes = 48 intervals)
FORECAST_HORIZON = 12  # 1 hour ahead prediction (60 minutes / 5 minutes = 12 intervals)
BATCH_SIZE = 16   # Reduced from 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_sequences(patient_data, seq_length=48, forecast_horizon=12):
    """Create sequences from patient data with multiple future values"""
    sequences = []
    
    # Get feature columns
    glucose = patient_data['glucose'].values
    basal = patient_data['basal'].values
    bolus = patient_data['bolus'].values
    carbs = patient_data['carbs'].values
    
    # Create sequences
    for i in range(len(glucose) - seq_length - forecast_horizon + 1):
        # Input sequences
        glucose_seq = glucose[i:i + seq_length]
        basal_seq = basal[i:i + seq_length]
        bolus_seq = bolus[i:i + seq_length]
        carbs_seq = carbs[i:i + seq_length]
        
        # Target sequence (next 12 values)
        target_seq = glucose[i + seq_length:i + seq_length + forecast_horizon]
        
        # Skip sequences with missing values
        if (np.isnan(glucose_seq).any() or np.isnan(target_seq).any() or
            np.isnan(basal_seq).any() or np.isnan(bolus_seq).any() or
            np.isnan(carbs_seq).any()):
            continue
        
        sequence = {
            'glucose': glucose_seq.reshape(-1, 1),
            'basal': basal_seq.reshape(-1, 1),
            'bolus': bolus_seq.reshape(-1, 1),
            'carbs': carbs_seq.reshape(-1, 1),
            'target': target_seq  # This will be 12 values
        }
        sequences.append(sequence)
    
    return sequences

class GlucoseDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            'glucose': torch.FloatTensor(sequence['glucose']),
            'basal': torch.FloatTensor(sequence['basal']),
            'bolus': torch.FloatTensor(sequence['bolus']),
            'carbs': torch.FloatTensor(sequence['carbs']),
            'target': torch.FloatTensor(sequence['target'])  # Will be shape [12]
        }

def preprocess_data(file_path, train_ratio=0.7, val_ratio=0.15):
    """Load and preprocess the data"""
    # Read data
    data = pd.read_csv(file_path)
    
    # Ensure we have the right column name
    if 'id' in data.columns and 'patient_id' not in data.columns:
        data = data.rename(columns={'id': 'patient_id'})
    
    # Sort by patient and time if time column exists
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'], format='%d/%m/%Y %H:%M')
        data = data.sort_values(by=['patient_id', 'time'])
    
    # Normalize data
    glucose_mean = data['glucose'].mean()
    glucose_std = data['glucose'].std()
    
    data['glucose'] = (data['glucose'] - glucose_mean) / glucose_std
    data['basal'] = (data['basal'] - data['basal'].mean()) / data['basal'].std()
    data['bolus'] = (data['bolus'] - data['bolus'].mean()) / data['bolus'].std()
    data['carbs'] = (data['carbs'] - data['carbs'].mean()) / data['carbs'].std()
    
    # Group by patient
    patient_groups = data.groupby('patient_id')
    
    # Create sequences for each patient
    all_sequences = []
    for _, patient_data in tqdm(patient_groups, desc="Processing patients"):
        patient_sequences = create_sequences(
            patient_data,
            seq_length=SEQ_LEN,
            forecast_horizon=FORECAST_HORIZON
        )
        all_sequences.extend(patient_sequences)
    
    # Log sequence creation results
    print(f"Created {len(all_sequences)} total sequences")
    
    # Split into train, validation, and test sets
    train_size = int(len(all_sequences) * train_ratio)
    val_size = int(len(all_sequences) * val_ratio)
    
    train_sequences = all_sequences[:train_size]
    val_sequences = all_sequences[train_size:train_size + val_size]
    test_sequences = all_sequences[train_size + val_size:]
    
    # Create datasets
    train_dataset = GlucoseDataset(train_sequences)
    val_dataset = GlucoseDataset(val_sequences)
    test_dataset = GlucoseDataset(test_sequences)
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} sequences")
    print(f"Validation: {len(val_dataset)} sequences")
    print(f"Test: {len(test_dataset)} sequences")
    
    return train_dataset, val_dataset, test_dataset

# Example usage
if __name__ == "__main__":
    file_path = 'full_patient_dataset.csv'
    train_dataset, val_dataset, test_dataset = preprocess_data(file_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Test a batch
    for batch in test_loader:
        print("Glucose shape:", batch['glucose'].shape)  # (batch_size, SEQ_LEN, 1)
        print("Target shape:", batch['target'].shape)    # (batch_size, 12)
        break