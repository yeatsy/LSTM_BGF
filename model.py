import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Optimized constants for MPS
BATCH_SIZE = 128  # Increased batch size for better MPS utilization
NUM_WORKERS = 4   # Multiple workers for data loading
PIN_MEMORY = True # Faster data transfer to GPU

def get_device():
    """Get the best available device: MPS, CUDA, or CPU"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable async data transfers
        torch.backends.mps.enable_async()
        return device
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128):  # Increased hidden size
        super(GlucoseLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,  # Added another layer for better capacity
            batch_first=True,
            dropout=0.2    # Moved dropout to LSTM
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Pre-allocate tensor for concatenation
        device = x['glucose'].device
        batch_size = x['glucose'].shape[0]
        seq_len = x['glucose'].shape[1]
        combined_input = torch.empty(
            (batch_size, seq_len, 4), 
            device=device, 
            dtype=torch.float32
        )
        
        # Faster concatenation
        combined_input[:, :, 0] = x['glucose'].squeeze(-1)
        combined_input[:, :, 1] = x['basal'].squeeze(-1)
        combined_input[:, :, 2] = x['bolus'].squeeze(-1)
        combined_input[:, :, 3] = x['carbs'].squeeze(-1)
        
        lstm_out, _ = self.lstm(combined_input)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = get_device()
    print(f"\nUsing device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision training
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
    best_val_loss = float('inf')
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', leave=False)
        
        for batch in train_pbar:
            # Move batch to device
            batch = {k: v.to(device, dtype=torch.float32, non_blocking=True) 
                    for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            # Mixed precision training
            with autocast(device_type='cpu' if device.type == 'mps' else device.type):
                outputs = model(batch)
                target = batch['target'].view(-1, 1)
                loss = criterion(outputs, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation', leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device, dtype=torch.float32, non_blocking=True) 
                        for k, v in batch.items()}
                outputs = model(batch)
                target = batch['target'].view(-1, 1)
                val_loss += criterion(outputs, target).item()
                val_pbar.set_postfix({'batch_loss': f'{val_loss/len(val_loader):.4f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_glucose_model.pth')
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })
    
    # Save final model
    torch.save(model.state_dict(), 'glucose_model.pth')
    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader):
    device = get_device()
    model.eval()
    predictions = []
    actuals = []
    
    test_pbar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in test_pbar:
            batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
            outputs = model(batch)
            # Reshape target to match output dimensions
            target = batch['target'].view(-1, 1)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f'\nTest RMSE: {rmse:.4f}')
    
    # Plot predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:100], label='Actual Glucose Levels')
    plt.plot(predictions[:100], label='Predicted Glucose Levels')
    plt.title('Glucose Level Prediction (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Standardized Glucose Level')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Starting training process...")
    
    # Load and preprocess data
    file_path = 'full_patient_dataset.csv'
    print(f"Loading data from {file_path}...")
    train_dataset, val_dataset, test_dataset = preprocess_data(file_path)
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Initialize and train model
    print("Initializing model...")
    model = GlucoseLSTM()
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    print("\nTraining complete! Saved models:")
    print("- Best model: best_glucose_model.pth")
    print("- Final model: glucose_model.pth")
    
    # Plot training progress
    print("\nPlotting training progress...")
    plot_losses(train_losses, val_losses)
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_loader)
    
    print("\nTraining and evaluation complete!")