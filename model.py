import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from datetime import datetime

# Adjusted constants for better stability
BATCH_SIZE = 32  # Reduced batch size
NUM_WORKERS = 0  # Disabled multiple workers initially
PIN_MEMORY = True

def get_device():
    """Get the best available device: MPS, CUDA, or CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
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

# Setup logging
def setup_logging(log_dir="logs"):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Create model directory
def setup_model_dir(base_dir="models"):
    """Setup model directory for saving checkpoints"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(base_dir, f'run_{timestamp}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, model_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best performance
    if is_best:
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        logging.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    # Save latest model (for resuming training)
    latest_path = os.path.join(model_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, model_dir=None):
    device = get_device()
    logging.info(f"Using device: {device}")
    logging.info(f"Number of training batches: {len(train_loader)}")
    logging.info(f"Number of validation batches: {len(val_loader)}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
            
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    
                    outputs = model(batch)
                    target = batch['target'].view(-1, 1)
                    loss = criterion(outputs, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_pbar.set_postfix({
                        'batch': f'{batch_idx}/{len(train_loader)}',
                        'loss': f'{loss.item():.4f}'
                    })
                    
                except Exception as e:
                    logging.error(f"Error in training batch {batch_idx}: {str(e)}")
                    raise e
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} Validation')
            
            with torch.no_grad():
                for batch in val_pbar:
                    batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
                    outputs = model(batch)
                    target = batch['target'].view(-1, 1)
                    val_loss += criterion(outputs, target).item()
            
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save checkpoint and best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if model_dir:
                save_checkpoint(
                    model, optimizer, epoch, 
                    train_loss, val_loss, 
                    model_dir, is_best
                )
            
            # Log progress
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}" +
                (" - Best Model!" if is_best else "")
            )
            
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise e
    
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
    # Setup logging and model directories
    log_file = setup_logging()
    model_dir = setup_model_dir()
    
    logging.info("Starting training process...")
    logging.info(f"Logs will be saved to: {log_file}")
    logging.info(f"Models will be saved to: {model_dir}")
    
    # Load and preprocess data
    file_path = 'full_patient_dataset.csv'
    logging.info(f"Loading data from {file_path}...")
    train_dataset, val_dataset, test_dataset = preprocess_data(file_path)
    
    logging.info(f"Dataset sizes:")
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    logging.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
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
    logging.info("Initializing model...")
    model = GlucoseLSTM()
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader,
        model_dir=model_dir
    )
    
    logging.info("\nTraining complete!")
    logging.info(f"Models saved in: {model_dir}")
    
    # Plot and save training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_loss.png'))
    plt.close()
    
    # Evaluate model
    logging.info("\nEvaluating model...")
    evaluate_model(model, test_loader)
    
    logging.info("Training and evaluation complete!")