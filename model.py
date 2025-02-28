import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data, SEQ_LEN, FORECAST_HORIZON
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import logging
import os
from datetime import datetime

# Add at the top of the file
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit

# Adjusted constants
BATCH_SIZE = 256
NUM_WORKERS = 0  # Set to 0 to debug data loading
PIN_MEMORY = False  # Disable pin_memory for debugging
MAX_GRAD_NORM = 1.0

def get_device():
    """Get the best available device: MPS, CUDA, or CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class GlucoseLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, dropout1_rate=0.3, dropout2_rate=0.2):
        super(GlucoseLSTM, self).__init__()
        
        # Single LSTM layer with strong regularization
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout between LSTM layers since we only have one
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Multiple dropout layers with different rates
        self.dropout1 = nn.Dropout(dropout1_rate)
        self.dropout2 = nn.Dropout(dropout2_rate)
        
        # Simpler fully connected layers
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
        
        # Apply first dropout to input
        combined_input = self.dropout1(combined_input)
        
        # LSTM layer
        lstm_out, _ = self.lstm(combined_input)
        last_output = lstm_out[:, -1, :]
        
        # Normalization and second dropout
        normalized = self.norm(last_output)
        dropped = self.dropout2(normalized)
        
        # Direct linear projection
        out = self.fc(dropped)
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

def calculate_glucose_metrics(predictions, actuals):
    """
    Calculate metrics specific to glucose prediction accuracy for 1-hour forecasts
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Clarke Error Grid Analysis zones (simplified)
    def in_clinical_error_zone(pred, actual):
        # Returns True if prediction would lead to appropriate treatment
        # This is a simplified version - you may want to implement full Clarke Error Grid
        error_margin = 0.2  # 20% margin
        return abs(pred - actual) <= actual * error_margin
    
    # Core metrics
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    
    # Clinical metrics
    in_range_accuracy = np.mean([in_clinical_error_zone(p, a) for p, a in zip(predictions, actuals)]) * 100
    
    # Trend accuracy (direction of change)
    pred_trends = np.diff(predictions.flatten())
    actual_trends = np.diff(actuals.flatten())
    trend_accuracy = np.mean(np.sign(pred_trends) == np.sign(actual_trends)) * 100
    
    # Extreme value metrics
    high_errors = np.mean(np.abs(predictions[actuals > np.percentile(actuals, 75)] - 
                                actuals[actuals > np.percentile(actuals, 75)])) 
    low_errors = np.mean(np.abs(predictions[actuals < np.percentile(actuals, 25)] - 
                               actuals[actuals < np.percentile(actuals, 25)]))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'in_range_%': in_range_accuracy,
        'trend_accuracy_%': trend_accuracy,
        'high_glucose_mae': high_errors,
        'low_glucose_mae': low_errors
    }

def validate_model(model, val_loader, criterion, device):
    """
    Perform validation with detailed metrics for 1-hour predictions
    """
    model.eval()
    val_loss = 0
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
            outputs = model(batch)
            target = batch['target'].view(-1, 1)
            
            # Calculate MSE loss
            loss = criterion(outputs, target)
            val_loss += loss.item()
            
            # Store predictions and actuals
            all_predictions.extend(outputs.cpu().numpy())
            all_actuals.extend(target.cpu().numpy())
    
    # Calculate average loss
    val_loss /= len(val_loader)
    
    # Calculate detailed metrics
    metrics = calculate_glucose_metrics(all_predictions, all_actuals)
    
    return val_loss, metrics

def hyperparameter_search(train_loader, val_loader, model_dir):
    """Perform grid search for hyperparameters"""
    # Define hyperparameter grid with smaller search space
    param_grid = {
        'hidden_size': [32, 48],  # Reduced sizes
        'learning_rate': [0.001, 0.0005],
        'weight_decay': [0.01, 0.02],
        'dropout_rates': [(0.2, 0.1), (0.3, 0.2)],
        'batch_size': [16, 32]  # Smaller batch sizes
    }
    
    results = []
    best_val_loss = float('inf')
    best_params = None
    
    logging.info("Starting hyperparameter search...")
    
    search_dir = os.path.join(model_dir, 'hyperparam_search')
    os.makedirs(search_dir, exist_ok=True)
    
    # Grid search
    for hidden_size in param_grid['hidden_size']:
        for lr in param_grid['learning_rate']:
            for wd in param_grid['weight_decay']:
                for d1, d2 in param_grid['dropout_rates']:
                    try:
                        model = GlucoseLSTM(
                            hidden_size=hidden_size,
                            dropout1_rate=d1,
                            dropout2_rate=d2
                        )
                        
                        train_losses, val_losses, metrics = train_model(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            num_epochs=10,
                            learning_rate=lr,
                            weight_decay=wd,
                            model_dir=search_dir
                        )
                        
                        best_epoch_val_loss = min(val_losses)
                        
                        # Convert numpy values to Python native types
                        metrics = {k: float(v) for k, v in metrics.items()}
                        
                        result = {
                            'hidden_size': hidden_size,
                            'learning_rate': lr,
                            'weight_decay': wd,
                            'dropout1_rate': d1,
                            'dropout2_rate': d2,
                            'best_val_loss': float(best_epoch_val_loss),  # Convert to native Python float
                            'final_metrics': metrics
                        }
                        results.append(result)
                        
                        if best_epoch_val_loss < best_val_loss:
                            best_val_loss = best_epoch_val_loss
                            best_params = result.copy()
                        
                        logging.info(f"\nTried parameters:")
                        logging.info(f"  Hidden Size: {hidden_size}")
                        logging.info(f"  Learning Rate: {lr}")
                        logging.info(f"  Weight Decay: {wd}")
                        logging.info(f"  Dropout Rates: ({d1}, {d2})")
                        logging.info(f"  Best Val Loss: {best_epoch_val_loss:.4f}")
                        
                        # Clear GPU memory
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        
                    except Exception as e:
                        logging.error(f"Error with parameters {hidden_size}, {lr}, {wd}, ({d1}, {d2}): {str(e)}")
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
                        continue
    
    # Save results
    import json
    with open(os.path.join(search_dir, 'search_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("\nHyperparameter search complete!")
    logging.info("\nBest parameters found:")
    for k, v in best_params.items():
        if k != 'final_metrics':  # Skip printing the full metrics
            logging.info(f"  {k}: {v}")
    
    return best_params

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.0005, 
                weight_decay=0.03, model_dir=None):
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Debug logging
    logging.info(f"Training with batch size: {train_loader.batch_size}")
    logging.info(f"Number of training batches: {len(train_loader)}")
    logging.info(f"Total training samples: {len(train_loader.dataset)}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_metrics = None
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            batch_count = 0
            
            # Debug first batch before training
            try:
                first_batch = next(iter(train_loader))
                logging.info("First batch loaded successfully")
                logging.info(f"First batch keys: {first_batch.keys()}")
                for k, v in first_batch.items():
                    logging.info(f"{k} shape: {v.shape}")
            except Exception as e:
                logging.error(f"Error loading first batch: {str(e)}")
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} Training')
            
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    # Debug first few batches
                    if epoch == 0 and batch_idx < 2:
                        logging.info(f"\nProcessing batch {batch_idx}")
                        logging.info(f"Batch keys: {batch.keys()}")
                    
                    # Move batch to device
                    batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
                    
                    # Debug inputs
                    if epoch == 0 and batch_idx == 0:
                        logging.info(f"Input glucose shape: {batch['glucose'].shape}")
                        logging.info(f"Target shape: {batch['target'].shape}")
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    outputs = model(batch)
                    target = batch['target'].view(-1, 1)
                    batch_loss = criterion(outputs, target)
                    
                    # Debug loss calculation
                    if epoch == 0 and batch_idx < 5:
                        logging.info(f"Batch {batch_idx} - Loss: {batch_loss.item():.4f}")
                        logging.info(f"Output shape: {outputs.shape}, Target shape: {target.shape}")
                    
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    
                    train_loss += batch_loss.item()
                    batch_count += 1
                    
                    # Update progress bar every batch
                    avg_loss = train_loss / (batch_idx + 1)
                    train_pbar.set_postfix({
                        'avg_loss': f'{avg_loss:.4f}',
                        'batch': f'{batch_idx}/{len(train_loader)}'
                    })
                    
                    # Clear memory less frequently
                    if device.type == 'mps' and batch_idx % 100 == 0:
                        torch.mps.empty_cache()
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    logging.error(f"Batch keys: {batch.keys()}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
            
            # Compute average loss for epoch
            train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
            logging.info(f"\nEpoch {epoch+1} average training loss: {train_loss:.4f}")
            
            # Validation phase with detailed metrics
            val_loss, metrics = validate_model(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Save best model based on clinical metrics
            clinical_score = (metrics['in_range_%'] + metrics['trend_accuracy_%']) / 2
            is_best = clinical_score > best_clinical_score if 'best_clinical_score' in locals() else True
            
            if is_best:
                best_clinical_score = clinical_score
                best_metrics = metrics
                if model_dir:
                    save_checkpoint(model, optimizer, epoch, train_loss, val_loss, model_dir, is_best)
            
            # Log progress with detailed metrics
            logging.info(
                f"Epoch {epoch+1}/{num_epochs}\n"
                f"  Train Loss: {train_loss:.4f}\n"
                f"  Val Loss: {val_loss:.4f}\n"
                f"  RMSE: {metrics['rmse']:.4f}\n"
                f"  In-Range Accuracy: {metrics['in_range_%']:.1f}%\n"
                f"  Trend Accuracy: {metrics['trend_accuracy_%']:.1f}%\n"
                f"  High Glucose MAE: {metrics['high_glucose_mae']:.4f}\n"
                f"  Low Glucose MAE: {metrics['low_glucose_mae']:.4f}"
                + (" - Best Model!" if is_best else "")
            )
            
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise e
    
    return train_losses, val_losses, best_metrics

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
    
    # Create data loaders with consistent settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Initialize model with best parameters
    model = GlucoseLSTM(
        hidden_size=48,
        dropout1_rate=0.2,
        dropout2_rate=0.1
    )
    
    # Train with best parameters and adjusted learning rate
    train_losses, val_losses, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=0.002,  # Increased learning rate for larger batch size
        weight_decay=0.02,
        model_dir=model_dir
    )
    
    logging.info("\nTraining complete!")
    logging.info(f"Models saved in: {model_dir}")
    logging.info("\nBest Model Metrics:")
    for metric, value in best_metrics.items():
        logging.info(f"  {metric}: {value:.4f}")
    
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