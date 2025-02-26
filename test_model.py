import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import GlucoseLSTM, get_device
from preprocess_data import preprocess_data, BATCH_SIZE, SEQ_LEN, FORECAST_HORIZON
from torch.utils.data import DataLoader
import pandas as pd

def load_model(model_path):
    """Load the trained model"""
    device = get_device()
    model = GlucoseLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def plot_single_prediction(model, sample_data, actual_value, patient_id=None):
    """Plot a single prediction with the input sequence"""
    device = get_device()
    model.eval()
    
    # Prepare input data
    with torch.no_grad():
        inputs = {k: v.unsqueeze(0).to(device) for k, v in sample_data.items() if k != 'target'}
        prediction = model(inputs).cpu().numpy()[0, 0]
    
    # Get the glucose sequence
    glucose_seq = sample_data['glucose'].numpy().flatten()
    
    # Create time points (assuming 5-minute intervals)
    time_points = np.arange(-len(glucose_seq), 1) * 5  # Past hours in minutes
    future_point = time_points[-1] + FORECAST_HORIZON * 5  # Future point (60 minutes ahead)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, glucose_seq, 'b-', label='Historical Glucose')
    plt.plot(time_points[-1], glucose_seq[-1], 'go', label='Last Known Value')
    plt.plot(future_point, prediction, 'ro', label='Prediction')
    plt.plot(future_point, actual_value, 'ko', label='Actual Value')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.fill_between([time_points[-1], future_point], 
                    [70, 70], [180, 180], 
                    color='green', alpha=0.1, 
                    label='Target Range')
    
    plt.title(f'Glucose Prediction (Patient {patient_id if patient_id else "Unknown"})')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose Level (standardized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_multiple_predictions(model, test_loader, num_samples=5):
    """Plot multiple predictions to show variety"""
    device = get_device()
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Get prediction
            inputs = {k: v[0:1].to(device) for k, v in batch.items() if k != 'target'}
            prediction = model(inputs).cpu().numpy()[0, 0]
            actual = batch['target'][0].numpy()
            
            # Get glucose sequence
            glucose_seq = batch['glucose'][0].numpy().flatten()
            
            # Create time points
            time_points = np.arange(-len(glucose_seq), 1) * 5
            future_point = time_points[-1] + FORECAST_HORIZON * 5
            
            # Plot
            axes[i].plot(time_points, glucose_seq, 'b-', label='Historical Glucose')
            axes[i].plot(time_points[-1], glucose_seq[-1], 'go', label='Last Known Value')
            axes[i].plot(future_point, prediction, 'ro', label='Prediction')
            axes[i].plot(future_point, actual, 'ko', label='Actual Value')
            
            axes[i].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            axes[i].fill_between([time_points[-1], future_point], 
                               [70, 70], [180, 180], 
                               color='green', alpha=0.1)
            
            axes[i].set_title(f'Sample {i+1}')
            axes[i].set_xlabel('Time (minutes)')
            axes[i].set_ylabel('Glucose Level')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_prediction_accuracy(model, test_loader, num_samples=100):
    """Analyze prediction accuracy with error distribution"""
    device = get_device()
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
            pred = model(inputs).cpu().numpy()
            predictions.extend(pred.flatten())
            actuals.extend(batch['target'].numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = predictions - actuals
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    ax1.scatter(actuals, predictions, alpha=0.5)
    ax1.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    ax1.set_title('Predicted vs Actual Values')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    
    # Error distribution
    sns.histplot(errors, kde=True, ax=ax2)
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Prediction Error')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Error Standard Deviation: {np.std(errors):.4f}")

if __name__ == "__main__":
    # Load data and model
    file_path = 'full_patient_dataset.csv'
    model_path = 'glucose_model.pth'  # Make sure this exists
    
    # Load test data
    _, _, test_dataset = preprocess_data(file_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load model
    model = load_model(model_path)
    
    # Get a single sample
    sample_batch = next(iter(test_loader))
    plot_single_prediction(model, sample_batch, sample_batch['target'][0])
    
    # Plot multiple predictions
    plot_multiple_predictions(model, test_loader, num_samples=5)
    
    # Analyze prediction accuracy
    analyze_prediction_accuracy(model, test_loader) 