import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import GlucoseLSTM, get_device, calculate_glucose_metrics
from preprocess_data import preprocess_data, BATCH_SIZE, SEQ_LEN, FORECAST_HORIZON
from torch.utils.data import DataLoader
import pandas as pd
import logging
from datetime import datetime
import os
import torch.nn as nn
from clarke_error_grid import clarke_error_grid_analysis, analyze_all_timesteps

def setup_test_logging(log_dir="test_logs"):
    """Setup logging for testing"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_results_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_best_model(model_dir):
    """Load the best model from training"""
    device = get_device()
    
    # Match the exact architecture of the saved model
    class OriginalGlucoseLSTM(nn.Module):
        def __init__(self, input_size=4, hidden_size=48, dropout1_rate=0.3, dropout2_rate=0.2):
            super(OriginalGlucoseLSTM, self).__init__()
            
            # LSTM for glucose sequence
            self.lstm_glucose = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout1_rate,
                batch_first=True
            )
            
            # Linear layers for prediction
            self.fc1 = nn.Linear(hidden_size + 3, hidden_size)
            self.dropout = nn.Dropout(dropout2_rate)
            self.fc2 = nn.Linear(hidden_size, 12)  # Output 12 values for 1-hour prediction
        
        def forward(self, x):
            # Process glucose sequence
            glucose_seq = x['glucose']
            lstm_out, _ = self.lstm_glucose(glucose_seq)
            lstm_last = lstm_out[:, -1, :]
            
            # Get other features
            basal = x['basal'][:, -1:, 0]
            bolus = x['bolus'][:, -1:, 0]
            carbs = x['carbs'][:, -1:, 0]
            
            # Combine features
            combined = torch.cat([lstm_last, basal, bolus, carbs], dim=1)
            
            # Make predictions for next 12 time steps
            x = self.fc1(combined)
            x = torch.relu(x)
            x = self.dropout(x)
            output = self.fc2(x)  # Shape: [batch_size, 12]
            
            return output
    
    # Create model with original architecture and correct dimensions
    model = OriginalGlucoseLSTM(
        input_size=4,
        hidden_size=48,
        dropout1_rate=0.3,
        dropout2_rate=0.2
    )
    
    model_path = os.path.join(model_dir, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def print_sample_predictions(predictions, actuals, inputs, num_samples=5):
    """Print a table of sample predictions with their sequences"""
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    def to_mgdl(standardized_value):
        return standardized_value * GLUCOSE_STD + GLUCOSE_MEAN
    
    logging.info("\nSample Predictions (in mg/dL):")
    logging.info("=" * 120)
    logging.info(f"{'Last Value':^15} | {'Trajectory (5-min intervals)':^90}")
    logging.info("-" * 120)
    
    sample_idx = np.random.choice(len(predictions), size=num_samples, replace=False)
    
    for idx in sample_idx:
        last_value = to_mgdl(inputs[idx].flatten()[-1])
        pred_trajectory = to_mgdl(predictions[idx])
        actual_trajectory = to_mgdl(actuals[idx])
        
        # Format trajectory values
        pred_str = " ".join([f"{v:6.1f}" for v in pred_trajectory])
        actual_str = " ".join([f"{v:6.1f}" for v in actual_trajectory])
        
        logging.info(f"{last_value:15.1f} | Pred:  {pred_str}")
        logging.info(f"{' ':15} | Actual:{actual_str}")
        logging.info("-" * 120)
    
    logging.info("=" * 120)

def plot_glucose_predictions(predictions, actuals, inputs, num_samples=6):
    """Plot glucose sequences with predictions and actuals"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Constants for converting standardized values to mg/dL
    GLUCOSE_MEAN = 150  # mg/dL
    GLUCOSE_STD = 50   # mg/dL
    
    def to_mgdl(standardized_value):
        return standardized_value * GLUCOSE_STD + GLUCOSE_MEAN
    
    # Randomly select samples
    sample_idx = np.random.choice(len(predictions), size=num_samples, replace=False)
    
    for idx, ax in zip(sample_idx, axes):
        # Get the sequence data
        glucose_seq = inputs[idx].flatten()
        pred_sequence = predictions[idx]  # Full 12-step prediction sequence
        actual_sequence = actuals[idx]    # Full 12-step actual sequence
        
        # Convert to mg/dL
        glucose_seq_mgdl = to_mgdl(glucose_seq)
        pred_sequence_mgdl = to_mgdl(pred_sequence)
        actual_sequence_mgdl = to_mgdl(actual_sequence)
        
        # Create time points (5-minute intervals)
        historical_times = np.arange(len(glucose_seq)) * 5  # Past time points
        future_times = np.arange(
            historical_times[-1] + 5,
            historical_times[-1] + 65,
            5
        )  # 12 future 5-min intervals
        
        # Plot historical glucose values
        ax.plot(historical_times, glucose_seq_mgdl, 'b-', 
                label='Historical Glucose', linewidth=2)
        
        # Plot vertical line at prediction start
        ax.axvline(x=historical_times[-1], color='gray', 
                  linestyle='--', alpha=0.5, label='Prediction Start')
        
        # Plot prediction and actual sequences
        ax.plot(future_times, pred_sequence_mgdl, 'r--', marker='o', 
               label='Predicted Glucose', linewidth=2, markersize=4)
        ax.plot(future_times, actual_sequence_mgdl, 'g:', marker='x', 
               label='Actual Glucose', linewidth=2, markersize=4)
        
        # Add target range zone (70-180 mg/dL)
        ax.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
        
        # Customize plot
        ax.set_title(
            f'Sample {idx+1}\n'
            f'Last Known: {glucose_seq_mgdl[-1]:.1f} mg/dL\n'
            f'60-min Pred: {pred_sequence_mgdl[-1]:.1f} mg/dL\n'
            f'60-min Actual: {actual_sequence_mgdl[-1]:.1f} mg/dL'
        )
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Glucose Level (mg/dL)')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to show reasonable glucose range
        ax.set_ylim(40, 400)
        
        # Add legend to first plot only
        if idx == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('glucose_predictions.png')
    plt.close()

def calculate_glucose_metrics_mgdl(predictions, actuals):
    """Calculate metrics in mg/dL for trajectory predictions"""
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    # Convert to mg/dL
    predictions_mgdl = predictions * GLUCOSE_STD + GLUCOSE_MEAN
    actuals_mgdl = actuals * GLUCOSE_STD + GLUCOSE_MEAN
    
    # Calculate metrics for each time step
    rmse_per_step = np.sqrt(np.mean((predictions_mgdl - actuals_mgdl) ** 2, axis=0))
    mae_per_step = np.mean(np.abs(predictions_mgdl - actuals_mgdl), axis=0)
    
    # Overall metrics
    rmse = np.mean(rmse_per_step)
    mae = np.mean(mae_per_step)
    
    # Clinical metrics
    in_range_mask = (actuals_mgdl >= 70) & (actuals_mgdl <= 180)
    in_range_accuracy = np.mean(
        np.abs(predictions_mgdl[in_range_mask] - actuals_mgdl[in_range_mask]) <= 20
    ) * 100
    
    # Trend accuracy (calculated between consecutive predictions)
    pred_trends = np.diff(predictions_mgdl, axis=1)
    actual_trends = np.diff(actuals_mgdl, axis=1)
    trend_accuracy = np.mean(np.sign(pred_trends) == np.sign(actual_trends)) * 100
    
    # Extreme value metrics
    high_mask = actuals_mgdl > 180
    low_mask = actuals_mgdl < 70
    
    high_glucose_mae = np.mean(np.abs(predictions_mgdl[high_mask] - actuals_mgdl[high_mask]))
    low_glucose_mae = np.mean(np.abs(predictions_mgdl[low_mask] - actuals_mgdl[low_mask]))
    
    # Hypo/Hyper detection rates
    hypo_detection = np.mean((predictions_mgdl < 70) & (actuals_mgdl < 70)) * 100
    hyper_detection = np.mean((predictions_mgdl > 180) & (actuals_mgdl > 180)) * 100
    
    # Time-specific metrics (5-min intervals)
    metrics_by_horizon = {
        f'rmse_{(i+1)*5}min': rmse_per_step[i] for i in range(len(rmse_per_step))
    }
    
    return {
        'rmse_mgdl': rmse,
        'mae_mgdl': mae,
        'in_range_%': in_range_accuracy,
        'trend_accuracy_%': trend_accuracy,
        'high_glucose_mae_mgdl': high_glucose_mae,
        'low_glucose_mae_mgdl': low_glucose_mae,
        'hypo_detection_%': hypo_detection,
        'hyper_detection_%': hyper_detection,
        **metrics_by_horizon
    }

def plot_prediction_analysis(predictions, actuals, inputs, metrics):
    """Create comprehensive visualization of model performance"""
    fig = plt.figure(figsize=(20, 12))
    
    # Constants for converting to mg/dL
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    def to_mgdl(standardized_value):
        return standardized_value * GLUCOSE_STD + GLUCOSE_MEAN
    
    # 1. Prediction vs Actual scatter for final values (60-min predictions)
    ax1 = plt.subplot(2, 2, 1)
    final_preds = predictions[:, -1]  # Last prediction (60 min)
    final_actuals = actuals[:, -1]    # Last actual value (60 min)
    
    final_preds_mgdl = to_mgdl(final_preds)
    final_actuals_mgdl = to_mgdl(final_actuals)
    
    ax1.scatter(final_actuals_mgdl, final_preds_mgdl, alpha=0.5, color='blue', s=10)
    ax1.plot([40, 400], [40, 400], 'r--')  # Perfect prediction line
    ax1.set_title('60-min Predictions vs Actuals')
    ax1.set_xlabel('Actual Values (mg/dL)')
    ax1.set_ylabel('Predicted Values (mg/dL)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(40, 400)
    ax1.set_ylim(40, 400)
    
    # 2. Error Distribution for all predictions
    ax2 = plt.subplot(2, 2, 2)
    all_preds_mgdl = to_mgdl(predictions.flatten())
    all_actuals_mgdl = to_mgdl(actuals.flatten())
    errors_mgdl = all_preds_mgdl - all_actuals_mgdl
    
    ax2.hist(errors_mgdl, bins=50, alpha=0.75, color='blue')
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Prediction Error (mg/dL)')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sample Predictions Over Time
    ax3 = plt.subplot(2, 2, 3)
    sample_idx = np.random.choice(len(predictions), size=5)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, color in zip(sample_idx, colors):
        # Historical values
        historical = to_mgdl(inputs[idx].flatten())
        historical_times = np.arange(len(historical)) * 5
        
        # Predicted and actual future values
        pred_sequence = to_mgdl(predictions[idx])
        actual_sequence = to_mgdl(actuals[idx])
        future_times = np.arange(len(historical), len(historical) + len(pred_sequence)) * 5
        
        # Plot
        ax3.plot(historical_times, historical, color=color, alpha=0.5)
        ax3.plot(future_times, pred_sequence, '--', color=color, marker='o', label=f'Pred {idx}')
        ax3.plot(future_times, actual_sequence, ':', color=color, marker='x', label=f'Actual {idx}')
    
    ax3.set_title('Sample Predictions Over Time')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Glucose Level (mg/dL)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(40, 400)
    if idx == 0:
        ax3.legend()
    
    # 4. Metrics Summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    metrics_text = (
        f"Model Performance Metrics:\n\n"
        f"RMSE: {metrics['rmse_mgdl']:.1f} mg/dL\n"
        f"MAE: {metrics['mae_mgdl']:.1f} mg/dL\n"
        f"In-Range Accuracy: {metrics['in_range_%']:.1f}%\n"
        f"Trend Accuracy: {metrics['trend_accuracy_%']:.1f}%\n"
        f"High Glucose MAE: {metrics['high_glucose_mae_mgdl']:.1f} mg/dL\n"
        f"Low Glucose MAE: {metrics['low_glucose_mae_mgdl']:.1f} mg/dL\n"
        f"Hypo Detection: {metrics['hypo_detection_%']:.1f}%\n"
        f"Hyper Detection: {metrics['hyper_detection_%']:.1f}%"
    )
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def calculate_in_range_accuracy(predictions, actuals, threshold=20):
    """Calculate percentage of predictions within threshold of actual values"""
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    # Convert to mg/dL
    predictions_mgdl = predictions * GLUCOSE_STD + GLUCOSE_MEAN
    actuals_mgdl = actuals * GLUCOSE_STD + GLUCOSE_MEAN
    
    return np.mean(np.abs(predictions_mgdl - actuals_mgdl) <= threshold) * 100

def evaluate_predictions(model, test_loader):
    """Evaluate model predictions with detailed metrics"""
    device = get_device()
    model.eval()
    
    # Verify we're using test data
    total_samples = len(test_loader.dataset)
    logging.info(f"\nEvaluating model on {total_samples} test samples")
    logging.info(f"Predicting {FORECAST_HORIZON * 5} minutes into the future")
    
    all_predictions = []
    all_actuals = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
            
            # Forward pass
            predictions = model(batch)  # Model expects dictionary input
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(batch['target'].cpu().numpy())
            all_inputs.extend(batch['glucose'].cpu().numpy())
    
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    inputs = np.array(all_inputs)
    
    logging.info(f"Test set size: {len(predictions)} samples")
    
    # Print sample predictions
    print_sample_predictions(predictions, actuals, inputs)
    
    # Plot glucose predictions
    plot_glucose_predictions(predictions, actuals, inputs)
    
    # Calculate metrics in mg/dL
    metrics = calculate_glucose_metrics_mgdl(predictions, actuals)
    
    # Add Clarke Error Grid Analysis (now using imported function)
    clarke_zones = clarke_error_grid_analysis(predictions, actuals)
    logging.info("\nClarke Error Grid Analysis:")
    for zone, percentage in clarke_zones.items():
        logging.info(f"  Zone {zone}: {percentage:.1f}%")
    
    # Add to metrics
    for zone, percentage in clarke_zones.items():
        metrics[f'clarke_{zone}_%'] = percentage
    
    # Log detailed metrics
    logging.info("\nDetailed Test Set Metrics (in mg/dL):")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.1f}")
    
    # Plot analysis
    plot_prediction_analysis(predictions, actuals, inputs, metrics)
    
    return metrics

def analyze_time_horizon_performance(predictions, actuals):
    """Analyze model performance degradation over prediction horizons"""
    metrics_by_horizon = {}
    
    # Calculate metrics for each 5-minute horizon
    for i in range(predictions.shape[1]):
        horizon_preds = predictions[:, i]
        horizon_actuals = actuals[:, i]
        
        horizon_metrics = {
            f'rmse_{(i+1)*5}min': np.sqrt(np.mean((horizon_preds - horizon_actuals) ** 2)),
            f'mae_{(i+1)*5}min': np.mean(np.abs(horizon_preds - horizon_actuals)),
            f'in_range_{(i+1)*5}min': calculate_in_range_accuracy(horizon_preds, horizon_actuals)
        }
        metrics_by_horizon.update(horizon_metrics)
    
    # Plot degradation over time
    plt.figure(figsize=(12, 6))
    rmse_values = [metrics_by_horizon[f'rmse_{(i+1)*5}min'] for i in range(predictions.shape[1])]
    plt.plot(range(5, 65, 5), rmse_values, marker='o')
    plt.title('Prediction Error by Time Horizon')
    plt.xlabel('Minutes into Future')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig('time_horizon_analysis.png')
    
    return metrics_by_horizon

def compare_with_baselines(model, test_loader):
    """Compare LSTM with simpler baseline models"""
    device = get_device()
    all_inputs = []
    all_targets = []
    lstm_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lstm_predictions.extend(model(batch).cpu().numpy())
            all_inputs.extend(batch['glucose'].cpu().numpy())
            all_targets.extend(batch['target'].cpu().numpy())
    
    inputs = np.array(all_inputs)
    targets = np.array(all_targets)
    lstm_predictions = np.array(lstm_predictions)
    
    # Baseline 1: Last value prediction (persistence model)
    last_value_preds = np.repeat(inputs[:, -1, 0].reshape(-1, 1), targets.shape[1], axis=1)
    
    # Baseline 2: Linear extrapolation
    slopes = (inputs[:, -1, 0] - inputs[:, -2, 0]).reshape(-1, 1)
    steps = np.arange(1, targets.shape[1] + 1).reshape(1, -1)
    linear_preds = inputs[:, -1, 0].reshape(-1, 1) + slopes * steps
    
    # Calculate metrics for each baseline
    baseline_metrics = {
        'lstm_model': calculate_glucose_metrics_mgdl(lstm_predictions, targets),
        'last_value': calculate_glucose_metrics_mgdl(last_value_preds, targets),
        'linear_extrapolation': calculate_glucose_metrics_mgdl(linear_preds, targets)
    }
    
    return baseline_metrics

def analyze_personalized_performance(predictions, actuals, patient_ids):
    """Analyze model performance per patient"""
    unique_patients = np.unique(patient_ids)
    patient_metrics = {}
    
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_preds = predictions[patient_mask]
        patient_actuals = actuals[patient_mask]
        
        if len(patient_preds) > 0:
            metrics = calculate_glucose_metrics_mgdl(patient_preds, patient_actuals)
            patient_metrics[patient] = metrics
    
    # Find patients with worst/best performance
    patient_rmse = {p: m['rmse_mgdl'] for p, m in patient_metrics.items()}
    best_patients = sorted(patient_rmse.items(), key=lambda x: x[1])[:5]
    worst_patients = sorted(patient_rmse.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return patient_metrics, best_patients, worst_patients

def collect_predictions(model, test_loader):
    """Collect all predictions, actuals and inputs from test loader"""
    device = get_device()
    all_predictions = []
    all_actuals = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch)
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(batch['target'].cpu().numpy())
            all_inputs.extend(batch['glucose'].cpu().numpy())
    
    return (np.array(all_predictions), 
            np.array(all_actuals), 
            np.array(all_inputs))

def evaluate_by_glucose_range(predictions, actuals):
    """Evaluate model performance stratified by glucose ranges"""
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    # Convert to mg/dL
    predictions_mgdl = predictions * GLUCOSE_STD + GLUCOSE_MEAN
    actuals_mgdl = actuals * GLUCOSE_STD + GLUCOSE_MEAN
    
    # Define ranges
    ranges = {
        'Hypoglycemia (<70 mg/dL)': (0, 70),
        'Target Range (70-180 mg/dL)': (70, 180),
        'Hyperglycemia (>180 mg/dL)': (180, float('inf'))
    }
    
    metrics = {}
    for range_name, (min_val, max_val) in ranges.items():
        mask = (actuals_mgdl >= min_val) & (actuals_mgdl < max_val)
        if not np.any(mask):
            continue
            
        range_preds = predictions_mgdl[mask]
        range_actuals = actuals_mgdl[mask]
        
        metrics[range_name] = {
            'count': np.sum(mask),
            'rmse': np.sqrt(np.mean((range_preds - range_actuals) ** 2)),
            'mae': np.mean(np.abs(range_preds - range_actuals)),
            'in_range_%': calculate_in_range_accuracy(
                (range_preds - GLUCOSE_MEAN) / GLUCOSE_STD,
                (range_actuals - GLUCOSE_MEAN) / GLUCOSE_STD
            )
        }
    
    return metrics

if __name__ == "__main__":
    # Setup logging
    log_file = setup_test_logging()
    logging.info("Starting model evaluation...")
    
    # Load data with same split as training
    file_path = 'full_patient_dataset.csv'
    logging.info(f"Loading data from {file_path}")
    train_dataset, val_dataset, test_dataset = preprocess_data(
        file_path,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    # Log dataset sizes to verify split
    logging.info(f"Dataset sizes:")
    logging.info(f"  Training samples: {len(train_dataset)}")
    logging.info(f"  Validation samples: {len(val_dataset)}")
    logging.info(f"  Test samples: {len(test_dataset)}")
    
    # Create test loader with same batch size as training
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load best model from the specified directory
    model_dir = "models/run_20250228_155402"  # Update with your model directory
    try:
        model = load_best_model(model_dir)
        logging.info(f"Successfully loaded model from {model_dir}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise
    
    # 1. Run standard evaluation with Clarke Error Grid
    logging.info("\n" + "="*80)
    logging.info("Standard Evaluation with Clarke Error Grid Analysis")
    logging.info("="*80)
    metrics = evaluate_predictions(model, test_loader)
    
    logging.info("\nStandard Test Set Metrics:")
    for metric, value in metrics.items():
        logging.info(f"  {metric}: {value:.1f}")
    
    # 2. Get predictions for additional analyses
    logging.info("\n" + "="*80)
    logging.info("Collecting predictions for additional analyses...")
    predictions, actuals, inputs = collect_predictions(model, test_loader)
    
    # New: Add timestep-specific Clarke Error Grid analysis
    logging.info("\n" + "="*80)
    logging.info("Generating Clarke Error Grids for Each Timestep")
    logging.info("="*80)
    
    # Create output directory based on model name
    model_name = os.path.basename(model_dir)
    clarke_output_dir = os.path.join('clarke_grids', model_name)
    os.makedirs(clarke_output_dir, exist_ok=True)
    
    # Generate Clarke Error Grids for each of the 12 timesteps
    timestep_results = analyze_all_timesteps(
        predictions, 
        actuals, 
        output_dir=clarke_output_dir
    )
    
    # Log the results for each timestep
    logging.info("\nClarke Error Grid Analysis by Timestep:")
    for result in timestep_results:
        minutes = result['minutes']
        logging.info(f"\n  {minutes}-Minute Predictions:")
        for zone in ['A', 'B', 'C', 'D', 'E', 'A+B']:
            logging.info(f"    Zone {zone}: {result[zone]:.1f}%")
    
    logging.info(f"\nClarke Error Grid visualizations saved to: {clarke_output_dir}")
    logging.info(f"Summary plot saved to: {os.path.join(clarke_output_dir, 'clarke_grid_summary.png')}")
    
    # 3. Time horizon analysis
    logging.info("\n" + "="*80)
    logging.info("Time Horizon Performance Analysis")
    logging.info("="*80)
    horizon_metrics = analyze_time_horizon_performance(predictions, actuals)
    
    # 4. Baseline comparison
    logging.info("\n" + "="*80)
    logging.info("Baseline Model Comparison")
    logging.info("="*80)
    baseline_metrics = compare_with_baselines(model, test_loader)
    
    logging.info("\nModel Comparison:")
    for model_name, metrics in baseline_metrics.items():
        logging.info(f"  {model_name}:")
        for metric, value in metrics.items():
            logging.info(f"    {metric}: {value:.1f}")
    
    # 5. Glucose range stratified analysis
    logging.info("\n" + "="*80)
    logging.info("Stratified Analysis by Glucose Range")
    logging.info("="*80)
    
    range_metrics = evaluate_by_glucose_range(predictions, actuals)
    
    logging.info("\nPerformance by Glucose Range:")
    for range_name, metrics in range_metrics.items():
        logging.info(f"  {range_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"    {metric}: {value:.1f}")
            else:
                logging.info(f"    {metric}: {value}")
    
    logging.info("\nEvaluation complete!") 