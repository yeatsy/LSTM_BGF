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
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    
    # Create model with original architecture
    model = GlucoseLSTM(
        input_size=1,
        hidden_size=128,  # Updated for new architecture
        num_layers=3,     # Updated for new architecture
        dropout1_rate=0.3,
        dropout2_rate=0.2,
        bidirectional=True  # Updated for new architecture
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
        
        # Format trajectory values with confidence intervals if available
        pred_str = " ".join([f"{v:6.1f}" for v in pred_trajectory])
        actual_str = " ".join([f"{v:6.1f}" for v in actual_trajectory])
        
        logging.info(f"{last_value:15.1f} | Pred:  {pred_str}")
        logging.info(f"{' ':15} | Actual:{actual_str}")
        logging.info("-" * 120)
    
    logging.info("=" * 120)

def calculate_glucose_metrics_mgdl(predictions, actuals):
    """Calculate comprehensive metrics in mg/dL for trajectory predictions"""
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
    
    # Clinical metrics with more detailed ranges
    in_range_mask = (actuals_mgdl >= 70) & (actuals_mgdl <= 180)
    tight_control_mask = (actuals_mgdl >= 80) & (actuals_mgdl <= 140)
    
    in_range_accuracy = np.mean(
        np.abs(predictions_mgdl[in_range_mask] - actuals_mgdl[in_range_mask]) <= 20
    ) * 100
    
    tight_control_accuracy = np.mean(
        np.abs(predictions_mgdl[tight_control_mask] - actuals_mgdl[tight_control_mask]) <= 15
    ) * 100
    
    # Trend accuracy with magnitude consideration
    pred_trends = np.diff(predictions_mgdl, axis=1)
    actual_trends = np.diff(actuals_mgdl, axis=1)
    
    # Basic trend accuracy (direction only)
    trend_accuracy = np.mean(np.sign(pred_trends) == np.sign(actual_trends)) * 100
    
    # Advanced trend accuracy (direction and magnitude)
    trend_mae = np.mean(np.abs(pred_trends - actual_trends))
    
    # Extreme value metrics with multiple thresholds
    severe_hypo_mask = actuals_mgdl < 54
    hypo_mask = (actuals_mgdl >= 54) & (actuals_mgdl < 70)
    high_mask = (actuals_mgdl > 180) & (actuals_mgdl <= 250)
    severe_high_mask = actuals_mgdl > 250
    
    # Calculate MAE for each range
    severe_hypo_mae = np.mean(np.abs(predictions_mgdl[severe_hypo_mask] - actuals_mgdl[severe_hypo_mask])) if np.any(severe_hypo_mask) else 0
    hypo_mae = np.mean(np.abs(predictions_mgdl[hypo_mask] - actuals_mgdl[hypo_mask])) if np.any(hypo_mask) else 0
    high_mae = np.mean(np.abs(predictions_mgdl[high_mask] - actuals_mgdl[high_mask])) if np.any(high_mask) else 0
    severe_high_mae = np.mean(np.abs(predictions_mgdl[severe_high_mask] - actuals_mgdl[severe_high_mask])) if np.any(severe_high_mask) else 0
    
    # Detection rates with multiple thresholds
    hypo_detection = {
        'severe': calculate_detection_metrics(predictions_mgdl < 54, actuals_mgdl < 54),
        'moderate': calculate_detection_metrics(predictions_mgdl < 70, actuals_mgdl < 70)
    }
    
    hyper_detection = {
        'high': calculate_detection_metrics(predictions_mgdl > 180, actuals_mgdl > 180),
        'severe': calculate_detection_metrics(predictions_mgdl > 250, actuals_mgdl > 250)
    }
    
    # Time-specific metrics
    metrics_by_horizon = {
        f'rmse_{(i+1)*5}min': rmse_per_step[i] for i in range(len(rmse_per_step))
    }
    
    # Lead time analysis for hypo/hyper events
    hypo_lead_time = calculate_lead_time(predictions_mgdl, actuals_mgdl, threshold=70, below=True)
    hyper_lead_time = calculate_lead_time(predictions_mgdl, actuals_mgdl, threshold=180, below=False)
    
    return {
        'rmse_mgdl': rmse,
        'mae_mgdl': mae,
        'in_range_%': in_range_accuracy,
        'tight_control_%': tight_control_accuracy,
        'trend_accuracy_%': trend_accuracy,
        'trend_mae': trend_mae,
        'severe_hypo_mae': severe_hypo_mae,
        'hypo_mae': hypo_mae,
        'high_mae': high_mae,
        'severe_high_mae': severe_high_mae,
        'hypo_detection': hypo_detection,
        'hyper_detection': hyper_detection,
        'hypo_lead_time': hypo_lead_time,
        'hyper_lead_time': hyper_lead_time,
        **metrics_by_horizon
    }

def calculate_detection_metrics(predictions, actuals):
    """Calculate detection metrics including precision, recall, and F1 score"""
    tn, fp, fn, tp = confusion_matrix(actuals.flatten(), predictions.flatten()).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }

def calculate_lead_time(predictions, actuals, threshold, below=True):
    """Calculate average lead time for detecting events"""
    lead_times = []
    
    for pred_seq, actual_seq in zip(predictions, actuals):
        # Find first occurrence of event in actual sequence
        actual_event = np.where(actual_seq < threshold if below else actual_seq > threshold)[0]
        if len(actual_event) > 0:
            actual_event = actual_event[0]
            
            # Find first prediction of event
            pred_event = np.where(pred_seq < threshold if below else pred_seq > threshold)[0]
            if len(pred_event) > 0:
                pred_event = pred_event[0]
                
                # Calculate lead time (in 5-minute intervals)
                lead_time = (actual_event - pred_event) * 5
                if lead_time > 0:  # Only count positive lead times
                    lead_times.append(lead_time)
    
    return np.mean(lead_times) if lead_times else 0

def plot_prediction_analysis(predictions, actuals, inputs, metrics):
    """Create comprehensive visualization of model performance"""
    fig = plt.figure(figsize=(20, 15))
    
    # Constants for converting to mg/dL
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    def to_mgdl(standardized_value):
        return standardized_value * GLUCOSE_STD + GLUCOSE_MEAN
    
    # 1. Prediction vs Actual scatter with Clarke Error Grid
    ax1 = plt.subplot(3, 2, 1)
    final_preds = to_mgdl(predictions[:, -1])
    final_actuals = to_mgdl(actuals[:, -1])
    
    plot_clarke_error_grid(ax1, final_actuals, final_preds)
    ax1.set_title('Clarke Error Grid Analysis (60-min Predictions)')
    
    # 2. Error Distribution with Normal Fit
    ax2 = plt.subplot(3, 2, 2)
    errors_mgdl = to_mgdl(predictions.flatten()) - to_mgdl(actuals.flatten())
    sns.histplot(errors_mgdl, kde=True, ax=ax2)
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Error (mg/dL)')
    
    # 3. RMSE by Prediction Horizon
    ax3 = plt.subplot(3, 2, 3)
    horizons = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    rmse_values = [metrics[f'rmse_{h}min'] for h in horizons]
    ax3.plot(horizons, rmse_values, marker='o')
    ax3.set_title('RMSE by Prediction Horizon')
    ax3.set_xlabel('Prediction Horizon (minutes)')
    ax3.set_ylabel('RMSE (mg/dL)')
    ax3.grid(True)
    
    # 4. ROC Curves for Hypo/Hyper Detection
    ax4 = plt.subplot(3, 2, 4)
    plot_detection_roc_curves(ax4, predictions, actuals)
    
    # 5. Detailed Metrics Table
    ax5 = plt.subplot(3, 2, (5, 6))
    ax5.axis('off')
    metrics_text = format_detailed_metrics(metrics)
    ax5.text(0.1, 0.95, metrics_text, fontsize=10, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig('test_results_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_clarke_error_grid(ax, actuals, predictions):
    """Plot Clarke Error Grid Analysis"""
    # Define zones
    ax.plot([0, 400], [0, 400], 'k--')  # 45 degree line
    ax.fill_between([0, 400], [0, 400*1.2], [0, 400*0.8], alpha=0.1, color='green', label='Zone A')
    ax.fill_between([0, 175], [70, 175], [0, 0], alpha=0.1, color='orange', label='Zone B')
    ax.fill_between([175, 400], [175, 400], [70, 70], alpha=0.1, color='orange')
    
    ax.scatter(actuals, predictions, alpha=0.5, s=10)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_detection_roc_curves(ax, predictions, actuals):
    """Plot ROC curves for hypo and hyper detection"""
    GLUCOSE_MEAN, GLUCOSE_STD = 150, 50
    predictions_mgdl = predictions * GLUCOSE_STD + GLUCOSE_MEAN
    actuals_mgdl = actuals * GLUCOSE_STD + GLUCOSE_MEAN
    
    # Calculate ROC curves
    fpr_hypo, tpr_hypo, _ = roc_curve((actuals_mgdl < 70).flatten(), 
                                     (predictions_mgdl < 70).flatten())
    fpr_hyper, tpr_hyper, _ = roc_curve((actuals_mgdl > 180).flatten(),
                                       (predictions_mgdl > 180).flatten())
    
    # Plot curves
    ax.plot(fpr_hypo, tpr_hypo, label=f'Hypo (AUC = {auc(fpr_hypo, tpr_hypo):.2f})')
    ax.plot(fpr_hyper, tpr_hyper, label=f'Hyper (AUC = {auc(fpr_hyper, tpr_hyper):.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curves for Event Detection')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(True)
    ax.legend()

def format_detailed_metrics(metrics):
    """Format metrics for display"""
    return (
        f"Detailed Performance Metrics:\n\n"
        f"Core Metrics:\n"
        f"  RMSE: {metrics['rmse_mgdl']:.1f} mg/dL\n"
        f"  MAE: {metrics['mae_mgdl']:.1f} mg/dL\n"
        f"  In-Range Accuracy: {metrics['in_range_%']:.1f}%\n"
        f"  Tight Control Accuracy: {metrics['tight_control_%']:.1f}%\n\n"
        f"Trend Analysis:\n"
        f"  Direction Accuracy: {metrics['trend_accuracy_%']:.1f}%\n"
        f"  Trend MAE: {metrics['trend_mae']:.1f} mg/dL/5min\n\n"
        f"Hypoglycemia Detection:\n"
        f"  Severe (<54 mg/dL):\n"
        f"    Precision: {metrics['hypo_detection']['severe']['precision']:.1f}%\n"
        f"    Recall: {metrics['hypo_detection']['severe']['recall']:.1f}%\n"
        f"    F1 Score: {metrics['hypo_detection']['severe']['f1']:.1f}%\n"
        f"  Moderate (<70 mg/dL):\n"
        f"    Precision: {metrics['hypo_detection']['moderate']['precision']:.1f}%\n"
        f"    Recall: {metrics['hypo_detection']['moderate']['recall']:.1f}%\n"
        f"    F1 Score: {metrics['hypo_detection']['moderate']['f1']:.1f}%\n\n"
        f"Lead Time Analysis:\n"
        f"  Hypo Warning: {metrics['hypo_lead_time']:.1f} min\n"
        f"  Hyper Warning: {metrics['hyper_lead_time']:.1f} min"
    )

def evaluate_predictions(model, test_loader):
    """Evaluate model predictions with detailed metrics"""
    device = get_device()
    model.eval()
    
    total_samples = len(test_loader.dataset)
    logging.info(f"\nEvaluating model on {total_samples} test samples")
    logging.info(f"Predicting {FORECAST_HORIZON * 5} minutes into the future")
    
    all_predictions = []
    all_actuals = []
    all_inputs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device, dtype=torch.float32) for k, v in batch.items()}
            predictions = model(batch)
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(batch['target'].cpu().numpy())
            all_inputs.extend(batch['glucose'].cpu().numpy())
    
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    inputs = np.array(all_inputs)
    
    logging.info(f"Test set size: {len(predictions)} samples")
    
    # Print sample predictions
    print_sample_predictions(predictions, actuals, inputs)
    
    # Calculate comprehensive metrics
    metrics = calculate_glucose_metrics_mgdl(predictions, actuals)
    
    # Log detailed metrics
    logging.info("\nDetailed Test Set Metrics (in mg/dL):")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {metric}:")
            for submetric, subvalue in value.items():
                if isinstance(subvalue, dict):
                    logging.info(f"    {submetric}:")
                    for k, v in subvalue.items():
                        logging.info(f"      {k}: {v:.1f}")
                else:
                    logging.info(f"    {submetric}: {subvalue:.1f}")
        else:
            logging.info(f"  {metric}: {value:.1f}")
    
    # Create detailed visualization
    plot_prediction_analysis(predictions, actuals, inputs, metrics)
    
    return metrics

if __name__ == "__main__":
    # Setup logging
    log_file = setup_test_logging()
    logging.info("Starting model evaluation...")
    
    # Load data
    file_path = 'full_patient_dataset.csv'
    logging.info(f"Loading data from {file_path}")
    train_dataset, val_dataset, test_dataset = preprocess_data(file_path)
    
    # Log dataset sizes
    logging.info(f"Dataset sizes:")
    logging.info(f"  Training samples: {len(train_dataset)}")
    logging.info(f"  Validation samples: {len(val_dataset)}")
    logging.info(f"  Test samples: {len(test_dataset)}")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Load best model
    model_dir = "models/run_20250228_155402"
    try:
        model = load_best_model(model_dir)
        logging.info(f"Successfully loaded model from {model_dir}")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise
    
    # Evaluate model
    logging.info("Evaluating model on test set...")
    metrics = evaluate_predictions(model, test_loader)
    
    logging.info("\nTest Set Final Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {metric}:")
            for submetric, subvalue in value.items():
                if isinstance(subvalue, dict):
                    logging.info(f"    {submetric}:")
                    for k, v in subvalue.items():
                        logging.info(f"      {k}: {v:.1f}")
                else:
                    logging.info(f"    {submetric}: {subvalue:.1f}")
        else:
            logging.info(f"  {metric}: {value:.1f}")
    
    logging.info("Evaluation complete!") 