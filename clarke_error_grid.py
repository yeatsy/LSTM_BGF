import numpy as np
import matplotlib.pyplot as plt
import os

def clarke_error_grid_analysis(predictions, actuals, plot=True, save_path='clarke_error_grid.png', 
                               title='Clarke Error Grid Analysis'):
    """
    Implement Clarke Error Grid Analysis for glucose predictions.
    
    The Clarke Error Grid divides the prediction-actual space into 5 zones:
    - Zone A: Clinically accurate, would lead to correct treatment
    - Zone B: Benign errors, would lead to benign or no treatment
    - Zone C: Overcorrection errors
    - Zone D: Dangerous failure to detect hypo/hyperglycemia
    - Zone E: Erroneous treatment, predicting opposite of actual condition
    
    Args:s
        predictions: Normalized glucose predictions
        actuals: Normalized actual glucose values
        plot: Whether to generate a plot
        save_path: Path to save the plot
        title: Title for the plot
        
    Returns:
        dict: Percentage of points in each zone
    """
    # Convert to mg/dL
    GLUCOSE_MEAN = 150
    GLUCOSE_STD = 50
    
    predictions_mgdl = predictions.flatten() * GLUCOSE_STD + GLUCOSE_MEAN
    actuals_mgdl = actuals.flatten() * GLUCOSE_STD + GLUCOSE_MEAN
    
    # Initialize counts for each zone
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total_points = len(predictions_mgdl)
    
    # Classify each point into a zone
    for pred, actual in zip(predictions_mgdl, actuals_mgdl):
        # Prevent division by zero
        if actual == 0:
            actual = 0.1
            
        # Zone A: Clinically accurate
        if (actual >= 70 and actual <= 180 and 
            (pred >= 0.7 * actual and pred <= 1.3 * actual)) or \
           (actual < 70 and pred < 70) or \
           (actual > 180 and pred > 180 and pred < 1.5 * actual):
            zones['A'] += 1
            
        # Zone B: Benign errors
        elif ((pred >= 70 and pred <= 180) and actual >= 0) or \
             (actual >= 70 and actual <= 180 and pred >= max(0, 1.5 * actual)) or \
             (actual >= 70 and actual <= 180 and pred <= 0.5 * actual and pred >= 0) or \
             (actual > 180 and pred <= 180 and pred >= 70) or \
             (actual < 70 and pred >= 70 and pred <= 180):
            zones['B'] += 1
            
        # Zone C: Overcorrection errors
        elif (actual >= 180 and pred < 70) or \
             (actual <= 70 and pred > 180):
            zones['C'] += 1
            
        # Zone D: Dangerous failure to detect
        elif (actual >= 70 and pred < 70) or \
             (actual <= 180 and pred > 180):
            zones['D'] += 1
            
        # Zone E: Erroneous treatment
        elif (actual <= 70 and pred >= actual) or \
             (actual >= 180 and pred <= actual):
            zones['E'] += 1
    
    # Calculate percentages
    zone_percentages = {zone: (count / total_points) * 100 for zone, count in zones.items()}
    
    # Generate plot if requested
    if plot:
        plt.figure(figsize=(10, 10))
        
        # Define the zones for plotting
        # Zone A
        plt.plot([0, 70], [0, 70], 'k-')  # lower bound
        plt.plot([0, 70], [0, 84], 'k-')  # upper bound
        plt.plot([70, 180], [56, 144], 'k-')  # lower bound
        plt.plot([70, 180], [84, 216], 'k-')  # upper bound
        plt.plot([180, 400], [144, 320], 'k-')  # lower bound
        plt.plot([180, 400], [216, 400], 'k-')  # upper bound
        
        # Zone B upper/lower bounds
        plt.plot([0, 70], [84, 180], 'k-')
        plt.plot([70, 290], [180, 400], 'k-')
        plt.plot([0, 70], [0, 56], 'k-')
        plt.plot([70, 180], [0, 56], 'k-')
        plt.plot([180, 400], [0, 144], 'k-')
        
        # Zone C, D, E boundaries
        plt.plot([70, 70], [0, 56], 'k-')
        plt.plot([70, 70], [84, 400], 'k-')
        plt.plot([180, 180], [0, 144], 'k-')
        plt.plot([180, 180], [216, 400], 'k-')
        
        # Add zone labels
        plt.text(30, 30, "A", fontsize=18)
        plt.text(30, 130, "B", fontsize=18)
        plt.text(30, 250, "C", fontsize=18)
        plt.text(130, 30, "D", fontsize=18)
        plt.text(250, 30, "E", fontsize=18)
        
        # Plot data points
        plt.scatter(actuals_mgdl, predictions_mgdl, c='b', alpha=0.4, s=10)
        
        # Add diagonal line (perfect prediction)
        plt.plot([0, 400], [0, 400], 'k--')
        
        # Set axis limits and labels
        plt.xlim(0, 400)
        plt.ylim(0, 400)
        plt.xlabel('Actual Glucose (mg/dL)')
        plt.ylabel('Predicted Glucose (mg/dL)')
        
        # Add zone percentage annotations
        plt.annotate(
            f"Zone Percentages:\n" + 
            f"A: {zone_percentages['A']:.1f}%\n" +
            f"B: {zone_percentages['B']:.1f}%\n" +
            f"C: {zone_percentages['C']:.1f}%\n" +
            f"D: {zone_percentages['D']:.1f}%\n" +
            f"E: {zone_percentages['E']:.1f}%",
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            va='top'
        )
        
        # Add risk regions (70-180 mg/dL is target range)
        plt.axhspan(0, 70, color='red', alpha=0.1, label='Hypoglycemia')
        plt.axhspan(180, 400, color='orange', alpha=0.1, label='Hyperglycemia')
        plt.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')
        
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    # Add combined A+B zone (clinically acceptable)
    zone_percentages['A+B'] = zone_percentages['A'] + zone_percentages['B']
    
    return zone_percentages

def analyze_all_timesteps(predictions, actuals, output_dir='clarke_grids'):
    """
    Generate Clarke Error Grid analysis for each prediction timestep.
    
    Args:
        predictions: Model predictions with shape [samples, timesteps]
        actuals: Actual values with shape [samples, timesteps]
        output_dir: Directory to save the grid plots
        
    Returns:
        list: List of dictionaries with zone percentages for each timestep
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    num_timesteps = predictions.shape[1]
    all_results = []
    
    for t in range(num_timesteps):
        # Extract predictions and actuals for this timestep
        preds_t = predictions[:, t:t+1]  # Keep 2D shape
        actuals_t = actuals[:, t:t+1]
        
        # Minutes into future
        minutes = (t + 1) * 5
        
        # Generate Clarke grid for this timestep
        save_path = os.path.join(output_dir, f'clarke_grid_{minutes}min.png')
        title = f'Clarke Error Grid: {minutes}-Minute Predictions'
        
        # Run analysis
        results = clarke_error_grid_analysis(
            preds_t, 
            actuals_t, 
            save_path=save_path,
            title=title
        )
        
        # Add timestep information
        results['timestep'] = t
        results['minutes'] = minutes
        
        all_results.append(results)
    
    # Create summary plot
    create_timestep_summary(all_results, output_dir)
    
    return all_results

def create_timestep_summary(results, output_dir):
    """Create a summary plot showing zone percentages across all timesteps"""
    minutes = [r['minutes'] for r in results]
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Plot each zone
    plt.plot(minutes, [r['A'] for r in results], 'o-', label='Zone A', linewidth=2)
    plt.plot(minutes, [r['B'] for r in results], 's-', label='Zone B', linewidth=2)
    plt.plot(minutes, [r['C'] for r in results], '^-', label='Zone C', linewidth=2)
    plt.plot(minutes, [r['D'] for r in results], 'd-', label='Zone D', linewidth=2)
    plt.plot(minutes, [r['E'] for r in results], 'x-', label='Zone E', linewidth=2)
    plt.plot(minutes, [r['A+B'] for r in results], '*-', label='Zone A+B (Clinical Acceptability)', linewidth=3)
    
    # Add clinical thresholds
    plt.axhline(y=95, color='g', linestyle='--', alpha=0.5, label='A+B Clinical Acceptability Target (95%)')
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='D+E Clinical Risk Threshold (5%)')
    
    # Customize plot
    plt.title('Clarke Error Grid Performance Across Prediction Horizons')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('Percentage of Predictions (%)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the summary
    plt.savefig(os.path.join(output_dir, 'clarke_grid_summary.png'))
    plt.close()

if __name__ == "__main__":
    # Example usage when run directly
    print("This module provides Clarke Error Grid Analysis functions.")
    print("Import and use in your evaluation pipeline.")
