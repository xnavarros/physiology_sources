# visualize_top_rois_grand_average_separate_figures.py

import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
output_base_dir = config['paths']['brain_plots_dir']
time_windows = config['params']['time_intervals']
source_estimates_dir_base = config['paths']['source_estimates_dir']
statistical_results_dir_base = config['paths']['statistics_dir']
suffix = str(config['params']['freq1']) + "_" + str(config['params']['freq2'])



# Define base directory for grand average ROI analysis results
grand_average_dir = op.join("roi_analysis", "grand_average")

# Define output directory for grand average figures
output_figures_dir = op.join("roi_figures", "grand_average")
if not op.exists(output_figures_dir):
    os.makedirs(output_figures_dir)


def load_roi_data(window_idx, condition):
    """Load grand average ROI data from CSV file."""
    csv_file = op.join(grand_average_dir, f"grand_average_{window_idx}_{condition}_{suffix}_roi.csv")
    if op.exists(csv_file):
        return pd.read_csv(csv_file)
    return None

def select_top_rois(roi_data, metric, n=5):
    """Select the top N ROIs based on a specific metric (mean or range activation)."""
    if roi_data is None or len(roi_data) == 0:
        return None
    
    # Sort ROIs by the specified metric (absolute value)
    sorted_indices = np.argsort(np.abs(roi_data[metric]))[::-1]
    top_rois = roi_data.iloc[sorted_indices[:n]]
    
    return top_rois

def plot_metric(top_rois, window_idx, condition, metric, output_dir):
    """Plot the top ROIs for a specific metric and save the figure as a PDF."""
    if top_rois is None or len(top_rois) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot for the specified metric
    ax.bar(top_rois['ROI'], top_rois[metric], color='b' if metric == 'Mean Activation' else 'g', alpha=0.6)
    
    # Add labels, title, and legend
    ax.set_xlabel('ROI')
    ax.set_ylabel(f'{metric} (μV/mm²)')
    ax.set_xticks(range(len(top_rois['ROI'])))
    ax.set_xticklabels(top_rois['ROI'], rotation=45, ha='right')
    
    # Add title
    plt.title(f"Top 5 ROIs - {condition} - Window {window_idx} - {metric} - Freq {suffix}")
    fig.tight_layout()
    
    # Save the figure as a PDF
    pdf_file = op.join(output_dir, f"top_rois_{condition}_{window_idx}_{suffix}_{metric.lower().replace(' ', '_')}.pdf")
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Loop through all time windows
    for i, window in enumerate(time_windows):
        window_idx = f'win_{i}'
        
        # Process VS condition
        roi_data_VS = load_roi_data(window_idx, 'VS')
        if roi_data_VS is not None:
            # Top 5 ROIs based on Mean Activation
            top_rois_mean_VS = select_top_rois(roi_data_VS, 'Mean Activation', n=5)
            plot_metric(top_rois_mean_VS, window_idx, 'VS', 'Mean Activation', output_figures_dir)
            
            # Top 5 ROIs based on Range Activation
            top_rois_range_VS = select_top_rois(roi_data_VS, 'Range Activation', n=5)
            plot_metric(top_rois_range_VS, window_idx, 'VS', 'Range Activation', output_figures_dir)
        
        # Process LD condition
        roi_data_LD = load_roi_data(window_idx, 'LD')
        if roi_data_LD is not None:
            # Top 5 ROIs based on Mean Activation
            top_rois_mean_LD = select_top_rois(roi_data_LD, 'Mean Activation', n=5)
            plot_metric(top_rois_mean_LD, window_idx, 'LD', 'Mean Activation', output_figures_dir)
            
            # Top 5 ROIs based on Range Activation
            top_rois_range_LD = select_top_rois(roi_data_LD, 'Range Activation', n=5)
            plot_metric(top_rois_range_LD, window_idx, 'LD', 'Range Activation', output_figures_dir)
        
        # Process VS-LD condition
        roi_data_diff = load_roi_data(window_idx, 'VS-LD')
        if roi_data_diff is not None:
            # Top 5 ROIs based on Mean Activation
            top_rois_mean_diff = select_top_rois(roi_data_diff, 'Mean Activation', n=5)
            plot_metric(top_rois_mean_diff, window_idx, 'VS-LD', 'Mean Activation', output_figures_dir)
            
            # Top 5 ROIs based on Range Activation
            top_rois_range_diff = select_top_rois(roi_data_diff, 'Range Activation', n=5)
            plot_metric(top_rois_range_diff, window_idx, 'VS-LD', 'Range Activation', output_figures_dir)
    
    print("Grand average ROI visualization completed.")