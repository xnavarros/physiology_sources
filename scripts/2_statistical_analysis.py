# statistical_analysis.py

import os
import os.path as op
import mne
import numpy as np
from mne.stats import spatio_temporal_cluster_test

# Define subject list
subjects = ['BJ25', 'JS08', 'LP26', 'MC05', 'MN23', 'OL04', 'SB27', 'TH24', 'VA14', 'VS06']

# Define base directory for raw data
base_data_dir = "/Users/xavier/work/data/Physiology"

# Define output directory for statistical results
output_base_dir = "statistical_results"

def run_statistics(stc_dict_VS, stc_dict_LD, fwd, time_windows, n_permutations=1000, threshold=4.5):
    adjacency_spatial = mne.spatial_src_adjacency(fwd['src'])
    cluster_results = {}
    
    for i in range(len(time_windows)):
        # Get mean activation for this window
        VS_data = np.array([np.mean(np.abs(stc.data), axis=1) for stc in stc_dict_VS[f'win_{i}']])
        LD_data = np.array([np.mean(np.abs(stc.data), axis=1) for stc in stc_dict_LD[f'win_{i}']])
        
        # Run cluster test
        X = [VS_data, LD_data]
        t_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
            X, adjacency=adjacency_spatial, n_jobs=None,
            n_permutations=n_permutations, threshold=threshold, out_type='mask'
        )
        
        # Create a mask for significant clusters
        significant_clusters = np.where(cluster_p_values < 0.01)[0]
        significant_mask = np.zeros_like(t_obs, dtype=bool)
        
        # Add significant clusters to the mask
        for cluster_idx in significant_clusters:
            significant_mask = np.logical_or(significant_mask, clusters[cluster_idx])
        
        # Store results including t-values and masks
        cluster_results[f'win_{i}'] = {
            'time_window': time_windows[i],
            'total_clusters': len(cluster_p_values),
            'significant_clusters': len(significant_clusters),
            'p_values': cluster_p_values.tolist(),
            't_obs': t_obs.tolist(),
            'significant_mask': significant_mask.tolist(),
            'has_significant': len(significant_clusters) > 0
        }
    
    return cluster_results

def save_statistics(cluster_results, output_dir):
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    
    for window_idx, results in cluster_results.items():
        np.save(op.join(output_dir, f"cluster_results_{window_idx}.npy"), results)

if __name__ == "__main__":
    # Define time windows
    time_windows = [[-1.25, -1], [-1, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0], [0, 0.25], [0.25, 0.5]]
    
    # Loop through all subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Define subject-specific paths
        source_estimates_dir = op.join("source_estimates", subject)
        fwd_fname = op.join(base_data_dir, subject, "derivatives", f"{subject}-fwd.fif")
        output_dir = op.join(output_base_dir, subject)
        
        # Load forward solution
        fwd = mne.read_forward_solution(fwd_fname)
        
        # Load source estimates for VS and LD conditions
        stc_dict_VS = {}
        stc_dict_LD = {}
        
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            stc_dict_VS[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir, f"stc_VS_{window_idx}_epoch{i}.stc")) for i in range(10)]
            stc_dict_LD[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir, f"stc_LD_{window_idx}_epoch{i}.stc")) for i in range(10)]
        
        # Run statistics
        cluster_results = run_statistics(stc_dict_VS, stc_dict_LD, fwd, time_windows)
        
        # Save results
        save_statistics(cluster_results, output_dir)
        
        print(f"Finished processing subject: {subject}")