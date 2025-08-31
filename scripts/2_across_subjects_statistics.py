# statistical_analysis.py

import os
import os.path as op
import mne
import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
output_base_dir = config['paths']['statistics_dir']
time_windows = config['params']['time_intervals']
n_permutations = config['group_stats']['n_permutations']
threshold = config['group_stats']['threshold']
significance_level = config['group_stats']['significance_level']
source_estimates_dir_base = config['paths']['source_estimates_dir']
suffix = f"{config['params']['freq1']}_{config['params']['freq2']}"

def run_group_statistics(all_subjects_data_VS, all_subjects_data_LD, adjacency_spatial, n_permutations, threshold, significance_level):
    """
    Runs a paired spatio-temporal cluster test on group-level data.
    """
    # The data should be in the format: (n_subjects, n_vertices)
    # We perform a paired test by running a 1-sample test on the difference.
    X_VS_power = np.array(all_subjects_data_VS)
    X_LD_power = np.array(all_subjects_data_LD)
    X_diff = X_VS_power - X_LD_power

    # Reshape for the test: (n_observations, n_times, n_vertices)
    # Since we averaged over the STC's time dimension, our "n_times" is 1.
    X_diff_reshaped = X_diff[:, np.newaxis, :]

    print("Running group-level cluster permutation test...")
    # Use tail=1 for a one-tailed test (VS > LD)
    t_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
        X_diff_reshaped,
        adjacency=adjacency_spatial,
        n_jobs=-1,
        n_permutations=n_permutations,
        threshold=threshold,
        out_type='mask',
        tail=1  # Test for positive effects (VS > LD)
    )
    
    significant_clusters = np.where(cluster_p_values < significance_level)[0]
    print(f"Found {len(significant_clusters)} significant clusters.")

    return {
        't_obs': t_obs.squeeze(),
        'clusters': clusters,
        'p_values': cluster_p_values,
        'significant_clusters': significant_clusters
    }

def save_statistics(results, output_dir, window_idx, suffix):
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    np.save(op.join(output_dir, f"group_cluster_results_{window_idx}_{suffix}.npy"), results)

if __name__ == "__main__":
    # We need one forward solution to get the adjacency matrix.
    # It's the same for all subjects, so we use the first one.
    fwd_fname = op.join(base_data_dir, subjects[0], f"{subjects[0]}-fwd.fif")
    fwd = mne.read_forward_solution(fwd_fname)
    adjacency_spatial = mne.spatial_src_adjacency(fwd['src'])

    # Loop through each time window to perform a separate group analysis
    for i, window in enumerate(time_windows):
        window_idx = f'win_{i}'
        print(f"\nProcessing group statistics for time window: {window_idx} ({window[0]}-{window[1]}s)")

        all_subjects_VS = []
        all_subjects_LD = []

        # Loop through all subjects to collect data for this time window
        for subject in subjects:
            try:
                stc_vs_fname = op.join(source_estimates_dir_base, subject, f"stc_avg_VS_{window_idx}_{suffix}.h5")
                stc_ld_fname = op.join(source_estimates_dir_base, subject, f"stc_avg_LD_{window_idx}_{suffix}.h5")
                
                stc_vs = mne.read_source_estimate(stc_vs_fname)
                stc_ld = mne.read_source_estimate(stc_ld_fname)
                
                # --- CORRECTED METHOD ---
                # 1. Square the data to get power.
                # 2. Take the mean power across the STC's time dimension.
                power_vs = np.mean(stc_vs.data ** 2, axis=1)
                power_ld = np.mean(stc_ld.data ** 2, axis=1)
                
                all_subjects_VS.append(power_vs)
                all_subjects_LD.append(power_ld)
            except OSError as e:
                print(f"  - Could not load data for subject {subject}. Skipping. Error: {e}")
                continue
        
        if len(all_subjects_VS) < 2:
            print(f"  - Not enough subjects ({len(all_subjects_VS)}) with data for this window. Skipping analysis.")
            continue

        # Run group-level statistics for this window
        cluster_results = run_group_statistics(
            all_subjects_VS, all_subjects_LD, adjacency_spatial,
            n_permutations, threshold, significance_level
        )
        
        # Save the single group-level result
        output_dir = op.join(output_base_dir, "group_results")
        save_statistics(cluster_results, output_dir, window_idx, suffix)
        
    print("\nFinished all statistical analyses.")