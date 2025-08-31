import os
import os.path as op
import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
from math import ceil
import scipy.interpolate

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    subjects = config['subjects']
    preprocessed_data_dir = config['paths']['preprocessed_data_dir']
    output_base_dir = config['paths']['statistics_dir']
    
    # Parameters from config
    freq1 = config['params']['freq1']
    freq2 = config['params']['freq2']
    time_windows = config['params']['time_intervals']
    # Stats params for cluster test
    n_permutations = config['group_stats']['n_permutations']
    threshold = config['group_stats']['threshold']
    significance_level = config['group_stats']['significance_level']
    
    freqs = np.arange(freq1, freq2 + 1, 1.0)
    n_cycles = freqs / 2.

    all_subjects_tfr_vs = []
    all_subjects_tfr_ld = []

    print("--- Calculating Sensor-Level Power for All Subjects ---")
    for subject in subjects:
        print(f"  - Processing subject: {subject}")
        try:
            vs_epochs_fname = op.join(preprocessed_data_dir, subject, f"{subject}_VS-epo.fif")
            ld_epochs_fname = op.join(preprocessed_data_dir, subject, f"{subject}_LD-epo.fif")
            
            epochs_VS = mne.read_epochs(vs_epochs_fname, preload=True)
            epochs_LD = mne.read_epochs(ld_epochs_fname, preload=True)

            # Use the new recommended .compute_tfr() method instead of the legacy tfr_morlet()
            tfr_vs = epochs_VS.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
            tfr_ld = epochs_LD.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
            
            all_subjects_tfr_vs.append(tfr_vs)
            all_subjects_tfr_ld.append(tfr_ld)

        except FileNotFoundError as e:
            print(f"    - Skipping subject {subject}, file not found: {e}")
            continue

    if not all_subjects_tfr_vs:
        print("\nNo data processed. Exiting.")
    else:
        # --- Create Grand Averages and Individual Differences ---
        grand_avg_vs = mne.grand_average(all_subjects_tfr_vs)
        grand_avg_ld = mne.grand_average(all_subjects_tfr_ld)
        grand_avg_diff = grand_avg_vs.copy()
        grand_avg_diff.data = grand_avg_vs.data - grand_avg_ld.data
        
        individual_diffs = [vs.copy() for vs in all_subjects_tfr_vs]
        for i in range(len(subjects)):
            individual_diffs[i].data = all_subjects_tfr_vs[i].data - all_subjects_tfr_ld[i].data

        output_dir = op.join(output_base_dir, "sensor_level_plots")
        if not op.exists(output_dir):
            os.makedirs(output_dir)

        # --- 1. PLOT INDIVIDUAL SUBJECTS TO CHECK FOR OUTLIERS ---
        print("\n--- Plotting Individual Subject Differences ---")
        n_subjects = len(subjects)
        n_cols = 4
        n_rows = ceil((n_subjects + 1) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        
        # Find a common color scale for all plots
        all_max = max([np.abs(d.data).max() for d in individual_diffs] + [np.abs(grand_avg_diff.data).max()])
        vmin, vmax = -all_max, all_max

        for i, (subj, diff) in enumerate(zip(subjects, individual_diffs)):
            ax = axes.flatten()[i]
            diff.plot_topomap(tmin=time_windows[0][0], tmax=time_windows[0][1], fmin=freq1, fmax=freq2, mode='logratio', axes=ax, show=False, vlim=[vmin, vmax])
            ax.set_title(subj)
        
        # Plot grand average at the end
        ax = axes.flatten()[n_subjects]
        grand_avg_diff.plot_topomap(tmin=time_windows[0][0], tmax=time_windows[0][1], fmin=freq1, fmax=freq2, mode='logratio', axes=ax, show=False, vlim=[vmin, vmax])
        ax.set_title("Grand Average")
        
        # Hide unused subplots
        for i in range(n_subjects + 1, len(axes.flatten())):
            axes.flatten()[i].axis('off')

        fig.tight_layout()
        fig_fname = op.join(output_dir, f"individual_power_diffs_{freq1}-{freq2}Hz.png")
        fig.savefig(fig_fname)
        plt.close(fig)
        print(f"Saved individual subject plot to {fig_fname}")

        # --- 2. RUN CLUSTER PERMUTATION TEST FOR SIGNIFICANCE ---
        print("\n--- Running Sensor-Level Cluster Permutation Test ---")
        
        # Get the list of channels that are common to all subjects from the grand_average
        common_channels = grand_avg_diff.ch_names
        
        # We will run a separate test for each time window
        for i, window in enumerate(time_windows):
            tmin, tmax = window
            print(f"\n  - Testing window: {tmin}s to {tmax}s")

            # Prepare data: average power within the window for each subject
            X_diff_power = []
            for diff in individual_diffs:
                # Create a copy and pick only the common channels
                diff_common = diff.copy().pick(common_channels)
                
                # Now crop to the window and average over time and frequency
                power_in_window = diff_common.crop(tmin=tmin, tmax=tmax).data.mean(axis=(1, 2))
                X_diff_power.append(power_in_window)
            
            # This will now work as all arrays have the same number of channels
            X = np.array(X_diff_power) # Shape: (n_subjects, n_common_channels)

            # Get channel adjacency for the common channels
            adjacency, _ = mne.channels.find_ch_adjacency(grand_avg_diff.info, ch_type='eeg')

            # Use a cluster test for 1D data (channels only)
            t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                X,
                adjacency=adjacency,
                n_permutations=n_permutations,
                threshold=threshold,
                n_jobs=-1
            )
            
            significant_clusters = np.where(cluster_p_values < significance_level)[0]
            print(f"    Found {len(significant_clusters)} significant clusters.")

            # --- NEW: Print details of significant clusters ---
            if significant_clusters.any():
                for cluster_idx in significant_clusters:
                    # Get the p-value for this specific cluster
                    p_val = cluster_p_values[cluster_idx]
                    
                    # Get the indices of the channels in this cluster
                    # For this 1D test, clusters[cluster_idx] is a tuple with one array
                    ch_indices = clusters[cluster_idx][0]
                    
                    # Get the actual channel names from the indices
                    cluster_ch_names = np.array(common_channels)[ch_indices]
                    
                    print(f"      - Cluster #{cluster_idx + 1}: p-value = {p_val:.4f}")
                    print(f"        - Number of channels: {len(cluster_ch_names)}")
                    print(f"        - Channels: {', '.join(cluster_ch_names)}")
            # --- END NEW ---

            # --- 3. PLOT GRAND AVERAGE WITH SIGNIFICANT CLUSTERS ---
            print("    Plotting grand average with significance mask...")
            
            # Create a mask from the significant clusters
            sig_ch_inds = np.array([], dtype=int)
            if significant_clusters.any():
                for cluster_idx in significant_clusters:
                    sig_ch_inds = np.union1d(sig_ch_inds, clusters[cluster_idx])
            
            sensor_mask = np.zeros(len(grand_avg_diff.ch_names), dtype=bool)
            if sig_ch_inds.any():
                sensor_mask[sig_ch_inds] = True

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            title = f'Power Difference (VS - LD)\n{tmin}-{tmax}s, {freq1}-{freq2}Hz'
            
            grand_avg_diff.plot_topomap(
                tmin=tmin, tmax=tmax, fmin=freq1, fmax=freq2, mode='logratio', 
                axes=ax, show=False, mask=sensor_mask, mask_params=dict(markersize=10, markerfacecolor='y')
            )
            ax.set_title(title)
            
            fig_fname = op.join(output_dir, f"sensor_power_diff_sig_win{i}_{freq1}-{freq2}Hz.png")
            fig.savefig(fig_fname)
            plt.close(fig)
            print(f"    Saved topomap with significance to {fig_fname}")

        print("\nSanity check complete.")