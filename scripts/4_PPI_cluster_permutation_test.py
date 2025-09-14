import os
import os.path as op
import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # --- 1. CONFIGURATION AND SETUP ---
    print("--- Setting up Spatio-Temporal Cluster Test ---")
    config = load_config()
    subjects = config['subjects']
    params = config['preprocessing_params']
    
    input_dir = config['paths']['preprocessed_slow_potentials_dir']
    # Create a dedicated output directory for these cluster results
    output_dir = op.join(config['paths']['results_dir'], 'cluster_permutation_test')
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. LOAD DATA ---
    all_evoked_vs = []
    all_evoked_ld = []
    print("\n--- Loading data for all subjects ---")
    for subject in subjects:
        try:
            vs_epochs_fname = op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif")
            ld_epochs_fname = op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif")
            
            epochs_vs = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
            epochs_ld = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)

            # Apply baseline correction from config
            baseline_window = tuple(params['baseline_timing_slow'])
            epochs_vs.apply_baseline(baseline=baseline_window)
            epochs_ld.apply_baseline(baseline=baseline_window)

            all_evoked_vs.append(epochs_vs.average())
            all_evoked_ld.append(epochs_ld.average())
            print(f"  - Loaded data for subject: {subject}")
        except FileNotFoundError:
            print(f"    - WARNING: Files not found for subject {subject}. Skipping.")
            continue
    
    if not all_evoked_vs:
        print("\nFATAL ERROR: No data found. Exiting analysis.")
        exit()

    # --- 3. PREPARE DATA FOR SPATIO-TEMPORAL CLUSTER TEST ---
    print("\n--- Preparing data for cluster analysis ---")
    # Define all channels of interest for the analysis
    channels_of_interest = [
        'F1', 'Fz', 'F2', 'F3', 'F4',
        'FC1', 'FC2', 'FC3', 'FC4',  # FCz removed
        'C1', 'Cz', 'C2', 'C3', 'C4',
        'CP1', 'CPz', 'CP2', 'CP3', 'CP4',
        'P1', 'Pz', 'P2', 'P3', 'P4'
    ]
    
    # Ensure all requested channels actually exist in the data
    info = all_evoked_vs[0].info
    channels_to_use = [ch for ch in channels_of_interest if ch in info['ch_names']]
    print(f"  - Using {len(channels_to_use)} channels for the test.")

    # Prepare the data array: (n_subjects, n_times, n_channels)
    X_vs = np.array([evk.copy().pick(channels_to_use).get_data() for evk in all_evoked_vs])
    X_ld = np.array([evk.copy().pick(channels_to_use).get_data() for evk in all_evoked_ld])
    
    # Calculate the difference and transpose for the stats function
    # The function expects (n_observations, n_times, n_vertices/channels)
    X_diff = (X_vs - X_ld)
    
    # Get channel adjacency information for all channels
    full_adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    
    # --- FIX: Subset the adjacency matrix to match the channels being tested ---
    # Find the indices of the channels we are using in the full list of channels
    ch_indices = [ch_names.index(ch) for ch in channels_to_use]
    # Create the subset of the adjacency matrix
    adjacency = full_adjacency[np.ix_(ch_indices, ch_indices)]
    # --- END FIX ---

    # --- 4. RUN THE CLUSTER-BASED PERMUTATION TEST ---
    print("\n--- Running Spatio-Temporal Cluster Permutation Test (this may take a while) ---")
    t_obs, clusters, cluster_p_values, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        X_diff,
        adjacency=adjacency,
        n_permutations=1024, # 1024 is a good number for exploration, consider >5000 for publication
        threshold=None,
        n_jobs=-1
    )

    # --- 5. VISUALIZE SIGNIFICANT CLUSTERS ---
    print("\n--- Processing and visualizing results ---")
    significant_clusters = np.where(cluster_p_values < 0.05)[0]
    print(f"  - Found {len(significant_clusters)} significant clusters.")

    if not len(significant_clusters):
        print("  - No significant clusters found.")
    
    grand_avg_diff = mne.grand_average(all_evoked_vs) - mne.grand_average(all_evoked_ld)
    times = grand_avg_diff.times

    # Loop through each significant cluster
    for i_clu, clu_idx in enumerate(significant_clusters):
        # Unpack cluster information
        time_inds, ch_inds = clusters[clu_idx]
        
        # Get cluster timing and channels
        ch_names_in_cluster = [info['ch_names'][i] for i in ch_inds]
        time_interval = (times[time_inds.min()], times[time_inds.max()])
        
        print(f"\n--- Visualizing Cluster #{i_clu + 1} (p-value: {cluster_p_values[clu_idx]:.3f}) ---")
        print(f"  - Time window: {time_interval[0]:.3f}s to {time_interval[1]:.3f}s")
        print(f"  - Channels involved: {len(ch_names_in_cluster)}")

        # Plot 1: Topography of the cluster effect
        # Calculate the average effect within the cluster's time window
        mean_effect_in_window = grand_avg_diff.copy().pick(info['ch_names']).data[:, time_inds].mean(axis=1)
        
        fig, ax_topo = plt.subplots(1, 1, figsize=(7, 6))
        mne.viz.plot_topomap(mean_effect_in_window, info, axes=ax_topo, show=False,
                             mask=np.array([ch in ch_names_in_cluster for ch in info['ch_names']]),
                             mask_params=dict(markersize=6))
        ax_topo.set_title(f"Topography of Cluster #{i_clu + 1} (p={cluster_p_values[clu_idx]:.3f})")
        fig_fname = op.join(output_dir, f"cluster_{i_clu+1}_topography.png")
        fig.savefig(fig_fname)
        plt.close(fig)
        print(f"  - Saved topography plot: {fig_fname}")

        # Plot 2: Waveform of the cluster
        fig, ax_wave = plt.subplots(figsize=(10, 6))
        # Average the difference wave across the channels in the cluster
        cluster_waveform = grand_avg_diff.copy().pick(ch_names_in_cluster).data.mean(axis=0)
        ax_wave.plot(times, cluster_waveform, label=f"Mean of {len(ch_names_in_cluster)} channels in cluster")
        ax_wave.axvspan(time_interval[0], time_interval[1], color='gray', alpha=0.3, label="Significant Time Window")
        ax_wave.axhline(0, color='k', linestyle='--', lw=1)
        ax_wave.axvline(0, color='r', linestyle='-', lw=1.5)
        ax_wave.legend()
        ax_wave.set_title(f"Waveform of Cluster #{i_clu + 1} (p={cluster_p_values[clu_idx]:.3f})")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude (µV)")
        fig_fname = op.join(output_dir, f"cluster_{i_clu+1}_waveform.png")
        fig.savefig(fig_fname)
        plt.close(fig)
        print(f"  - Saved waveform plot: {fig_fname}")

    print("\n--- Analysis Complete ---")