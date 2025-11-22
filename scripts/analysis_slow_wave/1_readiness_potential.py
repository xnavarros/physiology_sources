import os
import os.path as op
import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mne import Report

def load_config(config_path=None):
    """Loads the configuration file."""
    if config_path is None:
        # Construct path relative to this script's location
        script_dir = op.dirname(op.abspath(__file__))
        # Go up two levels to get to the project root
        root_dir = op.dirname(op.dirname(script_dir))
        config_path = op.join(root_dir, 'config', 'config.yaml')
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # --- 1. CONFIGURATION AND SETUP ---
    config = load_config()
    subjects = config['subjects']
    params = config['preprocessing_params']

    # --- FIX: Make all paths absolute from the project root ---
    script_dir = op.dirname(op.abspath(__file__))
    root_dir = op.dirname(op.dirname(script_dir))
    
    # Update paths in the config to be absolute
    for key, path_val in config['paths'].items():
        config['paths'][key] = op.join(root_dir, path_val)
    
    input_dir = config['paths']['preprocessed_slow_potentials_dir']
    output_dir = op.join(config['paths']['results_dir'], 'erp_analysis')
    erp_output_dir = op.join(root_dir, 'figures', 'subject_erp_plots')
    # --- END FIX ---

    if not op.exists(output_dir):
        os.makedirs(output_dir)
    if not op.exists(erp_output_dir):
        os.makedirs(erp_output_dir)

    # --- NEW: Initialize HTML Report ---
    report = Report(title='Readiness Potential Analysis Report: VS vs. LD', verbose=False)

    # Analysis parameters
    channel_of_interest = 'Cz' # For stats and grand average plot
    channels_to_plot = ['Fz', 'Cz', 'F3', 'F4', 'C3', 'C4'] # For subject-level plots
    topo_window = (-0.5, 0.0)
    # --- NEW: Define analysis windows from config ---
    time_windows = config.get('analysis_windows', {
        "early": (-1.5, -1.0),
        "mid": (-1.0, -0.5),
        "late": (-0.5, 0.0),
        "post": (0.0, 0.5)
    })
    # Ensure keys are lowercase for consistency if needed, or just use as is. 
    # The config has capitalized keys (Early, Mid...), let's normalize if necessary or just use them.
    # The script uses them for labels.


    all_evoked_vs = []
    all_evoked_ld = []

    print("--- Starting Readiness Potential (BP) Analysis ---")

    # --- 2. LOAD DATA AND CREATE INDIVIDUAL AVERAGES & PLOTS ---
    for subject in subjects:
        print(f"  - Processing subject: {subject}")
        try:
            vs_epochs_fname = op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif")
            ld_epochs_fname = op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif")
            
            epochs_vs = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
            epochs_ld = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)

            # --- MODIFIED: Apply baseline correction from config file ---
            baseline_window = tuple(params['baseline_timing_slow'])
            print(f"      Applying baseline correction from config: {baseline_window}s")
            epochs_vs.apply_baseline(baseline=baseline_window)
            epochs_ld.apply_baseline(baseline=baseline_window)
            # --- END MODIFIED ---

            # Get epoch counts for plotting
            n_epochs_vs = len(epochs_vs)
            n_epochs_ld = len(epochs_ld)
            print(f"      Found {n_epochs_vs} clean VS epochs and {n_epochs_ld} clean LD epochs.")

            # Create evoked objects from the clean epochs
            evoked_vs = epochs_vs.average()
            evoked_ld = epochs_ld.average()
            
            all_evoked_vs.append(evoked_vs)
            all_evoked_ld.append(evoked_ld)

            # --- Plot 1: Fz, Cz, Pz ERP comparison (using cleaned data) ---
            evokeds_dict_subj = {'VS': evoked_vs, 'LD': evoked_ld}
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            fig.suptitle(f"Readiness Potential (Average) for {subject}", fontsize=16)
            epoch_count_text = f"Epochs: VS={n_epochs_vs}, LD={n_epochs_ld}"
            mne.viz.plot_compare_evokeds(evokeds_dict_subj, picks='Fz', axes=axes[0], linestyles={'VS': '-', 'LD': '--'}, styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}}, show=False, invert_y=True, title="Channel Fz")
            axes[0].axvline(0, color='r', linestyle='-', lw=1.5); axes[0].text(0.02, 0.95, epoch_count_text, transform=axes[0].transAxes, fontsize=9, verticalalignment='top')
            mne.viz.plot_compare_evokeds(evokeds_dict_subj, picks='Cz', axes=axes[1], linestyles={'VS': '-', 'LD': '--'}, styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}}, show=False, invert_y=True, title="Channel Cz")
            axes[1].axvline(0, color='r', linestyle='-', lw=1.5); axes[1].text(0.02, 0.95, epoch_count_text, transform=axes[1].transAxes, fontsize=9, verticalalignment='top')
            mne.viz.plot_compare_evokeds(evokeds_dict_subj, picks='Pz', axes=axes[2], linestyles={'VS': '-', 'LD': '--'}, styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}}, show=False, invert_y=True, title="Channel Pz")
            axes[2].axvline(0, color='r', linestyle='-', lw=1.5); axes[2].text(0.02, 0.95, epoch_count_text, transform=axes[2].transAxes, fontsize=9, verticalalignment='top')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            subj_fig_fname = op.join(erp_output_dir, f"{subject}_Fz_Cz_Pz_waveforms.png")
            fig.savefig(subj_fig_fname)
            report.add_figure(fig=fig, title='ERP Waveforms (Fz, Cz, Pz)', section=f'Subject: {subject}', tags=('erp', 'waveform'))
            plt.close(fig)

            # --- MODIFIED: Create a single, combined butterfly plot for EEG and Respiratory signals ---
            
            # Define all channels to plot in order
            channels_to_plot = ['Debit', 'Pression', 'Fz', 'Cz', 'Pz']
            eeg_channels = ['Fz', 'Cz', 'Pz']

            fig_combo, axes_combo = plt.subplots(len(channels_to_plot), 2, figsize=(14, 18), sharex=True, sharey='row')
            fig_combo.suptitle(f"Combined EEG and Respiration Butterfly Plot for {subject}", fontsize=16)
            times = epochs_vs.times

            for i, chan in enumerate(channels_to_plot):
                is_eeg = chan in eeg_channels

                # --- VS condition on the left column ---
                ax_vs = axes_combo[i, 0]
                data_vs = epochs_vs.copy().pick(chan).get_data(copy=False).squeeze()
                if is_eeg:
                    data_vs *= 1e6 # Convert to µV
                
                ax_vs.plot(times, data_vs.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_vs.plot(times, data_vs.mean(axis=0), color='C0', linewidth=2)
                ax_vs.set_title(f"{chan} - VS ({n_epochs_vs} trials)")
                ax_vs.axvline(0, color='k', linestyle='--')
                
                # --- LD condition on the right column ---
                ax_ld = axes_combo[i, 1]
                data_ld = epochs_ld.copy().pick(chan).get_data(copy=False).squeeze()
                if is_eeg:
                    data_ld *= 1e6 # Convert to µV

                ax_ld.plot(times, data_ld.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_ld.plot(times, data_ld.mean(axis=0), color='C1', linewidth=2)
                ax_ld.set_title(f"{chan} - LD ({n_epochs_ld} trials)")
                ax_ld.axvline(0, color='k', linestyle='--')

                # --- Set Y-axis properties based on channel type ---
                if is_eeg:
                    ax_vs.invert_yaxis()
                    ax_ld.invert_yaxis()
                    ax_vs.set_ylim([40, -40])
                    ax_vs.set_ylabel("Amplitude (µV)")
                else:
                    ax_vs.set_ylabel("Amplitude (a.u.)")

            # Set common X-label
            axes_combo[-1, 0].set_xlabel("Time (s)")
            axes_combo[-1, 1].set_xlabel("Time (s)")

            fig_combo.tight_layout(rect=[0, 0.03, 1, 0.95])
            subj_combo_fname = op.join(erp_output_dir, f"{subject}_combined_butterfly_plot.png")
            fig_combo.savefig(subj_combo_fname)
            report.add_figure(fig=fig_combo, title='Combined Butterfly Plot', section=f'Subject: {subject}', tags=('butterfly', 'respiration'))
            plt.close(fig_combo)
            # --- END of combined plotting section ---

            # --- NEW: Joint Plot for each subject's difference wave ---
            diff_evoked_subj = mne.combine_evoked([evoked_vs, evoked_ld], weights=[1, -1])
            fig = diff_evoked_subj.plot_joint(
                title=f"Difference Wave (VS - LD) for {subject}",
                times=[-0.8, -0.5, -0.2, 0.0], # Use same time points as grand average
                show=False
            )
            subj_joint_fname = op.join(erp_output_dir, f"{subject}_joint_plot_diff.png")
            fig.savefig(subj_joint_fname)
            report.add_figure(fig=fig, title='Joint Plot (Difference)', section=f'Subject: {subject}', tags=('joint', 'difference'))
            plt.close(fig)
            # --- END NEW ---

            # --- MODIFIED: Plot 'Debit' and 'Pression' signals for artifact checking ---
            try:
                # --- Plot 1: Debit Signal ---
                fig_debit, axes_debit = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
                fig_debit.suptitle(f"Debit Signal for {subject}", fontsize=16)

                # VS condition (left)
                ax_vs_d = axes_debit[0]
                data_vs_d = epochs_vs.copy().pick('Debit').get_data(copy=False).squeeze()
                ax_vs_d.plot(times, data_vs_d.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_vs_d.plot(times, data_vs_d.mean(axis=0), color='C0', linewidth=2)
                ax_vs_d.set_title(f"VS Condition ({n_epochs_vs} trials)")
                ax_vs_d.axvline(0, color='k', linestyle='--')
                ax_vs_d.set_xlabel("Time (s)")
                ax_vs_d.set_ylabel("Amplitude (a.u.)")

                # LD condition (right)
                ax_ld_d = axes_debit[1]
                data_ld_d = epochs_ld.copy().pick('Debit').get_data(copy=False).squeeze()
                ax_ld_d.plot(times, data_ld_d.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_ld_d.plot(times, data_ld_d.mean(axis=0), color='C1', linewidth=2)
                ax_ld_d.set_title(f"LD Condition ({n_epochs_ld} trials)")
                ax_ld_d.axvline(0, color='k', linestyle='--')
                ax_ld_d.set_xlabel("Time (s)")

                fig_debit.tight_layout(rect=[0, 0.03, 1, 0.95])
                subj_debit_fname = op.join(erp_output_dir, f"{subject}_debit_plot.png")
                fig_debit.savefig(subj_debit_fname)
                report.add_figure(fig=fig_debit, title='Debit Signal', section=f'Subject: {subject}', tags=('respiration', 'qc'))
                plt.close(fig_debit)

                # --- Plot 2: Pression Signal ---
                fig_press, axes_press = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
                fig_press.suptitle(f"Pression Signal for {subject}", fontsize=16)

                # VS condition (left)
                ax_vs_p = axes_press[0]
                data_vs_p = epochs_vs.copy().pick('Pression').get_data(copy=False).squeeze()
                ax_vs_p.plot(times, data_vs_p.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_vs_p.plot(times, data_vs_p.mean(axis=0), color='C0', linewidth=2)
                ax_vs_p.set_title(f"VS Condition ({n_epochs_vs} trials)")
                ax_vs_p.axvline(0, color='k', linestyle='--')
                ax_vs_p.set_xlabel("Time (s)")
                ax_vs_p.set_ylabel("Amplitude (a.u.)")

                # LD condition (right)
                ax_ld_p = axes_press[1]
                data_ld_p = epochs_ld.copy().pick('Pression').get_data(copy=False).squeeze()
                ax_ld_p.plot(times, data_ld_p.T, color='gray', alpha=0.3, linewidth=0.5)
                ax_ld_p.plot(times, data_ld_p.mean(axis=0), color='C1', linewidth=2)
                ax_ld_p.set_title(f"LD Condition ({n_epochs_ld} trials)")
                ax_ld_p.axvline(0, color='k', linestyle='--')
                ax_ld_p.set_xlabel("Time (s)")

                fig_press.tight_layout(rect=[0, 0.03, 1, 0.95])
                subj_press_fname = op.join(erp_output_dir, f"{subject}_pression_plot.png")
                fig_press.savefig(subj_press_fname)
                report.add_figure(fig=fig_press, title='Pression Signal', section=f'Subject: {subject}', tags=('respiration', 'qc'))
                plt.close(fig_press)

            except Exception as e:
                print(f"    - WARNING: Could not plot Debit/Pression for subject {subject}. Error: {e}")

        except FileNotFoundError:
            print(f"    - WARNING: Cleaned slow-potential files not found for subject {subject}. Skipping.")
            continue
    
    if not all_evoked_vs:
        print("\nFATAL ERROR: No data found. Exiting analysis.")
        exit()

    # --- 3. COMPUTE AND PLOT GRAND AVERAGES ---
    print("\n--- Computing and Plotting Grand Averages ---")
    grand_avg_vs = mne.grand_average(all_evoked_vs)
    grand_avg_ld = mne.grand_average(all_evoked_ld)
    diff_evoked = mne.combine_evoked([grand_avg_vs, grand_avg_ld], weights=[1, -1])
    
    # --- MODIFIED: Create a multi-channel grand average ERP plot ---
    ga_channels_to_plot = ['Fz', 'Cz', 'Pz']
    evokeds_dict_ga = {'VS': grand_avg_vs, 'LD': grand_avg_ld}
    
    figs = mne.viz.plot_compare_evokeds(
        evokeds_dict_ga,
        picks=ga_channels_to_plot,
        linestyles={'VS': '-', 'LD': '--'},
        styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}},
        show=False,
        invert_y=True
    )
    fig = figs[0]
    fig.suptitle("Grand Average Readiness Potential", fontsize=16)
    # Add vertical line to each subplot
    for ax in fig.get_axes():
        ax.axvline(0, color='r', linestyle='-', lw=1.5, label='Movement Onset')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_fname = op.join(output_dir, "grand_average_bp_waveforms_Fz_Cz_Pz.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Waveforms (Fz, Cz, Pz)', section='Grand Average Results', tags=('erp', 'waveform'))
    plt.close(fig)
    print(f"  - Saved grand average Fz/Cz/Pz waveform plot.")

    # --- NEW: Grand average plot for Parietal channels ---
    ga_parietal_channels = ['P3', 'Pz', 'P4']
    figs = mne.viz.plot_compare_evokeds(
        evokeds_dict_ga,
        picks=ga_parietal_channels,
        linestyles={'VS': '-', 'LD': '--'},
        styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}},
        show=False,
        invert_y=True
    )
    fig = figs[0]
    fig.suptitle("Grand Average ERPs - Parietal Channels", fontsize=16)
    for ax in fig.get_axes():
        ax.axvline(0, color='r', linestyle='-', lw=1.5, label='Movement Onset')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_fname = op.join(output_dir, "grand_average_bp_waveforms_Parietal.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Waveforms (Parietal)', section='Grand Average Results', tags=('erp', 'waveform'))
    plt.close(fig)
    print(f"  - Saved grand average Parietal waveform plot.")
    # --- END NEW ---

    # --- NEW: Grand average plot for Frontal channels ---
    ga_frontal_channels = ['F3', 'Fz', 'F4']
    figs = mne.viz.plot_compare_evokeds(
        evokeds_dict_ga,
        picks=ga_frontal_channels,
        linestyles={'VS': '-', 'LD': '--'},
        styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}},
        show=False,
        invert_y=True
    )
    fig = figs[0]
    fig.suptitle("Grand Average ERPs - Frontal Channels", fontsize=16)
    for ax in fig.get_axes():
        ax.axvline(0, color='r', linestyle='-', lw=1.5, label='Movement Onset')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_fname = op.join(output_dir, "grand_average_bp_waveforms_Frontal.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Waveforms (Frontal)', section='Grand Average Results', tags=('erp', 'waveform'))
    plt.close(fig)
    print(f"  - Saved grand average Frontal waveform plot.")

    # --- NEW: Grand average plot for Central channels ---
    ga_central_channels = ['C3', 'Cz', 'C4']
    figs = mne.viz.plot_compare_evokeds(
        evokeds_dict_ga,
        picks=ga_central_channels,
        linestyles={'VS': '-', 'LD': '--'},
        styles={'VS': {"alpha": 1.0}, 'LD': {"alpha": 0.7}},
        show=False,
        invert_y=True
    )
    fig = figs[0]
    fig.suptitle("Grand Average ERPs - Central Channels", fontsize=16)
    for ax in fig.get_axes():
        ax.axvline(0, color='r', linestyle='-', lw=1.5, label='Movement Onset')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_fname = op.join(output_dir, "grand_average_bp_waveforms_Central.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Waveforms (Central)', section='Grand Average Results', tags=('erp', 'waveform'))
    plt.close(fig)
    print(f"  - Saved grand average Central waveform plot.")
    # --- END NEW ---

    # --- MODIFIED: Improve topography plot and add joint plot ---
    # This is now a multi-panel plot for each analysis window
    print("\n--- Plotting Grand Average Topographies for Analysis Windows ---")
    fig, axes = plt.subplots(1, len(time_windows), figsize=(5 * len(time_windows), 5), sharex=True, sharey=True)
    if len(time_windows) == 1: # Ensure axes is always a list
        axes = [axes]
    fig.suptitle("Grand Average Topography of VS-LD Difference", fontsize=16)

    for ax, (t_label, (tmin, tmax)) in zip(axes, time_windows.items()):
        diff_evoked.plot_topomap(
            times=(tmin + tmax) / 2, 
            average=tmax - tmin,
            axes=ax,
            show=False,
            colorbar=False # Add a single colorbar at the end
        )
        ax.set_title(f"{t_label.title()} Window\n({tmin}s to {tmax}s)")

    # Add a single colorbar to the figure
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=diff_evoked.data.min(), vmax=diff_evoked.data.max()))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, orientation='vertical', label='Amplitude (µV)')
    
    fig_fname = op.join(output_dir, "grand_average_bp_topography_diff_windows.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Topography (Difference per Window)', section='Grand Average Results', tags=('topomap', 'difference'))
    plt.close(fig)
    print(f"  - Saved grand average multi-window topography plot.")
    
    # Joint Plot (as before)
    fig = diff_evoked.plot_joint(
        title="Grand Average Difference Wave (VS - LD)",
        times=[-0.8, -0.5, -0.2, 0.0], # Specify time points for topomaps
        show=False
    )
    fig_fname = op.join(output_dir, "grand_average_bp_joint_plot_diff.png")
    fig.savefig(fig_fname)
    report.add_figure(fig=fig, title='GA Joint Plot (Difference)', section='Grand Average Results', tags=('joint', 'difference'))
    plt.close(fig)
    print(f"  - Saved grand average joint plot of the difference.")

    # --- 4. STATISTICAL ANALYSIS (RESTRUCTURED) ---
    print("\n" + "="*80)
    print("--- Running Spatio-Temporal Cluster Analysis ---")
    print("="*80)

    # --- NEW: Define a corrected alpha for multiple comparisons across windows ---
    n_windows = len(time_windows)
    alpha = 0.05
    corrected_alpha = alpha / n_windows
    print(f"--- Using Bonferroni-corrected alpha of {corrected_alpha:.4f} for {n_windows} time windows ---")

    # --- NEW: Define channel regions for reporting ---
    regions = {
        "Frontal": ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6'],
        "Central": ['T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6'],
        "Parietal": ['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
    }

    # Get channel adjacency for cluster tests
    eeg_channels = mne.pick_types(grand_avg_vs.info, eeg=True)
    adjacency, ch_names = mne.channels.find_ch_adjacency(grand_avg_vs.info, ch_type='eeg')
    
    group_results = {t_label: [] for t_label in time_windows}
    # --- MODIFIED: New data structure for detailed subject results ---
    subject_specific_results = {subject: {t_label: [] for t_label in time_windows} for subject in subjects}

    # Helper function to find cluster locations
    def get_cluster_regions(cluster_indices, ch_names, regions_dict):
        cluster_ch_names = [ch_names[i] for i in cluster_indices]
        found_regions = set()
        for region_name, region_channels in regions_dict.items():
            if any(ch in cluster_ch_names for ch in region_channels):
                found_regions.add(region_name)
        return sorted(list(found_regions))

    # --- NEW: Helper function to count channels per region in a cluster ---
    def count_channels_in_regions(cluster_ch_names, regions_dict):
        region_counts = {region_name: 0 for region_name in regions_dict}
        for ch_name in cluster_ch_names:
            for region_name, region_channels in regions_dict.items():
                if ch_name in region_channels:
                    region_counts[region_name] += 1
        return region_counts

    # --- NEW: Helper function to classify cluster strength ---
    def get_strength_descriptor(mass, channels, duration):
        """Provides a qualitative descriptor for a cluster's strength."""
        # Score based on cluster mass
        if mass > 1000: mass_score = 3
        elif mass > 500: mass_score = 2
        else: mass_score = 1
        
        # Score based on spatial spread (number of channels)
        if channels > 20: space_score = 3
        elif channels > 10: space_score = 2
        else: space_score = 1
            
        # Score based on temporal duration (ms)
        if duration > 300: time_score = 3
        elif duration > 150: time_score = 2
        else: time_score = 1
            
        total_score = mass_score + space_score + time_score
        
        if total_score >= 8: return "Very Strong"
        if total_score >= 6: return "Strong"
        if total_score >= 4: return "Moderate"
        return "Weak"

    # --- GROUP-LEVEL ANALYSIS ---
    print("\n--- Running GROUP-LEVEL Spatio-Temporal Cluster Tests ---")
    # Prepare data for all subjects (n_subjects, n_channels, n_times)
    X_vs_all = np.array([evk.copy().pick(ch_names).get_data() for evk in all_evoked_vs])
    X_ld_all = np.array([evk.copy().pick(ch_names).get_data() for evk in all_evoked_ld])
    X_diff_all = X_vs_all - X_ld_all

    for t_label, (tmin, tmax) in time_windows.items():
        print(f"\n--- Testing {t_label.upper()} window ({tmin}s to {tmax}s) at GROUP level ---")
        
        # Crop data to the current time window
        times = grand_avg_vs.times
        time_mask = (times >= tmin) & (times <= tmax)
        X_diff_window = X_diff_all[:, :, time_mask]
        
        # Transpose for the stats function: (n_subjects, n_times_in_window, n_channels)
        X = X_diff_window.transpose(0, 2, 1)

        t_obs, clusters, cluster_p, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            X, adjacency=adjacency, n_permutations=1000, n_jobs=-1
        )
        
        # --- MODIFIED: Use corrected alpha ---
        significant_clusters_idx = np.where(cluster_p < corrected_alpha)[0]
        
        if len(significant_clusters_idx) > 0:
            print(f"  >>> FOUND {len(significant_clusters_idx)} SIGNIFICANT GROUP CLUSTER(S) in {t_label.upper()} window.")
            for clu_idx in significant_clusters_idx:
                ch_inds = clusters[clu_idx][1]
                cluster_regions = get_cluster_regions(ch_inds, ch_names, regions)
                group_results[t_label].extend(cluster_regions)
            group_results[t_label] = sorted(list(set(group_results[t_label]))) # Get unique sorted list
        else:
            print(f"  - No significant group clusters found.")

    # --- SUBJECT-LEVEL ANALYSIS ---
    print("\n--- Running SUBJECT-LEVEL Spatio-Temporal Cluster Tests ---")
    for t_label, (tmin, tmax) in time_windows.items():
        print(f"\n--- Testing {t_label.upper()} window ({tmin}s to {tmax}s) for each SUBJECT ---")
        
        for i, subject in enumerate(subjects):
            try:
                # We need the original epochs for within-subject test
                vs_epochs_fname = op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif")
                ld_epochs_fname = op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif")
                epochs_vs = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
                epochs_ld = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)
                
                # Apply baseline
                epochs_vs.apply_baseline(baseline=baseline_window)
                epochs_ld.apply_baseline(baseline=baseline_window)

                # Crop epochs to the time window
                epochs_vs.crop(tmin, tmax)
                epochs_ld.crop(tmin, tmax)
                window_times = epochs_vs.times

                # Prepare data for permutation_cluster_test
                X_vs_subj = epochs_vs.pick(ch_names).get_data().transpose(0, 2, 1) # (n_epochs, n_times, n_channels)
                X_ld_subj = epochs_ld.pick(ch_names).get_data().transpose(0, 2, 1)
                
                t_obs_subj, clusters_subj, cluster_p_subj, _ = mne.stats.permutation_cluster_test(
                    [X_vs_subj, X_ld_subj], adjacency=adjacency, n_permutations=1000, n_jobs=-1
                )

                # --- MODIFIED: Use corrected alpha ---
                significant_clusters_subj_idx = np.where(cluster_p_subj < corrected_alpha)[0]
                if len(significant_clusters_subj_idx) > 0:
                    # --- MODIFIED: Extract detailed info for each significant cluster ---
                    for clu_idx in significant_clusters_subj_idx:
                        # Get temporal and spatial extent of the cluster
                        time_inds, ch_inds = clusters_subj[clu_idx]
                        
                        # --- FIX: Calculate cluster mass (sum of t-values) ---
                        cluster_mass = t_obs_subj[time_inds, ch_inds].sum()

                        # Temporal information
                        start_time = window_times[time_inds.min()]
                        end_time = window_times[time_inds.max()]
                        duration_ms = (end_time - start_time) * 1000
                        window_duration_ms = (tmax - tmin) * 1000
                        time_percentage = (duration_ms / window_duration_ms) * 100
                        
                        # --- FIX: Correctly identify unique channels ---
                        unique_ch_inds = np.unique(ch_inds)
                        cluster_ch_names = [ch_names[i] for i in unique_ch_inds]
                        num_unique_channels = len(cluster_ch_names)
                        region_counts = count_channels_in_regions(cluster_ch_names, regions)

                        # --- NEW: Get qualitative strength descriptor ---
                        strength = get_strength_descriptor(cluster_mass, num_unique_channels, duration_ms)

                        cluster_details = {
                            "p_value": cluster_p_subj[clu_idx],
                            "cluster_mass": cluster_mass,
                            "start_time_s": start_time,
                            "end_time_s": end_time,
                            "duration_ms": duration_ms,
                            "time_percentage": time_percentage,
                            "strength": strength,
                            "total_channels": num_unique_channels,
                            "region_counts": region_counts
                        }
                        subject_specific_results[subject][t_label].append(cluster_details)

            except Exception as e:
                print(f"    - WARNING: Could not process subject {subject} for {t_label} window. Error: {e}")

    # --- 5. FINAL SUMMARY REPORT ---
    print("\n" + "="*80)
    print("--- Generating Final HTML Report ---")
    print("="*80)

    # --- NEW: Build HTML for the analysis description ---
    desc_html = f"""
    <h2>Analysis Methods</h2>
    <h3>Preprocessing</h3>
    <p>Data was preprocessed using a pipeline defined in the configuration file. Key steps included:
    <ul>
        <li>Band-pass filtering from {params['l_freq_slow']} Hz to {params['h_freq_slow']} Hz.</li>
        <li>Independent Component Analysis (ICA) to identify and remove eye and heart artifacts.</li>
        <li>Epoching around the sigh onset from {params['tmin_slow']}s to {params['tmax_slow']}s.</li>
        <li>Automated epoch rejection using the Autoreject algorithm to remove remaining artifacts.</li>
        <li>Baseline correction applied using the interval: <code>{baseline_window}</code>.</li>
    </ul>
    </p>
    <h3>Statistical Analysis</h3>
    <p>Spatio-temporal cluster-based permutation tests were used to compare the Voluntary Sigh (VS) and Load (LD) conditions across four distinct time windows. This method corrects for multiple comparisons across time points and channels.</p>
    <ol>
        <li><b>Group Level:</b> A one-sample cluster test (<code>spatio_temporal_cluster_1samp_test</code>) was performed on the VS-LD difference waves across all subjects.</li>
        <li><b>Individual Level:</b> A within-subject paired cluster test (<code>permutation_cluster_test</code>) was performed for each participant, comparing their set of VS trials to their LD trials.</li>
    </ol>
    <p>For reporting, significant clusters were localized to one or more of three predefined regions: Frontal, Central, and Parietal.</p>
    """
    report.add_html(html=desc_html, title='Analysis Description', section='Methods')

    # Build HTML for the summary tables
    summary_html = "<h2>Statistical Analysis Summary</h2>"
    
    # Group Results
    summary_html += "<h3>Group-Level Results</h3><p>Significant differences (p < 0.05) were found in the following time windows:</p><ul>"
    found_group_sig = False
    for window, regions_found in group_results.items():
        if regions_found:
            summary_html += f"<li><b>{window.upper()}:</b> {', '.join(regions_found)}</li>"
            found_group_sig = True
    if not found_group_sig:
        summary_html += "<li>None</li>"
    summary_html += "</ul>"

    # --- MODIFIED: Reverted Subject Results table to a simpler version ---
    summary_html += f"""
    <h3>Subject-Level Results</h3>
    <p>The following table details the significant spatio-temporal clusters (p < {corrected_alpha:.4f}, Bonferroni-corrected for {n_windows} windows) found for each subject.</p>
    """
    # --- NEW: Detailed subject-level results table ---
    for subject, results_by_window in sorted(subject_specific_results.items()):
        summary_html += f"<h4>Subject: {subject}</h4>"
        has_any_sig = any(res for res in results_by_window.values())
        
        if not has_any_sig:
            summary_html += "<p>No significant clusters found in any window.</p>"
            continue

        summary_html += """
        <table class="table table-sm table-bordered">
            <thead class="thead-light">
                <tr>
                    <th>Window</th>
                    <th>p-value</th>
                    <th>Strength</th>
                    <th>Cluster Mass</th>
                    <th>Time Window (s)</th>
                    <th>Duration (ms)</th>
                    <th>% of Window</th>
                    <th># Unique Channels</th>
                    <th>Channel Distribution (F/C/P)</th>
                </tr>
            </thead>
            <tbody>
        """
        for t_label, clusters_list in results_by_window.items():
            if not clusters_list:
                continue
            
            # Sort clusters by p-value
            sorted_clusters = sorted(clusters_list, key=lambda x: x['p_value'])
            
            for i, cluster in enumerate(sorted_clusters):
                # Use rowspan for the window label if there are multiple clusters
                row_start = f'<tr style="background-color: #f8f9fa;">' if i == 0 else '<tr>'
                window_cell = f'<td rowspan="{len(sorted_clusters)}">{t_label.upper()}</td>' if i == 0 else ''
                
                p_val_str = f"{cluster['p_value']:.3f}"
                strength_str = cluster['strength']
                mass_str = f"{cluster['cluster_mass']:.1f}"
                time_win_str = f"[{cluster['start_time_s']:.3f}, {cluster['end_time_s']:.3f}]"
                duration_str = f"{cluster['duration_ms']:.0f}"
                time_perc_str = f"{cluster['time_percentage']:.1f}%"
                rc = cluster['region_counts']
                dist_str = f"{rc['Frontal']} / {rc['Central']} / {rc['Parietal']}"

                summary_html += f"""
                    {row_start}
                        {window_cell}
                        <td>{p_val_str}</td>
                        <td>{strength_str}</td>
                        <td>{mass_str}</td>
                        <td>{time_win_str}</td>
                        <td>{duration_str}</td>
                        <td>{time_perc_str}</td>
                        <td>{cluster['total_channels']}</td>
                        <td>{dist_str}</td>
                    </tr>
                """
        summary_html += "</tbody></table>"

    report.add_html(html=summary_html, title='Statistical Summary', section='Results')

    # Save the final report
    report_fname = op.join(output_dir, "report_readiness_potential.html")
    report.save(report_fname, overwrite=True, open_browser=False)

    print(f"\n--- Analysis Complete. HTML report saved to {report_fname} ---")

    # --- MODIFIED: Generate a simple MD summary file with region details ---
    print("\n--- Generating machine-readable summary file for cross-analysis ---")
    summary_md_fname = op.join(output_dir, "erp_subject_summary.md")
    with open(summary_md_fname, 'w') as f:
        for subject, results_dict in subject_specific_results.items():
            significant_effects = []
            # --- MODIFIED: This part now summarizes the detailed results for the MD file ---
            for t_label, clusters_list in results_dict.items():
                if clusters_list:
                    # For the simple MD file, we just care about the regions involved
                    found_regions_for_window = set()
                    for cluster in clusters_list:
                        for region, count in cluster['region_counts'].items():
                            if count > 0:
                                found_regions_for_window.add(region)
                    
                    if found_regions_for_window:
                        effect_str = f"{t_label.title()}({','.join(sorted(list(found_regions_for_window)))})"
                        significant_effects.append(effect_str)
            
            # Write the subject and their significant effects, comma-separated
            f.write(f"{subject}:{','.join(significant_effects)}\n")
    
    print(f"  - Subject summary saved to {summary_md_fname}")