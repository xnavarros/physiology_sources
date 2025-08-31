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
    config = load_config()
    subjects = config['subjects']
    
    input_dir = config['paths']['preprocessed_slow_potentials_dir']
    output_dir = op.join(config['paths']['results_dir'], 'readiness_potential')
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # --- NEW: Create a dedicated folder for subject ERP plots ---
    erp_output_dir = op.join('figures', 'subject_erp_plots')
    if not op.exists(erp_output_dir):
        os.makedirs(erp_output_dir)

    # Analysis parameters
    channel_of_interest = 'Cz' # For stats and grand average plot
    channels_to_plot = ['Fz', 'Cz', 'F3', 'F4', 'C3', 'C4'] # For subject-level plots
    topo_window = (-0.5, 0.0)

    all_evoked_vs = []
    all_evoked_ld = []

    print("--- Starting Readiness Potential (BP) Analysis ---")

    # --- 2. LOAD DATA AND CREATE INDIVIDUAL AVERAGES & PLOTS ---
    for subject in subjects:
        print(f"  - Processing subject: {subject}")
        try:
            vs_epochs_fname = op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif")
            ld_epochs_fname = op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif")
            
            # The data is already cleaned by AutoReject in the preprocessing script,
            # so we can load it directly without further rejection.
            epochs_vs = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
            epochs_ld = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)

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
                plt.close(fig_press)

            except Exception as e:
                print(f"    - WARNING: Could not plot Debit/Pression for subject {subject}. Error: {e}")
            # --- END MODIFIED ---

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
    plt.close(fig)
    print(f"  - Saved grand average Central waveform plot.")
    # --- END NEW ---

    # --- MODIFIED: Improve topography plot and add joint plot ---
    diff_evoked = mne.combine_evoked([grand_avg_vs, grand_avg_ld], weights=[1, -1])
    
    # Improved Topography Plot (as before)
    fig = diff_evoked.plot_topomap(
        times=topo_window, 
        average=topo_window[1] - topo_window[0],
        show=False
    )
    fig.set_size_inches(8, 6)
    fig.suptitle(f"Grand Average Topography of VS-LD Difference ({topo_window[0]}s to {topo_window[1]}s)", fontsize=14, y=0.98)
    fig_fname = op.join(output_dir, "grand_average_bp_topography_diff.png")
    fig.savefig(fig_fname)
    plt.close(fig)
    print(f"  - Saved grand average topography plot.")

    # --- NEW: Joint Plot to show time-course of the difference ---
    # This plot shows the difference wave (like a GFP) and topomaps at specific time points.
    fig = diff_evoked.plot_joint(
        title="Grand Average Difference Wave (VS - LD)",
        times=[-0.8, -0.5, -0.2, 0.0], # Specify time points for topomaps
        show=False
    )
    fig_fname = op.join(output_dir, "grand_average_bp_joint_plot_diff.png")
    fig.savefig(fig_fname)
    plt.close(fig)
    print(f"  - Saved grand average joint plot of the difference.")
    # --- END NEW ---

    # --- 4. STATISTICAL ANALYSIS ---
    print(f"--- Running Cluster Permutation Test on Channel '{channel_of_interest}' ---")
    
    X_vs = np.array([evk.copy().pick(channel_of_interest).get_data() for evk in all_evoked_vs]).squeeze()
    X_ld = np.array([evk.copy().pick(channel_of_interest).get_data() for evk in all_evoked_ld]).squeeze()
    X_diff = X_vs - X_ld

    t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X_diff, n_permutations=1024, threshold=None, n_jobs=-1
    )
    
    significant_clusters = np.where(cluster_p_values < 0.05)[0]
    print(f"  - Found {len(significant_clusters)} significant clusters.")

    fig, ax = plt.subplots(figsize=(10, 6))
    times = grand_avg_vs.times
    ax.plot(times, X_diff.mean(axis=0), label='VS - LD Difference')
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.axvline(0, color='r', linestyle='-', lw=1.5)
    
    for i_clu, clu_ts in enumerate(clusters):
        if cluster_p_values[i_clu] < 0.05:
            ax.axvspan(times[clu_ts[0].min()], times[clu_ts[0].max()], color='gray', alpha=0.3)
            print(f"    - Cluster #{i_clu+1}: p-value = {cluster_p_values[i_clu]:.3f}, "
                  f"from {times[clu_ts[0].min()]:.3f}s to {times[clu_ts[0].max()]:.3f}s")

    ax.legend()
    ax.set_title(f"Difference Wave at '{channel_of_interest}' with Significance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig_fname = op.join(output_dir, f"grand_average_bp_stats_{channel_of_interest}.png")
    fig.savefig(fig_fname)
    plt.close(fig)
    print(f"  - Saved stats plot.")

    print("\n--- Analysis Complete ---")