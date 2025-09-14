import os
import os.path as op
import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_test, permutation_cluster_test
from mne import Report

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # --- 1. CONFIGURATION AND SETUP ---
    print("--- Setting up Time-Frequency Analysis & Report Generation ---")
    config = load_config()
    subjects = config['subjects']
    
    input_dir = config['paths']['preprocessed_slow_potentials_dir']
    output_dir = op.join('figures', 'TF_analysis')
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    freq_min, freq_max = 8.0, 30.0
    freqs = np.arange(freq_min, freq_max, 1.0)
    n_cycles = freqs / 2.
    baseline_window = tuple(config['preprocessing_params']['baseline_timing_slow'])

    # --- 2. LOAD DATA AND COMPUTE AVERAGE TFRS ---
    all_tfr_vs, all_tfr_ld = [], []
    print("\n--- Loading data and computing subject-average TFRs ---")
    for subject in subjects:
        try:
            epochs_vs = mne.read_epochs(op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif"), preload=True, verbose=False)
            epochs_ld = mne.read_epochs(op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif"), preload=True, verbose=False)

            tfr_vs = epochs_vs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
            tfr_ld = epochs_ld.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
            
            tfr_vs.apply_baseline(baseline_window, mode='zscore')
            tfr_ld.apply_baseline(baseline_window, mode='zscore')

            all_tfr_vs.append(tfr_vs)
            all_tfr_ld.append(tfr_ld)
            print(f"  - Processed subject: {subject}")
        except FileNotFoundError:
            print(f"    - WARNING: Files not found for subject {subject}. Skipping.")
            continue

    if not all_tfr_vs:
        print("\nFATAL ERROR: No data found. Exiting analysis.")
        exit()

    # --- Initialize HTML Report ---
    report = Report(title='Time-Frequency Analysis Report: VS vs. LD', verbose=False)

    # --- 2.2. PLOTTING (as before, with adjusted scales) ---
    print("\n--- Plotting individual and grand average TFRs ---")
    # Individual Plots
    for i, subject in enumerate(subjects):
        if i >= len(all_tfr_vs): continue
        tfr_diff = all_tfr_vs[i].copy()
        tfr_diff.data = all_tfr_vs[i].data - all_tfr_ld[i].data
        # Widen the color scale to [-2.0, 2.0] to reduce sensitivity to noise
        fig_subj = tfr_diff.plot_topo(title=f'Subject {subject} - Difference (VS - LD)', show=False, vmin=-2.0, vmax=2.0)
        # This loop correctly adds a vertical line to each sensor's subplot
        for ax in fig_subj.axes:
            if hasattr(ax, 'axvline'): # Check if it's a plot axis, not a colorbar
                ax.axvline(0, color='w', linestyle='--', lw=1)
        fig_subj.set_size_inches(12, 8)
        report.add_figure(fig=fig_subj, title=f'Subject {subject}', section='Individual Results', tags=('individual',))
        plt.close(fig_subj)
    print("  - Added individual subject plots to report.")

    # Grand Average Plots
    grand_avg_vs = mne.grand_average(all_tfr_vs)
    grand_avg_ld = mne.grand_average(all_tfr_ld)
    grand_avg_diff = mne.grand_average(all_tfr_vs)
    grand_avg_diff.data = grand_avg_vs.data - grand_avg_ld.data
    # Widen the color scale to [-1.0, 1.0] for the grand average
    fig_diff = grand_avg_diff.plot_topo(title='Grand Average - Difference (VS - LD)', show=False, vmin=-1.0, vmax=1.0)
    # Add vertical line to each sensor's subplot in the grand average plot
    for ax in fig_diff.axes:
        if hasattr(ax, 'axvline'):
            ax.axvline(0, color='w', linestyle='--', lw=1)
    fig_diff.set_size_inches(12, 8)
    report.add_figure(fig=fig_diff, title='Grand Average Difference', section='Group Results', tags=('grand-average',))
    plt.close(fig_diff)
    print("  - Added grand average plot to report.")

    # --- 3. & 4. STATISTICAL ANALYSIS & RESULT STORAGE ---
    freq_bands = {"alpha": (8.0, 12.0), "beta": (13.0, 30.0)}
    time_windows = {"early": (-1.5, -1.0), "mid": (-1.0, -0.5), "late": (-0.5, 0.0), "post": (0.0, 0.5)}
    info = all_tfr_vs[0].info
    adjacency, ch_names_from_adj = mne.channels.find_ch_adjacency(info, ch_type='eeg')

    # --- NEW: Define a corrected alpha for multiple comparisons ---
    n_tests = len(time_windows) * len(freq_bands)
    alpha = 0.05
    corrected_alpha = alpha / n_tests
    print(f"\n--- Using Bonferroni-corrected alpha of {corrected_alpha:.4f} for {n_tests} tests (windows x bands) ---")

    # --- NEW: Define channel regions for reporting ---
    regions = {
        "Frontal": ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6'],
        "Central": ['T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6'],
        "Parietal": ['P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
    }
    # Get channel names from info, which is more reliable
    ch_names = info['ch_names']

    # --- NEW: Helper functions from ERP script ---
    def count_channels_in_regions(cluster_ch_names, regions_dict):
        region_counts = {region_name: 0 for region_name in regions_dict}
        for ch_name in cluster_ch_names:
            for region_name, region_channels in regions_dict.items():
                if ch_name in region_channels:
                    region_counts[region_name] += 1
        return region_counts

    def get_strength_descriptor(mass, channels, duration):
        """Provides a qualitative descriptor for a cluster's strength."""
        # Score based on cluster mass
        if abs(mass) > 1000: mass_score = 3
        elif abs(mass) > 500: mass_score = 2
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

    group_results = {}
    # --- MODIFIED: New data structure for detailed results ---
    subject_specific_results = {subject: {f"{band}-{t_label}": [] for t_label in time_windows for band in freq_bands} for subject in subjects}

    # Group-level analysis (remains high-level for now)
    for t_label, (tmin, tmax) in time_windows.items():
        for band, (fmin, fmax) in freq_bands.items():
            print(f"\n--- Running GROUP Cluster Test for {band.upper()} band in {t_label.upper()} window ---")
            X_vs = np.array([tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax).data.mean(axis=1) for tfr in all_tfr_vs])
            X_ld = np.array([tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax).data.mean(axis=1) for tfr in all_tfr_ld])
            X_diff = X_vs - X_ld
            X = X_diff.transpose(0, 2, 1)
            t_obs, clusters, cluster_p, _ = spatio_temporal_cluster_test([X], adjacency=adjacency, n_permutations=1024, n_jobs=-1)
            
            # Use corrected alpha for group test as well
            n_sig_clusters = len(np.where(cluster_p < corrected_alpha)[0])
            group_results[(t_label, band)] = n_sig_clusters
            if n_sig_clusters > 0:
                print(f"  >>> Found {n_sig_clusters} significant GROUP clusters.")

    # Subject-level analysis
    print("\n" + "="*80 + "\n--- Running WITHIN-SUBJECT Analysis ---\n" + "="*80)
    for t_label, (tmin, tmax) in time_windows.items():
        for band, (fmin, fmax) in freq_bands.items():
            n_sig_subjects = 0
            print(f"\n--- Testing {band.upper()} band in {t_label.upper()} window ---")
            for subject in subjects:
                try:
                    epochs_vs = mne.read_epochs(op.join(input_dir, subject, f"{subject}_VS-slow-epo.fif"), preload=True, verbose=False)
                    epochs_ld = mne.read_epochs(op.join(input_dir, subject, f"{subject}_LD-slow-epo.fif"), preload=True, verbose=False)

                    # Compute TFRs for each epoch (average=False)
                    tfr_vs_epochs = epochs_vs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, n_jobs=-1)
                    tfr_ld_epochs = epochs_ld.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, n_jobs=-1)
                    
                    tfr_vs_epochs.apply_baseline(baseline_window, mode='zscore')
                    tfr_ld_epochs.apply_baseline(baseline_window, mode='zscore')

                    # Crop to time window for this test
                    tfr_vs_win = tfr_vs_epochs.copy().crop(tmin=tmin, tmax=tmax)
                    tfr_ld_win = tfr_ld_epochs.copy().crop(tmin=tmin, tmax=tmax)
                    window_times = tfr_vs_win.times

                    # --- FIX: Manually crop by frequency and average over the frequency axis ---
                    # The .apply_freq_average() method is not available for EpochsTFR objects.
                    vs_data = tfr_vs_win.copy().crop(fmin=fmin, fmax=fmax).data.mean(axis=2)
                    ld_data = tfr_ld_win.copy().crop(fmin=fmin, fmax=fmax).data.mean(axis=2)
                    
                    # Data shape: (n_epochs, n_channels, n_times) -> transpose for test
                    X_subject = [vs_data.transpose(0, 2, 1), ld_data.transpose(0, 2, 1)]
                    
                    # Run cluster test for this subject
                    t_obs_subj, clusters_subj, cluster_p_subj, _ = permutation_cluster_test(X_subject, adjacency=adjacency, n_permutations=1000, n_jobs=-1)
                    
                    # --- MODIFIED: Use corrected alpha and extract full details ---
                    significant_clusters_idx = np.where(cluster_p_subj < corrected_alpha)[0]
                    if len(significant_clusters_idx) > 0:
                        n_sig_subjects += 1
                        
                        for clu_idx in significant_clusters_idx:
                            time_inds, ch_inds = clusters_subj[clu_idx]
                            cluster_mass = t_obs_subj[time_inds, ch_inds].sum()
                            
                            start_time = window_times[time_inds.min()]
                            end_time = window_times[time_inds.max()]
                            duration_ms = (end_time - start_time) * 1000
                            window_duration_ms = (tmax - tmin) * 1000
                            time_percentage = (duration_ms / window_duration_ms) * 100

                            unique_ch_inds = np.unique(ch_inds)
                            cluster_ch_names = [ch_names[i] for i in unique_ch_inds]
                            num_unique_channels = len(cluster_ch_names)
                            region_counts = count_channels_in_regions(cluster_ch_names, regions)
                            
                            strength = get_strength_descriptor(cluster_mass, num_unique_channels, duration_ms)

                            cluster_details = {
                                "p_value": cluster_p_subj[clu_idx],
                                "cluster_mass": cluster_mass,
                                "strength": strength,
                                "start_time_s": start_time,
                                "end_time_s": end_time,
                                "duration_ms": duration_ms,
                                "time_percentage": time_percentage,
                                "total_channels": num_unique_channels,
                                "region_counts": region_counts
                            }
                            subject_specific_results[subject][f"{band}-{t_label}"].append(cluster_details)

                except Exception as e:
                    print(f"  - Could not process subject {subject}. Error: {e}")
                    continue
            
            print(f"  >>> SUMMARY: {n_sig_subjects} out of {len(subjects)} subjects showed a significant difference.")

    # --- 5. GENERATE FINAL HTML REPORT ---
    print("\n--- Generating final HTML analysis report ---")
    
    # Dynamically create the time window list for the report
    time_window_list_html = ""
    for t_label, (tmin, tmax) in time_windows.items():
        time_window_list_html += f"<li><b>{t_label.title()}:</b> {tmin}s to {tmax}s</li>"

    # Build HTML for the analysis description
    desc_html = f"""
    <h3>Pre-processing</h3>
    <p>Data was pre-processed following a standard pipeline including band-pass filtering, ICA decomposition for artifact removal (EOG, ECG), epoching around the sigh onset, and automated artifact rejection using Autoreject.</p>
    <h3>Time-Frequency Analysis Parameters</h3>
    <ul>
        <li><b>Frequency Range:</b> {freq_min} Hz to {freq_max} Hz.</li>
        <li><b>Method:</b> Morlet wavelets (<code>n_cycles = freqs / 2</code>).</li>
        <li><b>Baseline Correction:</b> Z-score transformation relative to the <code>{baseline_window}</code> window.</li>
    </ul>
    <h3>Statistical Analysis</h3>
    <p>Two levels of statistical analysis were performed to compare the Voluntary Sigh (VS) and Load (LD) conditions:</p>
    <ol>
        <li><b>Group Level:</b> A non-parametric, cluster-based permutation test was performed on the subject-averaged data.</li>
        <li><b>Individual Level:</b> A within-subject cluster-based permutation test was performed for each participant.</li>
    </ol>
    <p>Both analyses were run separately for two frequency bands (Alpha: 8-12 Hz, Beta: 13-30 Hz) and the following time windows:</p>
    <ul>{time_window_list_html}</ul>
    <p>A Bonferroni correction was applied to account for the {n_tests} tests performed, resulting in a significance threshold of <b>p < {corrected_alpha:.4f}</b>.</p>
    """
    report.add_html(html=desc_html, title='Analysis Description', section='Methods')

    # --- MODIFIED: Remove old summary tables ---

    # --- NEW: Detailed subject-specific results tables ---
    subject_table_html = "<h3>Subject-Specific Effects</h3>"
    for subject, results_by_effect in sorted(subject_specific_results.items()):
        subject_table_html += f"<h4>Subject: {subject}</h4>"
        has_any_sig = any(res for res in results_by_effect.values())
        
        if not has_any_sig:
            subject_table_html += f"<p>No significant clusters found (p < {corrected_alpha:.4f}).</p>"
            continue

        subject_table_html += """
        <table class="table table-sm table-bordered">
            <thead class="thead-light">
                <tr>
                    <th>Effect (Band-Window)</th>
                    <th>p-value</th>
                    <th>Strength</th>
                    <th>Cluster Mass</th>
                    <th>Time Window (s)</th>
                    <th>Duration (ms)</th>
                    <th>% of Window</th>
                    <th># Channels</th>
                    <th>Distribution (F/C/P)</th>
                </tr>
            </thead>
            <tbody>
        """
        for effect_label, clusters_list in results_by_effect.items():
            if not clusters_list:
                continue
            
            sorted_clusters = sorted(clusters_list, key=lambda x: x['p_value'])
            
            for i, cluster in enumerate(sorted_clusters):
                row_start = f'<tr style="background-color: #f8f9fa;">' if i == 0 else '<tr>'
                effect_cell = f'<td rowspan="{len(sorted_clusters)}">{effect_label.replace("-", " ").title()}</td>' if i == 0 else ''
                
                p_val_str = f"{cluster['p_value']:.4f}"
                strength_str = cluster['strength']
                mass_str = f"{cluster['cluster_mass']:.1f}"
                time_win_str = f"[{cluster['start_time_s']:.3f}, {cluster['end_time_s']:.3f}]"
                duration_str = f"{cluster['duration_ms']:.0f}"
                time_perc_str = f"{cluster['time_percentage']:.1f}%"
                rc = cluster['region_counts']
                dist_str = f"{rc['Frontal']} / {rc['Central']} / {rc['Parietal']}"

                subject_table_html += f"""
                    {row_start}
                        {effect_cell}
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
        subject_table_html += "</tbody></table>"
    report.add_html(html=subject_table_html, title='Subject-Specific Results', section='Results')

    # Save the report
    report_dir = op.join('results', 'TF_analysis')
    if not op.exists(report_dir):
        os.makedirs(report_dir)
    report_path = op.join(report_dir, "report_TF_analysis.html")
    report.save(report_path, overwrite=True, open_browser=False)

    print(f"\n--- Analysis Complete. HTML report saved to {report_path} ---")

    # --- MODIFIED: Generate a simple MD summary file for cross-analysis ---
    print("\n--- Generating machine-readable summary file for cross-analysis ---")
    summary_md_fname = op.join(report_dir, "tfr_subject_summary.md")
    with open(summary_md_fname, 'w') as f:
        for subject, results_by_effect in subject_specific_results.items():
            significant_effects = []
            # --- MODIFIED: This part now summarizes the detailed results for the MD file ---
            for effect_label, clusters_list in results_by_effect.items():
                if clusters_list:
                    found_regions_for_effect = set()
                    for cluster in clusters_list:
                        for region, count in cluster['region_counts'].items():
                            if count > 0:
                                found_regions_for_effect.add(region)
                    
                    if found_regions_for_effect:
                        band, t_label = effect_label.split('-')
                        regions_str = ",".join(sorted(list(found_regions_for_effect)))
                        effect_str = f"{band.title()}-{t_label.title()}({regions_str})"
                        significant_effects.append(effect_str)
            
            f.write(f"{subject}:{','.join(significant_effects)}\n")
    
    print(f"  - Subject summary saved to {summary_md_fname}")