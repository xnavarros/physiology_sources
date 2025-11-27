import os
import os.path as op
import json
import mne
import yaml
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind, ttest_1samp
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import sys

# Add project root to path to import utils
sys.path.append(op.join(op.dirname(__file__), '..', '..'))
from scripts.utils.paper_data_manager import PaperDataManager

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
    """
    Main execution block for source analysis of Alpha and Beta band activity.
    This script performs a window-by-window source reconstruction to ensure stable results.
    """

    # --- 1. CONFIGURATION AND SETUP ---
    print("--- Setting up Source Analysis for Alpha/Beta Power ---")
    config = load_config()
    subjects = config['subjects']
    preprocessed_dir = config['paths']['preprocessed_slow_potentials_dir'] 
    output_dir = op.join('results', 'source_analysis_alpha_beta')
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    report = mne.Report(title='Source Analysis Report: Alpha & Beta Power', verbose=False)

    # --- Load profiles from JSON ---
    # Construct path relative to this script's location
    script_dir = op.dirname(op.abspath(__file__))
    root_dir = op.dirname(op.dirname(script_dir))
    json_path = op.join(root_dir, 'results', 'subject_profiles.json')
    
    if op.exists(json_path):
        with open(json_path, 'r') as f:
            profile_map = json.load(f)
        print(f"Loaded subject profiles from {json_path}")
    else:
        print(f"WARNING: Profile JSON not found at {json_path}. Using default hardcoded profiles.")
        profile_map = {
            "Early Preparer": ["MC05", "BJ25", "VS06"],
            "Late Preparer": ["JS08", "TH24", "VA14", "MN23", "SB27"],
            "Reactive Responder": ["LP26", "OL04"]
        }

    subject_to_profile = {sub: prof for prof, subs in profile_map.items() for sub in subs}

    # Initialize Data Manager
    data_manager = PaperDataManager(config['paths']['results_dir'])

    freq_bands = {'alpha': [8, 12], 'beta': [13, 30]}
    baseline_window = (-2.0, -1.5)
    active_windows = {"Early": (-1.5, -1.0), "Mid": (-1.0, -0.5), "Late": (-0.5, 0.0), "Post": (0.0, 0.5)}

    # --- 2. LOAD MODELS AND DATA ---
    print("\n--- Loading common fsaverage models and subject data ---")
    try:
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        fwd_fname = op.join(subjects_dir, "fsaverage-fwd.fif")
        fwd = mne.read_forward_solution(fwd_fname)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find model file: {e}"); exit()

    all_epochs_vs, all_epochs_ld = {}, {}
    for subject in subjects:
        try:
            vs_epochs_fname = op.join(preprocessed_dir, f"{subject}_VS-slow-epo.fif")
            ld_epochs_fname = op.join(preprocessed_dir, f"{subject}_LD-slow-epo.fif")
            all_epochs_vs[subject] = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
            all_epochs_ld[subject] = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)
        except FileNotFoundError:
            print(f"  - Warning: Could not find epoch files for subject {subject}. Skipping.")
            continue

    # --- 3. WINDOW-BY-WINDOW SOURCE RECONSTRUCTION ---
    print("\n--- Performing window-by-window source reconstruction ---")
    all_subject_results = []
    all_stc_results = {prof: {band: {win: [] for win in active_windows} for band in freq_bands} for prof in profile_map}

    grand_average_ld = mne.grand_average([e.average() for e in all_epochs_ld.values()])
    grand_average_ld.set_eeg_reference('average', projection=True)

    labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
    roi_names = ['precentral', 'postcentral', 'caudalmiddlefrontal', 'superiorfrontal', 'insula', 'superiortemporal']
    selected_labels = [lbl for lbl in labels if any(name in lbl.name for name in roi_names)]

    for band, (fmin, fmax) in freq_bands.items():
        print(f"\n-- Processing Frequency Band: {band.upper()} --")
        
        print("  - Creating baseline filter...")
        ga_baseline = grand_average_ld.copy().crop(tmin=baseline_window[0], tmax=baseline_window[1])
        epochs_for_csd_baseline = mne.EpochsArray(ga_baseline.data[None], ga_baseline.info, tmin=ga_baseline.tmin)
        csd_baseline_filt = mne.time_frequency.csd_fourier(epochs_for_csd_baseline, fmin=fmin, fmax=fmax, n_jobs=-1)
        dics_baseline_filt = mne.beamformer.make_dics(ga_baseline.info, fwd, csd_baseline_filt, reg=0.1, pick_ori='max-power', real_filter=True)

        for window_name, (tmin_win, tmax_win) in active_windows.items():
            print(f"  - Processing Window: {window_name} ({tmin_win}s to {tmax_win}s)")
            
            ga_active = grand_average_ld.copy().crop(tmin=tmin_win, tmax=tmax_win)
            epochs_for_csd_active = mne.EpochsArray(ga_active.data[None], ga_active.info, tmin=ga_active.tmin)
            csd_active_filt = mne.time_frequency.csd_fourier(epochs_for_csd_active, fmin=fmin, fmax=fmax, n_jobs=-1)
            dics_active_filt = mne.beamformer.make_dics(ga_active.info, fwd, csd_active_filt, reg=0.1, pick_ori='max-power', real_filter=True)

            for subject in all_epochs_ld.keys():
                epochs_vs = all_epochs_vs[subject]
                epochs_ld = all_epochs_ld[subject]

                csd_vs_baseline = mne.time_frequency.csd_fourier(epochs_vs.copy().crop(tmin=baseline_window[0], tmax=baseline_window[1]), fmin=fmin, fmax=fmax, n_jobs=-1)
                stc_vs_baseline, _ = mne.beamformer.apply_dics_csd(csd_vs_baseline, dics_baseline_filt)

                csd_vs_active = mne.time_frequency.csd_fourier(epochs_vs.copy().crop(tmin=tmin_win, tmax=tmax_win), fmin=fmin, fmax=fmax, n_jobs=-1)
                csd_ld_active = mne.time_frequency.csd_fourier(epochs_ld.copy().crop(tmin=tmin_win, tmax=tmax_win), fmin=fmin, fmax=fmax, n_jobs=-1)
                stc_vs_active, _ = mne.beamformer.apply_dics_csd(csd_vs_active, dics_active_filt)
                stc_ld_active, _ = mne.beamformer.apply_dics_csd(csd_ld_active, dics_active_filt)

                power_diff = stc_ld_active.data.mean(axis=1) - stc_vs_active.data.mean(axis=1)
                baseline_data = stc_vs_baseline.data.mean(axis=1)
                
                power_threshold = np.mean(baseline_data) * 0.01
                stable_mask = baseline_data > power_threshold
                percent_change_data = np.zeros_like(baseline_data)
                percent_change_data[stable_mask] = 100 * (power_diff[stable_mask] / baseline_data[stable_mask])
                
                stc_percent_change = mne.SourceEstimate(percent_change_data[:, np.newaxis], vertices=stc_vs_baseline.vertices, tmin=0, tstep=1, subject='fsaverage')

                profile = subject_to_profile[subject]
                all_stc_results[profile][band][window_name].append(stc_percent_change)
                roi_values = stc_percent_change.extract_label_time_course(selected_labels, src=fwd['src'], mode='mean')
                for i, label in enumerate(selected_labels):
                    all_subject_results.append({
                        "Subject": subject, "Profile": profile, "Region": label.name, 
                        "Window": window_name, "Band": band, "PowerChange": roi_values[i, 0]
                    })

                    # Save to Paper Data
                    data_manager.add_result(
                        analysis_type="alpha_beta_source_roi",
                        subject=subject,
                        metric_name=f"{label.name}_{window_name}_{band}",
                        value=float(roi_values[i, 0]),
                        metadata={
                            "profile": profile,
                            "region": label.name,
                            "window": window_name,
                            "band": band
                        }
                    )

    df_all_subjects = pd.DataFrame(all_subject_results)

    # --- 4. PERFORM WITHIN-GROUP (BASELINE) STATS ---
    print("\n--- Performing within-group (baseline) statistics ---")
    stats_vs_baseline = []
    for _, row in df_all_subjects[['Profile', 'Region', 'Window', 'Band']].drop_duplicates().iterrows():
        prof, region, window, band = row['Profile'], row['Region'], row['Window'], row['Band']
        data = df_all_subjects[(df_all_subjects['Profile'] == prof) & (df_all_subjects['Region'] == region) & (df_all_subjects['Window'] == window) & (df_all_subjects['Band'] == band)]['PowerChange']
        if len(data) > 1:
            t_stat, p_val = ttest_1samp(data, 0)
            stats_vs_baseline.append({
                'Comparison': f"{prof}_vs_Baseline", 'Profile': prof, 'Region': region, 'Window': window, 'Band': band,
                'MeanPowerChange': data.mean(), 'StdDev': data.std(), 'T-statistic': t_stat, 'p-value': p_val
            })
    df_stats_baseline = pd.DataFrame(stats_vs_baseline)
    significant_baseline_activations = df_stats_baseline[df_stats_baseline['p-value'] < 0.05]
    
    # --- 5. PERFORM BETWEEN-GROUP STATS & EXPORT DETAILED RESULTS ---
    print("\n--- Performing between-group tests and exporting detailed results ---")
    stats_between_groups = []
    for (prof1, prof2) in combinations(profile_map.keys(), 2):
        for _, row in df_all_subjects[['Region', 'Window', 'Band']].drop_duplicates().iterrows():
            region, window, band = row['Region'], row['Window'], row['Band']
            group1 = df_all_subjects[(df_all_subjects['Profile'] == prof1) & (df_all_subjects['Region'] == region) & (df_all_subjects['Window'] == window) & (df_all_subjects['Band'] == band)]['PowerChange']
            group2 = df_all_subjects[(df_all_subjects['Profile'] == prof2) & (df_all_subjects['Region'] == region) & (df_all_subjects['Window'] == window) & (df_all_subjects['Band'] == band)]['PowerChange']
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
                stats_between_groups.append({
                    'Comparison': f"{prof1}_vs_{prof2}", 'Region': region, 'Window': window, 'Band': band,
                    'Mean_1': group1.mean(), 'Mean_2': group2.mean(), 'T-statistic': t_stat, 'p-value': p_val
                })
    df_stats_groups = pd.DataFrame(stats_between_groups)

    # Save detailed stats to a text file (CSV)
    detailed_stats_fname = op.join(output_dir, 'detailed_statistics.csv')
    pd.concat([df_stats_baseline.drop(columns=['Profile']), df_stats_groups]).to_csv(detailed_stats_fname, index=False, float_format='%.4f')
    print(f"  - Detailed statistics saved to {detailed_stats_fname}")

    # --- 6. GENERATE PLOTS FOR SIGNIFICANT BASELINE-COMPARED ACTIVATIONS ---
    print("\n--- Generating plots for conditions with significant baseline-compared activations ---")
    brain_plots_dir = op.join(output_dir, 'brain_plots')
    if not op.exists(brain_plots_dir):
        os.makedirs(brain_plots_dir)

    # Define a custom progressive colormap
    colors = [
        (0.0, 'white'), (0.4, 'white'), (0.55, 'yellow'),
        (0.75, 'orange'), (0.9, 'red'), (1.0, 'darkred')
    ]
    custom_hot_cmap = LinearSegmentedColormap.from_list('custom_hot_progressive', colors)

    for profile_name, stcs_by_prof in all_stc_results.items():
        for band, stcs_by_band in stcs_by_prof.items():
            for window_name, stc_list in stcs_by_band.items():
                # FIX: Plot only if the condition has a significant difference from baseline
                has_significant_activation = not significant_baseline_activations[
                    (significant_baseline_activations['Profile'] == profile_name) &
                    (significant_baseline_activations['Band'] == band) &
                    (significant_baseline_activations['Window'] == window_name)
                ].empty

                if not stc_list or not has_significant_activation:
                    continue
                
                print(f"  - Plotting: {profile_name} - {band} - {window_name} (significant vs baseline)")
                grand_average_stc = sum(stc_list) / len(stc_list)
                
                local_max = grand_average_stc.data.max()
                local_min = grand_average_stc.data.min()
                local_abs_max = max(abs(local_max), abs(local_min))
                local_lim = np.ceil(local_abs_max / 10) * 10 if local_abs_max > 0 else 10

                norm, ticks = None, None
                if local_min >= 0:
                    colormap = custom_hot_cmap
                    clim = dict(kind='value', lims=[0, local_lim / 2, local_lim])
                    norm = Normalize(vmin=0, vmax=local_lim)
                    ticks = [0, local_lim]
                elif local_max <= 0:
                    colormap = 'Blues_r'
                    clim = dict(kind='value', lims=[-local_lim, -local_lim / 2, 0])
                    norm = Normalize(vmin=-local_lim, vmax=0)
                    ticks = [-local_lim, 0]
                else:
                    colormap = 'RdBu_r'
                    clim = dict(kind='value', lims=[-local_lim, 0, local_lim])
                    norm = TwoSlopeNorm(vmin=-local_lim, vcenter=0, vmax=local_lim)
                    ticks = [-local_lim, local_lim]

                # --- Plot 1: Brain (no colorbar) ---
                safe_profile_name = profile_name.replace(' ', '_').replace('/', '_')
                brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, surf='pial', hemi='split', views=['lat', 'med'], size=(800, 400), background='white', view_layout='horizontal')
                brain.add_data(grand_average_stc, colormap=colormap, clim=clim, colorbar=False, hemi='lh')
                brain.add_data(grand_average_stc, colormap=colormap, clim=clim, colorbar=False, hemi='rh')
                fig_fname = op.join(brain_plots_dir, f'{safe_profile_name}_{band}_{window_name}.png')
                brain.save_image(fig_fname)
                brain.close()

                # --- Plot 2: Standalone Colorbar ---
                cbar_fig = plt.figure(figsize=(8, 0.5))
                cbar_ax = cbar_fig.add_axes([0.25, 0.4, 0.5, 0.2])
                cbar = ColorbarBase(cbar_ax, cmap=colormap, norm=norm, orientation='horizontal')
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
                cbar_fname = op.join(brain_plots_dir, f'{safe_profile_name}_{band}_{window_name}_colorbar.png')
                cbar_fig.savefig(cbar_fname, transparent=True, dpi=100)
                plt.close(cbar_fig)

                # --- Add brain plot to report ---
                tmin_win, tmax_win = active_windows[window_name]
                report_title = f'{profile_name} - {band.upper()} - {window_name} [{tmin_win} {tmax_win}]'
                report.add_image(image=fig_fname, title=report_title, section='Brain Plots')

    # --- 7. ADD METHODOLOGY AND GENERATE SUMMARY TABLES ---
    print("\n--- Adding methodology and generating summary tables ---")
    # FIX: Update explanation to reflect baseline significance
    explanation_html = """
    <h3>Methodology and Interpretation (Alpha/Beta Power)</h3>
    <p>The values in the table represent the percentage change in power of the <strong>Loaded (LD) vs. Spontaneous (VS)</strong> difference, relative to a VS baseline period (-2.0s to -1.5s).</p>
    <p><code>% Change = 100 * ( (Power_LD_active - Power_VS_active) / Power_VS_baseline )</code></p>
    <p>A <strong>negative value</strong> (e.g., -40%) indicates power suppression (desynchronization). A <strong>positive value</strong> indicates power increase (synchronization).</p>
    <p>The asterisk <strong>(*)</strong> indicates a statistically significant difference (p < 0.05) of that activity compared to its own baseline (i.e., significantly different from zero).</p>
    """
    report.add_html(explanation_html, title="Analysis Methods", section="Summary Tables")

    # Create a set for quick lookup of significant baseline results
    significant_baseline_results = set(
        tuple(row) for row in significant_baseline_activations[['Profile', 'Region', 'Window', 'Band']].to_numpy()
    )

    for band in freq_bands.keys():
        df_band = df_all_subjects[df_all_subjects['Band'] == band]
        if df_band.empty: continue
        report.add_html(html=f"<h2>Results for {band.upper()} Band</h2>", title=f"{band.upper()} Band Results", section='Summary Tables')
        for profile_name in profile_map.keys():
            df_profile = df_band[df_band['Profile'] == profile_name]
            if df_profile.empty: continue
            
            pivot_table = df_profile.pivot_table(index='Region', columns='Window', values='PowerChange', aggfunc='mean')
            
            # FIX: Apply asterisk for significance vs baseline
            def add_asterisk(val, prof, region, window, band):
                if pd.isna(val): return ""
                is_significant = (prof, region, window, band) in significant_baseline_results
                return f"{val:.1f}*" if is_significant else f"{val:.1f}"
            
            formatted_table = pivot_table.copy()
            for region in formatted_table.index:
                for window in formatted_table.columns:
                    val = formatted_table.loc[region, window]
                    formatted_table.loc[region, window] = add_asterisk(val, profile_name, region, window, band)
            
            html_table = formatted_table.to_html(classes='table table-striped text-center', justify='center')
            report.add_html(html_table, title=f"Mean Power Change (%) for {profile_name} - {band.upper()} Band", section='Summary Tables')

    # --- 8. SAVE THE FINAL REPORT ---
    report_fname = op.join(output_dir, 'source_analysis_alpha_beta_report.html')
    report.save(report_fname, overwrite=True, open_browser=False)
    print(f"\n--- Complete report saved to {report_fname} ---")
    print("\n--- Analysis complete. ---")