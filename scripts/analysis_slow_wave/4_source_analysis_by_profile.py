import os
import os.path as op
import json
import mne
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind
import seaborn as sns

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
    Main execution block for source analysis based on subject profiles.
    
    This script performs the following steps:
    1.  Loads configuration, subject lists, and defines profile groups.
    2.  Loads pre-processed evoked data for each subject and a common forward model.
    3.  Computes a common LCMV spatial filter from the grand average of all subjects.
    4.  For each subject, it calculates Z-scored activity in predefined ROIs.
    5.  Performs t-tests to compare ROI Z-scores between profile groups.
    6.  Generates and saves a comprehensive HTML report containing:
        a. 3D brain plots of grand-averaged source activity for each profile.
        b. An explanation of the analysis methods.
        c. Summary tables of mean Z-scores with significance markers.
    """

    # --- 1. CONFIGURATION AND SETUP ---
    print("--- Setting up Source Analysis by Profile ---")
    config = load_config()
    subjects = config['subjects']

    # --- FIX: Make all paths absolute from the project root ---
    script_dir = op.dirname(op.abspath(__file__))
    root_dir = op.dirname(op.dirname(script_dir))
    
    # Update paths in the config to be absolute
    for key, path_val in config['paths'].items():
        config['paths'][key] = op.join(root_dir, path_val)
    
    preprocessed_dir = config['paths']['preprocessed_slow_potentials_dir']
    output_dir = op.join(root_dir, 'figures', 'source_analysis_profiles')
    # --- END FIX ---
    
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    report = mne.Report(title='Source Analysis Report by Profile', verbose=False)

    # --- Load profiles from JSON ---
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

    # --- 2. LOAD MODELS AND DATA ---
    print("\n--- Loading common fsaverage models and subject data ---")
    try:
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        fwd_fname = op.join(subjects_dir, "fsaverage-fwd.fif")
        fwd = mne.read_forward_solution(fwd_fname)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find model file: {e}"); exit()

    all_evokeds = {}
    for subject in subjects:
        try:
            vs_epochs_fname = op.join(preprocessed_dir, f"{subject}_VS-slow-epo.fif")
            ld_epochs_fname = op.join(preprocessed_dir, f"{subject}_LD-slow-epo.fif")
            epochs_vs = mne.read_epochs(vs_epochs_fname, preload=True, verbose=False)
            epochs_ld = mne.read_epochs(ld_epochs_fname, preload=True, verbose=False)
            all_evokeds[subject] = mne.combine_evoked([epochs_vs.average(), epochs_ld.average()], weights=[1, -1])
        except FileNotFoundError:
            continue

    # --- 3. COMPUTE LCMV SPATIAL FILTERS ---
    print("\n--- Computing LCMV spatial filters from grand average ---")
    grand_average_all = mne.grand_average(list(all_evokeds.values()))
    grand_average_all.set_eeg_reference('average', projection=True)
    epochs_for_cov = mne.EpochsArray(grand_average_all.data[None], grand_average_all.info, tmin=grand_average_all.tmin)
    data_cov = mne.compute_covariance(epochs_for_cov, tmin=-0.5, tmax=0.5, method='shrunk', rank=None)
    filters = mne.beamformer.make_lcmv(grand_average_all.info, fwd, data_cov, reg=0.025, pick_ori='max-power', weight_norm='unit-noise-gain')

    # --- 4. PROCESS EACH SUBJECT INDIVIDUALLY TO GET ROI Z-SCORES ---
    print("\n--- Processing each subject to extract ROI Z-scores ---")
    all_subject_results = []
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
    
    # --- Load ROIs and Windows from config ---
    roi_names = config.get('rois', ['precentral', 'postcentral', 'caudalmiddlefrontal', 'superiorfrontal', 'insula', 'superiortemporal'])
    active_windows = config.get('analysis_windows', {"Early": (-1.5, -1.0), "Mid": (-1.0, -0.5), "Late": (-0.5, 0.0), "Post": (0.0, 0.5)})
    
    selected_labels = [lbl for lbl in labels if any(name in lbl.name for name in roi_names)]
    baseline_window = (-2.0, -1.0) # This could also be in config, but let's stick to the requested changes for now


    for subject, evoked in all_evokeds.items():
        evoked.set_eeg_reference('average', projection=True)
        stc_subject = mne.beamformer.apply_lcmv(evoked, filters)
        stc_baseline = stc_subject.copy().crop(tmin=baseline_window[0], tmax=baseline_window[1])
        stc_baseline_abs = mne.SourceEstimate(data=np.abs(stc_baseline.data), vertices=stc_baseline.vertices, tmin=stc_baseline.tmin, tstep=stc_baseline.tstep, subject=stc_baseline.subject)
        baseline_tc = stc_baseline_abs.extract_label_time_course(selected_labels, src=fwd['src'], mode='mean')
        baseline_means = np.mean(baseline_tc, axis=1)
        baseline_stds = np.std(baseline_tc, axis=1)
        for i, label in enumerate(selected_labels):
            for window_name, (tmin, tmax) in active_windows.items():
                stc_mean_window = stc_subject.copy().crop(tmin=tmin, tmax=tmax).mean()
                stc_abs = mne.SourceEstimate(data=np.abs(stc_mean_window.data), vertices=stc_mean_window.vertices, tmin=stc_mean_window.tmin, tstep=stc_mean_window.tstep, subject=stc_mean_window.subject)
                activity = stc_abs.extract_label_time_course(label, src=fwd['src'], mode='mean')[0, 0]
                z_score = (activity - baseline_means[i]) / baseline_stds[i] if baseline_stds[i] > 1e-12 else 0.0
                all_subject_results.append({"Subject": subject, "Profile": subject_to_profile[subject], "Region": label.name, "Window": window_name, "Z-Score": z_score})

    df_all_subjects = pd.DataFrame(all_subject_results)
    
    # --- 5. PERFORM STATISTICAL TESTS BETWEEN GROUPS ---
    print("\n--- Performing statistical tests between profile groups ---")
    significant_results = set()
    comparisons = df_all_subjects[['Region', 'Window']].drop_duplicates()
    for (prof1, prof2) in combinations(profile_map.keys(), 2):
        for _, row in comparisons.iterrows():
            region, window = row['Region'], row['Window']
            group1_scores = df_all_subjects[(df_all_subjects['Profile'] == prof1) & (df_all_subjects['Region'] == region) & (df_all_subjects['Window'] == window)]['Z-Score']
            group2_scores = df_all_subjects[(df_all_subjects['Profile'] == prof2) & (df_all_subjects['Region'] == region) & (df_all_subjects['Window'] == window)]['Z-Score']
            if len(group1_scores) > 1 and len(group2_scores) > 1:
                stat, p_val = ttest_ind(group1_scores, group2_scores, equal_var=False)
                if p_val < 0.05:
                    significant_results.add((prof1, region, window))
                    significant_results.add((prof2, region, window))

    # --- 6. GENERATE 3D SOURCE PLOTS AND ADD TO REPORT ---
    print("\n--- Generating 3D source plots for each profile ---")
    for profile_name, subject_list in profile_map.items():
        profile_evokeds = [all_evokeds[subj] for subj in subject_list if subj in all_evokeds]
        if not profile_evokeds: continue
        
        profile_grand_average = mne.grand_average(profile_evokeds)
        profile_grand_average.set_eeg_reference('average', projection=True)
        stc = mne.beamformer.apply_lcmv(profile_grand_average, filters)
        
        significant_region_names = {region for prof, region, window in significant_results if prof == profile_name}
        significant_labels = [lbl for lbl in selected_labels if lbl.name in significant_region_names]
        
        if not significant_labels:
            print(f"  - No significant ROIs to plot for {profile_name}. Skipping 3D plot.")
            continue

        n_lh_verts = len(stc.vertices[0])
        mask = np.zeros(len(stc.data), dtype=bool)
        for label in significant_labels:
            if label.hemi == 'lh':
                idx = np.searchsorted(stc.vertices[0], label.vertices)
                mask[idx] = True
            elif label.hemi == 'rh':
                idx = np.searchsorted(stc.vertices[1], label.vertices)
                mask[idx + n_lh_verts] = True
        
        stc_masked_data = stc.data * mask[:, np.newaxis]
        stc_masked = mne.SourceEstimate(stc_masked_data, vertices=stc.vertices, tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')

        _, peak_time = stc_masked.copy().crop(tmin=-0.5, tmax=0.5).get_peak(mode='abs', time_as_index=False)
        
        clim = dict(kind='value', center=0.0)
        
        # CHANGE: Use 'pial' surface for a more realistic brain model
        brain = mne.viz.Brain('fsaverage', hemi='split', surf='pial', subjects_dir=subjects_dir, size=(800, 400), views=['lat', 'med'])
        
        if np.any(stc_masked.lh_data):
            brain.add_data(stc_masked.lh_data, vertices=stc_masked.vertices[0], hemi='lh', colormap='mne', smoothing_steps=10, time=stc_masked.times, time_label=None, clim=clim, colorbar_kwargs={'label_font_size': 8})
        if np.any(stc_masked.rh_data):
            brain.add_data(stc_masked.rh_data, vertices=stc_masked.vertices[1], hemi='rh', colormap='mne', smoothing_steps=10, time=stc_masked.times, time_label=None, clim=clim, colorbar_kwargs={'label_font_size': 8})

        brain.set_time(peak_time)
        
        fig_fname = op.join(output_dir, f"temp_source_plot_{profile_name.replace(' ', '_')}.png")
        brain.save_image(fig_fname)
        brain.close()

        report.add_image(image=fig_fname, title=f"Significant Source Activity for {profile_name}", section='Group Source Plots')

    # --- 7. ADD METHODOLOGY EXPLANATION TO REPORT ---
    print("\n--- Adding methodology explanation to report ---")
    explanation_html = """
    <h3>Methodology and Interpretation</h3>
    <p>The following tables summarize the normalized brain activity (Z-scores) for each profile group within specific Regions of Interest (ROIs) and time windows.</p>
    <h4>Baseline Calculation</h4>
    <p>The baseline activity for each ROI was calculated from the pre-stimulus period of <strong>-2.0s to -1.0s</strong>. The mean and standard deviation of the absolute source power within this period were computed for each subject's ROI.</p>
    <h4>Z-Score Calculation</h4>
    <p>The Z-score for an active window was then computed using the formula: <code>Z = (activity_in_window - baseline_mean) / baseline_standard_deviation</code>.</p>
    <h4>Interpretation</h4>
    <p>A Z-score represents how many standard deviations an ROI's activity is from its own baseline average. For example, a Z-score of 2.0 means the activity is 2 standard deviations <em>higher</em> than its baseline. A value near 0 indicates activity similar to baseline.</p>
    <p>The asterisk <strong>(*)</strong> next to a Z-score indicates that this value is part of a statistically significant difference (p < 0.05) when compared to at least one other profile group in the same region and time window.</p>
    """
    report.add_html(explanation_html, title="Analysis Methods", section="Summary Tables")

    # --- 8. GENERATE SUMMARY TABLES AND ADD TO REPORT ---
    print("\n--- Generating final summary tables ---")
    for profile_name in profile_map.keys():
        df_profile = df_all_subjects[df_all_subjects['Profile'] == profile_name]
        pivot_z_score = df_profile.pivot_table(index='Region', columns='Window', values='Z-Score', aggfunc='mean')
        def add_asterisk(val, prof, region, window):
            return f"{val:.2f}*" if (prof, region, window) in significant_results else f"{val:.2f}"
        formatted_table = pivot_z_score.copy()
        for region in formatted_table.index:
            for window in formatted_table.columns:
                val = formatted_table.loc[region, window]
                formatted_table.loc[region, window] = add_asterisk(val, profile_name, region, window)
        
        # --- THESE LINES PRINT THE TABLES TO YOUR CONSOLE ---
        print(f"\n--- Mean Z-Scores for {profile_name} (* indicates significant difference from another group) ---")
        print(formatted_table)
        
        html_table = formatted_table.to_html(classes='table table-striped text-center', justify='center')
        report.add_html(html_table, title=f"Mean Z-Scores for {profile_name}", section='Summary Tables')

    # --- 9. SAVE THE FINAL REPORT ---
    report_fname = op.join(output_dir, 'source_analysis_report.html')
    report.save(report_fname, overwrite=True, open_browser=False)
    print(f"\n--- Complete report saved to {report_fname} ---")
    print("\n--- Analysis complete. ---")