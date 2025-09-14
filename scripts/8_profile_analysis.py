import os
import os.path as op
import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
from mne import Report

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # --- 1. CONFIGURATION AND SETUP ---
    config = load_config()
    subjects = config['subjects']
    
    # Define the subject profiles based on the cross-analysis
    profiles = {
        "Co-localized Early Preparers": ['BJ25', 'MC05', 'VS06'],
        "Late Preparers": ['MN23', 'VA14', 'JS08', 'TH24'],
        "Reactive Responders": ['LP26', 'OL04', 'SB27']
    }

    # Define input and output paths
    erp_input_dir = config['paths']['preprocessed_slow_potentials_dir']
    tfr_input_dir = config['paths']['preprocessed_tfr_dir']
    output_dir = op.join(config['paths']['results_dir'], 'profile_analysis')
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the HTML Report
    report = Report(title='Subject Profile Analysis Report', verbose=False)

    print("--- Starting Subject Profile Analysis ---")

    # --- 2. LOAD ALL SUBJECT DATA (ERP & TFR) ---
    all_evoked_vs = {}
    all_evoked_ld = {}
    all_tfr_vs = {}
    all_tfr_ld = {}

    print("\n--- Loading data for all subjects ---")
    for subject in subjects:
        try:
            # Load ERP data
            evoked_vs = mne.read_evokeds(op.join(erp_input_dir, subject, f"{subject}_VS-slow-ave.fif"), verbose=False)[0]
            evoked_ld = mne.read_evokeds(op.join(erp_input_dir, subject, f"{subject}_LD-slow-ave.fif"), verbose=False)[0]
            all_evoked_vs[subject] = evoked_vs
            all_evoked_ld[subject] = evoked_ld

            # Load TFR data
            tfr_vs = mne.time_frequency.read_tfrs(op.join(tfr_input_dir, subject, f"{subject}_VS-tfr.h5"), verbose=False)[0]
            tfr_ld = mne.time_frequency.read_tfrs(op.join(tfr_input_dir, subject, f"{subject}_LD-tfr.h5"), verbose=False)[0]
            all_tfr_vs[subject] = tfr_vs
            all_tfr_ld[subject] = tfr_ld
            print(f"  - Loaded ERP and TFR data for {subject}")
        except FileNotFoundError as e:
            print(f"  - WARNING: Could not find all data for subject {subject}. Skipping. Error: {e}")
            continue

    # --- 3. COMPUTE GRAND AVERAGES AND PLOTS FOR EACH PROFILE ---
    print("\n--- Computing grand averages and generating plots for each profile ---")
    
    profile_data = {}

    for profile_name, subject_list in profiles.items():
        print(f"\n--- Processing Profile: {profile_name} ---")
        
        # Collect data for subjects in this profile
        profile_evokeds_vs = [all_evoked_vs[s] for s in subject_list if s in all_evoked_vs]
        profile_evokeds_ld = [all_evoked_ld[s] for s in subject_list if s in all_evoked_ld]
        profile_tfrs_vs = [all_tfr_vs[s] for s in subject_list if s in all_tfr_vs]
        profile_tfrs_ld = [all_tfr_ld[s] for s in subject_list if s in all_tfr_ld]

        if not profile_evokeds_vs or not profile_tfrs_vs:
            print("  - Not enough data to process this profile. Skipping.")
            continue

        # --- ERP Grand Average ---
        ga_erp_vs = mne.grand_average(profile_evokeds_vs)
        ga_erp_ld = mne.grand_average(profile_evokeds_ld)
        ga_erp_diff = mne.combine_evoked([ga_erp_vs, ga_erp_ld], weights=[1, -1])

        # --- TFR Grand Average ---
        ga_tfr_vs = mne.grand_average(profile_tfrs_vs)
        ga_tfr_ld = mne.grand_average(profile_tfrs_ld)
        ga_tfr_diff = ga_tfr_vs.copy()
        ga_tfr_diff.data = ga_tfr_vs.data - ga_tfr_ld.data
        
        profile_data[profile_name] = {'erp_diff': ga_erp_diff, 'tfr_diff': ga_tfr_diff}

    # --- 4. GENERATE REPORT ---
    print("\n--- Generating HTML Report ---")

    # Add descriptive text for the profiles
    profiling_html = """
    <h2>Subject Profile Descriptions</h2>
    <p>Based on the timing and location of significant ERP and TFR effects from previous analyses, subjects were grouped into three distinct profiles representing different neural strategies.</p>
    
    <h3>Profile 1: Co-localized Early Preparers</h3>
    <p><b>Subjects:</b> BJ25, MC05, VS06</p>
    <p><b>Characteristics:</b> These subjects exhibit the strongest preparatory strategy. They show a sustained readiness potential (ERP) and an early, specific change in beta power (TFR) in overlapping brain regions (Central/Frontal). This indicates a unified, proactive motor strategy.</p>
    
    <h3>Profile 2: Late/Motor-Focused Preparers</h3>
    <p><b>Subjects:</b> MN23, VA14, JS08, TH24</p>
    <p><b>Characteristics:</b> These subjects delay their specific motor preparation. Their key oscillatory changes (beta power decrease) only appear in the mid-to-late preparatory windows, suggesting a "just-in-time" ramping up of motor-related activity.</p>
    
    <h3>Profile 3: Reactive Responders</h3>
    <p><b>Subjects:</b> LP26, OL04, SB27</p>
    <p><b>Characteristics:</b> This group is defined by a lack of significant preparatory activity. Brain activity differences only appear *after* the action is complete, suggesting their response is to the sensory feedback of the sigh itself, not in anticipation of it.</p>
    """
    report.add_html(html=profiling_html, title='Profile Descriptions', section='Profiles')

    # Generate and add comparison plots to the report
    # --- ERP Comparison Plot ---
    fig_erp, axes_erp = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig_erp.suptitle("Profile Comparison: Grand Average ERP Difference (VS - LD) at Cz", fontsize=16)
    for i, (profile_name, data) in enumerate(profile_data.items()):
        data['erp_diff'].plot(picks='Cz', axes=axes_erp[i], show=False, spatial_colors=True, gfp=False)
        axes_erp[i].set_title(profile_name)
        axes_erp[i].axvline(0, color='r', linestyle='--')
    axes_erp[0].invert_yaxis()
    report.add_figure(fig=fig_erp, title='ERP Difference Wave Comparison', section='Comparison Plots', tags=('erp', 'comparison'))
    plt.close(fig_erp)

    # --- TFR Comparison Plot (Beta Band) ---
    fig_tfr, axes_tfr = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig_tfr.suptitle("Profile Comparison: Grand Average TFR Difference (VS - LD) over Cz (Beta Band)", fontsize=16)
    # Find common color scale limits for TFR plots
    vmin = min(d['tfr_diff'].data.min() for d in profile_data.values())
    vmax = max(d['tfr_diff'].data.max() for d in profile_data.values())
    
    for i, (profile_name, data) in enumerate(profile_data.items()):
        data['tfr_diff'].plot(picks='Cz', fmin=13, fmax=30, tmin=-1.5, tmax=0.5, 
                              mode='logratio', axes=axes_tfr[i], show=False, 
                              colorbar=False, vmin=vmin, vmax=vmax)
        axes_tfr[i].set_title(profile_name)
    
    # Add a single colorbar
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig_tfr.colorbar(sm, ax=axes_tfr, shrink=0.8, orientation='vertical', label='Power (log ratio)')
    report.add_figure(fig=fig_tfr, title='TFR Difference Comparison (Beta)', section='Comparison Plots', tags=('tfr', 'comparison', 'beta'))
    plt.close(fig_tfr)

    # Save the final report
    report_fname = op.join(output_dir, "report_profile_analysis.html")
    report.save(report_fname, overwrite=True, open_browser=False)

    print(f"\n--- Analysis Complete. Profile report saved to {report_fname} ---")