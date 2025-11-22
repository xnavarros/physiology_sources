# 0_preprocess_slow_potentials.py
# This script is specifically tailored for analyzing slow potentials like the readiness potential.
# It uses a very low high-pass filter and a long pre-stimulus epoch window.

import os
import os.path as op
import mne
import numpy as np
import yaml
import pandas as pd
from mne_icalabel import label_components
from autoreject import AutoReject 
from joblib import Parallel, delayed

def load_config(config_path=None):
    """Loads the configuration file."""
    if config_path is None:
        # Construct path relative to this script's location
        script_dir = op.dirname(op.abspath(__file__))
        # The config file is in the parent directory of the 'scripts' folder
        root_dir = op.dirname(script_dir)
        config_path = op.join(root_dir, 'config', 'config.yaml')
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_prepare_raw(data_fname, params):
    """
    Loads raw data, performs filtering suitable for slow potentials, and segments data.
    """
    print("Step 1: Loading and preparing raw data for SLOW POTENTIAL analysis...")
    raw = mne.io.read_raw_brainvision(data_fname + '.vhdr', preload=True)
    raw.set_channel_types({'EOG': 'eog', 'ECG': 'ecg', 'PS': 'emg', 'Pression': 'resp', 'Debit': 'resp'})
    
    # --- MODIFIED FILTERING FOR SLOW POTENTIALS ---
    print(f"Applying filter for slow potentials: {params['l_freq_slow']} - {params['h_freq_slow']} Hz")
    filt_raw = raw.copy().filter(l_freq=params['l_freq_slow'], h_freq=params['h_freq_slow'])
    if params['apply_notch']:
        filt_raw.notch_filter(freqs=np.arange(50, 251, 50), picks='eeg', method='spectrum_fit', filter_length='5s', trans_bandwidth=2)
    
    # --- Referencing Scheme (made robust for all subjects) ---
    print("Applying common average reference...")
    if 'A2' not in filt_raw.ch_names:
        print("  - 'A2' not found. Adding it as a reference channel.")
        filt_raw.add_reference_channels(ref_channels=['A2'])
    else:
        print("  - 'A2' already exists as a data channel. Proceeding.")
    filt_raw.set_eeg_reference(ref_channels='average')
    filt_raw.set_montage('standard_1020', on_missing="ignore")

    # --- Event Extraction (remains the same logic) ---
    events, event_id = mne.events_from_annotations(raw)
    try:
        t_endVS = events[events[:, 2] == event_id["Comment/endVS"], 0][0] / raw.info['sfreq']
        t_startLD = events[events[:, 2] == event_id["Comment/startLD"], 0][0] / raw.info['sfreq']
        t_endLD = events[events[:, 2] == event_id["Comment/endLD"], 0][0] / raw.info['sfreq']
    except (IndexError, KeyError) as e:
        print(f"Error finding event markers: {e}. Please check your annotations.")
        return None, None, None, None, None # Return 5 values to prevent crash

    raw_VS = filt_raw.copy().crop(tmax=t_endVS)
    raw_LD = filt_raw.copy().crop(tmin=t_startLD, tmax=t_endLD)

    start_samp_vs = raw_VS.first_samp
    end_samp_vs = raw_VS.last_samp
    start_samp_ld = raw_LD.first_samp
    end_samp_ld = raw_LD.last_samp

    events_VS = events[(events[:, 0] >= start_samp_vs) & (events[:, 0] <= end_samp_vs)]
    events_LD = events[(events[:, 0] >= start_samp_ld) & (events[:, 0] <= end_samp_ld)]

    events_VS[:, 0] -= start_samp_vs
    events_LD[:, 0] -= start_samp_ld
    
    # --- MODIFICATION: Use the correct "original" marker name from the data ---
    inspiration_marker_name = 'Response/R128' 
    if inspiration_marker_name not in event_id:
        print(f"FATAL ERROR: Inspiration marker '{inspiration_marker_name}' not found in data annotations.")
        return None, None, None, None, None # Return 5 values to prevent crash
        
    print("Raw data and events correctly segmented.")
    return raw_VS, raw_LD, events_VS, events_LD, event_id


def run_preprocessing_pipeline(raw_VS, raw_LD, events_VS, events_LD, event_id, params, ar_params):
    """
    Runs the full preprocessing pipeline for slow potentials.
    """
    log_info = {}
    print("\nStep 2: Concatenating VS and LD conditions...")
    split_point_samples = len(raw_VS.times)
    raw_combined = mne.concatenate_raws([raw_VS, raw_LD])
    events_LD[:, 0] += split_point_samples
    events_combined = np.concatenate((events_VS, events_LD))
    
    print("\nStep 3: Interpolating bad channels (placeholder)...")
    raw_combined.info['bads'] = []
    # --- FIX: Initialize the log entry for interpolated channels ---
    log_info['interpolated_channels'] = "N/A" # Or [] if you prefer
    
    # --- MODIFIED EPOCHING FOR SLOW POTENTIALS ---
    print("\nStep 4: Creating long epochs around inspiration marker...")
    # --- MODIFICATION: Use the correct "original" marker name from the data ---
    inspiration_marker_name = 'Response/R128'
    epochs = mne.Epochs(raw_combined, events_combined, event_id={inspiration_marker_name: event_id[inspiration_marker_name]}, 
                        tmin=params['tmin_slow'], tmax=params['tmax_slow'],
                        baseline=None, preload=True) # Baseline is applied AFTER cleaning
    log_info['initial_epochs_VS'] = int((epochs.events[:, 0] < split_point_samples).sum())
    log_info['initial_epochs_LD'] = int((epochs.events[:, 0] >= split_point_samples).sum())

    # --- Step 5: ICA artifact removal (using the "Filter-Fit-Apply" method) ---
    if params['apply_ica']:
        print("\nStep 5: Performing ICA with ICALabel and Respiratory Channels...")
        
        # 1. Create a temporary copy of the data filtered for ICALabel (1-100 Hz)
        print("  - Creating temporary 1-100Hz filtered data for ICA fitting...")
        epochs_for_ica = epochs.copy().filter(l_freq=1.0, h_freq=100.0)
        
        # --- FIX: Fit ICA on EEG channels ONLY ---
        picks_for_ica = mne.pick_types(epochs_for_ica.info, eeg=True) # <-- This is the fix
        ica = mne.preprocessing.ICA(n_components=params['n_ica_components'], method="infomax", random_state=97, fit_params=dict(extended=True))
        ica.fit(epochs_for_ica, picks=picks_for_ica)

        # 2. Find artifact components using ICALabel (for EOG, ECG)
        ic_labels = label_components(epochs_for_ica, ica, method='iclabel')
        exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]
        
        # --- NEW: Find artifact components by correlating with respiratory channels ---
        print("  - Finding components correlated with respiratory signals...")
        resp_channels = ['Debit', 'Pression']
        for ch_name in resp_channels:
            if ch_name in epochs.ch_names:
                # Find correlation between ICs and the respiratory channel
                inds, scores = ica.find_bads_eog(epochs, ch_name=ch_name, threshold=3.0) # Using find_bads_eog as a generic correlator
                if inds:
                    print(f"    - Found {len(inds)} components correlated with {ch_name}.")
                    exclude_idx.extend(inds)

        # Remove duplicates and sort
        exclude_idx = sorted(list(set(exclude_idx)))
        log_info['rejected_ica_components'] = len(exclude_idx)
        
        # 3. Apply the ICA solution to the original (0.1-40Hz) data
        print(f"  - Applying ICA solution to the original slow-potential data (removing {len(exclude_idx)} components)...")
        ica.apply(epochs, exclude=exclude_idx)
    else:
        log_info['rejected_ica_components'] = 'N/A'

    # --- STEP 6: AutoReject (unchanged) ---
    print("\nStep 6: Cleaning epochs with AutoReject...")
    ar = AutoReject(n_interpolate=ar_params['n_interpolate'], consensus=ar_params['consensus'], random_state=42, n_jobs=-1, verbose='tqdm')
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    print(f"AutoReject finished. {len(reject_log.bad_epochs)} epochs were rejected.")

    # --- NEW STEP: Apply Baseline Correction ---
    print("\nApplying baseline correction for slow potential analysis...")
    epochs_clean.apply_baseline(baseline=tuple(params['baseline_timing_slow']))

    # --- Step 7: Split epochs and finalize logs ---
    print("\nStep 7: Splitting cleaned epochs back into VS and LD conditions...")
    epochs_VS_clean = epochs_clean[epochs_clean.events[:, 0] < split_point_samples]
    epochs_LD_clean = epochs_clean[epochs_clean.events[:, 0] >= split_point_samples]
    
    log_info['final_epochs_VS'] = len(epochs_VS_clean)
    log_info['final_epochs_LD'] = len(epochs_LD_clean)
    log_info['rejected_epochs_autoreject_VS'] = log_info['initial_epochs_VS'] - log_info['final_epochs_VS']
    log_info['rejected_epochs_autoreject_LD'] = log_info['initial_epochs_LD'] - log_info['final_epochs_LD']
    
    print(f"Final clean epochs: {len(epochs_VS_clean)} for VS, {len(epochs_LD_clean)} for LD.")
    return epochs_VS_clean, epochs_LD_clean, log_info

def process_subject(subject, params, ar_params):
    """
    A wrapper function to process a single subject.
    This function can be called in parallel.
    """
    print("\n" + "="*50)
    print(f"Processing subject: {subject}")
    print("="*50)
    
    # Define file paths
    data_fname = op.join(params['base_data_dir'], subject, f"{subject}_CONTINU_64Ch_A2Ref")
    output_dir = params['preprocessed_slow_potentials_dir']
    
    try:
        # Load and prepare raw data
        raw_VS, raw_LD, events_VS, events_LD, event_id = load_and_prepare_raw(data_fname, params)
        
        # If data loading fails, return an error log
        if raw_VS is None:
            raise ValueError("Failed during data loading and preparation.")

        # Run the main preprocessing pipeline
        epochs_VS_clean, epochs_LD_clean, log_info = run_preprocessing_pipeline(
            raw_VS, raw_LD, events_VS, events_LD, event_id, params, ar_params
        )
        
        # Save the cleaned epochs
        epochs_VS_clean.save(op.join(output_dir, f"{subject}_VS-slow-epo.fif"), overwrite=True)
        epochs_LD_clean.save(op.join(output_dir, f"{subject}_LD-slow-epo.fif"), overwrite=True)
        
        # Add subject info to the log and return
        log_info['subject'] = subject
        return log_info

    except Exception as e:
        print(f"!!! FAILED to process subject {subject}. Error: {e}")
        # Log the failure and return
        return {'subject': subject, 'status': 'FAILED', 'error': str(e)}

if __name__ == "__main__":
    config = load_config()
    subjects = config['subjects']
    params = config['preprocessing_params']
    ar_params = config['autoreject_params']

    # --- FIX: Make all paths absolute from the project root ---
    script_dir = op.dirname(op.abspath(__file__))
    root_dir = op.dirname(script_dir)
    
    # Update paths in the config to be absolute
    for key, path_val in config['paths'].items():
        config['paths'][key] = op.join(root_dir, path_val)
    
    params.update(config['paths'])
    # --- END FIX ---

    print("--- OVERRIDING EPOCH AND BASELINE PARAMETERS ---")
    params['tmin_slow'] = -2.0
    params['tmax_slow'] = 0.5
    params['baseline_timing_slow'] = (-2.0, -1.5)

    print(f"  - New epoch time: {params['tmin_slow']}s to {params['tmax_slow']}s")
    print(f"  - New baseline: {params['baseline_timing_slow'][0]}s to {params['baseline_timing_slow'][1]}s")

    output_dir = params['preprocessed_slow_potentials_dir']
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # --- PARALLEL PROCESSING ---
    # Use joblib to run the processing for each subject in parallel.
    # n_jobs=-1 tells joblib to use all available CPU cores.
    print("\n--- Starting Parallel Processing for All Subjects ---")
    all_logs = Parallel(n_jobs=-1)(
        delayed(process_subject)(subject, params, ar_params) for subject in subjects
    )
    # --- END PARALLEL PROCESSING ---

    # Save the logs to a CSV file
    log_df = pd.DataFrame(all_logs)
    log_df.to_csv(op.join(output_dir, 'preprocessing_log_slow_potentials.csv'), index=False)
    
    print("\n--- Preprocessing complete for all subjects. ---")