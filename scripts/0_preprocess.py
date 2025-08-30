# preprocessing_pipeline.py

import os
import os.path as op
import mne
import numpy as np
import yaml
import pandas as pd
from mne_icalabel import label_components
from autoreject import AutoReject 

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_and_prepare_raw(data_fname, params):
    """
    Loads raw data, performs initial filtering, and correctly segments data and events
    into VS and LD conditions.
    """
    print("Step 1: Loading and preparing raw data...")
    raw = mne.io.read_raw_brainvision(data_fname + '.vhdr', preload=True)
    raw.set_channel_types({'EOG': 'eog', 'ECG': 'ecg', 'PS': 'emg', 'Pression': 'resp', 'Debit': 'resp'})
    raw.set_montage('standard_1020', on_missing="ignore")

    # Perform initial filtering on the whole raw object
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
    if params['apply_notch']:
        print("Applying notch filter...")
        filt_raw.notch_filter(freqs=np.arange(50, 251, 50), picks='eeg', method='spectrum_fit', filter_length='5s', trans_bandwidth=2)
    filt_raw = filt_raw.set_eeg_reference("average", projection=True)
    filt_raw.apply_proj()

    # --- CORRECTED EVENT EXTRACTION LOGIC ---
    # 1. Get all events and the one true event_id mapping from the ENTIRE recording
    events, event_id = mne.events_from_annotations(raw)

    # 2. Get the start/end times (in seconds) for the conditions
    try:
        t_endVS = events[events[:, 2] == event_id["Comment/endVS"], 0][0] / raw.info['sfreq']
        t_startLD = events[events[:, 2] == event_id["Comment/startLD"], 0][0] / raw.info['sfreq']
        t_endLD = events[events[:, 2] == event_id["Comment/endLD"], 0][0] / raw.info['sfreq']
    except (IndexError, KeyError) as e:
        print(f"Error finding event markers: {e}. Please check your annotations.")
        return None, None, None, None

    # 3. Crop the continuous raw data
    raw_VS = filt_raw.copy().crop(tmax=t_endVS)
    raw_LD = filt_raw.copy().crop(tmin=t_startLD, tmax=t_endLD)

    # 4. Filter the original events array based on the time windows of the new cropped files
    # MNE's crop function adjusts the first_samp attribute, which we use to find the new sample boundaries
    start_samp_vs = raw_VS.first_samp
    end_samp_vs = raw_VS.last_samp
    start_samp_ld = raw_LD.first_samp
    end_samp_ld = raw_LD.last_samp

    events_VS = events[(events[:, 0] >= start_samp_vs) & (events[:, 0] <= end_samp_vs)]
    events_LD = events[(events[:, 0] >= start_samp_ld) & (events[:, 0] <= end_samp_ld)]

    # 5. Adjust event sample numbers to be relative to the start of their respective cropped files
    # This is critical for mne.Epochs to work correctly on the cropped raw objects
    events_VS[:, 0] -= start_samp_vs
    events_LD[:, 0] -= start_samp_ld
    
    print(f"Found {len(events_LD[events_LD[:, 2] == event_id['Response/R128']])} 'Response/R128' triggers in the LD period.")
    print("Raw data and events correctly segmented into VS and LD conditions.")
    return raw_VS, raw_LD, events_VS, events_LD


def run_preprocessing_pipeline(raw_VS, raw_LD, events_VS, events_LD, params, ar_params):
    """
    Runs the full preprocessing pipeline using AutoReject for epoch cleaning.
    """
    # ... (Steps 1-4 are unchanged: Concatenate, Bad Channels, Epoching) ...
    # Initialize log, concatenate, handle bad channels, create epochs
    log_info = {}
    print("\nStep 2: Concatenating VS and LD conditions...")
    split_point_samples = len(raw_VS.times)
    raw_combined = mne.concatenate_raws([raw_VS, raw_LD])
    events_LD[:, 0] += split_point_samples
    events_combined = np.concatenate((events_VS, events_LD))
    
    print("\nStep 3: Detecting and interpolating bad channels...")
    bad_channels = [] # Placeholder
    raw_combined.info['bads'] = bad_channels
    log_info['interpolated_channels'] = ', '.join(bad_channels) if bad_channels else 'None'
    if raw_combined.info['bads']:
        raw_combined.interpolate_bads(reset_bads=True, mode='accurate')


    print("\nStep 4: Creating epochs without baseline correction for ICA compatibility...")
    # UPDATED: Use the correct event name 'Response/R128' which corresponds to the integer ID 1128
    epochs = mne.Epochs(raw_combined, events_combined, event_id={'Response/R128': 1128}, 
                        tmin=params['tmin'], tmax=params['tmax'],
                        baseline=None, preload=True, reject_by_annotation=True)
    log_info['initial_epochs_VS'] = int((epochs.events[:, 0] < split_point_samples).sum())
    log_info['initial_epochs_LD'] = int((epochs.events[:, 0] >= split_point_samples).sum())


    # --- Step 5: ICA artifact removal is unchanged ---
    if params['apply_ica']:
        print("\nStep 5: Performing ICA with ICALabel...")
        ica = mne.preprocessing.ICA(n_components=params['n_ica_components'], method="infomax",
                                    random_state=97, fit_params=dict(extended=True))
        ica.fit(epochs)
        ic_labels = label_components(epochs, ica, method='iclabel')
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        log_info['rejected_ica_components'] = len(exclude_idx)
        ica.apply(epochs, exclude=exclude_idx)
    else:
        log_info['rejected_ica_components'] = 'N/A'

    # (Optional) Apply baseline correction AFTER ICA
    if params.get('apply_baseline_after_ica', False):
        epochs.apply_baseline(baseline=tuple(params['baseline_timing']))

    # --- STEP 6: REPLACED with AUTOREJECT ---
    print("\nStep 6: Cleaning epochs with AutoReject algorithm...")
    ar = AutoReject(
        n_interpolate=ar_params['n_interpolate'],
        consensus=ar_params['consensus'],
        random_state=42,
        n_jobs=-1,  # Use all available CPU cores
        verbose='tqdm' # Shows a progress bar
    )
    
    # Fit autoreject and transform the epochs object
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    print(f"AutoReject finished. {len(reject_log.bad_epochs)} epochs were rejected.")

    # --- Step 7: Split epochs and finalize logs ---
    print("\nStep 7: Splitting cleaned epochs back into VS and LD conditions...")
    epochs_VS_clean = epochs_clean[epochs_clean.events[:, 0] < split_point_samples]
    epochs_LD_clean = epochs_clean[epochs_clean.events[:, 0] >= split_point_samples]
    
    # Log final counts and calculate rejected epochs per condition
    log_info['final_epochs_VS'] = len(epochs_VS_clean)
    log_info['final_epochs_LD'] = len(epochs_LD_clean)
    log_info['rejected_epochs_autoreject_VS'] = log_info['initial_epochs_VS'] - log_info['final_epochs_VS']
    log_info['rejected_epochs_autoreject_LD'] = log_info['initial_epochs_LD'] - log_info['final_epochs_LD']
    
    print(f"Final clean epochs: {len(epochs_VS_clean)} for VS, {len(epochs_LD_clean)} for LD.")
    
    return epochs_VS_clean, epochs_LD_clean, log_info

if __name__ == "__main__":
    config = load_config()
    # ... (loading paths and subjects is the same) ...
    subjects = config['subjects']
    base_data_dir = config['paths']['base_data_dir']
    output_base_dir = config['paths']['preprocessed_dir']
    params = config['preprocessing_params']
    ar_params = config['autoreject_params'] # <-- Load new autoreject params

    log_data = []
    log_fname = op.join(output_base_dir, 'preprocessing_log.csv')

    for subject in subjects:
        # ... (subject loop setup is the same) ...
        print(f"\n==========================================")
        print(f"Processing subject: {subject}")
        print(f"==========================================")
        data_fname = op.join(base_data_dir, subject, f"{subject}_CONTINU_64Ch_A2Ref")
        output_dir = op.join(output_base_dir, subject)
        if not op.exists(output_dir):
            os.makedirs(output_dir)
        raw_VS, raw_LD, events_VS, events_LD = load_and_prepare_raw(data_fname, params)
        
        if raw_VS is not None:
            # Pass autoreject params to the pipeline function
            epochs_VS_clean, epochs_LD_clean, log_info = run_preprocessing_pipeline(
                raw_VS, raw_LD, events_VS, events_LD, params, ar_params
            )
            # ... (saving epochs is the same) ...
            epochs_VS_clean.save(op.join(output_dir, f"{subject}_VS-epo.fif"), overwrite=True)
            epochs_LD_clean.save(op.join(output_dir, f"{subject}_LD-epo.fif"), overwrite=True)
            
            # --- Update the log dictionary ---
            subject_log = {
                'subject': subject,
                'interpolated_channels': log_info['interpolated_channels'],
                'rejected_ica_components': log_info['rejected_ica_components'],
                'initial_epochs_VS': log_info['initial_epochs_VS'],
                'rejected_epochs_VS': log_info['rejected_epochs_autoreject_VS'],
                'final_epochs_VS': log_info['final_epochs_VS'],
                'initial_epochs_LD': log_info['initial_epochs_LD'],
                'rejected_epochs_LD': log_info['rejected_epochs_autoreject_LD'],
                'final_epochs_LD': log_info['final_epochs_LD'],
            }
            log_data.append(subject_log)
    
    # --- Write the updated log file ---
    if log_data:
        log_df = pd.DataFrame(log_data)
        log_df = log_df[['subject', 'interpolated_channels', 'rejected_ica_components', 
                         'initial_epochs_VS', 'rejected_epochs_VS', 'final_epochs_VS',
                         'initial_epochs_LD', 'rejected_epochs_LD', 'final_epochs_LD']]
        log_df.to_csv(log_fname, index=False)
        print(f"\n==========================================")
        print(f"Preprocessing complete. Log file saved to: {log_fname}")
        print(f"==========================================")