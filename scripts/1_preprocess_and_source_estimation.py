# preprocess_and_source_estimation.py

import os
import os.path as op
import mne
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse
import yaml

def load_config(config_path="scripts/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
output_base_dir = config['paths']['source_estimates_dir']
params = config['params']


def load_and_preprocess(data_fname, fwd_fname, params):
    raw = mne.io.read_raw_brainvision(data_fname + '.vhdr', preload=True)
    raw.set_channel_types({'EOG': 'eog', 'ECG': 'ecg', 'PS': 'emg', 'Pression': 'resp', 'Debit': 'resp'})
    raw.set_montage('standard_1020', on_missing="ignore")
    
    fwd = mne.read_forward_solution(fwd_fname)
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
    
    if params['apply_notch']:
        filt_raw.notch_filter(freqs=np.arange(50, 251, 50), picks='eeg', method='spectrum_fit', filter_length='5s', trans_bandwidth=2)
    
    filt_raw = filt_raw.set_eeg_reference("average")
    filt_raw.set_eeg_reference('average', projection=True)
    filt_raw.apply_proj()
    
    events, event_id = mne.events_from_annotations(raw)
    time_endVS = ((events[events[:, 2] == event_id["Comment/endVS"], 0]) / 1000).astype(int)[0]
    time_startLD = ((events[events[:, 2] == event_id["Comment/startLD"], 0]) / 1000).astype(int)[0]
    time_endLD = ((events[events[:, 2] == event_id["Comment/endLD"], 0]) / 1000).astype(int)[0]
    
    filt_VS = filt_raw.copy().crop(tmax=time_endVS)
    filt_LD = filt_raw.copy().crop(tmin=time_startLD, tmax=time_endLD)
    
    filt_VS.load_data()
    filt_LD.load_data()
    
    events_VS, _ = mne.events_from_annotations(filt_VS)
    events_LD, _ = mne.events_from_annotations(filt_LD)
    
    return filt_VS, filt_LD, events_VS, events_LD, fwd

def apply_frequency_filter(raw_VS, raw_LD, params):
    if params['apply_ica']:
        # Initialize ICA
        ica = mne.preprocessing.ICA(
            n_components=params['n_ica_components'],
            method="infomax",
            random_state=97,
            fit_params=dict(extended=True),
        )
        
        # Fit ICA to VS condition
        ica.fit(raw_VS)
        
        # Find and exclude bad components using ICLabel
        ica_labels = ica.labels_  # This contains the ICLabel classifications
        exclude_idx = [idx for idx, label in enumerate(ica_labels) if "brain" not in label]
        
        # Apply ICA to VS condition
        reconst_VS = raw_VS.copy().filter(params['freq1'], params['freq2'])
        ica.apply(reconst_VS, exclude=exclude_idx)
        
        # Fit ICA to LD condition
        ica.fit(raw_LD)
        
        # Find and exclude bad components using ICLabel
        ica_labels = ica.labels_  # This contains the ICLabel classifications
        exclude_idx = [idx for idx, label in enumerate(ica_labels) if "brain" not in label]
        
        # Apply ICA to LD condition
        reconst_LD = raw_LD.copy().filter(params['freq1'], params['freq2'])
        ica.apply(reconst_LD, exclude=exclude_idx)
    else:
        # If ICA is not applied, just filter the data
        reconst_VS = raw_VS.copy().filter(params['freq1'], params['freq2'])
        reconst_LD = raw_LD.copy().filter(params['freq1'], params['freq2'])
    
    return reconst_VS, reconst_LD

def create_epochs(raw_VS, raw_LD, events_VS, events_LD, params):
    event_mapping = {'1128': 1128}
    tmin = -1.5
    tmax = 0.5
    baseline = params['baseline_intervals'] if params['apply_td_baseline'] else None
    
    epochs_VS = mne.Epochs(raw_VS, events_VS, event_mapping, tmin, tmax, baseline=baseline, preload=True)
    epochs_LD = mne.Epochs(raw_LD, events_LD, event_mapping, tmin, tmax, baseline=baseline, preload=True)
    
    return epochs_VS, epochs_LD

def compute_source_estimates(epochs_VS, epochs_LD, fwd, time_windows):
    rank_VS = mne.compute_rank(epochs_VS, tol=1e-6, tol_kind="relative")
    rank_LD = mne.compute_rank(epochs_LD, tol=1e-6, tol_kind="relative")
    
    covs_VS = [mne.compute_covariance(epochs_VS, tmin=win[0], tmax=win[1], method="shrunk", rank=rank_VS, verbose=False) for win in time_windows]
    covs_LD = [mne.compute_covariance(epochs_LD, tmin=win[0], tmax=win[1], method="shrunk", rank=rank_LD, verbose=False) for win in time_windows]
    common_covs = [cov_vs + cov_ld for cov_vs, cov_ld in zip(covs_VS, covs_LD)]
    
    stc_dict_VS = {}
    for i, ((tmin, tmax), cov) in enumerate(zip(time_windows, common_covs)):
        filters = mne.beamformer.make_lcmv(epochs_VS.info, fwd, cov, reg=0.05, noise_cov=None, pick_ori="max-power", verbose=False)
        epochs_win = epochs_VS.copy().crop(tmin=tmin, tmax=tmax)
        stc_dict_VS[f'win_{i}'] = mne.beamformer.apply_lcmv_epochs(epochs_win, filters)
    
    stc_dict_LD = {}
    for i, ((tmin, tmax), cov) in enumerate(zip(time_windows, common_covs)):
        filters = mne.beamformer.make_lcmv(epochs_LD.info, fwd, cov, reg=0.05, noise_cov=None, pick_ori="max-power", verbose=False)
        epochs_win = epochs_LD.copy().crop(tmin=tmin, tmax=tmax)
        stc_dict_LD[f'win_{i}'] = mne.beamformer.apply_lcmv_epochs(epochs_win, filters)
    
    return stc_dict_VS, stc_dict_LD

def save_source_estimates(stc_dict_VS, stc_dict_LD, output_dir):
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    
    for window_idx, stc_list in stc_dict_VS.items():
        for i, stc in enumerate(stc_list):
            stc.save(op.join(output_dir, f"stc_VS_{window_idx}_epoch{i}.stc"), overwrite=True)
    
    for window_idx, stc_list in stc_dict_LD.items():
        for i, stc in enumerate(stc_list):
            stc.save(op.join(output_dir, f"stc_LD_{window_idx}_epoch{i}.stc"), overwrite=True)

if __name__ == "__main__":
    # Loop through all subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Define subject-specific paths
        data_fname = op.join(base_data_dir, subject, f"{subject}_CONTINU_64Ch_A2Ref")
        fwd_fname = op.join(base_data_dir, subject, f"{subject}-fwd.fif")
        output_dir = op.join(output_base_dir, subject)
        
        # Run preprocessing and source estimation
        filt_VS, filt_LD, events_VS, events_LD, fwd = load_and_preprocess(data_fname, fwd_fname, params)
        reconst_VS, reconst_LD = apply_frequency_filter(filt_VS, filt_LD, params)
        epochs_VS, epochs_LD = create_epochs(reconst_VS, reconst_LD, events_VS, events_LD, params)
        stc_dict_VS, stc_dict_LD = compute_source_estimates(epochs_VS, epochs_LD, fwd, params['time_intervals'])
        save_source_estimates(stc_dict_VS, stc_dict_LD, output_dir)
        
        print(f"Finished processing subject: {subject}")