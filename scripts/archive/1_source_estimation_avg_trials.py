# source_estimation_avg_trials.py

import os
import os.path as op
import mne
import numpy as np
from mne.minimum_norm import make_inverse_operator, apply_inverse
import h5py
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
# Define path for preprocessed data
preprocessed_data_dir = config['paths']['preprocessed_data_dir']
output_base_dir = config['paths']['source_estimates_dir']
params = config['params']

def apply_frequency_filter(epochs_VS, epochs_LD, params):
    """Apply band-pass filter to the epoched data."""
    filtered_epochs_VS = epochs_VS.copy().filter(l_freq=params['freq1'], h_freq=params['freq2'])
    filtered_epochs_LD = epochs_LD.copy().filter(l_freq=params['freq1'], h_freq=params['freq2'])
    return filtered_epochs_VS, filtered_epochs_LD

def compute_source_estimates(epochs_VS, epochs_LD, fwd, time_windows, output_dir, suffix):
    """
    Computes and saves the AVERAGE source estimate across epochs for each time window.
    This is the memory-safe and disk-safe method.
    """
    # --- Compute separate covariance matrices for each condition and time window ---
    covs_VS = [mne.compute_covariance(epochs_VS, tmin=win[0], tmax=win[1], method="shrunk", rank=None, verbose=False) for win in time_windows]
    covs_LD = [mne.compute_covariance(epochs_LD, tmin=win[0], tmax=win[1], method="shrunk", rank=None, verbose=False) for win in time_windows]

    # --- Combine the covariances for a common filter ---
    common_covs = [cov_vs + cov_ld for cov_vs, cov_ld in zip(covs_VS, covs_LD)]

    # --- Average epochs to create Evoked objects (the key memory-saving step) ---
    evoked_VS = epochs_VS.average()
    evoked_LD = epochs_LD.average()

    # --- Loop through time windows, create filter, and apply to the single Evoked object ---
    for i, cov in enumerate(common_covs):
        window_idx = f'win_{i}'
        
        # VS Condition
        filters = mne.beamformer.make_lcmv(evoked_VS.info, fwd, cov, reg=0.05, noise_cov=None, pick_ori="max-power", verbose=False)
        stc_avg_vs = mne.beamformer.apply_lcmv(evoked_VS, filters)
        stc_avg_vs.save(op.join(output_dir, f"stc_avg_VS_{window_idx}_{suffix}.h5"), ftype='h5', overwrite=True)

        # LD Condition
        filters = mne.beamformer.make_lcmv(evoked_LD.info, fwd, cov, reg=0.05, noise_cov=None, pick_ori="max-power", verbose=False)
        stc_avg_ld = mne.beamformer.apply_lcmv(evoked_LD, filters)
        stc_avg_ld.save(op.join(output_dir, f"stc_avg_LD_{window_idx}_{suffix}.h5"), ftype='h5', overwrite=True)

    print("Finished computing and saving all source estimates.")

if __name__ == "__main__":
    # Loop through all subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Define subject-specific paths
        fwd_fname = op.join(base_data_dir, subject, f"{subject}-fwd.fif")
        output_dir = op.join(output_base_dir, subject)
        if not op.exists(output_dir):
            os.makedirs(output_dir)

        # Load preprocessed and epoched data
        vs_epochs_fname = op.join(preprocessed_data_dir, subject, f"{subject}_VS-epo.fif")
        ld_epochs_fname = op.join(preprocessed_data_dir, subject, f"{subject}_LD-epo.fif")
        
        epochs_VS = mne.read_epochs(vs_epochs_fname, preload=True)
        epochs_LD = mne.read_epochs(ld_epochs_fname, preload=True)
        fwd = mne.read_forward_solution(fwd_fname)

        # Run source estimation pipeline
        filtered_epochs_VS, filtered_epochs_LD = apply_frequency_filter(epochs_VS, epochs_LD, params)
        
        # Define suffix for saving files
        suffix = f"{params['freq1']}_{params['freq2']}"
        
        # Compute and save source estimates in a memory-efficient way
        compute_source_estimates(filtered_epochs_VS, filtered_epochs_LD, fwd, params['time_intervals'], output_dir, suffix)
        
        print(f"Finished processing subject: {subject}")