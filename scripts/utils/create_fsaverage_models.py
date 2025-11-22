import os
import os.path as op
import mne
import yaml

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("--- Creating common fsaverage forward model and noise covariance ---")
    config = load_config()
    subjects = config['subjects']
    preprocessed_dir = config['paths']['preprocessed_slow_potentials_dir']
    
    # --- 1. PREPARE A MASTER INFO OBJECT ---
    print("\n--- Preparing a master Info object ---")
    try:
        representative_subject = subjects[0]
        epochs_fname = op.join(preprocessed_dir, f"{representative_subject}_VS-slow-epo.fif")
        info = mne.io.read_info(epochs_fname)
        print(f"Loaded base info from subject: {representative_subject}")

        # Apply all necessary modifications to create the final master info
        if 'A2' in info.ch_names:
            print("  - Setting channel 'A2' to type 'misc'.")
            info.set_channel_types({'A2': 'misc'})
        
        # --- MODIFIED: Use .clear() to modify the list in-place ---
        info['projs'].clear()
        
        montage_path = "/Users/xavier/work/physiology_sources/data/raw/physiology_updated_positions.sfp"
        if op.exists(montage_path):
            print(f"  - Applying custom montage from: {montage_path}")
            montage = mne.channels.read_custom_montage(montage_path)
            info.set_montage(montage, on_missing='warn')
        else:
            print(f"  - WARNING: Custom montage file not found. Using default positions.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load representative epoch file. {e}")
        exit()

    # --- 2. CREATE AND SAVE THE FORWARD SOLUTION (using master info) ---
    print("\n--- Creating the forward solution ---")
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
    fsaverage_subject = 'fsaverage'
    src = mne.setup_source_space(fsaverage_subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
    model = mne.make_bem_model(subject=fsaverage_subject, ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    trans = 'fsaverage' 
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    
    # Save forward solution in subjects_dir
    fwd_fname = op.join(subjects_dir, "fsaverage-fwd.fif")
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
    print(f"Forward solution saved to: {fwd_fname}")

    # --- 3. CREATE AND SAVE A COMPATIBLE NOISE COVARIANCE (using master info) ---
    print("\n--- Creating a compatible noise covariance matrix ---")
    epochs = mne.read_epochs(epochs_fname, preload=True, verbose=False)
    # CRITICAL: Use the master info object that has cleared projectors
    epochs.info = info 
    
    # Compute and save the covariance directly. No rebuilding needed.
    noise_cov = mne.compute_covariance(epochs, tmax=0.0, method='shrunk', rank=None, verbose=False)
    cov_fname = op.join(subjects_dir, "fsaverage-cov.fif")
    mne.write_cov(cov_fname, noise_cov, overwrite=True)
    print(f"Compatible noise covariance saved to: {cov_fname}")
    
    print("\n--- Model creation complete. ---")