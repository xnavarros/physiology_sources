import os.path as op
import mne

print("--- Attempting to fix the fsaverage covariance file ---")

try:
    # Define the path to the covariance file
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
    cov_fname = op.join(subjects_dir, "fsaverage-cov.fif")
    print(f"Loading covariance file: {cov_fname}")

    # Load the existing (broken) covariance object
    noise_cov = mne.read_cov(cov_fname)

    # Check if the 'projs' key is missing
    if 'projs' not in noise_cov:
        print("  - 'projs' key is missing. Rebuilding the Covariance object...")

        # Re-create the object from its own data, but add the 'projs' key
        # This is the most robust way to ensure the structure is correct.
        fixed_cov = mne.Covariance(
            data=noise_cov.data,
            ch_names=noise_cov.ch_names,
            bads=noise_cov.get('bads', []),
            projs=[],  # Explicitly add the empty projectors list
            nfree=noise_cov.nfree,
            verbose=False
        )

        # Overwrite the original file with the fixed version
        mne.write_cov(cov_fname, fixed_cov, overwrite=True)
        print(f"  - Successfully fixed and saved the file.")
    else:
        print("  - 'projs' key already exists. File appears to be correct.")

except Exception as e:
    print(f"An error occurred: {e}")

print("\n--- Fix attempt complete. ---")