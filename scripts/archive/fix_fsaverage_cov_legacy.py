import os.path as op
import mne

subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
cov_fname = op.join(subjects_dir, "fsaverage-cov.fif")

print(f"Loading covariance file: {cov_fname}")
cov = mne.read_cov(cov_fname)

print("Rebuilding covariance object with explicit 'projs' key...")
fixed_cov = mne.Covariance(
    cov.data,
    cov.ch_names,           # <-- positional argument, not keyword
    cov.get('bads', []),
    [],                     # projs
    cov.nfree,
    verbose=False
)

mne.write_cov(cov_fname, fixed_cov, overwrite=True)
print("Covariance file fixed and saved.")