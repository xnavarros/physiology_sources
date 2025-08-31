# 4_visualization.py
import os
import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt
import gc
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_grand_average(stc, output_dir, window_idx, suffix, clim):
    """
    Plots the grand-average source estimate on the fsaverage brain using a fixed color scale.
    """
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    try:
        plt.close('all')
        
        brain = stc.plot(
            subject='fsaverage',
            hemi='both',
            time_viewer=False,
            views=['lat', 'med'],
            subjects_dir=os.environ.get('SUBJECTS_DIR'),
            surface='pial',
            background='white',
            foreground='black',
            view_layout="horizontal",
            size=[1600, 400],
            clim=clim,
            colormap='coolwarm',
            smoothing_steps=10,
            title=f"Grand Average Power Difference (VS - LD) - Window: {window_idx}",
            add_data_kwargs={'time_label': None}
        )

        fig_fname = op.join(output_dir, f"grand_avg_diff_{window_idx}_{suffix}.png")
        brain.save_image(fig_fname)
        print(f"Saved grand average brain plot to {fig_fname}")
        
        brain.close()
        
    except Exception as e:
        print(f"Error plotting brain: {e}")
    
    plt.close('all')
    gc.collect()

if __name__ == "__main__":
    config = load_config()
    subjects = config['subjects']
    source_estimates_dir_base = config['paths']['source_estimates_dir']
    output_base_dir = config['paths']['brain_plots_dir']
    time_windows = config['params']['time_intervals']
    suffix = f"{config['params']['freq1']}_{config['params']['freq2']}"

    # Set SUBJECTS_DIR environment variable for fsaverage
    subjects_dir = os.path.expanduser("~/mne_data/MNE-fsaverage-data")
    if not op.exists(subjects_dir):
        raise FileNotFoundError(f"fsaverage directory not found at {subjects_dir}. Please check the path or download it.")
    os.environ['SUBJECTS_DIR'] = subjects_dir

    # --- PASS 1: Calculate all grand averages and find the global scale ---
    grand_averages = {}
    print("--- Pass 1: Calculating all grand averages ---")
    for i, window in enumerate(time_windows):
        window_idx = f'win_{i}'
        print(f"  - Processing window: {window_idx}")

        all_subject_diffs = []
        for subject in subjects:
            try:
                stc_vs_fname = op.join(source_estimates_dir_base, subject, f"stc_avg_VS_{window_idx}_{suffix}.h5")
                stc_ld_fname = op.join(source_estimates_dir_base, subject, f"stc_avg_LD_{window_idx}_{suffix}.h5")
                
                stc_vs = mne.read_source_estimate(stc_vs_fname)
                stc_ld = mne.read_source_estimate(stc_ld_fname)
                
                stc_diff = stc_vs.copy()
                stc_diff.data = (stc_vs.data ** 2) - (stc_ld.data ** 2)
                all_subject_diffs.append(stc_diff)
            except OSError:
                continue
        
        if all_subject_diffs:
            grand_averages[window_idx] = sum(all_subject_diffs) / len(all_subject_diffs)

    if not grand_averages:
        print("No data found for any window. Exiting.")
    else:
        # Determine the global color scale based on the 98th percentile
        all_data = np.concatenate([stc.data.ravel() for stc in grand_averages.values()])
        global_max = np.percentile(np.abs(all_data), 98)
        if global_max == 0: global_max = 1.0 # Avoid division by zero
        
        global_clim = dict(kind='value', lims=[-global_max, 0, global_max])
        print(f"\nGlobal color scale set to [{-global_max:.2e}, {global_max:.2e}] based on 98th percentile.")

        # --- PASS 2: Plot all grand averages using the fixed global scale ---
        print("\n--- Pass 2: Plotting all grand averages with fixed scale ---")
        output_dir = op.join(output_base_dir, "grand_average")
        for window_idx, grand_avg_stc in grand_averages.items():
            plot_grand_average(grand_avg_stc, output_dir, window_idx, suffix, global_clim)
        
    print("\nFinished creating all grand average plots.")