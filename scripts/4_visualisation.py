# 4_visualization.py
import os
import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt
import gc, vtk
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config
config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
output_base_dir = config['paths']['brain_plots_dir']
time_windows = config['params']['time_intervals']
source_estimates_dir_base = config['paths']['source_estimates_dir']
statistical_results_dir_base = config['paths']['statistics_dir']
suffix = str(config['params']['freq1']) + "_" + str(config['params']['freq2'])


# Set SUBJECTS_DIR environment variable
subjects_dir = op.expanduser("~/mne_data/subjects")  # Default FreeSurfer subjects directory
if not op.exists(subjects_dir):
    subjects_dir = op.expanduser("~/subjects")  # Alternative FreeSurfer subjects directory
if not op.exists(subjects_dir):
    raise FileNotFoundError("FreeSurfer subjects directory not found. Please set SUBJECTS_DIR.")
os.environ['SUBJECTS_DIR'] = subjects_dir

def plot_brain_views(stc, mask, output_dir, subject, window_idx, condition, views=None):
    if views is None:
        views = ['rostral', 'lateral', 'medial', 'dorsal', 'ventral']
    
    masked_stc = stc.copy()
    
    if isinstance(mask, list):
        mask = np.array(mask)
    
    if not np.any(mask):
        print(f"Warning: No significant clusters found for {subject}, {window_idx}, {condition}")
        return None
    
    if not op.exists(output_dir):
        os.makedirs(output_dir)
    
    data = masked_stc.data.copy()
    significant_values = data[mask]
    mean = np.mean(significant_values)
    std = np.std(significant_values)
    
    if std > 0:
        data[mask] = (data[mask] - mean) / std
    else:
        print("Warning, std(data)<=0")        
    
    data_mask = np.zeros_like(data, dtype=bool)
    for i in range(data.shape[1]):
        data_mask[:, i] = mask
    
    data[~data_mask] = 0
    masked_stc.data = data
    
    brain_figures = []
    
    try:
        plt.close('all')
        
        brain = masked_stc.plot(
            subject='fsaverage',
            hemi='both',
            time_viewer=False,
            views=views,
            subjects_dir=os.environ['SUBJECTS_DIR'],
            surface='pial',
            alpha=1.0,
            background='white',
            foreground='black',
            view_layout="horizontal",
            size=[1600, 320],
            clim='auto',
            colormap='coolwarm',
            smoothing_steps=10,
            title=f"{subject} - {condition} - {window_idx}",
            add_data_kwargs={'time_label': None}
        )
        # Access PyVista colorbar actor
        if brain.plotter and brain.plotter.scalar_bar:
            cbar_actor = brain.plotter.scalar_bar
            # Create a new text property
            text_prop = vtk.vtkTextProperty()
            text_prop.SetFontSize(10)  # Adjust font size
            text_prop.SetBold(False)  # Optional: remove bold text
            text_prop.SetItalic(False)
            text_prop.SetColor([0, 0, 0])# Optional: remove italic text
            # Apply text property to colorbar labels
            cbar_actor.SetLabelTextProperty(text_prop)     
            # Move colorbar (X, Y)
            cbar_actor.SetPosition(0.1, 0) # Adjust position (values between 0 and 1)
            cbar_actor.SetWidth(0.8)  # Adjust colorbar width


        fig_fname = op.join(output_dir, f"{subject}_{condition}_{window_idx}_{suffix}.png")
        brain.save_image(fig_fname)  
        brain_figures.append(brain)
        
        print(f"Saved brain plot to {fig_fname}")
        
        brain.close()
        del brain
        
    except Exception as e:
        print(f"Error plotting brain: {e}")
        import traceback
        traceback.print_exc()
    
    plt.close('all')
    gc.collect()
    
    return brain_figures

if __name__ == "__main__":
    # Loop through all subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Define subject-specific paths
        source_estimates_dir = op.join(source_estimates_dir_base, subject)
        statistical_results_dir = op.join(statistical_results_dir_base, subject)
        print(f"Statistical results directory: {statistical_results_dir}")
        output_dir = op.join(output_base_dir, subject)
        print(f"Output results directory: {output_dir}")

        # Load significant masks
        significant_masks = {}
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            results_file = op.join(statistical_results_dir, f"cluster_results_{window_idx}_{suffix}.npy")
            print(f"Loading results from {results_file}")
            if op.exists(results_file):
                results = np.load(results_file, allow_pickle=True).item()
                # Convert significant_mask to a NumPy array
                significant_masks[window_idx] = np.array(results['significant_mask'])

        
        # Load source estimates for VS and LD conditions
        stc_dict_VS = {}
        stc_dict_LD = {}
        
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            print(f"Loading source estimates for {window_idx}")
            stc_dict_VS[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir_base, subject, f"stc_VS_{window_idx}_epoch{i}_{suffix}.h5")) for i in range(10)]
            stc_dict_LD[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir_base, subject, f"stc_LD_{window_idx}_epoch{i}_{suffix}.h5")) for i in range(10)]
        
        # Plot brain views for each time window
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            if window_idx in significant_masks and np.any(significant_masks[window_idx]):
                # For VS condition
                if window_idx in stc_dict_VS and len(stc_dict_VS[window_idx]) > 0:
                    stc_VS = stc_dict_VS[window_idx][0]
                    plot_brain_views(stc_VS, significant_masks[window_idx], output_dir, subject, window_idx, 'VS')
                
                # For LD condition
                if window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0:
                    stc_LD = stc_dict_LD[window_idx][0]
                    plot_brain_views(stc_LD, significant_masks[window_idx], output_dir, subject, window_idx, 'LD')
                
                # For difference between conditions
                if (window_idx in stc_dict_VS and len(stc_dict_VS[window_idx]) > 0 and
                    window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0):
                    diff_stc = stc_dict_VS[window_idx][0].copy()
                    diff_stc.data = stc_dict_VS[window_idx][0].data - stc_dict_LD[window_idx][0].data
                    plot_brain_views(diff_stc, significant_masks[window_idx], output_dir, subject, window_idx, 'VS-LD')
        
        print(f"Finished processing subject: {subject}")