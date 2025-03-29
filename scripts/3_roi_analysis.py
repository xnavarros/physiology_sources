# roi_analysis.py

import os
import os.path as op
import mne
import numpy as np
import pandas as pd

# Define subject list
subjects = ['BJ25', 'JS08', 'LP26', 'MC05', 'MN23', 'OL04', 'SB27', 'TH24', 'VA14', 'VS06']

# Define base directory for raw data
base_data_dir = "/Users/xavier/work/data/Physiology"

# Define output directory for ROI analysis
output_base_dir = "roi_analysis"

def extract_roi_data(stc, mask, labels, roi_names=None, scaling_factor=1e6):
    if isinstance(mask, list):
        mask = np.array(mask)
    
    roi_data = {
        'mean': {},
        'max': {},
        'std': {},
        'range': {},
        'active_vertices': {},
        'total_vertices': {}
    }
    
    for label in labels:
        if roi_names is not None and label.name not in roi_names:
            continue
        
        vertices_in_label = label.get_vertices_used()
        hemi_prefix = 0 if label.hemi == 'lh' else len(stc.lh_vertno)
        
        stc_indices = np.where(np.in1d(stc.vertices[0 if label.hemi == 'lh' else 1], vertices_in_label))[0]
        if label.hemi == 'rh':
            stc_indices += len(stc.lh_vertno)
        
        if len(stc_indices) == 0:
            continue
        
        label_data = stc.data[stc_indices]
        label_mask = mask[stc_indices]
        
        if not np.any(label_mask):
            roi_data['mean'][label.name] = 0.0
            roi_data['max'][label.name] = 0.0
            roi_data['std'][label.name] = 0.0
            roi_data['range'][label.name] = 0.0
            roi_data['active_vertices'][label.name] = 0
            roi_data['total_vertices'][label.name] = len(stc_indices)
            continue
        
        significant_data = label_data[label_mask]
        significant_data = significant_data * scaling_factor
        
        roi_data['mean'][label.name] = np.mean(significant_data) if len(significant_data) > 0 else 0.0
        roi_data['max'][label.name] = np.max(significant_data) if len(significant_data) > 0 else 0.0
        roi_data['std'][label.name] = np.std(significant_data) if len(significant_data) > 0 else 0.0
        roi_data['range'][label.name] = (np.max(significant_data) - np.min(significant_data)) if len(significant_data) > 1 else 0.0
        roi_data['active_vertices'][label.name] = np.sum(label_mask)
        roi_data['total_vertices'][label.name] = len(stc_indices)
    
    return roi_data

def save_roi_data_to_csv(roi_data, filename, condition=None, window_idx=None, scaling_type='μV/mm²'):
    columns = pd.MultiIndex.from_product([['mean', 'max', 'std', 'range', 'active_vertices', 'total_vertices'], ['value']])
    roi_names = set()
    for metric in roi_data.keys():
        roi_names.update(roi_data[metric].keys())
    roi_names = sorted(roi_names)
    
    df = pd.DataFrame(index=roi_names, columns=columns)
    
    for roi in roi_names:
        for metric in roi_data.keys():
            if roi in roi_data[metric]:
                df.loc[roi, (metric, 'value')] = roi_data[metric][roi]
    
    metadata = {'scaling': f"Values scaled to {scaling_type}"}
    
    if condition is not None:
        metadata['condition'] = condition
    if window_idx is not None:
        metadata['window'] = window_idx
    
    meta_df = pd.DataFrame([metadata])
    meta_df.to_csv(filename, index=False)
    df.to_csv(filename, mode='a')
    
    return filename

def analyze_roi_for_subject(subject, stc_dict_VS, stc_dict_LD, significant_masks, time_windows, output_dir=None, scaling_factor=1e6):
    if output_dir is None:
        output_dir = "roi_analysis"
    
    subject_dir = op.join(output_dir, subject)
    if not op.exists(subject_dir):
        os.makedirs(subject_dir)
    
    subjects_dir = os.environ.get('SUBJECTS_DIR')
    if not subjects_dir:
        subjects_dir = op.expanduser('~/mne_data/subjects')
        os.environ['SUBJECTS_DIR'] = subjects_dir
    
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)
    roi_results = {}
    
    for i, window in enumerate(time_windows):
        window_idx = f'win_{i}'
        roi_results[window_idx] = {}
        
        if window_idx not in significant_masks:
            continue
        
        mask = significant_masks[window_idx]
        
        if window_idx in stc_dict_VS and len(stc_dict_VS[window_idx]) > 0:
            stc_VS = stc_dict_VS[window_idx][0]
            roi_data_VS = extract_roi_data(stc_VS, mask, labels, scaling_factor=scaling_factor)
            roi_results[window_idx]['VS'] = roi_data_VS
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_VS_roi.csv")
            save_roi_data_to_csv(roi_data_VS, csv_file, 'VS', window_idx, scaling_type='μV/mm²')
        
        if window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0:
            stc_LD = stc_dict_LD[window_idx][0]
            roi_data_LD = extract_roi_data(stc_LD, mask, labels, scaling_factor=scaling_factor)
            roi_results[window_idx]['LD'] = roi_data_LD
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_LD_roi.csv")
            save_roi_data_to_csv(roi_data_LD, csv_file, 'LD', window_idx, scaling_type='μV/mm²')
        
        if (window_idx in stc_dict_VS and len(stc_dict_VS[window_idx]) > 0 and
            window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0):
            diff_stc = stc_dict_VS[window_idx][0].copy()
            diff_stc.data = stc_dict_VS[window_idx][0].data - stc_dict_LD[window_idx][0].data
            roi_data_diff = extract_roi_data(diff_stc, mask, labels, scaling_factor=scaling_factor)
            roi_results[window_idx]['VS-LD'] = roi_data_diff
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_VS-LD_roi.csv")
            save_roi_data_to_csv(roi_data_diff, csv_file, 'VS-LD', window_idx, scaling_type='μV/mm²')
    
    return roi_results

if __name__ == "__main__":
    # Define time windows
    time_windows = [[-1.25, -1], [-1, -0.75], [-0.75, -0.5], [-0.5, -0.25], [-0.25, 0], [0, 0.25], [0.25, 0.5]]
    
    # Loop through all subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Define subject-specific paths
        source_estimates_dir = op.join("source_estimates", subject)
        statistical_results_dir = op.join("statistical_results", subject)
        output_dir = op.join(output_base_dir, subject)
        
        # Load significant masks
        significant_masks = {}
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            results_file = op.join(statistical_results_dir, f"cluster_results_{window_idx}.npy")
            if op.exists(results_file):
                results = np.load(results_file, allow_pickle=True).item()
                significant_masks[window_idx] = results['significant_mask']
        
        # Load source estimates for VS and LD conditions
        stc_dict_VS = {}
        stc_dict_LD = {}
        
        for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
            stc_dict_VS[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir, f"stc_VS_{window_idx}_epoch{i}.stc")) for i in range(10)]
            stc_dict_LD[window_idx] = [mne.read_source_estimate(op.join(source_estimates_dir, f"stc_LD_{window_idx}_epoch{i}.stc")) for i in range(10)]
        
        # Perform ROI analysis
        roi_results = analyze_roi_for_subject(subject, stc_dict_VS, stc_dict_LD, significant_masks, time_windows, output_dir)
        
        print(f"Finished processing subject: {subject}")