# grand_average_analysis2.py
import os
import os.path as op
import mne
import numpy as np
import pandas as pd
import json
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
subjects = config['subjects']
base_data_dir = config['paths']['base_data_dir']
output_base_dir = config['paths']['roi_dir']
time_windows = config['params']['time_intervals']
source_estimates_dir_base = config['paths']['source_estimates_dir']
statistical_results_dir_base = config['paths']['statistics_dir']
suffix = str(config['params']['freq1']) + "_" + str(config['params']['freq2'])


# Define output directory for grand average analysis
output_base_dir = "roi_analysis"

def extract_roi_data(stc, mask, labels, roi_names=None, scaling_factor=1e6):
    if isinstance(mask, list):
        mask = np.array(mask)
    
    roi_data = {
        'ROI': [],
        'Mean Activation': [],
        'Max Activation': [],
        'Std Activation': [],
        'Range Activation': [],
        'Active Vertices': [],
        'Total Vertices': []
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
            roi_data['ROI'].append(label.name)
            roi_data['Mean Activation'].append(0.0)
            roi_data['Max Activation'].append(0.0)
            roi_data['Std Activation'].append(0.0)
            roi_data['Range Activation'].append(0.0)
            roi_data['Active Vertices'].append(0)
            roi_data['Total Vertices'].append(len(stc_indices))
            continue
        
        significant_data = label_data[label_mask]
        significant_data = significant_data * scaling_factor
        
        roi_data['ROI'].append(label.name)
        roi_data['Mean Activation'].append(np.mean(significant_data) if len(significant_data) > 0 else 0.0)
        roi_data['Max Activation'].append(np.max(significant_data) if len(significant_data) > 0 else 0.0)
        roi_data['Std Activation'].append(np.std(significant_data) if len(significant_data) > 0 else 0.0)
        roi_data['Range Activation'].append((np.max(significant_data) - np.min(significant_data)) if len(significant_data) > 1 else 0.0)
        roi_data['Active Vertices'].append(np.sum(label_mask))
        roi_data['Total Vertices'].append(len(stc_indices))
    
    return pd.DataFrame(roi_data)

def save_roi_data_to_csv(roi_data, filename, metadata_filename, condition=None, window_idx=None, scaling_type='μV/mm²'):
    # Save ROI data to CSV
    roi_data.to_csv(filename, index=False)
    
    # Save metadata to a separate JSON file
    metadata = {
        'scaling': f"Values scaled to {scaling_type}",
        'condition': condition,
        'window': window_idx
    }
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)

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
            
            # Save to CSV
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_VS_roi.csv")
            metadata_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_VS_metadata.json")
            save_roi_data_to_csv(roi_data_VS, csv_file, metadata_file, 'VS', window_idx, scaling_type='μV/mm²')
        
        if window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0:
            stc_LD = stc_dict_LD[window_idx][0]
            roi_data_LD = extract_roi_data(stc_LD, mask, labels, scaling_factor=scaling_factor)
            roi_results[window_idx]['LD'] = roi_data_LD
            
            # Save to CSV
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_LD_roi.csv")
            metadata_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_LD_metadata.json")
            save_roi_data_to_csv(roi_data_LD, csv_file, metadata_file, 'LD', window_idx, scaling_type='μV/mm²')
        
        if (window_idx in stc_dict_VS and len(stc_dict_VS[window_idx]) > 0 and
            window_idx in stc_dict_LD and len(stc_dict_LD[window_idx]) > 0):
            diff_stc = stc_dict_VS[window_idx][0].copy()
            diff_stc.data = stc_dict_VS[window_idx][0].data - stc_dict_LD[window_idx][0].data
            roi_data_diff = extract_roi_data(diff_stc, mask, labels, scaling_factor=scaling_factor)
            roi_results[window_idx]['VS-LD'] = roi_data_diff
            
            # Save to CSV
            csv_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_VS-LD_roi.csv")
            metadata_file = op.join(subject_dir, f"{subject}_{window_idx}_{suffix}_VS-LD_metadata.json")
            save_roi_data_to_csv(roi_data_diff, csv_file, metadata_file, 'VS-LD', window_idx, scaling_type='μV/mm²')
    
    return roi_results

def compute_grand_average(subjects, time_windows, condition):
    grand_average_stcs = {}
    
    for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
        stc_list = []
        
        for subject in subjects:
            stc_files = [op.join(source_estimates_dir_base, subject, f"stc_{condition}_{window_idx}_epoch{i}_{suffix}.h5") for i in range(10)]
            stc_list.extend([mne.read_source_estimate(f) for f in stc_files])
        
        # Compute grand average
        grand_average_stc = stc_list[0].copy()
        grand_average_stc.data = np.mean([stc.data for stc in stc_list], axis=0)
        grand_average_stcs[window_idx] = grand_average_stc
    
    return grand_average_stcs

if __name__ == "__main__":
    # Compute grand average for VS and LD conditions
    grand_average_stcs_VS = compute_grand_average(subjects, time_windows, "VS")
    grand_average_stcs_LD = compute_grand_average(subjects, time_windows, "LD")
    
    # Load significant masks (assuming the same mask for all subjects)
    significant_masks = {}
    for window_idx in [f'win_{i}' for i in range(len(time_windows))]:
        results_file = op.join(statistical_results_dir_base, subjects[0], f"cluster_results_{window_idx}_{suffix}.npy")
        if op.exists(results_file):
            results = np.load(results_file, allow_pickle=True).item()
            significant_masks[window_idx] = np.array(results['significant_mask'])
    
    # Extract ROI data from grand average
    grand_average_roi_results = {}
    for window_idx in grand_average_stcs_VS:
        stc_VS = grand_average_stcs_VS[window_idx]
        stc_LD = grand_average_stcs_LD[window_idx]
        mask = significant_masks[window_idx]
        
        labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=os.environ.get('SUBJECTS_DIR'))
        
        roi_data_VS = extract_roi_data(stc_VS, mask, labels, scaling_factor=1e6)
        roi_data_LD = extract_roi_data(stc_LD, mask, labels, scaling_factor=1e6)
        
        diff_stc = stc_VS.copy()
        diff_stc.data = stc_VS.data - stc_LD.data
        roi_data_diff = extract_roi_data(diff_stc, mask, labels, scaling_factor=1e6)
        
        grand_average_roi_results[window_idx] = {
            'VS': roi_data_VS,
            'LD': roi_data_LD,
            'VS-LD': roi_data_diff
        }
    
    # Save grand average ROI data to CSV
    grand_average_dir = op.join(output_base_dir, "grand_average")
    if not op.exists(grand_average_dir):
        os.makedirs(grand_average_dir)
    
    for window_idx in grand_average_roi_results:
        for condition in grand_average_roi_results[window_idx]:
            roi_data = grand_average_roi_results[window_idx][condition]
            csv_file = op.join(grand_average_dir, f"grand_average_{window_idx}_{condition}_roi.csv")
            metadata_file = op.join(grand_average_dir, f"grand_average_{window_idx}_{condition}_metadata.json")
            save_roi_data_to_csv(roi_data, csv_file, metadata_file, condition, window_idx, scaling_type='μV/mm²')
    
    print("Grand average ROI analysis completed.")