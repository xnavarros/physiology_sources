import os
import os.path as op
import mne
import numpy as np
import yaml
import json
from mne import Report

def load_config(config_path=None):
    if config_path is None:
        script_dir = op.dirname(op.abspath(__file__))
        root_dir = op.dirname(op.dirname(script_dir))
        config_path = op.join(root_dir, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def morph_stc(stc, subject, subjects_dir, fsaverage_vertices):
    """Morphs an STC to fsaverage."""
    morph = mne.compute_source_morph(
        stc, subject_from=subject, subject_to='fsaverage',
        subjects_dir=subjects_dir, spacing=fsaverage_vertices,
        smooth=10, verbose=False
    )
    return morph.apply(stc)

if __name__ == "__main__":
    config = load_config()
    subjects_dir = config['paths']['mri_dir']
    source_dir = config['paths']['source_estimates_dir']
    output_dir = op.join(config['paths']['results_dir'], 'profile_analysis', 'source_averages')
    
    if not op.exists(output_dir):
        os.makedirs(output_dir)

    # Load profiles
    profiles_json_path = op.join(config['paths']['results_dir'], 'subject_profiles.json')
    with open(profiles_json_path, 'r') as f:
        profiles = json.load(f)

    # Prepare fsaverage vertices for morphing
    # We need to know the source space spacing used. Assuming 'oct6' or similar.
    # To be safe, we can load fsaverage source space if available, or let compute_source_morph handle it.
    # For simplicity, we'll let compute_source_morph handle the target vertices by default (fsaverage, grade 5 usually).
    
    report = Report(title='Profile Source Reconstruction', verbose=False)

    for profile_name, subject_list in profiles.items():
        print(f"\n--- Processing Profile: {profile_name} (N={len(subject_list)}) ---")
        
        morphed_stcs = []
        
        for subject in subject_list:
            stc_path = op.join(source_dir, subject, f"{subject}_VS-LD-beta-source.stc") # Assuming this naming convention from previous steps
            # If .stc doesn't exist, try .h5 or look for the folder structure
            # Based on workspace info: source_estimates/BJ25/
            # We need to find the actual file. Let's assume a standard name or search.
            
            # Search for a likely STC file
            subj_source_path = op.join(source_dir, subject)
            found_stc = None
            if op.exists(subj_source_path):
                for f in os.listdir(subj_source_path):
                    if f.endswith('-lh.stc') or f.endswith('-rh.stc'):
                        # MNE saves as -lh.stc and -rh.stc, but we load by the base name
                        base_name = f[:-7]
                        # We prefer the difference map if available, or we might need to compute it.
                        # Let's look for "beta" and "source" in the name
                        if "beta" in base_name:
                            found_stc = op.join(subj_source_path, base_name)
                            break
            
            if found_stc:
                print(f"  - Loading {found_stc} for {subject}")
                try:
                    stc = mne.read_source_estimate(found_stc)
                    stc_morphed = morph_stc(stc, subject, subjects_dir, fsaverage_vertices=None)
                    morphed_stcs.append(stc_morphed)
                except Exception as e:
                    print(f"    Error loading/morphing {subject}: {e}")
            else:
                print(f"  - No suitable STC file found for {subject}")

        if morphed_stcs:
            print(f"  - Averaging {len(morphed_stcs)} subjects...")
            # Average
            data = np.mean([s.data for s in morphed_stcs], axis=0)
            stc_avg = morphed_stcs[0].copy()
            stc_avg.subject = 'fsaverage'
            stc_avg.data = data
            
            # Save
            out_fname = op.join(output_dir, f"profile_{profile_name.replace(' ', '_').replace('/', '-')}_avg")
            stc_avg.save(out_fname)
            
            # Plot
            brain = stc_avg.plot(
                subject='fsaverage', subjects_dir=subjects_dir,
                hemi='both', views=['lateral', 'medial'],
                time_label=f"{profile_name} (N={len(morphed_stcs)})",
                smoothing_steps=5, clim=dict(kind='percent', lims=[90, 95, 99])
            )
            
            # Save screenshot for report
            img_fname = out_fname + ".png"
            brain.save_image(img_fname)
            brain.close()
            
            report.add_image(image=img_fname, title=f"{profile_name} (N={len(morphed_stcs)})")
        else:
            print("  - No data to average for this profile.")

    report_path = op.join(output_dir, "report_profile_source.html")
    report.save(report_path, overwrite=True, open_browser=False)
    print(f"\nReport saved to {report_path}")
