import os
import os.path as op
from collections import defaultdict
import re
import sys
import json

# Add project root to path to import utils
sys.path.append(op.join(op.dirname(__file__), '..', '..'))
from scripts.utils.paper_data_manager import PaperDataManager

def parse_summary_file(filepath):
    """
    Parses a summary file (either ERP or TFR) into a dictionary.
    Handles both new and old formats:
    New: SubjectID:Effect1(Region),Effect2(Region),...
    Old: SubjectID:Effect1,Effect2,...
    """
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header lines or malformed lines
            if ':' not in line or line.startswith("Subject") or line.startswith("-"):
                continue
            subject, effects_str = line.strip().split(':', 1)
            
            # Robust splitting: split by comma only if not inside parentheses
            effects = []
            current = []
            depth = 0
            for char in effects_str:
                if char == '(':
                    depth += 1
                    current.append(char)
                elif char == ')':
                    depth -= 1
                    current.append(char)
                elif char == ',' and depth == 0:
                    effects.append("".join(current).strip())
                    current = []
                else:
                    current.append(char)
            if current:
                effects.append("".join(current).strip())
            
            data[subject] = effects
    return data

def get_features(effects):
    """Extracts features (timing, location) from a list of effect strings."""
    features = {
        'early': set(), 'mid': set(), 'late': set(), 'post': set()
    }
    for eff in effects:
        if not eff: continue # Skip empty effect strings

        # --- FIX: Safely parse regions ---
        regions = set()
        match = re.search(r'\((.*?)\)', eff)
        if match:
            # Split regions by comma and strip whitespace
            regions.update([r.strip() for r in match.group(1).split(',')])
        
        # If no regions found (or just empty parens), add a placeholder 
        # so the timing set is not empty (truthy)
        if not regions:
            regions.add("Global")
        
        if 'Early' in eff or 'early' in eff:
            features['early'].update(regions)
        if 'Mid' in eff or 'mid' in eff:
            features['mid'].update(regions)
        if 'Late' in eff or 'late' in eff:
            features['late'].update(regions)
        if 'Post' in eff or 'post' in eff:
            features['post'].update(regions)
            
    return features

def classify_subject(erp_features, tfr_features):
    """Classifies a subject into a profile based on their ERP and TFR features."""
    
    # --- Timing Features ---
    has_prep_erp = erp_features['early'] or erp_features['mid'] or erp_features['late']
    has_prep_tfr = tfr_features['early'] or tfr_features['mid'] or tfr_features['late']
    has_early_tfr = bool(tfr_features['early'])
    
    # --- Spatial Overlap Feature ---
    # Check for non-empty intersection
    early_overlap = erp_features['early'] and tfr_features['early'] and (erp_features['early'] & tfr_features['early'])
    
    # --- Classification Logic ---
    if has_prep_erp and has_early_tfr:
        if early_overlap:
            return "Co-localized Early Preparer"
        else:
            return "Distributed Early Preparer"
    
    if not has_prep_erp and not has_prep_tfr:
        if tfr_features['post'] or erp_features['post']:
            return "Reactive Responder"
        else:
            return "Non-Responder"

    if has_prep_tfr and not has_early_tfr:
        return "Late/Motor-Focused Preparer"
        
    if has_prep_erp and not has_prep_tfr:
        return "Sustained ERP Responder (No specific TFR prep)"

    return "Mixed/Other"

if __name__ == "__main__":
    # Corrected path for erp_summary.md which is inside 'erp_analysis' not 'readiness_potential'
    erp_summary_path = op.join('results', 'erp_analysis', 'erp_subject_summary.md')
    tfr_summary_path = op.join('results', 'TF_analysis', 'tfr_subject_summary.md')
    report_path = op.join('results', 'cross_analysis_report.md')

    if not op.exists(erp_summary_path):
        print(f"Error: Cannot find ERP summary file at {erp_summary_path}. Please run the readiness potential analysis first.")
        exit()

    erp_data = parse_summary_file(erp_summary_path)
    
    if op.exists(tfr_summary_path):
        tfr_data = parse_summary_file(tfr_summary_path)
        print(f"Loaded TFR summary from {tfr_summary_path}")
    else:
        print(f"WARNING: TFR summary file not found at {tfr_summary_path}. Proceeding with ERP data only.")
        tfr_data = defaultdict(list) # Empty data for TFR

    # Group subjects by profile
    profiles = defaultdict(list)
    # Use all subjects found in ERP data (since TFR might be empty)
    all_subjects = sorted(list(set(erp_data.keys()) | set(tfr_data.keys())))

    # Initialize Data Manager
    results_dir = op.join('results')
    if not op.exists(results_dir):
        os.makedirs(results_dir)
    data_manager = PaperDataManager(results_dir)

    for subject in all_subjects:
        erp_features = get_features(erp_data[subject])
        tfr_features = get_features(tfr_data[subject])
        profile = classify_subject(erp_features, tfr_features)
        profiles[profile].append(subject)
        print(f"Subject {subject} classified as: {profile}")

        # Save to Paper Data
        # Convert sets to lists for JSON serialization
        erp_serializable = {k: list(v) for k, v in erp_features.items()}
        tfr_serializable = {k: list(v) for k, v in tfr_features.items()}
        
        data_manager.add_result(
            analysis_type="profiling",
            subject=subject,
            metric_name="profile_classification",
            value=profile,
            metadata={
                "erp_features": erp_serializable,
                "tfr_features": tfr_serializable
            }
        )

    # --- Generate Report ---
    with open(report_path, 'w') as f:
        f.write("# Cross-Analysis Subject Profiling Report\n\n")
        f.write("This report groups subjects into profiles based on the timing and location of their significant ERP and TFR effects.\n\n")
        
        for profile_name, subject_list in sorted(profiles.items()):
            f.write(f"## Profile: {profile_name}\n\n")
            f.write(f"**Subjects:** {', '.join(subject_list)}\n\n")
            f.write("**Characteristics:**\n")
            if profile_name == "Co-localized Early Preparer":
                f.write("- Shows significant ERP and TFR effects early in the preparatory period.\n")
                f.write("- Crucially, the brain regions of these early effects overlap.\n")
                f.write("- **Interpretation:** Strong evidence for a specific, localized preparatory strategy involving both slow potentials and oscillatory changes.\n\n")
            elif profile_name == "Distributed Early Preparer":
                f.write("- Shows significant ERP and TFR effects early in the preparatory period.\n")
                f.write("- However, the brain regions of these effects do *not* overlap.\n")
                f.write("- **Interpretation:** Suggests parallel but spatially distinct preparatory processes (e.g., motor preparation in one area, cognitive load in another).\n\n")
            elif profile_name == "Reactive Responder":
                f.write("- Shows no significant effects during preparation.\n")
                f.write("- Effects only appear *after* the sigh is executed.\n")
                f.write("- **Interpretation:** Employs a reactive strategy with minimal neural preparation.\n\n")
            elif profile_name == "Late/Motor-Focused Preparer":
                f.write("- TFR effects emerge primarily in the mid-to-late preparatory windows.\n")
                f.write("- Often associated with Beta band desynchronization in motor areas.\n")
                f.write("- **Interpretation:** Preparation seems to ramp up closer to the motor event.\n\n")
            elif profile_name == "Sustained ERP Responder (No specific TFR prep)":
                f.write("- Shows a widespread ERP difference throughout the trial.\n")
                f.write("- Lacks specific preparatory TFR effects.\n")
                f.write("- **Interpretation:** May reflect a general state of alertness or cognitive load without specific oscillatory motor preparation signals.\n\n")
            elif profile_name == "Non-Responder":
                f.write("- Shows no significant effects in any analysis.\n")
                f.write("- **Interpretation:** The experimental manipulation did not elicit a detectable difference in brain activity for these subjects.\n\n")
            else: # Mixed/Other
                f.write("- This group shows a mix of effects that does not fit into a clear category.\n\n")

    print(f"Cross-analysis report saved to: {report_path}")

    # --- NEW: Save profiles to a JSON file for other scripts to use ---
    import json
    json_path = op.join('results', 'subject_profiles.json')
    
    # Convert defaultdict to regular dict for JSON serialization
    profiles_dict = {k: v for k, v in profiles.items()}
    
    with open(json_path, 'w') as f:
        json.dump(profiles_dict, f, indent=4)
    print(f"Subject profiles saved to JSON: {json_path}")

