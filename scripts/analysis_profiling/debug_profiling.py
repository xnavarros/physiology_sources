
from collections import defaultdict
import re

def parse_summary_line(line):
    if ':' not in line: return None, []
    subject, effects_str = line.strip().split(':', 1)
    # Naive split
    effects = effects_str.split(',') if effects_str else []
    return subject, effects

def get_features(effects):
    features = {
        'early': set(), 'mid': set(), 'late': set(), 'post': set()
    }
    for eff in effects:
        if not eff: continue
        
        # Naive region extraction
        regions = set()
        if '(' in eff:
            found = re.findall(r'\((.*?)\)', eff)
            if found and found[0]:
                regions.update(found[0].split(','))
        
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
    has_prep_erp = erp_features['early'] or erp_features['mid'] or erp_features['late']
    has_prep_tfr = tfr_features['early'] or tfr_features['mid'] or tfr_features['late']
    has_early_tfr = bool(tfr_features['early'])
    early_overlap = erp_features['early'] and tfr_features['early'] and (erp_features['early'] & tfr_features['early'])
    
    print(f"  has_prep_erp: {bool(has_prep_erp)}")
    print(f"  has_prep_tfr: {bool(has_prep_tfr)}")
    print(f"  has_early_tfr: {has_early_tfr}")
    
    if has_prep_erp and has_early_tfr:
        if early_overlap: return "Co-localized Early Preparer"
        else: return "Distributed Early Preparer"
    
    if not has_prep_erp and not has_prep_tfr:
        if tfr_features['post'] or erp_features['post']: return "Reactive Responder"
        else: return "Non-Responder"

    if has_prep_tfr and not has_early_tfr:
        return "Late/Motor-Focused Preparer"
        
    if has_prep_erp and not has_prep_tfr:
        return "Sustained ERP Responder (No specific TFR prep)"

    return "Mixed/Other"

# Test Data from files
erp_line = "JS08:Early(Central,Frontal,Parietal)"
tfr_line = "JS08:Low Beta-Late(Central,Frontal,Parietal),High Beta-Late(Central,Frontal,Parietal),Alpha-Post(Central,Frontal,Parietal),Low Beta-Post(Central,Frontal,Parietal),High Beta-Post(Central,Frontal,Parietal)"

print("--- JS08 Analysis ---")
s, erp_eff = parse_summary_line(erp_line)
print(f"ERP Effects parsed: {erp_eff}")
erp_feat = get_features(erp_eff)
print(f"ERP Features: {erp_feat}")

s, tfr_eff = parse_summary_line(tfr_line)
print(f"TFR Effects parsed: {tfr_eff}")
tfr_feat = get_features(tfr_eff)
print(f"TFR Features: {tfr_feat}")

profile = classify_subject(erp_feat, tfr_feat)
print(f"Profile: {profile}")
