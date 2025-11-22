#!/bin/bash
# This script runs the complete analysis pipeline for slow potentials.
# It executes the preprocessing script first, followed by the specific
# slow-wave analyses in a logical sequence.

# Function to ask for confirmation if output exists
run_step() {
    local step_desc="$1"
    local check_path="$2"
    local cmd="$3"

    echo ""
    echo "$step_desc"
    
    if [ -e "$check_path" ]; then
        echo "Output found at: $check_path"
        # Read user input with a prompt, compatible with zsh/bash
        echo -n "Output already exists. Re-run? (y/N): "
        read response
        case "$response" in
            [yY][eE][sS]|[yY]) 
                echo "Re-running..."
                $cmd
                ;;
            *)
                echo "Skipping..."
                ;;
        esac
    else
        echo "No previous output found. Running..."
        $cmd
    fi
}

# Define paths (matching config.yaml structure)
PREPROC_DIR="data/preprocessed_slow_potentials"
ERP_RESULTS_DIR="results/erp_analysis"
CLUSTER_RESULTS_DIR="results/cluster_permutation_test"
PROFILES_JSON="results/subject_profiles.json"
SOURCE_FIG_DIR="figures/source_analysis_profiles"

# STEP 1
run_step "--- STEP 1/5: Preprocessing ---" "$PREPROC_DIR" "python scripts/0_preprocess.py"

# STEP 2
run_step "--- STEP 2/5: Readiness Potential (ERP) Analysis ---" "$ERP_RESULTS_DIR" "python scripts/analysis_slow_wave/1_readiness_potential.py"

# STEP 3
run_step "--- STEP 3/5: PPI Spatio-Temporal Cluster Test ---" "$CLUSTER_RESULTS_DIR" "python scripts/analysis_slow_wave/2_ppi_cluster_test.py"

# STEP 4
run_step "--- STEP 4/5: Subject Profiling ---" "$PROFILES_JSON" "python scripts/analysis_profiling/1_run_profiling.py"

# STEP 5
run_step "--- STEP 5/5: Slow Wave Source Analysis by Profile ---" "$SOURCE_FIG_DIR" "python scripts/analysis_slow_wave/4_source_analysis_by_profile.py"

echo ""
echo "--- Slow Wave Analysis Pipeline Complete ---"
