#!/bin/bash
# This script runs the complete analysis pipeline for Alpha/Beta oscillations.
# It assumes preprocessing (Step 1 of slow wave pipeline) has already been run.

# Function to ask for confirmation if output exists
run_step() {
    local step_desc="$1"
    local check_path="$2"
    local cmd="$3"

    echo ""
    echo "$step_desc"
    
    if [ -e "$check_path" ]; then
        echo "Output found at: $check_path"
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

# Define paths
TFR_RESULTS_DIR="results/TF_analysis"
PROFILES_JSON="results/subject_profiles.json"
SOURCE_RESULTS_DIR="results/source_analysis_alpha_beta"

echo "--- Alpha/Beta Analysis Pipeline ---"
echo "Note: This pipeline uses the preprocessed data from the slow-wave analysis."

# STEP 1
run_step "--- STEP 1/3: Time-Frequency Analysis (Sensor Level) ---" "$TFR_RESULTS_DIR" "python scripts/analysis_alpha_beta/1_time_frequency.py"

# STEP 2
# We re-run profiling here because the TFR analysis might have updated the summary file
run_step "--- STEP 2/3: Update Subject Profiling (incorporating TFR results) ---" "$PROFILES_JSON" "python scripts/analysis_profiling/1_run_profiling.py"

# STEP 3
run_step "--- STEP 3/3: Source Analysis (Alpha/Beta) by Profile ---" "$SOURCE_RESULTS_DIR" "python scripts/analysis_alpha_beta/2_source_analysis.py"

echo ""
echo "--- Alpha/Beta Analysis Pipeline Complete ---"
