# Project Context & Domain Knowledge

## Terminology & Abbreviations
*   **VS**: "Ventilation Spontanée" (French) -> Spontaneous Breathing. For writing repors, use 'SV' instead. This is the control condition where subjects breathe normally without resistance.
*   **LD**: "Loaded Breathing" (English). This is the experimental condition where an inspiratory load (resistence) is applied to the breathing.
*   **PPI**: Pre-Inspiratory Potentials. For reports use the acronym "PIP". These are slow cortical potentials (similar to Readiness Potentials or Bereitschaftspotential) that precede the onset of inspiration.
*   **RP**: Readiness Potential. A slow negative shift in EEG activity preceding a voluntary movement. In this study, we are investigating the "Respiratory Readiness Potential" (RRP) or PIP.

## Scientific Goals
1.  **Compare VS vs. LD**: To determine if the cortical preparation for breathing (PIP) is different when breathing is difficult (Loaded) compared to normal (Spontaneous).
2.  **Profile Analysis**: Subjects are categorized into profiles based on when their cortical activity begins relative to the breath.
3.  **Frequency Bands**:
    *   **Slow Potentials (< 1 Hz)**: The primary marker of motor preparation (CNV/RP).
    *   **Alpha (8-12 Hz) / Beta (13-30 Hz)**: Markers of cortical activation/idling. Desynchronization (power decrease) usually indicates active processing.

## Data Structure
*   **Epochs**: Time-locked to the onset of inspiration (Marker: `Response/R128`).
*   **Baseline**: In the analysis -2s to -1.5s pre-inspiration periods.
*   **Preprocessing**:
    *   **Slow Potentials**: 0.1 - 40 Hz filter. Preserves DC shifts.
    *   **Alpha/Beta**: Extracted from the same 0.1-40 Hz preprocessed data.

## Analysis Pipelines
1.  **Slow Wave Analysis**: Focuses on the time-domain waveform (ERP) of the PPI.
2.  **Alpha/Beta Analysis**: Focuses on the time-frequency power changes (ERD/ERS) associated with breathing.
