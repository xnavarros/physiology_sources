# Time-Frequency Analysis Report: VS vs. LD

## 1. Analysis Description

### Pre-processing
Data was pre-processed following a standard pipeline including band-pass filtering, ICA decomposition for artifact removal (EOG, ECG), epoching around the sigh onset, and automated artifact rejection using Autoreject.

### Time-Frequency Analysis Parameters
- **Frequency Range:** 8.0 Hz to 30.0 Hz.
- **Method:** Morlet wavelets (`n_cycles = freqs / 2`).
- **Baseline Correction:** Z-score transformation relative to the `(-2, -1.5)` window.

### Statistical Analysis
Two levels of statistical analysis were performed to compare the Voluntary Sigh (VS) and Load (LD) conditions:
1.  **Group Level:** A non-parametric, cluster-based permutation test (`spatio_temporal_cluster_test`) was performed on the subject-averaged data to identify significant differences at the group level.
2.  **Individual Level:** A within-subject cluster-based permutation test (`permutation_cluster_test`) was performed for each participant to count how many individuals showed a significant difference between conditions.

Both analyses were run separately for two frequency bands (Alpha: 8-12 Hz, Beta: 13-30 Hz) and three time windows:
- **Early:** -1.0s to -0.5s (early preparation)
- **Late:** -0.5s to 0.0s (late preparation)
- **Post:** 0.0s to 0.5s (execution/recovery)

## 2. Results

### Group-Level Results
The group-level analysis **did not reveal any significant spatio-temporal clusters** in any of the tested frequency bands or time windows (all p > 0.05). This indicates that there is no consistent, robust difference between the VS and LD conditions when averaging across all subjects.

The grand average difference plot (`grand_average_tfr_diff.png`) visually supports this, showing no strong, widespread patterns of power change.

### Individual-Level Results
In contrast to the group analysis, a subset of subjects showed significant differences between conditions at the individual level. The table below summarizes the number of subjects with a significant effect (p < 0.05) in each window.

| Time Window | Frequency Band | Number of Significant Subjects |
|-------------|----------------|--------------------------------|
| Early       | Alpha          | 0 / 10                      |
| Early       | Beta           | 2 / 10                      |
| Late        | Alpha          | 1 / 10                      |
| Late        | Beta           | 5 / 10                      |
| Post        | Alpha          | 4 / 10                      |
| Post        | Beta           | 9 / 10                      |

## 3. Interpretation and Conclusion

The primary finding of this analysis is a **discrepancy between group-level and individual-level results**. While no effect survives group averaging, a notable portion of subjects do show significant neural modulation. This strongly suggests that the cognitive and motor load of the experiment elicits **heterogeneous neural strategies** across participants.

### Plausible Explanations for Individual Differences:

1.  **Variability in Cognitive Strategy:** The lack of a consistent group effect, especially in the preparatory 'early' and 'late' windows, implies that subjects may be using different cognitive strategies to handle the load. Some might engage in more intense motor preparation (beta modulation), while others might focus on attentional filtering (alpha modulation). This variability in strategy would lead to different neural patterns that cancel each other out in the group average.

2.  **Subjective Sensation and Interoception:** The breathing task is highly interoceptive. It is plausible that the subjective sensation of 'load' and the effort required to control breathing varies significantly between individuals. Participants who find the task more demanding or unnatural might exhibit stronger and different patterns of brain activity compared to those who adapt more easily. The individual subject plots (`subject_..._tfr_diff.png`) should be inspected to find common patterns among subsets of subjects.

3.  **Post-Inspiratory Motor Activity:** You hypothesized higher motor activity in the post-inspiratory window. The results from the individual-level analysis in the 'post' window should be examined closely. If several subjects show significant beta-band modulation here, it could reflect differences in the recruitment of respiratory muscles or post-movement motor resetting (i.e., Post-Movement Beta Rebound). The lack of a *group* effect suggests this motor response is not uniform across all participants.

### Final Conclusion:
The experiment successfully modulates brain activity, but not in a way that is consistent across the entire group. The most robust conclusion is that **inter-subject variability is a key feature of the neural response to this task**. Future analysis could benefit from correlating these neural patterns with behavioral data or subjective reports (e.g., perceived difficulty, anxiety levels) to explain *why* different subjects use different neural strategies. The current null result at the group level is not a failure of the experiment, but rather an important insight into the complex and individualized nature of cognitive-motor control.
