# Manuscript-to-Code Mapping

| Manuscript element | Repository implementation |
|---|---|
| Section 3.1 / Table 1 | `src/config.py`: selected variables and aliases |
| Section 3.2 | `basic_quality_control()` + `build_preprocessor()` |
| Section 3.3 / Eq. (1) | `elbow_analysis()` + `fit_kmeans()` |
| Figure 1 | `outputs/figure_1_elbow.png` |
| Figure 2 | `pca_plot()` |
| Section 3.4 / Eq. (2)-(3) | `evaluate_random_forest()` |
| Section 3.5 | 80/20 stratified hold-out in `evaluate_random_forest()` |
| Tables 2-3 | `classification_report.csv`, `confusion_matrix.csv` |
| Section 3.6 / Eq. (4) | `generate_lime_explanations()` |
| Section 3.7 / Eq. (5)-(8) | `prioritize_within_clusters()` |
| Tables 4-6 | `table_top10_cluster_0.csv` etc. |
| Section 4.5 / Table 7 | `simulate_prioritization()` |

## Scientific caveats

This code deliberately does not claim exact numerical reproduction when the manuscript omits an
implementation detail. In particular, the exact original Random Forest hyperparameters, random seed,
simulation distributions, LIME kernel settings, and semantic cluster-label assignment are not stated
in the paper.

The repository is therefore a **transparent reconstruction**, suitable as a reproducibility base.
