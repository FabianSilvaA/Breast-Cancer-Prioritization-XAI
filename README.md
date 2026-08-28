# Machine Learning and Explainable AI for Breast Cancer Patient Prioritization

This repository provides a **reproducible implementation of the computational methodology described in the manuscript**:

> *Machine Learning and Explainable AI for Breast Cancer Patient Prioritization: An Intelligent Decision-Support Framework*

The implementation was reconstructed from the methodological description, equations, tables, and reported workflow in the manuscript to support transparency and reproducibility.

## Scope

The repository implements the full analytical pipeline described in the article:

1. data quality control;
2. use of the variables selected in Table 1;
3. numerical encoding and standardization;
4. K-Means elbow analysis for `k = 1,...,10`;
5. K-Means stratification with `k = 3`;
6. PCA visualization of the three clusters;
7. Random Forest classification of cluster-derived labels;
8. 80/20 hold-out evaluation;
9. precision, recall, F1-score, accuracy, and confusion matrix;
10. Random Forest feature importance;
11. patient-level LIME explanations;
12. Euclidean-distance-based within-cluster prioritization;
13. proof-of-concept simulation comparing random and prioritized selection.

## Data

The dataset used in this study is publicly available through **Mendeley Data**:

**Breast cancer risk factors in Cuban women**  
Version 1, published 30 August 2024  
DOI: `10.17632/7jhddnpz2p.1`  
Dataset page: https://data.mendeley.com/datasets/7jhddnpz2p/1  
License: **CC BY 4.0**

The Mendeley Data record describes the resource as breast-cancer risk-factor data from Cuban patients collected through medical electronic records.

Download the original dataset from the Mendeley Data record and place/export the analysis file as:

```text
data/breast_cancer.csv
```

If its downloaded filename or format differs, rename/export it to CSV or provide its actual path to `run_pipeline.py`.

The analysis uses the manuscript-selected variables: Age, Menarche, Menopause age, Age at first birth, Breastfeeding duration, First-degree relatives with breast cancer, Atypical hyperplasia, Histological classification, BI-RADS category, BMI, and Weight.

Column aliases can be adjusted in `src/config.py`. The included `synthetic_schema_example.csv` is only a schema example and is not clinical data.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Execution

```bash
python run_pipeline.py --data data/breast_cancer.csv --output outputs
```

## Main outputs

The pipeline generates:

- `figure_1_elbow.png`
- `figure_2_pca_clusters.png`
- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `feature_importance.csv`
- `feature_importance.png`
- `lime_patient_13.csv/png`
- `lime_patient_26.csv/png`
- `all_patient_prioritization.csv`
- `table_top10_cluster_0.csv`
- `table_top10_cluster_1.csv`
- `table_top10_cluster_2.csv`
- `table_7_simulation_summary_reconstructed.csv`

## Important reproducibility statement

The original analysis scripts were not available when this repository was prepared. Therefore, this repository should be understood as a **faithful reconstruction of the computational workflow described in the manuscript**, rather than a byte-for-byte copy of the original analysis environment.

Whenever the manuscript does not report an implementation parameter, the repository uses an explicit and documented reproducibility setting. These choices are centralized in `src/config.py` and are not presented as unpublished original parameters.

The following aspects should therefore be interpreted carefully:

### Random Forest configuration
The manuscript reports bootstrap sampling and random feature subsets but does not specify all hyperparameters. The reconstructed implementation uses a fixed random seed and 500 trees to provide stable, reproducible behavior.

### Feature selection
The manuscript reports Pearson-correlation-based feature selection and Table 1 identifies the variables retained. Because the exact original feature-selection target is not recoverable from the article alone, the repository uses the variables explicitly marked as selected in Table 1.

### Risk-label assignment
K-Means labels are arbitrary integers. The repository follows the manuscript's reported interpretation:
`Cluster 0 = Low`, `Cluster 1 = Medium`, and `Cluster 2 = High`.
The semantic interpretation of these clusters should be verified from centroid profiles when the authorized data are available.

### Simulation
The manuscript specifies the design and reported results of the 300-patient proof-of-concept simulation but does not provide every random-generation parameter. The simulation code therefore reproduces the experimental logic, not an artificial forced match to the published numerical values.

## Reproducibility and scientific use

The classification metrics quantify the Random Forest's agreement with cluster-derived labels. They should not be interpreted as independent clinical validation against an external gold standard.

The simulation is a proof-of-concept operational evaluation and should not be interpreted as prospective clinical validation.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   └── synthetic_schema_example.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   └── pipeline.py
├── outputs/
│   └── .gitkeep
├── run_pipeline.py
├── requirements.txt
├── MANUSCRIPT_CODE_MAP.md
├── REPRODUCIBILITY_STATEMENT.md
├── CITATION.cff
└── README.md
```

## Citation

If this repository is used in connection with the manuscript, please cite the final published article.

## License

The code can be released under an MIT License if the authors decide to do so. No patient-level clinical data are distributed.
