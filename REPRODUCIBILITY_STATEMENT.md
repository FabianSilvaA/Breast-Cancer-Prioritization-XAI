# Reproducibility Statement

This repository was prepared to provide a transparent and executable implementation of the analytical
workflow described in the manuscript.

The implementation follows the published methodological description, including:

- the variables identified as selected in Table 1;
- the 30% missingness threshold;
- categorical encoding and standardization;
- K-Means clustering with three strata;
- elbow analysis;
- PCA visualization;
- Random Forest classification;
- an 80/20 hold-out evaluation;
- LIME-based local explanations;
- Euclidean-distance-based prioritization;
- the 300-patient proof-of-concept prioritization simulation.

The repository does **not** claim that undocumented hyperparameters are the exact settings of the
original computational environment. Where the manuscript is silent, explicit reproducibility choices
are used and documented.

This distinction is important because scientific reproducibility requires both transparency and
traceability. The repository therefore avoids fabricating undocumented details solely to force exact
numerical agreement with the article.

When the authorized dataset is supplied, the implementation can be used to verify the reported
analytical behavior and to generate the principal figures, tables, and evaluation outputs.


## Public dataset

The source dataset is publicly available from Mendeley Data at `https://data.mendeley.com/datasets/7jhddnpz2p/1` (DOI `10.17632/7jhddnpz2p.1`, Version 1, CC BY 4.0). This repository links to the authoritative record to preserve provenance and licensing information.
