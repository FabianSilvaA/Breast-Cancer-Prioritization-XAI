from dataclasses import dataclass

RANDOM_STATE = 42
MISSING_THRESHOLD = 0.30
N_CLUSTERS = 3
TEST_SIZE = 0.20

# Variables marked "Yes" in Table 1 of the manuscript.
SELECTED_FEATURES = [
    "age",
    "menarche",
    "menopause_age",
    "age_first_birth",
    "breastfeeding_duration",
    "first_degree_relatives_bc",
    "atypical_hyperplasia",
    "histological_classification",
    "birads",
    "bmi",
    "weight",
]

# Adapt aliases to the exact headers in the authorized dataset.
COLUMN_ALIASES = {
    "age": ["age", "Age", "edad"],
    "menarche": ["menarche", "Menarche", "menarquia"],
    "menopause_age": ["menopause_age", "Menopause age", "menopause", "menopausia"],
    "age_first_birth": ["age_first_birth", "Agefirst", "agefirst", "Age at first birth"],
    "breastfeeding_duration": ["breastfeeding_duration", "Breastfeeding duration", "breastfeeding"],
    "first_degree_relatives_bc": ["first_degree_relatives_bc", "Nrelbc", "nrelbc"],
    "atypical_hyperplasia": ["atypical_hyperplasia", "Atypical hyperplasia"],
    "histological_classification": ["histological_classification", "Histological classification"],
    "birads": ["birads", "BIRADS", "BI-RADS", "BIRADS category"],
    "bmi": ["bmi", "BMI", "Body Mass Index"],
    "weight": ["weight", "Weight", "peso"],
}

# Types inferred from Table 1.
CONTINUOUS_FEATURES = [
    "age",
    "menarche",
    "age_first_birth",
    "breastfeeding_duration",
    "bmi",
    "weight",
]

CATEGORICAL_OR_ORDINAL_FEATURES = [
    "menopause_age",
    "first_degree_relatives_bc",
    "atypical_hyperplasia",
    "histological_classification",
    "birads",
]

# Literal mapping reported by the manuscript.
# IMPORTANT: K-Means IDs are arbitrary; authors must verify this mapping against the original analysis.
CLUSTER_RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}
RISK_WEIGHT = {"Low": 1.0, "Medium": 2.0, "High": 3.0}

RF_PARAMS = dict(
    n_estimators=500,
    random_state=RANDOM_STATE,
    bootstrap=True,
    n_jobs=-1,
    class_weight=None,
)

# Illustrative LIME row positions. The paper names Patients 13 and 26.
# Python uses zero-based positions, so manuscript patient 13 -> row 12, patient 26 -> row 25
# only if patient numbering corresponds exactly to row order after preprocessing.
LIME_PATIENT_POSITIONS = [12, 25]
