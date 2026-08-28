from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import (
    RANDOM_STATE,
    MISSING_THRESHOLD,
    N_CLUSTERS,
    TEST_SIZE,
    SELECTED_FEATURES,
    COLUMN_ALIASES,
    CONTINUOUS_FEATURES,
    CATEGORICAL_OR_ORDINAL_FEATURES,
    CLUSTER_RISK_MAP,
    RISK_WEIGHT,
    RF_PARAMS,
    LIME_PATIENT_POSITIONS,
)


def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known dataset headers to canonical names."""
    rename = {}
    lower_lookup = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            key = str(alias).strip().lower()
            if key in lower_lookup:
                found = lower_lookup[key]
                break
        if found is not None:
            rename[found] = canonical

    out = df.rename(columns=rename).copy()
    missing = [c for c in SELECTED_FEATURES if c not in out.columns]
    if missing:
        raise ValueError(
            "The following manuscript-selected variables were not found in the dataset: "
            + ", ".join(missing)
            + ". Update COLUMN_ALIASES in src/config.py."
        )
    return out


def basic_quality_control(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the manuscript's broad QC description:
    - remove exact duplicate observations;
    - exclude variables above 30% missingness;
    - retain the manuscript-selected variables.

    Rows are not automatically dropped solely because one field is missing; remaining
    missing values are imputed in the preprocessing pipeline to avoid unnecessary loss.
    """
    df = df.drop_duplicates().copy()

    miss_rate = df.isna().mean()
    too_missing = set(miss_rate[miss_rate > MISSING_THRESHOLD].index)
    required_too_missing = [c for c in SELECTED_FEATURES if c in too_missing]
    if required_too_missing:
        raise ValueError(
            "Selected manuscript variables exceed the 30% missingness threshold: "
            + ", ".join(required_too_missing)
        )

    return df[SELECTED_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", numeric, CONTINUOUS_FEATURES),
            ("categorical", categorical, CATEGORICAL_OR_ORDINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return CONTINUOUS_FEATURES + CATEGORICAL_OR_ORDINAL_FEATURES


def elbow_analysis(X: np.ndarray, output_dir: Path) -> pd.DataFrame:
    rows = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km.fit(X)
        rows.append({"k": k, "wcss": float(km.inertia_)})

    elbow = pd.DataFrame(rows)
    elbow.to_csv(output_dir / "elbow_wcss.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(elbow["k"], elbow["wcss"], marker="o")
    ax.axvline(3, linestyle="--")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_title("Elbow Method")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_elbow.png", dpi=300)
    plt.close(fig)
    return elbow


def fit_kmeans(X: np.ndarray) -> KMeans:
    return KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit(X)


def pca_plot(X: np.ndarray, cluster_ids: np.ndarray, output_dir: Path) -> None:
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == cluster
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=18,
            alpha=0.7,
            label=f"Cluster {cluster} ({CLUSTER_RISK_MAP.get(int(cluster), cluster)})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Patient Clusters Projected onto the First Two Principal Components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_pca_clusters.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(
        {
            "PC1": proj[:, 0],
            "PC2": proj[:, 1],
            "cluster_id": cluster_ids,
            "risk_level": [CLUSTER_RISK_MAP[int(c)] for c in cluster_ids],
        }
    ).to_csv(output_dir / "pca_coordinates.csv", index=False)


def evaluate_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
):
    """
    Reconstructed hold-out evaluation.

    IMPORTANT:
    The manuscript does not state whether K-Means was fitted before or after the 80/20 split.
    This function evaluates the RF on unseen observations given cluster-derived labels.

    For a stricter leakage-resistant prospective workflow, K-Means should be fitted on the
    training partition only, and test observations should be assigned to the fitted centroids.
    """
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    report_dict = classification_report(
        y_test,
        pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=["True Low", "True Medium", "True High"],
        columns=["Pred Low", "Pred Medium", "Pred High"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High"])
    disp.plot(ax=ax, values_format="d")
    ax.set_title("Random Forest Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(output_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = fi.sort_values("importance", ascending=True)
    ax.barh(ordered["feature"], ordered["importance"])
    ax.set_xlabel("Mean Decrease in Impurity Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close(fig)

    summary = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "test_class_counts": {
            str(int(c)): int((y_test == c).sum()) for c in np.unique(y_test)
        },
    }
    (output_dir / "rf_summary.json").write_text(json.dumps(summary, indent=2))

    joblib.dump(rf, output_dir / "random_forest.joblib")
    return rf, train_idx, test_idx, pred, report_df, cm_df


def prioritize_within_clusters(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    centroids: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for i, (x, cluster) in enumerate(zip(X, cluster_ids), start=1):
        d = float(np.linalg.norm(x - centroids[int(cluster)]))
        risk = CLUSTER_RISK_MAP[int(cluster)]
        rows.append(
            {
                "patient_row": i,
                "cluster_id": int(cluster),
                "risk_level": risk,
                "distance_to_centroid": d,
            }
        )

    result = pd.DataFrame(rows)

    result["normalized_distance"] = 0.0
    for cluster in sorted(result["cluster_id"].unique()):
        mask = result["cluster_id"] == cluster
        d = result.loc[mask, "distance_to_centroid"]
        d_min, d_max = float(d.min()), float(d.max())
        if np.isclose(d_max, d_min):
            norm = np.zeros(len(d))
        else:
            norm = (d - d_min) / (d_max - d_min)
        result.loc[mask, "normalized_distance"] = norm

    result["risk_weight"] = result["risk_level"].map(RISK_WEIGHT)
    result["weighted_prioritization"] = (
        result["normalized_distance"] * result["risk_weight"]
    )

    # Within each risk group, the manuscript tables rank larger values first.
    result["within_cluster_rank"] = (
        result.groupby("cluster_id")["weighted_prioritization"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Operational priority: risk stratum first, then descending within-cluster score.
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    result["_risk_order"] = result["risk_level"].map(risk_order)
    result = result.sort_values(
        ["_risk_order", "weighted_prioritization"],
        ascending=[True, False],
    ).drop(columns="_risk_order")

    result.to_csv(output_dir / "all_patient_prioritization.csv", index=False)

    for cluster in [0, 1, 2]:
        top10 = (
            result[result["cluster_id"] == cluster]
            .sort_values("weighted_prioritization", ascending=False)
            .head(10)
        )
        top10.to_csv(
            output_dir / f"table_top10_cluster_{cluster}.csv",
            index=False,
        )
    return result


def generate_lime_explanations(
    rf: RandomForestClassifier,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        warnings.warn(f"LIME not available; explanations skipped: {exc}")
        return

    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=["Low", "Medium", "High"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    for manuscript_patient, pos in [(13, LIME_PATIENT_POSITIONS[0]), (26, LIME_PATIENT_POSITIONS[1])]:
        if pos >= len(X):
            continue

        exp = explainer.explain_instance(
            X[pos],
            rf.predict_proba,
            num_features=min(10, X.shape[1]),
            top_labels=3,
        )

        # Save textual contributions for reproducibility.
        rows = []
        for label in sorted(exp.local_exp.keys()):
            for feat_idx, weight in exp.local_exp[label]:
                rows.append(
                    {
                        "patient": manuscript_patient,
                        "class_id": int(label),
                        "class_name": ["Low", "Medium", "High"][int(label)],
                        "feature": feature_names[int(feat_idx)],
                        "lime_weight": float(weight),
                    }
                )
        pd.DataFrame(rows).to_csv(
            output_dir / f"lime_patient_{manuscript_patient}.csv",
            index=False,
        )

        fig = exp.as_pyplot_figure(label=int(rf.predict(X[[pos]])[0]))
        fig.suptitle(f"LIME Analysis for Patient {manuscript_patient}")
        fig.tight_layout()
        fig.savefig(output_dir / f"lime_patient_{manuscript_patient}.png", dpi=300)
        plt.close(fig)


def simulate_prioritization(
    output_dir: Path,
    n_patients: int = 300,
    capacity: int = 60,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Proof-of-concept reconstruction.

    The manuscript does not provide exact distributions or the severity formula, so this
    reproduces the experimental design rather than the exact reported 0.92/1.66 and 14/60 values.
    """
    rng = np.random.default_rng(random_state)

    # Balanced illustrative population; authors should replace with the original simulation design.
    risk_level = rng.choice(["Low", "Medium", "High"], size=n_patients, p=[0.40, 0.40, 0.20])
    age = np.clip(rng.normal(58, 12, n_patients), 25, 90)
    bmi = np.clip(rng.normal(27, 5, n_patients), 16, 50)
    comorbidities = rng.poisson(1.2, n_patients)

    base = np.select(
        [risk_level == "Low", risk_level == "Medium", risk_level == "High"],
        [0.55, 1.05, 1.55],
        default=0.55,
    )
    severity = np.clip(
        base
        + 0.008 * (age - 50)
        + 0.015 * np.maximum(bmi - 25, 0)
        + 0.12 * comorbidities
        + rng.normal(0, 0.15, n_patients),
        0,
        None,
    )

    sim = pd.DataFrame(
        {
            "patient": np.arange(1, n_patients + 1),
            "age": age,
            "bmi": bmi,
            "comorbidities": comorbidities,
            "risk_level": risk_level,
            "severity": severity,
        }
    )

    random_sel = sim.sample(n=capacity, random_state=random_state)

    order = pd.Categorical(
        sim["risk_level"],
        categories=["High", "Medium", "Low"],
        ordered=True,
    )
    prioritized = (
        sim.assign(_risk_order=order)
        .sort_values(["_risk_order", "severity"], ascending=[True, False])
        .head(capacity)
    )

    summary = pd.DataFrame(
        [
            {
                "Scenario": "Random Selection",
                "Average Severity": random_sel["severity"].mean(),
                "Total High-Risk Patients Selected": (random_sel["risk_level"] == "High").sum(),
            },
            {
                "Scenario": "Prioritized Selection",
                "Average Severity": prioritized["severity"].mean(),
                "Total High-Risk Patients Selected": (prioritized["risk_level"] == "High").sum(),
            },
        ]
    )
    sim.to_csv(output_dir / "simulation_population.csv", index=False)
    random_sel.to_csv(output_dir / "simulation_random_selected.csv", index=False)
    prioritized.to_csv(output_dir / "simulation_prioritized_selected.csv", index=False)
    summary.to_csv(output_dir / "table_7_simulation_summary_reconstructed.csv", index=False)
    return summary


def run_pipeline(data_path: str | Path, output_dir: str | Path = "outputs"):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_path)
    resolved = resolve_columns(raw)
    X_df = basic_quality_control(resolved)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_df)
    feature_names = feature_names_from_preprocessor(preprocessor)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")

    elbow_analysis(X, output_dir)

    kmeans = fit_kmeans(X)
    cluster_ids = kmeans.labels_.astype(int)
    joblib.dump(kmeans, output_dir / "kmeans.joblib")

    cluster_profile = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
    cluster_profile.insert(0, "cluster_id", range(N_CLUSTERS))
    cluster_profile.insert(
        1,
        "risk_level",
        [CLUSTER_RISK_MAP[i] for i in range(N_CLUSTERS)],
    )
    cluster_profile.to_csv(output_dir / "cluster_centroids_standardized.csv", index=False)

    pca_plot(X, cluster_ids, output_dir)

    rf, train_idx, test_idx, pred, report_df, cm_df = evaluate_random_forest(
        X, cluster_ids, feature_names, output_dir
    )

    prioritization = prioritize_within_clusters(
        X, cluster_ids, kmeans.cluster_centers_, output_dir
    )

    generate_lime_explanations(rf, X, feature_names, output_dir)
    sim_summary = simulate_prioritization(output_dir)

    manifest = {
        "n_raw_rows": int(len(raw)),
        "n_rows_after_qc": int(len(X_df)),
        "n_features": int(X.shape[1]),
        "features": feature_names,
        "cluster_counts": {
            str(int(c)): int((cluster_ids == c).sum()) for c in sorted(np.unique(cluster_ids))
        },
        "risk_mapping_assumption": CLUSTER_RISK_MAP,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Pipeline completed.")
    print(json.dumps(manifest, indent=2))
    print("\nSimulation proof-of-concept:")
    print(sim_summary.to_string(index=False))

    return {
        "preprocessor": preprocessor,
        "kmeans": kmeans,
        "random_forest": rf,
        "prioritization": prioritization,
        "classification_report": report_df,
        "confusion_matrix": cm_df,
        "simulation_summary": sim_summary,
    }
