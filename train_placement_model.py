"""
AI Student Placement Predictor
Generic Placement Model Training Script

This script supports multiple academic branches using a common ML schema.

Expected real dataset file, if available:
    placement_training_data.csv

Required columns:
    age
    gender
    ug_degree
    ug_branch
    ug_cgpa
    backlogs
    internships
    projects
    certifications
    coding_skills
    communication_skills
    aptitude_score
    domain_skill_1
    domain_skill_2
    domain_skill_3
    domain_skill_4
    domain_skill_5
    placed

If placement_training_data.csv is not present, the script creates a
synthetic demonstration dataset. Synthetic results must NOT be interpreted
as real-world employment probabilities.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ============================================================
# FILES
# ============================================================

DATA_FILE = "placement_training_data.csv"

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULE_FILE = "recommendation_rules.pkl"

RANDOM_STATE = 42


# ============================================================
# COMMON MODEL FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "age",
    "ug_cgpa",
    "backlogs",
    "internships",
    "projects",
    "certifications",
    "coding_skills",
    "communication_skills",
    "aptitude_score",
    "domain_skill_1",
    "domain_skill_2",
    "domain_skill_3",
    "domain_skill_4",
    "domain_skill_5",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ug_degree",
    "ug_branch",
]

FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "placed"


# ============================================================
# SUPPORTED BRANCHES
# ============================================================

BRANCHES = [
    "Computer Science",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Machine Learning",
    "Cyber Security",
    "Software Engineering",
    "Computer Applications",
    "Mechanical Engineering",
    "Automobile Engineering",
    "Production Engineering",
    "Industrial Engineering",
    "Aeronautical Engineering",
    "Aerospace Engineering",
    "Electrical Engineering",
    "Electronics Engineering",
    "Electronics and Communication Engineering",
    "Biomedical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Mathematics",
    "Statistics",
    "Physics",
    "Chemistry",
    "Environmental Science",
    "Biotechnology",
    "Microbiology",
    "Biochemistry",
    "Biological Sciences",
    "Life Sciences",
    "Genetics",
    "Botany",
    "Zoology",
    "Food Science and Nutrition",
    "Food Technology",
    "Nutrition and Dietetics",
    "Economics",
    "Commerce",
    "Business Administration",
    "Finance",
    "Accounting",
    "Management",
    "Marketing",
    "Human Resources",
    "Psychology",
    "English",
    "Political Science",
    "Sociology",
    "History",
    "Public Administration",
    "Other",
]


DEGREES = [
    "BE",
    "BTech",
    "BSc",
    "BCA",
    "BBA",
    "BCom",
    "BA",
    "Other",
]


GENDERS = [
    "Male",
    "Female",
    "Other",
]


# ============================================================
# BRANCH EFFECTS
# ============================================================

# These values are ONLY used when generating the synthetic
# demonstration dataset.
#
# They are not presented as real-world placement statistics.

BRANCH_EFFECTS = {
    "Computer Science": 0.35,
    "Information Technology": 0.30,
    "Data Science": 0.30,
    "Artificial Intelligence": 0.32,
    "Machine Learning": 0.32,
    "Cyber Security": 0.28,
    "Software Engineering": 0.32,
    "Computer Applications": 0.18,
    "Mechanical Engineering": 0.08,
    "Automobile Engineering": 0.05,
    "Production Engineering": 0.04,
    "Industrial Engineering": 0.06,
    "Aeronautical Engineering": 0.10,
    "Aerospace Engineering": 0.12,
    "Electrical Engineering": 0.12,
    "Electronics Engineering": 0.14,
    "Electronics and Communication Engineering": 0.16,
    "Biomedical Engineering": 0.04,
    "Civil Engineering": 0.02,
    "Chemical Engineering": 0.04,
    "Mathematics": 0.08,
    "Statistics": 0.12,
    "Physics": 0.04,
    "Chemistry": 0.02,
    "Environmental Science": 0.00,
    "Biotechnology": 0.02,
    "Microbiology": 0.00,
    "Biochemistry": 0.02,
    "Biological Sciences": -0.02,
    "Life Sciences": 0.00,
    "Genetics": 0.02,
    "Botany": -0.04,
    "Zoology": -0.04,
    "Food Science and Nutrition": 0.00,
    "Food Technology": 0.02,
    "Nutrition and Dietetics": 0.00,
    "Economics": 0.08,
    "Commerce": 0.06,
    "Business Administration": 0.08,
    "Finance": 0.12,
    "Accounting": 0.08,
    "Management": 0.08,
    "Marketing": 0.06,
    "Human Resources": 0.04,
    "Psychology": 0.02,
    "English": -0.02,
    "Political Science": -0.02,
    "Sociology": -0.02,
    "History": -0.04,
    "Public Administration": 0.00,
    "Other": 0.00,
}


# ============================================================
# HELPERS
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_one_hot_encoder():
    """
    Handles compatibility between different scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_dataset(n_samples=5000, random_state=RANDOM_STATE):
    """
    Generate a synthetic demonstration dataset.

    IMPORTANT:
    This is NOT real placement data.

    It exists so the complete application can be trained and tested
    before a real historical placement dataset is available.
    """

    rng = np.random.default_rng(random_state)

    rows = []

    for _ in range(n_samples):

        age = int(np.clip(rng.normal(22, 1.5), 18, 30))

        gender = rng.choice(
            GENDERS,
            p=[0.50, 0.48, 0.02],
        )

        ug_degree = rng.choice(
            DEGREES,
            p=[
                0.20,  # BE
                0.28,  # BTech
                0.15,  # BSc
                0.10,  # BCA
                0.06,  # BBA
                0.07,  # BCom
                0.08,  # BA
                0.06,  # Other
            ],
        )

        ug_branch = rng.choice(BRANCHES)

        ug_cgpa = float(
            np.clip(
                rng.normal(7.2, 1.1),
                4.0,
                10.0,
            )
        )

        # Most students have zero or one backlog.
        backlogs = int(
            np.clip(
                rng.poisson(0.65),
                0,
                6,
            )
        )

        internships = int(
            np.clip(
                rng.poisson(1.0),
                0,
                4,
            )
        )

        projects = int(
            np.clip(
                rng.poisson(2.0),
                0,
                7,
            )
        )

        certifications = int(
            np.clip(
                rng.poisson(1.7),
                0,
                7,
            )
        )

        coding_skills = int(
            np.clip(
                np.rint(rng.normal(5.8, 1.9)),
                1,
                10,
            )
        )

        communication_skills = int(
            np.clip(
                np.rint(rng.normal(6.0, 1.7)),
                1,
                10,
            )
        )

        aptitude_score = float(
            np.clip(
                rng.normal(65, 15),
                20,
                100,
            )
        )

        domain_skills = np.clip(
            np.rint(rng.normal(6.0, 1.8, size=5)),
            1,
            10,
        ).astype(int)

        domain_skill_1 = int(domain_skills[0])
        domain_skill_2 = int(domain_skills[1])
        domain_skill_3 = int(domain_skills[2])
        domain_skill_4 = int(domain_skills[3])
        domain_skill_5 = int(domain_skills[4])

        domain_average = float(np.mean(domain_skills))

        branch_effect = BRANCH_EFFECTS.get(
            ug_branch,
            0.0,
        )

        # ----------------------------------------------------
        # Synthetic latent placement score
        # ----------------------------------------------------
        #
        # The weights are deliberately distributed across many
        # features so one slider does not completely dominate.
        #
        # Again: these are synthetic assumptions, NOT real-world
        # placement coefficients.
        #
        latent_score = (
            -4.7
            + 0.62 * (ug_cgpa - 7.0)
            - 0.48 * backlogs
            + 0.27 * internships
            + 0.17 * projects
            + 0.08 * certifications
            + 0.13 * (coding_skills - 5)
            + 0.12 * (communication_skills - 5)
            + 0.018 * (aptitude_score - 50)
            + 0.10 * (domain_average - 5)
            + branch_effect
            + rng.normal(0, 0.75)
        )

        probability = float(sigmoid(latent_score))

        placed = int(
            rng.random() < probability
        )

        rows.append(
            {
                "age": age,
                "gender": gender,
                "ug_degree": ug_degree,
                "ug_branch": ug_branch,
                "ug_cgpa": round(ug_cgpa, 2),
                "backlogs": backlogs,
                "internships": internships,
                "projects": projects,
                "certifications": certifications,
                "coding_skills": coding_skills,
                "communication_skills": communication_skills,
                "aptitude_score": round(aptitude_score, 1),
                "domain_skill_1": domain_skill_1,
                "domain_skill_2": domain_skill_2,
                "domain_skill_3": domain_skill_3,
                "domain_skill_4": domain_skill_4,
                "domain_skill_5": domain_skill_5,
                "placed": placed,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REAL DATA VALIDATION
# ============================================================

def validate_dataset(df):
    required_columns = FEATURE_NAMES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The dataset is missing required column(s):\n"
            + "\n".join(f"- {column}" for column in missing)
        )

    df = df.copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Categorical cleanup
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Target conversion
    # --------------------------------------------------------

    if df[TARGET].dtype == object:

        target_map = {
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "placed": 1,
            "not placed": 0,
            "not_placed": 0,
            "y": 1,
            "n": 0,
        }

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_map)
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove invalid target rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[TARGET]
    )

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Keep only valid binary target values
    # --------------------------------------------------------

    df = df[
        df[TARGET].isin([0, 1])
    ].copy()

    if len(df) < 100:
        raise ValueError(
            "The dataset contains fewer than 100 valid rows."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "The 'placed' column must contain both 0 and 1 classes."
        )

    return df


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    base_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=0.7,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                calibrated_model,
            ),
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def calculate_cross_validation_auc(X, y):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        build_model(),
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return {
        "mean_roc_auc": round(
            float(scores.mean()),
            4,
        ),
        "std_roc_auc": round(
            float(scores.std()),
            4,
        ),
        "fold_scores": [
            round(float(score), 4)
            for score in scores
        ],
    }


# ============================================================
# RECOMMENDATION RULES
# ============================================================

def create_recommendation_rules():

    return {
        "cgpa": {
            "message": (
                "Work on maintaining or improving CGPA through "
                "consistent academic preparation and revision."
            )
        },
        "backlogs": {
            "message": (
                "Prioritize clearing academic backlogs because "
                "some placement opportunities may have eligibility restrictions."
            )
        },
        "internships": {
            "message": (
                "Seek relevant internships, industry projects, "
                "research work, or supervised practical experience."
            )
        },
        "projects": {
            "message": (
                "Build practical projects related to your branch "
                "and target career so that you can demonstrate applied skills."
            )
        },
        "certifications": {
            "message": (
                "Consider relevant certifications that strengthen "
                "your target career direction rather than collecting unrelated certificates."
            )
        },
        "communication_skills": {
            "message": (
                "Practice communication, presentations, group discussions, "
                "technical explanations, and mock interviews."
            )
        },
        "aptitude_score": {
            "message": (
                "Practice quantitative aptitude, logical reasoning, "
                "verbal reasoning, and timed problem-solving regularly."
            )
        },
    }


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    df,
    dataset_mode,
):

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Training mode : {dataset_mode}"
    )

    print(
        f"Rows          : {len(df)}"
    )

    print(
        f"Columns       : {len(df.columns)}"
    )

    print(
        f"Placed = 1    : {(df[TARGET] == 1).sum()}"
    )

    print(
        f"Placed = 0    : {(df[TARGET] == 0).sum()}"
    )

    print(
        f"Placement rate: {df[TARGET].mean() * 100:.2f}%"
    )

    print("\nBranch distribution:")

    branch_counts = (
        df["ug_branch"]
        .value_counts()
        .sort_index()
    )

    print(
        branch_counts.to_string()
    )


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    metrics,
    cv_metrics,
    dataset_mode,
    dataset_size,
):

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    joblib.dump(
        FEATURE_NAMES,
        FEATURE_FILE,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": (
            "Calibrated Logistic Regression "
            "with Standardized Numeric and One-Hot Categorical Features"
        ),
        "dataset_mode": dataset_mode,
        "dataset_size": int(dataset_size),
        "target_column": TARGET,
        "feature_names": FEATURE_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "evaluation_metrics": metrics,
        "cross_validation": cv_metrics,
        "random_state": RANDOM_STATE,
        "probability_note": (
            "The model probability is an estimated calibrated "
            "classification probability for the population represented "
            "by the training data. It is not a guarantee of employment."
        ),
        "synthetic_warning": (
            "Synthetic probabilities are demonstration-only and "
            "must not be interpreted as real-world placement probabilities."
            if dataset_mode == "synthetic_demo"
            else None
        ),
        "domain_skill_design": (
            "domain_skill_1 through domain_skill_5 represent the "
            "five branch-specific skill ratings. Their labels are "
            "determined by the application UI based on ug_branch."
        ),
        "coding_skill_design": (
            "coding_skills is an independent feature and is not derived "
            "from the branch-specific domain skill ratings."
        ),
    }

    joblib.dump(
        metadata,
        METADATA_FILE,
    )

    # --------------------------------------------------------
    # Recommendation rules
    # --------------------------------------------------------

    rules = create_recommendation_rules()

    joblib.dump(
        rules,
        RULE_FILE,
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("🎓 AI STUDENT PLACEMENT PREDICTOR")
    print("GENERIC ALL-BRANCH MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load real data if available
    # --------------------------------------------------------

    if os.path.exists(DATA_FILE):

        print(
            f"\nFound real dataset: {DATA_FILE}"
        )

        df = pd.read_csv(
            DATA_FILE
        )

        df = validate_dataset(
            df
        )

        dataset_mode = "real_dataset"

    else:

        print(
            f"\n{DATA_FILE} was not found."
        )

        print(
            "Creating a synthetic demonstration dataset..."
        )

        df = generate_synthetic_dataset(
            n_samples=5000,
            random_state=RANDOM_STATE,
        )

        dataset_mode = "synthetic_demo"

    print_dataset_summary(
        df,
        dataset_mode,
    )

    # --------------------------------------------------------
    # Split X / y
    # --------------------------------------------------------

    X = df[
        FEATURE_NAMES
    ].copy()

    y = df[
        TARGET
    ].astype(int)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    print(
        "\nFitting calibrated logistic regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating test set..."
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    print("\nClassification Report:")

    print(
        metrics["classification_report"]
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print(
        "\nRunning 5-fold cross-validation..."
    )

    cv_metrics = calculate_cross_validation_auc(
        X,
        y,
    )

    print(
        f"Mean CV ROC-AUC: "
        f"{cv_metrics['mean_roc_auc']:.4f}"
    )

    print(
        f"CV ROC-AUC Std : "
        f"{cv_metrics['std_roc_auc']:.4f}"
    )

    print(
        "Fold scores:",
        cv_metrics["fold_scores"],
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    print(
        "\nSaving model artifacts..."
    )

    metadata = save_artifacts(
        model=model,
        metrics=metrics,
        cv_metrics=cv_metrics,
        dataset_mode=dataset_mode,
        dataset_size=len(df),
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nCreated:"
    )

    print(
        f"  ✓ {MODEL_FILE}"
    )

    print(
        f"  ✓ {FEATURE_FILE}"
    )

    print(
        f"  ✓ {METADATA_FILE}"
    )

    print(
        f"  ✓ {RULE_FILE}"
    )

    print(
        "\nModel features:"
    )

    for index, feature in enumerate(
        FEATURE_NAMES,
        start=1,
    ):
        print(
            f"  {index:02d}. {feature}"
        )

    print(
        "\nDataset mode:",
        metadata["dataset_mode"],
    )

    if dataset_mode == "synthetic_demo":

        print("\n⚠️ IMPORTANT:")
        print(
            "This model was trained using synthetic demonstration data."
        )
        print(
            "Its probabilities are NOT real-world employment probabilities."
        )
        print(
            "For meaningful real-world evaluation, replace the synthetic"
        )
        print(
            "dataset with historical placement outcome data."
        )

    else:

        print(
            "\n✓ Model was trained using placement_training_data.csv."
        )

    print(
        "\nYou can now use the generated .pkl files with streamlit_app.py."
    )


if __name__ == "__main__":
    main()
