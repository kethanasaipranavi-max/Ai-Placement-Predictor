"""
AI Student Placement Predictor
Training Pipeline - Large Placement Dataset

Expected source:
Kaggle Student Placement Prediction dataset
8,000 training records + 2,000 test records.

The trained model is intentionally limited to features that
the Streamlit application actually collects.
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULES_FILE = "recommendation_rules.pkl"


# ============================================================
# FEATURES USED BY STREAMLIT
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

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "placed"


# ============================================================
# BRANCH SKILLS
# ============================================================

BRANCH_SKILLS = {
    "Computer Science": [
        "Programming",
        "Data Structures & Algorithms",
        "Databases",
        "Software Development",
        "Problem Solving",
    ],
    "Information Technology": [
        "Programming",
        "Networking",
        "Databases",
        "Cloud Computing",
        "System Administration",
    ],
    "Data Science": [
        "Python / Programming",
        "Statistics",
        "Data Analysis",
        "Machine Learning",
        "Data Visualization",
    ],
    "Artificial Intelligence": [
        "Programming",
        "Machine Learning",
        "Deep Learning",
        "Mathematics",
        "Data Analysis",
    ],
    "Machine Learning": [
        "Python",
        "Machine Learning",
        "Statistics",
        "Deep Learning",
        "Data Processing",
    ],
    "Cyber Security": [
        "Networking",
        "Cyber Security Concepts",
        "Linux",
        "Security Tools",
        "Ethical Hacking",
    ],
    "Software Engineering": [
        "Programming",
        "Software Design",
        "Databases",
        "Web Development",
        "Problem Solving",
    ],
    "Computer Applications": [
        "Programming",
        "Databases",
        "Web Development",
        "Software Applications",
        "Problem Solving",
    ],
    "Mechanical Engineering": [
        "CAD / Design",
        "Thermodynamics",
        "Manufacturing",
        "Machine Design",
        "Production Processes",
    ],
    "Electrical Engineering": [
        "Circuit Analysis",
        "Power Systems",
        "Control Systems",
        "PLC / Automation",
        "Electrical Design",
    ],
    "Civil Engineering": [
        "Structural Engineering",
        "AutoCAD",
        "Surveying",
        "Construction Management",
        "Quantity Estimation",
    ],
    "Electronics Engineering": [
        "Circuit Design",
        "Embedded Systems",
        "Digital Electronics",
        "Microcontrollers",
        "Communication Systems",
    ],
    "Electronics and Communication Engineering": [
        "Circuit Design",
        "Embedded Systems",
        "Digital Electronics",
        "Communication Systems",
        "Signal Processing",
    ],
    "Commerce": [
        "Accounting",
        "Taxation",
        "Financial Analysis",
        "Auditing",
        "Business Knowledge",
    ],
    "Finance": [
        "Financial Analysis",
        "Accounting",
        "Investment Analysis",
        "Financial Modeling",
        "Banking Knowledge",
    ],
    "Business Administration": [
        "Business Strategy",
        "Marketing",
        "Finance",
        "Communication",
        "Management",
    ],
    "Other": [
        "Technical Knowledge",
        "Problem Solving",
        "Communication",
        "Analytical Skills",
        "Professional Skills",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name):
    """Normalize a column name for matching."""
    name = str(name).strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalized_columns(df):
    """Create normalized column mapping."""
    return {
        normalize_name(col): col
        for col in df.columns
    }


def find_column(df, aliases, required=False):
    """
    Find a source column from a list of possible names.
    """
    mapping = normalized_columns(df)

    # Exact normalized match
    for alias in aliases:
        key = normalize_name(alias)
        if key in mapping:
            return mapping[key]

    # Partial match
    for alias in aliases:
        key = normalize_name(alias)

        for normalized, original in mapping.items():
            if key in normalized or normalized in key:
                return original

    if required:
        raise ValueError(
            f"Could not find required column.\n"
            f"Possible names: {aliases}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def numeric_from_column(df, column, default=0):
    """
    Convert a source column to numeric safely.
    """
    if column is None:
        return pd.Series(default, index=df.index, dtype=float)

    values = (
        df[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(values, errors="coerce")


def scale_to_10(series, source_min=None, source_max=None):
    """
    Convert arbitrary numeric values to approximately 0-10.
    """
    s = pd.to_numeric(series, errors="coerce")

    if source_min is not None:
        s = s.clip(lower=source_min)

    if source_max is not None:
        s = s.clip(upper=source_max)

    current_min = s.min()
    current_max = s.max()

    if pd.isna(current_min) or pd.isna(current_max):
        return pd.Series(5.0, index=series.index)

    if current_max == current_min:
        return pd.Series(5.0, index=series.index)

    result = (s - current_min) / (current_max - current_min) * 10.0

    return result.clip(0, 10)


def map_gender(value):
    value = str(value).strip().lower()

    if value in ["male", "m", "man", "1"]:
        return "Male"

    if value in ["female", "f", "woman", "2"]:
        return "Female"

    return "Other"


def map_degree(value):
    value = str(value).strip().lower()

    if "b.tech" in value or "btech" in value:
        return "BTech"

    if "b.e" in value or value in ["be", "bachelor of engineering"]:
        return "BE"

    if "bca" in value:
        return "BCA"

    if "b.sc" in value or "bsc" in value:
        return "BSc"

    if "bba" in value:
        return "BBA"

    if "b.com" in value or "bcom" in value:
        return "BCom"

    if value == "ba" or "bachelor of arts" in value:
        return "BA"

    return "Other"


def clean_branch(value):
    """
    Convert common branch names to the names expected by Streamlit.
    """
    value = str(value).strip().lower()

    replacements = {
        "cse": "Computer Science",
        "computer science": "Computer Science",
        "computer science engineering": "Computer Science",

        "it": "Information Technology",
        "information technology": "Information Technology",

        "ds": "Data Science",
        "data science": "Data Science",

        "ai": "Artificial Intelligence",
        "artificial intelligence": "Artificial Intelligence",

        "ml": "Machine Learning",
        "machine learning": "Machine Learning",

        "cyber security": "Cyber Security",
        "cybersecurity": "Cyber Security",

        "software engineering": "Software Engineering",
        "computer applications": "Computer Applications",

        "me": "Mechanical Engineering",
        "mechanical": "Mechanical Engineering",
        "mechanical engineering": "Mechanical Engineering",

        "ee": "Electrical Engineering",
        "electrical": "Electrical Engineering",
        "electrical engineering": "Electrical Engineering",

        "ece": "Electronics and Communication Engineering",
        "electronics and communication": "Electronics and Communication Engineering",
        "electronics and communication engineering":
            "Electronics and Communication Engineering",

        "electronics": "Electronics Engineering",
        "electronics engineering": "Electronics Engineering",

        "civil": "Civil Engineering",
        "civil engineering": "Civil Engineering",

        "commerce": "Commerce",
        "finance": "Finance",
        "business administration": "Business Administration",
    }

    if value in replacements:
        return replacements[value]

    for key, result in replacements.items():
        if key in value:
            return result

    return "Other"


# ============================================================
# FIND TRAINING FILE
# ============================================================

def locate_training_file():
    """
    Locate the uploaded Kaggle training CSV.
    """
    preferred_names = [
        "train.csv",
        "placement_train.csv",
        "student_placement_train.csv",
        "student_placement_prediction_train.csv",
    ]

    for name in preferred_names:
        if os.path.exists(name):
            return name

    csv_files = [
        f for f in os.listdir(".")
        if f.lower().endswith(".csv")
    ]

    if len(csv_files) == 1:
        return csv_files[0]

    if not csv_files:
        raise FileNotFoundError(
            "\nNo CSV training dataset found.\n"
            "Upload the Kaggle training CSV into Colab and run again."
        )

    print("Multiple CSV files found:")
    for f in csv_files:
        print(" -", f)

    # Prefer a file containing "train"
    train_candidates = [
        f for f in csv_files
        if "train" in f.lower()
    ]

    if len(train_candidates) == 1:
        return train_candidates[0]

    raise FileNotFoundError(
        "\nCould not automatically choose the training CSV.\n"
        f"Available CSV files: {csv_files}\n"
        "Rename the correct training file to train.csv."
    )


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(raw_df):
    print("\nPreparing dataset...")
    print("Original shape:", raw_df.shape)

    df = pd.DataFrame(index=raw_df.index)

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age_col = find_column(
        raw_df,
        ["age", "student_age"],
        required=False
    )

    df["age"] = numeric_from_column(
        raw_df,
        age_col,
        default=21
    ).fillna(21).clip(17, 60)

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender_col = find_column(
        raw_df,
        ["gender", "sex"],
        required=False
    )

    if gender_col:
        df["gender"] = raw_df[gender_col].apply(map_gender)
    else:
        df["gender"] = "Other"

    # --------------------------------------------------------
    # DEGREE
    # --------------------------------------------------------

    degree_col = find_column(
        raw_df,
        [
            "degree",
            "ug_degree",
            "education",
            "highest_degree"
        ],
        required=False
    )

    if degree_col:
        df["ug_degree"] = raw_df[degree_col].apply(map_degree)
    else:
        df["ug_degree"] = "Other"

    # --------------------------------------------------------
    # BRANCH
    # --------------------------------------------------------

    branch_col = find_column(
        raw_df,
        [
            "branch",
            "ug_branch",
            "stream",
            "specialization",
            "department",
            "field_of_study"
        ],
        required=False
    )

    if branch_col:
        df["ug_branch"] = raw_df[branch_col].apply(clean_branch)
    else:
        df["ug_branch"] = "Other"

    # --------------------------------------------------------
    # CGPA
    # --------------------------------------------------------

    cgpa_col = find_column(
        raw_df,
        [
            "cgpa",
            "ug_cgpa",
            "academic_cgpa",
            "gpa"
        ],
        required=True
    )

    df["ug_cgpa"] = numeric_from_column(
        raw_df,
        cgpa_col
    )

    # Handle percentage-like CGPA if necessary
    if df["ug_cgpa"].median(skipna=True) > 10:
        df["ug_cgpa"] = df["ug_cgpa"] / 10.0

    df["ug_cgpa"] = df["ug_cgpa"].clip(0, 10)

    # --------------------------------------------------------
    # BACKLOGS
    # --------------------------------------------------------

    backlog_col = find_column(
        raw_df,
        [
            "backlogs",
            "backlog",
            "academic_backlogs",
            "number_of_backlogs"
        ],
        required=False
    )

    df["backlogs"] = numeric_from_column(
        raw_df,
        backlog_col,
        default=0
    ).fillna(0).clip(0, 20)

    # --------------------------------------------------------
    # INTERNSHIPS
    # --------------------------------------------------------

    internship_col = find_column(
        raw_df,
        [
            "internships",
            "internship",
            "internships_count",
            "number_of_internships"
        ],
        required=False
    )

    df["internships"] = numeric_from_column(
        raw_df,
        internship_col,
        default=0
    ).fillna(0).clip(0, 20)

    # --------------------------------------------------------
    # PROJECTS
    # --------------------------------------------------------

    project_col = find_column(
        raw_df,
        [
            "projects",
            "project",
            "projects_count",
            "number_of_projects"
        ],
        required=False
    )

    df["projects"] = numeric_from_column(
        raw_df,
        project_col,
        default=0
    ).fillna(0).clip(0, 30)

    # --------------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------------

    certification_col = find_column(
        raw_df,
        [
            "certifications",
            "certification",
            "certifications_count",
            "number_of_certifications"
        ],
        required=False
    )

    df["certifications"] = numeric_from_column(
        raw_df,
        certification_col,
        default=0
    ).fillna(0).clip(0, 30)

    # --------------------------------------------------------
    # CODING SKILLS
    # --------------------------------------------------------

    coding_col = find_column(
        raw_df,
        [
            "coding",
            "coding_skill",
            "coding_skill_score",
            "coding_ability",
            "programming_skill"
        ],
        required=False
    )

    if coding_col:
        coding_raw = numeric_from_column(raw_df, coding_col)
        df["coding_skills"] = scale_to_10(coding_raw)
    else:
        df["coding_skills"] = 5.0

    # --------------------------------------------------------
    # COMMUNICATION
    # --------------------------------------------------------

    communication_col = find_column(
        raw_df,
        [
            "communication",
            "communication_skill",
            "communication_skill_score",
            "communication_skills"
        ],
        required=False
    )

    if communication_col:
        communication_raw = numeric_from_column(
            raw_df,
            communication_col
        )
        df["communication_skills"] = scale_to_10(
            communication_raw
        )
    else:
        df["communication_skills"] = 5.0

    # --------------------------------------------------------
    # APTITUDE
    # --------------------------------------------------------

    aptitude_col = find_column(
        raw_df,
        [
            "aptitude",
            "aptitude_score",
            "aptitude_test",
            "aptitude_test_score"
        ],
        required=False
    )

    if aptitude_col:
        aptitude_raw = numeric_from_column(
            raw_df,
            aptitude_col
        )

        # Keep as 0-100 if it already looks like percentage
        if aptitude_raw.median(skipna=True) <= 10:
            aptitude_raw = aptitude_raw * 10

        df["aptitude_score"] = aptitude_raw.clip(0, 100)
    else:
        df["aptitude_score"] = 50.0

    # --------------------------------------------------------
    # TECHNICAL SKILL
    # --------------------------------------------------------

    technical_col = find_column(
        raw_df,
        [
            "technical_skill",
            "technical_skills",
            "technical_skill_score",
            "technical_knowledge",
            "technical_ability"
        ],
        required=False
    )

    if technical_col:
        technical = scale_to_10(
            numeric_from_column(raw_df, technical_col)
        )
    else:
        technical = df["coding_skills"].copy()

    # --------------------------------------------------------
    # OTHER SKILLS
    # --------------------------------------------------------

    logical_col = find_column(
        raw_df,
        [
            "logical_reasoning",
            "logical_reasoning_score",
            "reasoning_score"
        ],
        required=False
    )

    if logical_col:
        logical = scale_to_10(
            numeric_from_column(raw_df, logical_col)
        )
    else:
        logical = scale_to_10(df["aptitude_score"])

    hackathon_col = find_column(
        raw_df,
        [
            "hackathons",
            "hackathon",
            "hackathon_count"
        ],
        required=False
    )

    if hackathon_col:
        hackathons = scale_to_10(
            numeric_from_column(raw_df, hackathon_col)
        )
    else:
        hackathons = pd.Series(
            3.0,
            index=raw_df.index
        )

    github_col = find_column(
        raw_df,
        [
            "github",
            "github_activity",
            "github_repos",
            "github_repositories"
        ],
        required=False
    )

    if github_col:
        github = scale_to_10(
            numeric_from_column(raw_df, github_col)
        )
    else:
        github = pd.Series(
            3.0,
            index=raw_df.index
        )

    leadership_col = find_column(
        raw_df,
        [
            "leadership",
            "leadership_score"
        ],
        required=False
    )

    if leadership_col:
        leadership = scale_to_10(
            numeric_from_column(raw_df, leadership_col)
        )
    else:
        leadership = df["communication_skills"].copy()

    # --------------------------------------------------------
    # CREATE FIVE DOMAIN SKILLS
    #
    # These are kept compatible with your Streamlit UI.
    # They are constructed from technical / coding / reasoning /
    # GitHub / practical experience signals available in the
    # larger dataset.
    # --------------------------------------------------------

    df["domain_skill_1"] = (
        0.50 * df["coding_skills"]
        + 0.30 * technical
        + 0.20 * github
    )

    df["domain_skill_2"] = (
        0.50 * logical
        + 0.30 * df["aptitude_score"] / 10
        + 0.20 * technical
    )

    df["domain_skill_3"] = (
        0.50 * technical
        + 0.30 * df["coding_skills"]
        + 0.20 * hackathons
    )

    df["domain_skill_4"] = (
        0.40 * github
        + 0.30 * hackathons
        + 0.30 * leadership
    )

    df["domain_skill_5"] = (
        0.40 * df["communication_skills"]
        + 0.30 * technical
        + 0.30 * logical
    )

    for col in [
        "domain_skill_1",
        "domain_skill_2",
        "domain_skill_3",
        "domain_skill_4",
        "domain_skill_5",
    ]:
        df[col] = df[col].clip(0, 10)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_col = find_column(
        raw_df,
        [
            "placed",
            "placement",
            "placement_status",
            "placement_outcome",
            "is_placed"
        ],
        required=True
    )

    target = raw_df[target_col]

    if target.dtype == object:
        target = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
                "placed": 1,
                "not placed": 0,
                "true": 1,
                "false": 0,
            })
        )
    else:
        target = pd.to_numeric(
            target,
            errors="coerce"
        )

    target = target.where(
        target.isin([0, 1])
    )

    valid = target.notna()

    df = df.loc[valid].copy()
    target = target.loc[valid].astype(int)

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .astype(str)
            .replace("nan", "Other")
            .fillna("Other")
        )

    print("\nPrepared dataset:")
    print("Rows:", len(df))
    print("Features:", len(MODEL_FEATURES))
    print("\nTarget distribution:")
    print(target.value_counts().sort_index())

    print("\nPlacement rate:")
    print(f"{target.mean() * 100:.2f}%")

    return df, target


# ============================================================
# TRAIN MODELS
# ============================================================

def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    return ColumnTransformer(
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
        ]
    )


def build_models(preprocessor):
    models = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.7,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # HistGradientBoosting works only with numeric arrays,
    # so use a separate preprocessing pipeline.
    hgb_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            )
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    models["HistGradientBoosting"] = Pipeline(
        steps=[
            (
                "preprocessor",
                hgb_preprocessor
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# RECOMMENDATION RULES
# ============================================================

def create_recommendation_rules():
    return {
        "cgpa": {
            "threshold": 7.5,
            "message": "Improve CGPA through consistent academic preparation."
        },
        "backlogs": {
            "threshold": 0,
            "message": "Clear pending backlogs and maintain academic consistency."
        },
        "internships": {
            "threshold": 1,
            "message": "Complete an internship or practical industry experience."
        },
        "projects": {
            "threshold": 2,
            "message": "Build more practical projects and publish strong work."
        },
        "certifications": {
            "threshold": 1,
            "message": "Add relevant certifications aligned with your target role."
        },
        "coding_skills": {
            "threshold": 6,
            "message": "Strengthen programming and problem-solving skills."
        },
        "communication_skills": {
            "threshold": 6,
            "message": "Practice communication, interviews and presentation skills."
        },
        "aptitude_score": {
            "threshold": 60,
            "message": "Practice quantitative, logical and verbal aptitude."
        },
        "domain_skills": {
            "threshold": 6,
            "message": "Strengthen branch-specific technical skills."
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI STUDENT PLACEMENT PREDICTOR")
    print("Large Placement Dataset Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Locate CSV
    # --------------------------------------------------------

    data_file = locate_training_file()

    print("\nUsing training file:")
    print(data_file)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    raw_df = pd.read_csv(data_file)

    print("\nRaw dataset loaded successfully.")
    print("Rows:", raw_df.shape[0])
    print("Columns:", raw_df.shape[1])

    print("\nAvailable columns:")
    for col in raw_df.columns:
        print(" -", col)

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_dataset(raw_df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\nTrain rows:", len(X_train))
    print("Test rows :", len(X_test))

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    models = build_models(
        preprocessor
    )

    results = {}

    # --------------------------------------------------------
    # Train and compare
    # --------------------------------------------------------

    for name, model in models.items():

        print("\n" + "-" * 70)
        print("Training:", name)
        print("-" * 70)

        model.fit(
            X_train,
            y_train
        )

        results[name] = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

    # --------------------------------------------------------
    # Select best model by ROC-AUC
    # --------------------------------------------------------

    best_name = max(
        results,
        key=lambda name: results[name]["roc_auc"]
    )

    best_model = models[best_name]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print("Selected:", best_name)
    print(
        "ROC-AUC:",
        f"{results[best_name]['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print("\nRunning 5-fold stratified cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(
        "CV ROC-AUC scores:",
        np.round(cv_scores, 4)
    )

    print(
        "Mean CV ROC-AUC:",
        f"{cv_scores.mean():.4f}"
    )

    print(
        "CV ROC-AUC std:",
        f"{cv_scores.std():.4f}"
    )

    # --------------------------------------------------------
    # Calibrate best model
    # --------------------------------------------------------

    print("\nCalibrating final model probabilities...")

    calibrated_model = CalibratedClassifierCV(
        estimator=best_model,
        method="sigmoid",
        cv=5,
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        calibrated_model,
        MODEL_FILE
    )

    joblib.dump(
        MODEL_FEATURES,
        FEATURE_FILE
    )

    metadata = {
        "model_type": "CalibratedClassifierCV",
        "base_model": best_name,
        "dataset_mode": "large_placement_dataset",
        "training_file": data_file,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "total_rows": int(len(X)),
        "feature_count": len(MODEL_FEATURES),
        "features": MODEL_FEATURES,
        "target": TARGET,
        "test_metrics": results[best_name],
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "cv_roc_auc_scores": [
            float(x) for x in cv_scores
        ],
        "synthetic_demo_warning": (
            "The source competition dataset is a structured "
            "placement prediction dataset and should not be "
            "described as verified real-world student records."
        ),
        "probability_warning": (
            "The displayed probability is a calibrated model "
            "estimate based on the training dataset. It is not "
            "a guarantee of employment."
        ),
        "branch_skill_mapping": BRANCH_SKILLS,
    }

    joblib.dump(
        metadata,
        METADATA_FILE
    )

    joblib.dump(
        create_recommendation_rules(),
        RULES_FILE
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("\nCreated files:")

    for file_name in [
        MODEL_FILE,
        FEATURE_FILE,
        METADATA_FILE,
        RULES_FILE,
    ]:
        print(" ✓", file_name)

    print("\nModel features:")

    for i, feature in enumerate(
        MODEL_FEATURES,
        start=1
    ):
        print(
            f"{i:02d}. {feature}"
        )

    print("\nBest model:")
    print(best_name)

    print(
        "\nTest ROC-AUC:",
        f"{results[best_name]['roc_auc']:.4f}"
    )

    print(
        "Mean CV ROC-AUC:",
        f"{cv_scores.mean():.4f}"
    )

    print(
        "\nPlacement rate:",
        f"{y.mean() * 100:.2f}%"
    )

    print(
        "\nThe four generated .pkl files are ready "
        "for the Streamlit application."
    )


if __name__ == "__main__":
    main()
