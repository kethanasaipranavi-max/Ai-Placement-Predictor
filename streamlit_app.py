import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
RULE_FILE = "recommendation_rules.pkl"


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "student_profile" not in st.session_state:
    st.session_state.student_profile = None

if "ai_career_advice" not in st.session_state:
    st.session_state.ai_career_advice = None

if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = None


# ============================================================
# BRANCH SKILLS
# ============================================================

def get_branch_skills(branch):

    skill_mapping = {

        "Computer Science": [
            "Programming",
            "Data Structures & Algorithms",
            "Databases",
            "Software Development",
            "Problem Solving"
        ],

        "Information Technology": [
            "Programming",
            "Networking",
            "Databases",
            "Cloud Computing",
            "System Administration"
        ],

        "Data Science": [
            "Python / Programming",
            "Statistics",
            "Data Analysis",
            "Machine Learning",
            "Data Visualization"
        ],

        "Artificial Intelligence": [
            "Programming",
            "Machine Learning",
            "Deep Learning",
            "Mathematics",
            "Data Analysis"
        ],

        "Machine Learning": [
            "Python",
            "Machine Learning",
            "Statistics",
            "Deep Learning",
            "Data Processing"
        ],

        "Cyber Security": [
            "Networking",
            "Cyber Security Concepts",
            "Linux",
            "Security Tools",
            "Ethical Hacking"
        ],

        "Software Engineering": [
            "Programming",
            "Software Design",
            "Databases",
            "Web Development",
            "Problem Solving"
        ],

        "Computer Applications": [
            "Programming",
            "Databases",
            "Web Development",
            "Software Applications",
            "Problem Solving"
        ],

        "Mechanical Engineering": [
            "CAD / Design",
            "Thermodynamics",
            "Manufacturing",
            "Machine Design",
            "Production Processes"
        ],

        "Automobile Engineering": [
            "Automobile Systems",
            "CAD / Design",
            "Engine Technology",
            "Manufacturing",
            "Vehicle Diagnostics"
        ],

        "Production Engineering": [
            "Manufacturing",
            "Production Planning",
            "Quality Control",
            "Industrial Processes",
            "CAD"
        ],

        "Industrial Engineering": [
            "Operations Management",
            "Production Systems",
            "Quality Management",
            "Supply Chain",
            "Industrial Analysis"
        ],

        "Aeronautical Engineering": [
            "Aerodynamics",
            "Aircraft Systems",
            "CAD",
            "Propulsion",
            "Manufacturing"
        ],

        "Aerospace Engineering": [
            "Aerodynamics",
            "Aircraft Design",
            "Propulsion",
            "CAD",
            "Space Systems"
        ],

        "Electrical Engineering": [
            "Circuit Analysis",
            "Power Systems",
            "Control Systems",
            "PLC / Automation",
            "Electrical Design"
        ],

        "Electronics Engineering": [
            "Electronic Circuits",
            "Embedded Systems",
            "PCB Design",
            "Microcontrollers",
            "Instrumentation"
        ],

        "Electronics and Communication Engineering": [
            "Communication Systems",
            "Embedded Systems",
            "Electronics",
            "Signal Processing",
            "VLSI"
        ],

        "Biomedical Engineering": [
            "Biomedical Instrumentation",
            "Medical Devices",
            "Electronics",
            "Clinical Engineering",
            "Signal Processing"
        ],

        "Civil Engineering": [
            "Structural Engineering",
            "AutoCAD",
            "Surveying",
            "Construction Management",
            "Quantity Estimation"
        ],

        "Chemical Engineering": [
            "Chemical Processes",
            "Thermodynamics",
            "Process Engineering",
            "Plant Operations",
            "Industrial Safety"
        ],

        "Mathematics": [
            "Mathematical Analysis",
            "Statistics",
            "Problem Solving",
            "Quantitative Reasoning",
            "Research Methods"
        ],

        "Statistics": [
            "Statistical Analysis",
            "Probability",
            "Data Interpretation",
            "Research Methods",
            "Quantitative Analysis"
        ],

        "Physics": [
            "Laboratory Techniques",
            "Instrumentation",
            "Electronics",
            "Scientific Analysis",
            "Research Methods"
        ],

        "Chemistry": [
            "Analytical Chemistry",
            "Laboratory Techniques",
            "Chemical Analysis",
            "Instrumentation",
            "Research Methods"
        ],

        "Environmental Science": [
            "Environmental Analysis",
            "Sustainability",
            "Environmental Monitoring",
            "Research Methods",
            "Data Analysis"
        ],

        "Biotechnology": [
            "Laboratory Techniques",
            "Molecular Biology",
            "Biotechnology Methods",
            "Research Skills",
            "Scientific Analysis"
        ],

        "Microbiology": [
            "Microbiology Techniques",
            "Laboratory Skills",
            "Culture Techniques",
            "Research Methods",
            "Scientific Analysis"
        ],

        "Biochemistry": [
            "Biochemical Techniques",
            "Laboratory Skills",
            "Chemical Analysis",
            "Research Methods",
            "Scientific Analysis"
        ],

        "Biological Sciences": [
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis",
            "Biological Techniques",
            "Data Interpretation"
        ],

        "Life Sciences": [
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis",
            "Biological Techniques",
            "Data Interpretation"
        ],

        "Genetics": [
            "Genetics",
            "Molecular Biology",
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis"
        ],

        "Botany": [
            "Plant Biology",
            "Laboratory Skills",
            "Research Methods",
            "Field Research",
            "Scientific Analysis"
        ],

        "Zoology": [
            "Animal Biology",
            "Laboratory Skills",
            "Research Methods",
            "Field Research",
            "Scientific Analysis"
        ],

        "Food Science and Nutrition": [
            "Nutrition Science",
            "Food Analysis",
            "Laboratory Skills",
            "Diet Planning",
            "Food Safety"
        ],

        "Food Technology": [
            "Food Processing",
            "Food Safety",
            "Quality Control",
            "Laboratory Analysis",
            "Manufacturing"
        ],

        "Nutrition and Dietetics": [
            "Clinical Nutrition",
            "Diet Planning",
            "Nutrition Assessment",
            "Food Science",
            "Communication"
        ],

        "Economics": [
            "Economic Analysis",
            "Statistics",
            "Financial Analysis",
            "Research Methods",
            "Quantitative Analysis"
        ],

        "Commerce": [
            "Accounting",
            "Taxation",
            "Financial Analysis",
            "Auditing",
            "Business Knowledge"
        ],

        "Business Administration": [
            "Business Strategy",
            "Marketing",
            "Operations",
            "Management",
            "Business Analysis"
        ],

        "Finance": [
            "Financial Analysis",
            "Accounting",
            "Investment Analysis",
            "Financial Modeling",
            "Banking Knowledge"
        ],

        "Accounting": [
            "Accounting",
            "Taxation",
            "Auditing",
            "Financial Reporting",
            "Financial Analysis"
        ],

        "Management": [
            "Leadership",
            "Operations",
            "Business Strategy",
            "Project Management",
            "Decision Making"
        ],

        "Marketing": [
            "Marketing Strategy",
            "Digital Marketing",
            "Market Research",
            "Brand Management",
            "Consumer Analysis"
        ],

        "Human Resources": [
            "Recruitment",
            "HR Operations",
            "Employee Relations",
            "Talent Management",
            "Organizational Skills"
        ],

        "Psychology": [
            "Psychological Assessment",
            "Research Methods",
            "Counseling Skills",
            "Behavioral Analysis",
            "Data Interpretation"
        ],

        "English": [
            "Writing",
            "Communication",
            "Editing",
            "Research",
            "Presentation Skills"
        ],

        "Political Science": [
            "Political Analysis",
            "Research Methods",
            "Public Policy",
            "International Relations",
            "Communication"
        ],

        "Sociology": [
            "Social Research",
            "Research Methods",
            "Data Analysis",
            "Community Studies",
            "Communication"
        ],

        "History": [
            "Historical Research",
            "Research Methods",
            "Writing",
            "Analysis",
            "Documentation"
        ],

        "Public Administration": [
            "Public Policy",
            "Administration",
            "Governance",
            "Research",
            "Management"
        ]
    }

    return skill_mapping.get(
        branch,
        [
            "Core Domain Knowledge",
            "Practical Skills",
            "Research Skills",
            "Industry Knowledge",
            "Problem Solving"
        ]
    )


# ============================================================
# CAREER INTERESTS
# ============================================================

def get_career_interests(branch):

    if branch in [
        "Computer Science",
        "Information Technology",
        "Data Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Cyber Security",
        "Software Engineering",
        "Computer Applications"
    ]:
        return [
            "Software Development",
            "Data Science",
            "Artificial Intelligence",
            "Cyber Security",
            "Cloud Computing",
            "Data Analytics",
            "Other"
        ]

    if branch in [
        "Mechanical Engineering",
        "Automobile Engineering",
        "Production Engineering",
        "Industrial Engineering"
    ]:
        return [
            "Design Engineering",
            "Manufacturing",
            "Automotive Engineering",
            "Production Engineering",
            "Operations",
            "Other"
        ]

    if branch in [
        "Electrical Engineering",
        "Electronics Engineering",
        "Electronics and Communication Engineering",
        "Biomedical Engineering"
    ]:
        return [
            "Embedded Systems",
            "Electronics Design",
            "Power Systems",
            "Automation",
            "VLSI",
            "Instrumentation",
            "Other"
        ]

    if branch == "Civil Engineering":
        return [
            "Structural Engineering",
            "Construction Management",
            "Site Engineering",
            "Infrastructure",
            "Surveying",
            "Other"
        ]

    if branch == "Chemical Engineering":
        return [
            "Process Engineering",
            "Plant Operations",
            "Chemical Analysis",
            "Industrial Safety",
            "Research",
            "Other"
        ]

    if branch in [
        "Biotechnology",
        "Microbiology",
        "Biochemistry",
        "Biological Sciences",
        "Life Sciences",
        "Genetics",
        "Botany",
        "Zoology"
    ]:
        return [
            "Research",
            "Laboratory Work",
            "Biotechnology",
            "Pharmaceutical Industry",
            "Quality Control",
            "Other"
        ]

    if branch in [
        "Food Science and Nutrition",
        "Food Technology",
        "Nutrition and Dietetics"
    ]:
        return [
            "Food Industry",
            "Nutrition",
            "Quality Control",
            "Food Safety",
            "Research",
            "Clinical Nutrition",
            "Other"
        ]

    if branch in [
        "Commerce",
        "Finance",
        "Accounting",
        "Economics"
    ]:
        return [
            "Accounting",
            "Finance",
            "Banking",
            "Financial Analysis",
            "Auditing",
            "Business Analytics",
            "Other"
        ]

    if branch in [
        "Business Administration",
        "Management",
        "Marketing",
        "Human Resources"
    ]:
        return [
            "Management",
            "Marketing",
            "Human Resources",
            "Operations",
            "Business Analytics",
            "Finance",
            "Other"
        ]

    return [
        "Research",
        "Teaching",
        "Government Sector",
        "Industry",
        "Higher Studies",
        "Other"
    ]


# ============================================================
# MODEL BRANCH MAPPING
# ============================================================

def map_branch_for_model(branch):

    """
    The placement model was trained using grouped branch categories.
    We keep the original student branch separately for Gemini guidance.
    """

    if branch in [
        "Computer Science",
        "Information Technology",
        "Computer Applications",
        "Software Engineering",
        "Data Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Cyber Security"
    ]:
        return "CS"

    if branch in [
        "Mechanical Engineering",
        "Automobile Engineering",
        "Production Engineering",
        "Industrial Engineering",
        "Aeronautical Engineering",
        "Aerospace Engineering"
    ]:
        return "Mechanical"

    if branch in [
        "Electrical Engineering",
        "Electronics Engineering",
        "Electronics and Communication Engineering",
        "Biomedical Engineering"
    ]:
        return "Electrical"

    if branch in [
        "Civil Engineering"
    ]:
        return "Civil"

    if branch in [
        "Chemical Engineering"
    ]:
        return "Chemical"

    return "Other"


# ============================================================
# DEGREE MAPPING HELPERS
# ============================================================

def degree_aliases(degree):

    aliases = {
        "BE": [
            "BE",
            "B.E",
            "B.E.",
            "Bachelor of Engineering"
        ],

        "BTech": [
            "BTech",
            "B.Tech",
            "B.Tech.",
            "Bachelor of Technology"
        ],

        "BSc": [
            "BSc",
            "B.Sc",
            "B.Sc.",
            "Bachelor of Science"
        ],

        "BCA": [
            "BCA",
            "Bachelor of Computer Applications"
        ],

        "BBA": [
            "BBA",
            "Bachelor of Business Administration"
        ],

        "BCom": [
            "BCom",
            "B.Com",
            "B.Com.",
            "Bachelor of Commerce"
        ],

        "BA": [
            "BA",
            "B.A",
            "B.A.",
            "Bachelor of Arts"
        ]
    }

    return aliases.get(degree, [degree])


# ============================================================
# LOAD MODEL ONLY WHEN NEEDED
# ============================================================

@st.cache_resource
def load_components():

    model = joblib.load(MODEL_FILE)

    feature_names = joblib.load(FEATURE_FILE)

    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns.tolist()

    elif isinstance(feature_names, pd.Series):
        feature_names = feature_names.tolist()

    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    elif isinstance(feature_names, dict):
        if "feature_names" in feature_names:
            feature_names = feature_names["feature_names"]
        elif "features" in feature_names:
            feature_names = feature_names["features"]
        else:
            feature_names = list(feature_names.keys())

    feature_names = [
        str(x)
        for x in feature_names
    ]

    recommendation_rules = {}

    if os.path.exists(RULE_FILE):

        try:
            recommendation_rules = joblib.load(
                RULE_FILE
            )
        except Exception:
            recommendation_rules = {}

    return (
        model,
        feature_names,
        recommendation_rules
    )


# ============================================================
# MODEL FEATURE DISCOVERY
# ============================================================

RAW_FEATURE_NAMES = {
    "gender",
    "age",
    "degree",
    "branch",
    "cgpa",
    "backlogs",
    "internships",
    "certifications",
    "coding_skills",
    "communication_skills",
    "aptitude_score",
    "projects",
    "cgpa_category"
}


def get_object_feature_names(obj):

    names = getattr(
        obj,
        "feature_names_in_",
        None
    )

    if names is not None:

        try:
            return [
                str(x)
                for x in names
            ]
        except Exception:
            pass

    return None


def get_model_expected_features(model):

    """
    Find the actual columns used when the fitted estimator was trained.

    This is critical because the feature pickle can contain columns that
    are not necessarily the exact columns accepted by the fitted model.
    """

    # 1. Direct estimator
    names = get_object_feature_names(model)

    if names:
        return names

    # 2. Pipeline / imblearn Pipeline
    named_steps = getattr(
        model,
        "named_steps",
        None
    )

    if named_steps:

        for _, step in named_steps.items():

            names = get_object_feature_names(step)

            if names:
                return names

    # 3. Generic pipeline steps
    steps = getattr(
        model,
        "steps",
        None
    )

    if steps:

        for _, step in steps:

            names = get_object_feature_names(step)

            if names:
                return names

    return None


# ============================================================
# RAW STUDENT DATA
# ============================================================

def build_raw_model_dataframe(student):

    model_branch = map_branch_for_model(
        student["ug_branch"]
    )

    raw = {

        "gender": student["gender"],

        "age": student["age"],

        "degree": student["ug_degree"],

        "branch": model_branch,

        "cgpa": student["ug_cgpa"],

        "backlogs": student["backlogs"],

        "internships": student["internships"],

        "certifications": student["certifications"],

        "coding_skills": student["coding_skills"],

        "communication_skills": (
            student["communication_skills"]
        ),

        "aptitude_score": (
            student["aptitude_score"]
        ),

        "projects": student["projects"]
    }

    df = pd.DataFrame([raw])

    df["cgpa_category"] = df["cgpa"].apply(
        get_cgpa_category
    )

    return df


# ============================================================
# CGPA CATEGORY
# ============================================================

def get_cgpa_category(cgpa):

    if cgpa < 6:
        return "Low"

    if cgpa < 7.5:
        return "Good"

    return "Excellent"


# ============================================================
# BUILD ENCODED DATAFRAME
# ============================================================

def build_encoded_candidates(student):

    model_branch = map_branch_for_model(
        student["ug_branch"]
    )

    raw = {

        "gender": student["gender"],

        "age": student["age"],

        "degree": student["ug_degree"],

        "branch": model_branch,

        "cgpa": student["ug_cgpa"],

        "backlogs": student["backlogs"],

        "internships": student["internships"],

        "certifications": student["certifications"],

        "coding_skills": student["coding_skills"],

        "communication_skills": (
            student["communication_skills"]
        ),

        "aptitude_score": (
            student["aptitude_score"]
        ),

        "projects": student["projects"]
    }

    base_df = pd.DataFrame([raw])

    base_df["cgpa_category"] = (
        base_df["cgpa"].apply(
            get_cgpa_category
        )
    )

    candidates = []

    # --------------------------------------------------------
    # Candidate 1: normal grouped branch
    # --------------------------------------------------------

    candidate = pd.get_dummies(
        base_df.copy(),
        columns=[
            "gender",
            "degree",
            "branch",
            "cgpa_category"
        ],
        dtype=int
    )

    candidates.append(candidate)

    # --------------------------------------------------------
    # Candidate 2: long original branch
    # --------------------------------------------------------

    original_df = base_df.copy()

    original_df["branch"] = student[
        "ug_branch"
    ]

    original_encoded = pd.get_dummies(
        original_df,
        columns=[
            "gender",
            "degree",
            "branch",
            "cgpa_category"
        ],
        dtype=int
    )

    candidates.append(
        original_encoded
    )

    # --------------------------------------------------------
    # Candidate 3: common abbreviated branch names
    # --------------------------------------------------------

    abbreviated = base_df.copy()

    abbreviation = {
        "Computer Science": "CS",
        "Information Technology": "IT",
        "Data Science": "DS",
        "Artificial Intelligence": "AI",
        "Machine Learning": "ML",
        "Cyber Security": "Cyber Security",
        "Software Engineering": "Software Engineering",
        "Computer Applications": "BCA",
        "Mechanical Engineering": "Mechanical",
        "Automobile Engineering": "Automobile",
        "Production Engineering": "Production",
        "Industrial Engineering": "Industrial",
        "Electrical Engineering": "Electrical",
        "Electronics Engineering": "Electronics",
        "Electronics and Communication Engineering": "ECE",
        "Biomedical Engineering": "Biomedical",
        "Civil Engineering": "Civil",
        "Chemical Engineering": "Chemical"
    }

    abbreviated["branch"] = abbreviation.get(
        student["ug_branch"],
        student["ug_branch"]
    )

    abbreviated_encoded = pd.get_dummies(
        abbreviated,
        columns=[
            "gender",
            "degree",
            "branch",
            "cgpa_category"
        ],
        dtype=int
    )

    candidates.append(
        abbreviated_encoded
    )

    return candidates


# ============================================================
# ALIGN ENCODED DATA TO MODEL FEATURES
# ============================================================

def align_encoded_dataframe(
    encoded_df,
    expected_features
):

    aligned = pd.DataFrame(
        0,
        index=[0],
        columns=expected_features,
        dtype=float
    )

    for column in encoded_df.columns:

        column_name = str(column)

        if column_name in aligned.columns:

            aligned.loc[
                0,
                column_name
            ] = encoded_df.iloc[0][column]

    return aligned


# ============================================================
# INTELLIGENT MODEL INPUT
# ============================================================

def prepare_model_input(
    model,
    student,
    artifact_features
):

    raw_df = build_raw_model_dataframe(
        student
    )

    expected_features = (
        get_model_expected_features(
            model
        )
    )

    # --------------------------------------------------------
    # CASE 1
    # Model explicitly exposes raw training columns
    # --------------------------------------------------------

    if expected_features:

        expected_set = set(
            expected_features
        )

        raw_overlap = (
            expected_set.intersection(
                RAW_FEATURE_NAMES
            )
        )

        # If model expects raw columns such as degree/branch/gender,
        # send raw dataframe exactly in that order.
        if len(raw_overlap) >= 3:

            missing = [
                col
                for col in expected_features
                if col not in raw_df.columns
            ]

            if not missing:

                return raw_df[
                    expected_features
                ], "raw"

    # --------------------------------------------------------
    # CASE 2
    # Model expects encoded features
    # --------------------------------------------------------

    candidates = build_encoded_candidates(
        student
    )

    if expected_features:

        best_candidate = None
        best_overlap = -1

        expected_set = set(
            expected_features
        )

        for candidate in candidates:

            overlap = len(
                set(candidate.columns)
                .intersection(
                    expected_set
                )
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_candidate = candidate

        if best_candidate is not None:

            aligned = align_encoded_dataframe(
                best_candidate,
                expected_features
            )

            return aligned, "encoded_model_features"

    # --------------------------------------------------------
    # CASE 3
    # Fall back to artifact feature names
    # --------------------------------------------------------

    for candidate in candidates:

        aligned = align_encoded_dataframe(
            candidate,
            artifact_features
        )

        return aligned, "encoded_artifact_features"

    raise ValueError(
        "Unable to construct model input."
    )


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def get_positive_probability(
    model,
    model_input,
    prediction
):

    if not hasattr(
        model,
        "predict_proba"
    ):

        return None

    probabilities = model.predict_proba(
        model_input
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:

        return None

    row = probabilities[0]

    classes = getattr(
        model,
        "classes_",
        None
    )

    if classes is not None:

        classes = list(classes)

        # Prefer class 1 when present.
        if 1 in classes:

            index = classes.index(1)

            return float(
                row[index]
            )

        # Sometimes labels are strings.
        for target in [
            "1",
            "Placed",
            "PLACED",
            "Yes",
            "YES",
            True
        ]:

            if target in classes:

                index = classes.index(
                    target
                )

                return float(
                    row[index]
                )

    # Binary classifier fallback.
    if len(row) == 2:

        return float(row[1])

    # If there is only one probability,
    # use it according to prediction.
    if len(row) == 1:

        value = float(row[0])

        return value

    return None


# ============================================================
# READINESS
# ============================================================

def get_readiness_level(probability):

    if probability is None:

        return (
            "Prediction available",
            "🔵"
        )

    if probability >= 0.75:

        return (
            "High Placement Readiness",
            "🟢"
        )

    if probability >= 0.50:

        return (
            "Moderate Placement Readiness",
            "🟡"
        )

    return (
        "Needs Improvement",
        "🔴"
    )


# ============================================================
# PROFILE STRENGTHS
# ============================================================

def get_profile_strengths(student):

    strengths = []

    if student["ug_cgpa"] >= 8:
        strengths.append(
            f"Strong academic performance with a CGPA of {student['ug_cgpa']:.2f}."
        )

    if student["backlogs"] == 0:
        strengths.append(
            "No current academic backlogs."
        )

    if student["internships"] >= 1:
        strengths.append(
            f"{student['internships']} internship(s) provide practical exposure."
        )

    if student["projects"] >= 2:
        strengths.append(
            f"{student['projects']} projects demonstrate practical experience."
        )

    if student["certifications"] >= 2:
        strengths.append(
            f"{student['certifications']} certifications show continued learning."
        )

    if student["coding_skills"] >= 7:
        strengths.append(
            "Good programming/computational skill level."
        )

    if student["communication_skills"] >= 7:
        strengths.append(
            "Good communication skill level."
        )

    if student["aptitude_score"] >= 75:
        strengths.append(
            "Strong aptitude performance."
        )

    if not strengths:

        strengths.append(
            "The profile provides a foundation that can be strengthened through focused preparation."
        )

    return strengths


# ============================================================
# PROFILE DEVELOPMENT AREAS
# ============================================================

def get_profile_gaps(student):

    gaps = []

    if student["ug_cgpa"] < 7:

        gaps.append(
            "Academic performance"
        )

    if student["backlogs"] > 0:

        gaps.append(
            "Backlog clearance"
        )

    if student["internships"] == 0:

        gaps.append(
            "Industry/internship exposure"
        )

    if student["projects"] < 2:

        gaps.append(
            "Practical project experience"
        )

    if student["certifications"] == 0:

        gaps.append(
            "Relevant certifications"
        )

    if student["coding_skills"] < 6:

        gaps.append(
            "Technical/computational skills"
        )

    if student["communication_skills"] < 6:

        gaps.append(
            "Communication and interview skills"
        )

    if student["aptitude_score"] < 60:

        gaps.append(
            "Aptitude preparation"
        )

    return gaps


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    student,
    recommendation_rules
):

    recommendations = []

    gaps = get_profile_gaps(
        student
    )

    for gap in gaps:

        rule_key = None

        if gap == "Technical/computational skills":
            rule_key = "coding_skills"

        elif gap == "Communication and interview skills":
            rule_key = "communication_skills"

        elif gap == "Aptitude preparation":
            rule_key = "aptitude_score"

        elif gap == "Academic performance":
            rule_key = "cgpa"

        elif gap == "Backlog clearance":
            rule_key = "backlogs"

        elif gap == "Industry/internship exposure":
            rule_key = "internships"

        elif gap == "Practical project experience":
            rule_key = "projects"

        elif gap == "Relevant certifications":
            rule_key = "certifications"

        if (
            rule_key
            and isinstance(
                recommendation_rules,
                dict
            )
            and rule_key in recommendation_rules
        ):

            rule = recommendation_rules[
                rule_key
            ]

            if isinstance(rule, dict):

                message = rule.get(
                    "message"
                )

                if message:
                    recommendations.append(
                        message
                    )

    # Always provide useful fallback recommendations.
    fallback_map = {

        "Technical/computational skills":
            "Strengthen the technical skills most relevant to your branch and target career.",

        "Communication and interview skills":
            "Practice structured answers, presentations, group discussions and mock interviews.",

        "Aptitude preparation":
            "Practice quantitative aptitude, logical reasoning and verbal reasoning regularly.",

        "Academic performance":
            "Focus on improving current academic performance and maintaining a consistent CGPA.",

        "Backlog clearance":
            "Prioritize clearing academic backlogs because they can affect eligibility for some opportunities.",

        "Industry/internship exposure":
            "Seek a relevant internship, industry project, research project or supervised practical experience.",

        "Practical project experience":
            "Build branch-specific projects that demonstrate practical application of your knowledge.",

        "Relevant certifications":
            "Consider certifications that directly support your chosen career direction."
    }

    for gap in gaps:

        fallback = fallback_map.get(
            gap
        )

        if (
            fallback
            and fallback not in recommendations
        ):

            recommendations.append(
                fallback
            )

    return recommendations[:5]


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client(api_key):

    from google import genai

    return genai.Client(
        api_key=api_key.strip()
    )


# ============================================================
# GEMINI RESPONSE EXTRACTION
# ============================================================

def extract_gemini_text(interaction):

    # google-genai Interactions API
    text_value = getattr(
        interaction,
        "output_text",
        None
    )

    if text_value:

        return str(
            text_value
        ).strip()

    # Try output objects.
    output = getattr(
        interaction,
        "output",
        None
    )

    if output:

        collected = []

        if isinstance(
            output,
            list
        ):

            items = output

        else:

            items = [output]

        for item in items:

            item_text = getattr(
                item,
                "text",
                None
            )

            if item_text:
                collected.append(
                    str(item_text)
                )

            content = getattr(
                item,
                "content",
                None
            )

            if content:

                if isinstance(
                    content,
                    list
                ):

                    for part in content:

                        part_text = getattr(
                            part,
                            "text",
                            None
                        )

                        if part_text:
                            collected.append(
                                str(part_text)
                            )

        if collected:

            return "\n".join(
                collected
            ).strip()

    return ""


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    student,
    prediction_context=None
):

    branch = student[
        "ug_branch"
    ]

    branch_skills = get_branch_skills(
        branch
    )

    branch_skill_lines = []

    for skill in branch_skills:

        score_key = f"skill_{skill}"

        score = student.get(
            score_key,
            5
        )

        branch_skill_lines.append(
            f"- {skill}: {score}/10"
        )

    branch_skills_text = "\n".join(
        branch_skill_lines
    )

    critical_gaps = []

    for skill in branch_skills:

        score = student.get(
            f"skill_{skill}",
            5
        )

        if score <= 4:

            critical_gaps.append(
                f"{skill} ({score}/10)"
            )

    development = []

    for skill in branch_skills:

        score = student.get(
            f"skill_{skill}",
            5
        )

        if 5 <= score <= 6:

            development.append(
                f"{skill} ({score}/10)"
            )

    if critical_gaps:

        critical_gap_text = "\n".join(
            f"- {x}"
            for x in critical_gaps
        )

    else:

        critical_gap_text = (
            "- No critical branch-skill gap based on the self-assessment."
        )

    if development:

        development_text = "\n".join(
            f"- {x}"
            for x in development
        )

    else:

        development_text = (
            "- Continue strengthening the existing skill profile."
        )

    if prediction_context:

        prediction_text = prediction_context

    else:

        prediction_text = (
            "Placement prediction was not requested. "
            "Provide career guidance based on the student profile only."
        )

    return f"""
You are an expert student career counselor and employability advisor.

Your task is to provide personalized, realistic and field-specific career guidance.

IMPORTANT RULES:

1. The student's actual academic branch is the primary domain context.
2. Career interest and target career goal should strongly influence recommendations.
3. Consider both UG and PG education when available.
4. Do NOT automatically assume the student is from Computer Science.
5. Do NOT recommend software-development projects to non-software students unless their stated career goal specifically requires it.
6. Project ideas must be appropriate for the student's actual academic field.
7. Give EXACTLY 3 project ideas.
8. Do not mention internal machine-learning encoding, feature names, SHAP internals, or implementation details.
9. Do not mention resume analysis.
10. Do not guarantee employment or placement.
11. Do not invent achievements that are not in the profile.
12. Identify realistic skill gaps.
13. Make the guidance practical and actionable.
14. Use the student's branch, skills, education, interests and target career goal together.
15. If the target career is different from the academic branch, explain the bridge skills needed.
16. For non-CS branches, keep the technical recommendations appropriate to that branch.
17. The answer should be useful for a student preparing for internships, jobs or higher studies.

============================================================
STUDENT PROFILE
============================================================

Age:
{student["age"]}

Gender:
{student["gender"]}

UG Degree:
{student["ug_degree"]}

UG Branch:
{student["ug_branch"]}

UG CGPA:
{student["ug_cgpa"]}

Postgraduate Education:
{
    student["pg_degree"]
    if student["has_pg"]
    else "Not Applicable"
}

PG Specialization:
{
    student["pg_branch"]
    if student["has_pg"]
    else "Not Applicable"
}

PG CGPA:
{
    student["pg_cgpa"]
    if student["has_pg"]
    else "Not Applicable"
}

============================================================
CAREER DIRECTION
============================================================

Career Interest:
{student["career_interest"]}

Target Career Goal:
{
    student["target_career_goal"]
    if student["target_career_goal"]
    else "Not specified"
}

============================================================
PLACEMENT PROFILE
============================================================

{prediction_text}

Projects:
{student["projects"]}

Internships:
{student["internships"]}

Certifications:
{student["certifications"]}

Backlogs:
{student["backlogs"]}

Coding / Computational Skills:
{student["coding_skills"]}/10

Communication Skills:
{student["communication_skills"]}/10

Aptitude Score:
{student["aptitude_score"]}/100

============================================================
ACTUAL BRANCH-SPECIFIC SKILLS
============================================================

Academic Branch:
{branch}

Relevant branch skills:

{branch_skills_text}

============================================================
CRITICAL SKILL GAPS
============================================================

{critical_gap_text}

============================================================
SKILLS NEEDING DEVELOPMENT
============================================================

{development_text}

============================================================
REQUIRED RESPONSE
============================================================

Use the following headings exactly:

## 1. Overall Profile Assessment

Assess the student's current academic, practical and career profile.

## 2. Career Direction

Explain how the student's branch, career interest and target goal fit together.

## 3. Recommended Career Paths

Give relevant career paths based on the actual branch and stated interests.

## 4. Top Strengths

Identify the strongest current advantages.

## 5. Skill Gap Analysis

Identify the most important gaps and explain why they matter for the target career.

## 6. Areas to Improve

Give specific and practical improvement actions.

## 7. 30-Day Improvement Plan

Create a Week 1, Week 2, Week 3 and Week 4 plan.

## 8. Technical Topics to Study

Recommend branch-specific technical topics.

## 9. Industry Tools and Professional Skills

Recommend relevant tools, software, laboratory methods, business tools, engineering tools, analytical tools or professional skills based on the actual branch.

## 10. Project Ideas

Give EXACTLY 3 project ideas.

Each project must include:
- Project title
- What the student should build/do
- Skills demonstrated
- Why it is relevant to the student's target career

The three projects must be appropriate for the student's actual branch.

## 11. Interview Preparation

Cover:
- Core technical/domain questions
- HR questions
- Communication
- Project explanation
- Internship explanation
- Aptitude preparation where relevant

Keep recommendations specific to the student's field.

Finish with a short practical action summary.

Do not guarantee placement.
"""


# ============================================================
# GENERATE GEMINI GUIDANCE
# ============================================================

def generate_ai_guidance(
    student,
    prediction_context=None
):

    api_key = st.session_state.get(
        "gemini_api_key",
        ""
    )

    if not api_key:

        raise ValueError(
            "Please enter your Gemini API key."
        )

    prompt = build_gemini_prompt(
        student,
        prediction_context
    )

    client = get_gemini_client(
        api_key
    )

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        store=False
    )

    answer = extract_gemini_text(
        interaction
    )

    if not answer:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return answer


# ============================================================
# BUILD STUDENT PROFILE
# ============================================================

def build_student_profile(
    gender,
    age,
    ug_degree,
    ug_branch,
    ug_cgpa,
    has_pg,
    pg_degree,
    pg_branch,
    pg_cgpa,
    backlogs,
    internships,
    projects,
    certifications,
    coding_skills,
    communication_skills,
    aptitude_score,
    career_interest,
    target_career_goal,
    domain_scores
):

    student = {

        "gender": gender,

        "age": age,

        # Normalized names used by the model layer.
        "ug_degree": ug_degree,

        "degree": ug_degree,

        "ug_branch": ug_branch,

        "branch": map_branch_for_model(
            ug_branch
        ),

        "ug_cgpa": float(
            ug_cgpa
        ),

        "cgpa": float(
            ug_cgpa
        ),

        "has_pg": has_pg,

        "pg_degree": pg_degree,

        "pg_branch": pg_branch,

        "pg_cgpa": (
            float(pg_cgpa)
            if has_pg
            else None
        ),

        "backlogs": int(
            backlogs
        ),

        "internships": int(
            internships
        ),

        "projects": int(
            projects
        ),

        "certifications": int(
            certifications
        ),

        "coding_skills": int(
            coding_skills
        ),

        "communication_skills": int(
            communication_skills
        ),

        "aptitude_score": int(
            aptitude_score
        ),

        "career_interest": (
            career_interest
        ),

        "target_career_goal": (
            target_career_goal.strip()
        )
    }

    for skill, score in domain_scores.items():

        student[
            f"skill_{skill}"
        ] = score

    return student


# ============================================================
# RUN PLACEMENT PREDICTION
# ============================================================

def run_prediction(student):

    (
        model,
        artifact_features,
        recommendation_rules
    ) = load_components()

    model_input, input_type = (
        prepare_model_input(
            model,
            student,
            artifact_features
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input
    )[0]

    probability = (
        get_positive_probability(
            model,
            model_input,
            prediction
        )
    )

    # --------------------------------------------------------
    # Prediction text
    # --------------------------------------------------------

    prediction_number = None

    try:
        prediction_number = int(
            prediction
        )
    except Exception:
        pass

    if prediction_number == 1:

        prediction_text = (
            "LIKELY PLACED"
        )

    elif prediction_number == 0:

        prediction_text = (
            "NOT PLACED"
        )

    else:

        prediction_text = str(
            prediction
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    strengths = get_profile_strengths(
        student
    )

    gaps = get_profile_gaps(
        student
    )

    recommendations = (
        generate_recommendations(
            student,
            recommendation_rules
        )
    )

    return {

        "prediction": prediction,

        "prediction_text": prediction_text,

        "probability": probability,

        "model_input_type": input_type,

        "model_features": list(
            model_input.columns
        )
        if hasattr(
            model_input,
            "columns"
        )
        else [],

        "strengths": strengths,

        "gaps": gaps,

        "recommendations": recommendations
    }


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 AI Student Placement Predictor"
)

st.write(
    """
Predict placement readiness from an academic and employability profile,
or use Gemini independently for personalized career guidance.
"""
)

st.info(
    """
⚠️ Placement predictions are based on the trained model and should be
treated as decision-support information, not a guarantee of employment.
"""
)


# ============================================================
# ANALYSIS MODE
# ============================================================

st.divider()

st.subheader(
    "Choose Analysis Mode"
)

mode = st.radio(
    "What would you like to use?",
    [
        "🔮 Placement Prediction + AI Guidance",
        "🤖 AI Career Guidance Only"
    ],
    horizontal=True
)

if mode == "🤖 AI Career Guidance Only":

    st.success(
        "AI-only mode is active. You can generate Gemini career guidance without running the placement prediction model."
    )


# ============================================================
# GEMINI API KEY
# ============================================================

st.subheader(
    "🔑 Gemini AI"
)

api_key = st.text_input(
    "Gemini API Key",
    type="password",
    value=st.session_state.get(
        "gemini_api_key",
        ""
    ),
    help="Your key is used only for the current Streamlit session."
)

if api_key:

    st.session_state.gemini_api_key = (
        api_key.strip()
    )

else:

    st.caption(
        "Gemini guidance requires a Gemini API key."
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

st.divider()

st.header(
    "📋 Student Profile"
)


# ============================================================
# EDUCATION
# ============================================================

st.subheader(
    "🎓 Education"
)

col1, col2, col3 = st.columns(3)

with col1:

    ug_degree = st.selectbox(
        "UG Degree",
        [
            "BE",
            "BTech",
            "BSc",
            "BCA",
            "BBA",
            "BCom",
            "BA",
            "Other"
        ]
    )

with col2:

    branches = [

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

        "Other"
    ]

    ug_branch = st.selectbox(
        "UG Branch / Major",
        branches
    )

with col3:

    ug_cgpa = st.number_input(
        "UG CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )


# ============================================================
# PG
# ============================================================

has_pg = st.checkbox(
    "I have postgraduate education"
)

pg_degree = "Not Applicable"
pg_branch = "Not Applicable"
pg_cgpa = 0.0

if has_pg:

    pg1, pg2, pg3 = st.columns(3)

    with pg1:

        pg_degree = st.selectbox(
            "PG Degree",
            [
                "MTech",
                "MSc",
                "MCA",
                "MBA",
                "MCom",
                "MA",
                "Other"
            ]
        )

    with pg2:

        pg_branch = st.text_input(
            "PG Specialization",
            placeholder="Example: Data Science"
        )

    with pg3:

        pg_cgpa = st.number_input(
            "PG CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.subheader(
    "👤 Personal Information"
)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=60,
        value=22
    )


# ============================================================
# PLACEMENT PROFILE
# ============================================================

st.subheader(
    "💼 Placement Profile"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=20,
        value=0
    )

with col2:

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=20,
        value=1
    )

with col3:

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=30,
        value=2
    )

with col4:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=30,
        value=2
    )


col1, col2, col3 = st.columns(3)

with col1:

    coding_skills = st.slider(
        "Programming / Computational Skills",
        min_value=1,
        max_value=10,
        value=5
    )

with col2:

    communication_skills = st.slider(
        "Communication Skills",
        min_value=1,
        max_value=10,
        value=5
    )

with col3:

    aptitude_score = st.slider(
        "Aptitude Score",
        min_value=0,
        max_value=100,
        value=60
    )


# ============================================================
# CAREER DIRECTION
# ============================================================

st.subheader(
    "🧭 Career Direction"
)

career_options = get_career_interests(
    ug_branch
)

career_interest = st.selectbox(
    "Career Interest",
    career_options
)

target_career_goal = st.text_input(
    "Target Career Goal",
    placeholder="Example: Data Analyst, Food Safety Officer, Financial Analyst, Research Scientist"
)


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

st.subheader(
    f"🧠 {ug_branch} Skills"
)

st.caption(
    "Rate your current confidence in the skills relevant to your actual academic branch."
)

branch_skills = get_branch_skills(
    ug_branch
)

domain_scores = {}

skill_columns = st.columns(
    len(branch_skills)
)

for index, skill in enumerate(
    branch_skills
):

    with skill_columns[index]:

        domain_scores[
            skill
        ] = st.slider(
            skill,
            min_value=1,
            max_value=10,
            value=5,
            key=f"domain_{skill}"
        )


# ============================================================
# BUILD STUDENT
# ============================================================

student = build_student_profile(

    gender=gender,

    age=age,

    ug_degree=ug_degree,

    ug_branch=ug_branch,

    ug_cgpa=ug_cgpa,

    has_pg=has_pg,

    pg_degree=pg_degree,

    pg_branch=pg_branch,

    pg_cgpa=pg_cgpa,

    backlogs=backlogs,

    internships=internships,

    projects=projects,

    certifications=certifications,

    coding_skills=coding_skills,

    communication_skills=communication_skills,

    aptitude_score=aptitude_score,

    career_interest=career_interest,

    target_career_goal=target_career_goal,

    domain_scores=domain_scores
)


# ============================================================
# ACTION BUTTON
# ============================================================

st.divider()

if mode == "🤖 AI Career Guidance Only":

    generate_button = st.button(
        "🤖 Generate AI Career Guidance",
        use_container_width=True,
        type="primary"
    )

else:

    generate_button = st.button(
        "🔮 Predict Placement Readiness",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# AI-ONLY MODE
# ============================================================

if (
    generate_button
    and mode == "🤖 AI Career Guidance Only"
):

    st.session_state.student_profile = student
    st.session_state.ai_mode = "only"

    try:

        with st.spinner(
            "Gemini is preparing personalized career guidance..."
        ):

            advice = generate_ai_guidance(
                student,
                prediction_context=None
            )

        st.session_state.ai_career_advice = (
            advice
        )

    except Exception as e:

        st.error(
            f"Gemini guidance failed: {e}"
        )


# ============================================================
# PLACEMENT MODE
# ============================================================

if (
    generate_button
    and mode == "🔮 Placement Prediction + AI Guidance"
):

    st.session_state.student_profile = student
    st.session_state.ai_mode = "placement"
    st.session_state.ai_career_advice = None

    try:

        with st.spinner(
            "Running placement prediction..."
        ):

            result = run_prediction(
                student
            )

        st.session_state.prediction_result = (
            result
        )

    except Exception as e:

        st.session_state.prediction_result = None

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.warning(
            """
The app now uses the fitted model's own feature_names_in_ when
available, so category columns such as branch_AI, degree_BCA and
gender_Female are not blindly sent to the model.
"""
        )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

result = st.session_state.prediction_result

if (
    result is not None
    and st.session_state.ai_mode == "placement"
):

    st.divider()

    st.header(
        "🎯 Placement Prediction Result"
    )

    probability = result[
        "probability"
    ]

    prediction_text = result[
        "prediction_text"
    ]

    if probability is not None:

        probability_percent = (
            probability * 100
        )

        readiness, readiness_emoji = (
            get_readiness_level(
                probability
            )
        )

    else:

        probability_percent = None

        readiness, readiness_emoji = (
            get_readiness_level(
                None
            )
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            prediction_text
        )

    with col2:

        if probability_percent is not None:

            st.metric(
                "Placement Probability",
                f"{probability_percent:.2f}%"
            )

        else:

            st.metric(
                "Placement Probability",
                "Not available"
            )

    with col3:

        st.metric(
            "Placement Readiness",
            f"{readiness_emoji} {readiness}"
        )


    # ========================================================
    # PROFILE STRENGTHS
    # ========================================================

    st.subheader(
        "💪 Your Strengths"
    )

    for strength in result[
        "strengths"
    ]:

        st.success(
            strength
        )


    # ========================================================
    # SKILL GAPS
    # ========================================================

    st.subheader(
        "📈 Areas to Improve"
    )

    gaps = result[
        "gaps"
    ]

    if gaps:

        for gap in gaps:

            st.warning(
                gap
            )

    else:

        st.success(
            "No major profile gaps were identified from the supplied inputs."
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "💡 Personalized Recommendations"
    )

    recommendations = result[
        "recommendations"
    ]

    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"**{index}.** {recommendation}"
            )

    else:

        st.info(
            "No additional rule-based recommendations are available."
        )


    # ========================================================
    # MODEL INPUT STATUS
    # ========================================================

    with st.expander(
        "Model compatibility information"
    ):

        st.write(
            "Input preparation mode:",
            result["model_input_type"]
        )

        if result["model_features"]:

            st.write(
                "Number of features sent to the model:",
                len(
                    result[
                        "model_features"
                    ]
                )
            )


    # ========================================================
    # AI GUIDANCE AFTER PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Personalized AI Career Guidance"
    )

    st.write(
        "Gemini can now interpret your placement result together with your academic branch, career interest, skills and goals."
    )

    ai_button = st.button(
        "🧠 Generate Personalized Gemini Guidance",
        use_container_width=True
    )

    if ai_button:

        try:

            if probability is not None:

                prediction_context = f"""
Placement Prediction:
{prediction_text}

Placement Probability:
{probability_percent:.2f}%

Placement Readiness:
{readiness}
"""

            else:

                prediction_context = f"""
Placement Prediction:
{prediction_text}

The model did not provide a probability score.
"""

            with st.spinner(
                "Gemini is preparing personalized career guidance..."
            ):

                advice = generate_ai_guidance(
                    student,
                    prediction_context
                )

            st.session_state.ai_career_advice = (
                advice
            )

        except Exception as e:

            st.error(
                f"Gemini guidance failed: {e}"
            )


# ============================================================
# DISPLAY AI-ONLY OR POST-PREDICTION GUIDANCE
# ============================================================

if st.session_state.ai_career_advice:

    st.divider()

    st.header(
        "🧠 Personalized AI Career Guidance"
    )

    st.markdown(
        st.session_state.ai_career_advice
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Student Placement Predictor | Machine Learning + Gemini Career Guidance"
)
