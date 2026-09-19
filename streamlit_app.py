import os
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
    layout="wide",
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULES_FILE = "recommendation_rules.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_FILE)
    metadata = joblib.load(METADATA_FILE)
    recommendation_rules = joblib.load(RULES_FILE)

    return model, feature_names, metadata, recommendation_rules


try:
    model, feature_names, metadata, recommendation_rules = load_artifacts()
except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# BRANCH-SPECIFIC DOMAIN SKILLS
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
    "Automobile Engineering": [
        "Automotive Design",
        "Automotive Systems",
        "CAD / Design",
        "Manufacturing",
        "Vehicle Technology",
    ],
    "Production Engineering": [
        "Manufacturing",
        "Production Planning",
        "Quality Control",
        "CAD / Design",
        "Industrial Processes",
    ],
    "Industrial Engineering": [
        "Operations Research",
        "Production Planning",
        "Quality Control",
        "Process Optimization",
        "Supply Chain",
    ],
    "Aeronautical Engineering": [
        "Aerodynamics",
        "Aircraft Design",
        "CAD / Design",
        "Propulsion",
        "Materials",
    ],
    "Aerospace Engineering": [
        "Aerodynamics",
        "Aircraft Design",
        "Propulsion",
        "Flight Mechanics",
        "Materials",
    ],
    "Electrical Engineering": [
        "Circuit Analysis",
        "Power Systems",
        "Control Systems",
        "PLC / Automation",
        "Electrical Design",
    ],
    "Electronics Engineering": [
        "Circuit Design",
        "Digital Electronics",
        "Microcontrollers",
        "Embedded Systems",
        "Communication Systems",
    ],
    "Electronics and Communication Engineering": [
        "Circuit Design",
        "Digital Electronics",
        "Communication Systems",
        "Embedded Systems",
        "Signal Processing",
    ],
    "Biomedical Engineering": [
        "Biomedical Instrumentation",
        "Medical Devices",
        "Biology",
        "Electronics",
        "Signal Processing",
    ],
    "Civil Engineering": [
        "Structural Engineering",
        "AutoCAD",
        "Surveying",
        "Construction Management",
        "Quantity Estimation",
    ],
    "Chemical Engineering": [
        "Chemical Processes",
        "Thermodynamics",
        "Process Control",
        "Fluid Mechanics",
        "Plant Design",
    ],
    "Mathematics": [
        "Mathematical Analysis",
        "Algebra",
        "Calculus",
        "Statistics",
        "Problem Solving",
    ],
    "Statistics": [
        "Statistics",
        "Probability",
        "Data Analysis",
        "Statistical Modeling",
        "R / Python",
    ],
    "Physics": [
        "Classical Physics",
        "Quantum Physics",
        "Mathematics",
        "Experimental Methods",
        "Data Analysis",
    ],
    "Chemistry": [
        "Organic Chemistry",
        "Inorganic Chemistry",
        "Physical Chemistry",
        "Laboratory Skills",
        "Analytical Chemistry",
    ],
    "Environmental Science": [
        "Environmental Analysis",
        "Ecology",
        "Environmental Management",
        "GIS",
        "Sustainability",
    ],
    "Biotechnology": [
        "Molecular Biology",
        "Genetics",
        "Biochemistry",
        "Laboratory Skills",
        "Bioinformatics",
    ],
    "Microbiology": [
        "Microbiology",
        "Molecular Biology",
        "Laboratory Skills",
        "Biochemistry",
        "Biotechnology",
    ],
    "Biochemistry": [
        "Biochemistry",
        "Molecular Biology",
        "Laboratory Skills",
        "Analytical Chemistry",
        "Biotechnology",
    ],
    "Biological Sciences": [
        "Biology",
        "Genetics",
        "Molecular Biology",
        "Laboratory Skills",
        "Data Analysis",
    ],
    "Life Sciences": [
        "Biology",
        "Molecular Biology",
        "Genetics",
        "Laboratory Skills",
        "Research Methods",
    ],
    "Genetics": [
        "Genetics",
        "Molecular Biology",
        "Bioinformatics",
        "Biotechnology",
        "Laboratory Skills",
    ],
    "Botany": [
        "Plant Biology",
        "Plant Physiology",
        "Ecology",
        "Taxonomy",
        "Laboratory Skills",
    ],
    "Zoology": [
        "Animal Biology",
        "Ecology",
        "Genetics",
        "Taxonomy",
        "Laboratory Skills",
    ],
    "Food Science and Nutrition": [
        "Food Chemistry",
        "Food Safety",
        "Nutrition",
        "Food Processing",
        "Quality Control",
    ],
    "Food Technology": [
        "Food Processing",
        "Food Safety",
        "Food Chemistry",
        "Quality Control",
        "Product Development",
    ],
    "Nutrition and Dietetics": [
        "Nutrition",
        "Diet Planning",
        "Food Science",
        "Clinical Nutrition",
        "Public Health",
    ],
    "Economics": [
        "Microeconomics",
        "Macroeconomics",
        "Econometrics",
        "Data Analysis",
        "Financial Analysis",
    ],
    "Commerce": [
        "Accounting",
        "Taxation",
        "Financial Analysis",
        "Auditing",
        "Business Knowledge",
    ],
    "Business Administration": [
        "Business Management",
        "Marketing",
        "Finance",
        "Operations",
        "Business Analytics",
    ],
    "Finance": [
        "Financial Analysis",
        "Accounting",
        "Investment Analysis",
        "Financial Modeling",
        "Banking Knowledge",
    ],
    "Accounting": [
        "Financial Accounting",
        "Cost Accounting",
        "Taxation",
        "Auditing",
        "Financial Analysis",
    ],
    "Management": [
        "Business Management",
        "Leadership",
        "Operations",
        "Marketing",
        "Business Analytics",
    ],
    "Marketing": [
        "Marketing Strategy",
        "Digital Marketing",
        "Market Research",
        "Consumer Behavior",
        "Sales",
    ],
    "Human Resources": [
        "Recruitment",
        "Employee Relations",
        "HR Management",
        "Performance Management",
        "Communication",
    ],
    "Psychology": [
        "Psychology",
        "Research Methods",
        "Statistics",
        "Counseling",
        "Communication",
    ],
    "English": [
        "Communication",
        "Writing",
        "Literature",
        "Research",
        "Presentation",
    ],
    "Political Science": [
        "Political Analysis",
        "Public Policy",
        "Research",
        "Communication",
        "Data Analysis",
    ],
    "Sociology": [
        "Social Research",
        "Data Analysis",
        "Communication",
        "Research Methods",
        "Social Theory",
    ],
    "History": [
        "Historical Research",
        "Research Methods",
        "Writing",
        "Critical Analysis",
        "Documentation",
    ],
    "Public Administration": [
        "Public Policy",
        "Administration",
        "Governance",
        "Communication",
        "Research",
    ],
    "Other": [
        "Domain Knowledge",
        "Technical Skills",
        "Problem Solving",
        "Communication",
        "Professional Skills",
    ],
}


# ============================================================
# OPTIONS
# ============================================================

BRANCHES = list(BRANCH_SKILLS.keys())

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
# HELPER FUNCTIONS
# ============================================================

def clamp_score(value):
    return max(0.0, min(10.0, float(value)))


def get_readiness_level(score):
    if score >= 8:
        return "Excellent"
    elif score >= 6:
        return "Good"
    elif score >= 4:
        return "Developing"
    else:
        return "Needs Improvement"


def generate_recommendations(
    branch,
    domain_scores,
    coding,
    communication,
    aptitude,
    cgpa,
    internships,
    projects,
    certifications,
):
    recommendations = []

    skill_names = BRANCH_SKILLS.get(branch, BRANCH_SKILLS["Other"])

    for skill_name, score in zip(skill_names, domain_scores):
        if score < 4:
            recommendations.append(
                f"Focus strongly on {skill_name}. Your current score is {score:.1f}/10."
            )
        elif score < 6:
            recommendations.append(
                f"Improve {skill_name} through projects, practice, and structured learning."
            )

    if coding < 6:
        recommendations.append(
            "Improve coding skills through regular problem solving and practical programming projects."
        )

    if communication < 6:
        recommendations.append(
            "Practice communication, presentations, interviews, and explaining technical concepts."
        )

    if aptitude < 60:
        recommendations.append(
            "Practice quantitative aptitude, logical reasoning, and verbal reasoning."
        )

    if cgpa < 7:
        recommendations.append(
            "Try to improve academic performance and maintain a consistent CGPA."
        )

    if internships == 0:
        recommendations.append(
            "Consider gaining internship or practical industry experience."
        )

    if projects < 2:
        recommendations.append(
            "Build more practical projects related to your degree and branch."
        )

    if certifications == 0:
        recommendations.append(
            "Consider completing relevant industry certifications."
        )

    if not recommendations:
        recommendations.append(
            "Maintain your current preparation and continue building practical experience."
        )

    return recommendations[:8]


def make_input_dataframe(
    age,
    gender,
    degree,
    branch,
    cgpa,
    backlogs,
    internships,
    projects,
    certifications,
    coding,
    communication,
    aptitude,
    domain_scores,
):
    data = {
        "age": [age],
        "ug_cgpa": [cgpa],
        "backlogs": [backlogs],
        "internships": [internships],
        "projects": [projects],
        "certifications": [certifications],
        "coding_skills": [coding],
        "communication_skills": [communication],
        "aptitude_score": [aptitude],
        "domain_skill_1": [domain_scores[0]],
        "domain_skill_2": [domain_scores[1]],
        "domain_skill_3": [domain_scores[2]],
        "domain_skill_4": [domain_scores[3]],
        "domain_skill_5": [domain_scores[4]],
        "gender": [gender],
        "ug_degree": [degree],
        "ug_branch": [branch],
    }

    df = pd.DataFrame(data)

    # Ensure exact model feature order.
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0

    df = df[feature_names]

    return df


# ============================================================
# HEADER
# ============================================================

st.title("🎓 AI Student Placement Predictor")

st.markdown(
    """
Predict placement readiness using academic performance, skills,
projects, internships, certifications, aptitude, communication,
and branch-specific domain skills.
"""
)

st.info(
    "ℹ️ This application uses a machine-learning model trained on "
    "synthetic demonstration data. The predicted percentage is a "
    "model score/probability, not a guaranteed real-world employment probability."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📊 Model Information")

    training_mode = metadata.get("training_mode", "Unknown")
    st.write(f"**Training mode:** {training_mode}")

    if "dataset_rows" in metadata:
        st.write(f"**Training rows:** {metadata['dataset_rows']}")

    if "roc_auc" in metadata:
        st.write(f"**ROC-AUC:** {metadata['roc_auc']:.3f}")

    st.markdown("---")

    st.write("### Model Features")
    st.write("17 input features")

    st.write(
        """
- Academic performance
- Backlogs
- Internships
- Projects
- Certifications
- Coding skills
- Communication
- Aptitude
- 5 branch-specific skills
- Gender
- Degree
- Branch
"""
    )


# ============================================================
# STUDENT DETAILS
# ============================================================

st.header("👨‍🎓 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=17,
        max_value=40,
        value=21,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        GENDERS,
    )

with col2:
    degree = st.selectbox(
        "UG Degree",
        DEGREES,
    )

    branch = st.selectbox(
        "UG Branch",
        BRANCHES,
    )

with col3:
    cgpa = st.number_input(
        "UG CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )


# ============================================================
# EXPERIENCE / ACHIEVEMENTS
# ============================================================

st.header("💼 Experience & Achievements")

col1, col2, col3 = st.columns(3)

with col1:
    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

with col2:
    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=2,
        step=1,
    )

with col3:
    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
    )


# ============================================================
# GENERAL SKILLS
# ============================================================

st.header("🧠 General Skills")

col1, col2, col3 = st.columns(3)

with col1:
    coding_skills = st.slider(
        "Coding Skills",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
    )

with col2:
    communication_skills = st.slider(
        "Communication Skills",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
    )

with col3:
    aptitude_score = st.slider(
        "Aptitude Score",
        min_value=0.0,
        max_value=100.0,
        value=65.0,
        step=1.0,
    )


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

st.header("🎯 Branch-Specific Domain Skills")

st.write(
    f"Rate your current knowledge in the five core areas for "
    f"**{branch}**."
)

skill_names = BRANCH_SKILLS.get(branch, BRANCH_SKILLS["Other"])

domain_scores = []

skill_cols = st.columns(5)

for i, skill_name in enumerate(skill_names):
    with skill_cols[i]:
        score = st.slider(
            skill_name,
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            key=f"domain_skill_{i}_{branch}",
        )

        domain_scores.append(score)


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Placement Readiness",
    type="primary",
    use_container_width=True,
)


if predict_button:

    input_df = make_input_dataframe(
        age=age,
        gender=gender,
        degree=degree,
        branch=branch,
        cgpa=cgpa,
        backlogs=backlogs,
        internships=internships,
        projects=projects,
        certifications=certifications,
        coding=coding_skills,
        communication=communication_skills,
        aptitude=aptitude_score,
        domain_scores=domain_scores,
    )

    try:
        probability = float(model.predict_proba(input_df)[0][1])

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))
        st.stop()

    placement_percentage = probability * 100

    # ========================================================
    # RESULT
    # ========================================================

    st.header("📈 Placement Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Placement Model Score",
            f"{placement_percentage:.1f}%",
        )

    with result_col2:
        predicted_class = int(model.predict(input_df)[0])

        if predicted_class == 1:
            st.success("🟢 Model classification: Placement-ready")
        else:
            st.warning("🟡 Model classification: Needs further preparation")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        "This percentage is the model's calibrated classification probability "
        "based on the training data distribution. It is not a guaranteed "
        "employment probability."
    )

    # ========================================================
    # READINESS SCORE
    # ========================================================

    domain_average = float(np.mean(domain_scores))

    readiness_components = [
        cgpa / 10,
        coding_skills / 10,
        communication_skills / 10,
        aptitude_score / 100,
        domain_average / 10,
    ]

    readiness_score = float(np.mean(readiness_components) * 100)

    st.header("🎯 Overall Preparation Readiness")

    readiness_col1, readiness_col2 = st.columns(2)

    with readiness_col1:
        st.metric(
            "Readiness Score",
            f"{readiness_score:.1f}/100",
        )

    with readiness_col2:
        st.metric(
            "Domain Skill Average",
            f"{domain_average:.1f}/10",
        )

    st.progress(
        min(max(readiness_score / 100, 0.0), 1.0)
    )

    readiness_level = get_readiness_level(readiness_score / 10)

    if readiness_score >= 80:
        st.success(f"🌟 Readiness Level: {readiness_level}")
    elif readiness_score >= 60:
        st.info(f"👍 Readiness Level: {readiness_level}")
    elif readiness_score >= 40:
        st.warning(f"📚 Readiness Level: {readiness_level}")
    else:
        st.error(f"⚠️ Readiness Level: {readiness_level}")

    # ========================================================
    # SKILL BREAKDOWN
    # ========================================================

    st.header("📊 Skill Breakdown")

    skill_data = pd.DataFrame(
        {
            "Skill": skill_names,
            "Score": domain_scores,
        }
    )

    st.dataframe(
        skill_data,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.header("💡 Personalized Recommendations")

    recommendations = generate_recommendations(
        branch=branch,
        domain_scores=domain_scores,
        coding=coding_skills,
        communication=communication_skills,
        aptitude=aptitude_score,
        cgpa=cgpa,
        internships=internships,
        projects=projects,
        certifications=certifications,
    )

    for i, recommendation in enumerate(recommendations, start=1):
        st.write(f"**{i}.** {recommendation}")

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("🔎 View Prediction Input Data"):

        display_df = input_df.copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Student Placement Predictor • Machine Learning + Streamlit"
)

st.caption(
    "Model trained using synthetic demonstration data. "
    "For real-world deployment, train and validate using historical placement outcomes."
)
