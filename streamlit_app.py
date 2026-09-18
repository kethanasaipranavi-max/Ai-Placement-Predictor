
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# BRANCH-SPECIFIC SKILLS
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

    if branch in [
        "Civil Engineering"
    ]:
        return [
            "Structural Engineering",
            "Construction Management",
            "Site Engineering",
            "Infrastructure",
            "Surveying",
            "Other"
        ]

    if branch in [
        "Chemical Engineering"
    ]:
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
# MAP ACTUAL BRANCH TO MODEL BRANCH
# ============================================================

def map_branch_for_model(branch):

    computer_branches = [
        "Computer Science",
        "Information Technology",
        "Data Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Cyber Security",
        "Software Engineering",
        "Computer Applications"
    ]

    electrical_branches = [
        "Electrical Engineering",
        "Electronics Engineering",
        "Electronics and Communication Engineering",
        "Biomedical Engineering"
    ]

    mechanical_branches = [
        "Mechanical Engineering",
        "Automobile Engineering",
        "Production Engineering",
        "Industrial Engineering",
        "Aeronautical Engineering",
        "Aerospace Engineering"
    ]

    if branch in computer_branches:
        return "CS"

    if branch in electrical_branches:
        return "Electrical"

    if branch in mechanical_branches:
        return "Mechanical"

    if branch == "Civil Engineering":
        return "Mechanical"

    return "DS"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_components():

    model = joblib.load(
        "placement_prediction_final.pkl"
    )

    feature_names = joblib.load(
        "placement_feature_names_final.pkl"
    )

    recommendation_rules = {}

    if os.path.exists("recommendation_rules.pkl"):
        try:
            recommendation_rules = joblib.load(
                "recommendation_rules.pkl"
            )
        except Exception:
            recommendation_rules = {}

    return model, list(feature_names), recommendation_rules


try:

    model, feature_names, recommendation_rules = load_components()

except Exception as e:

    st.error(
        "Unable to load the placement prediction model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_student_data(student, feature_columns):

    df = pd.DataFrame([{
        "gender": student["gender"],
        "age": student["age"],
        "degree": student["degree"],
        "branch": map_branch_for_model(student["ug_branch"]),
        "cgpa": student["cgpa"],
        "backlogs": student["backlogs"],
        "internships": student["internships"],
        "certifications": student["certifications"],
        "coding_skills": student["coding_skills"],
        "communication_skills": student["communication_skills"],
        "aptitude_score": student["aptitude_score"],
        "projects": student["projects"]
    }])

    def cgpa_category(cgpa):

        if cgpa < 6:
            return "Low"

        if cgpa < 7.5:
            return "Good"

        return "Excellent"

    df["cgpa_category"] = df["cgpa"].apply(
        cgpa_category
    )

    categorical_columns = [
        "gender",
        "degree",
        "branch",
        "cgpa_category"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    for column in feature_columns:

        if column not in df.columns:
            df[column] = 0

    df = df[feature_columns]

    return df


# ============================================================
# PLACEMENT PREDICTION
# ============================================================

def predict_placement(student):

    processed = prepare_student_data(
        student,
        feature_names
    )

    prediction = int(
        model.predict(processed)[0]
    )

    try:

        probability = float(
            model.predict_proba(processed)[0][1]
        )

    except Exception:

        probability = float(prediction)

    return prediction, probability, processed


# ============================================================
# RELIABILITY
# ============================================================

def get_reliability(probability):

    distance = abs(probability - 0.5) * 2

    if distance >= 0.70:
        return "High"

    if distance >= 0.40:
        return "Moderate"

    return "Low"


# ============================================================
# READINESS
# ============================================================

def get_readiness(probability):

    if probability >= 0.75:
        return "High Placement Readiness", "🟢"

    if probability >= 0.50:
        return "Moderate Placement Readiness", "🟡"

    return "Needs Improvement", "🔴"


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

def analyze_skill_gaps(branch_skills):

    critical = []
    development = []
    strengths = []

    for skill, score in branch_skills.items():

        if score <= 3:

            critical.append({
                "skill": skill,
                "score": score
            })

        elif score <= 6:

            development.append({
                "skill": skill,
                "score": score
            })

        else:

            strengths.append({
                "skill": skill,
                "score": score
            })

    return {
        "critical_gaps": critical,
        "development_needed": development,
        "strengths": strengths
    }


# ============================================================
# RULE-BASED RECOMMENDATIONS
# ============================================================

def generate_recommendations(student):

    recommendations = []

    if student["cgpa"] < 7:
        recommendations.append(
            "Work on improving academic performance and maintaining a stronger CGPA."
        )

    if student["backlogs"] > 0:
        recommendations.append(
            "Clear pending backlogs and maintain a clean academic record."
        )

    if student["internships"] == 0:
        recommendations.append(
            "Gain practical exposure through internships, industry projects, or field training."
        )

    if student["projects"] < 2:
        recommendations.append(
            "Build at least two practical projects related to your academic field and career goal."
        )

    if student["certifications"] == 0:
        recommendations.append(
            "Complete relevant certifications that support your chosen career direction."
        )

    if student["aptitude_score"] < 60:
        recommendations.append(
            "Practice quantitative aptitude, logical reasoning, and problem-solving regularly."
        )

    if student["communication_skills"] < 6:
        recommendations.append(
            "Improve communication, presentation, group discussion, and interview skills."
        )

    if student["coding_skills"] < 5:
        recommendations.append(
            "Develop stronger programming or computational skills where relevant to your career direction."
        )

    if not recommendations:

        recommendations.append(
            "Continue strengthening practical experience, advanced skills, projects, and interview preparation."
        )

    return recommendations


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "student_data" not in st.session_state:
    st.session_state.student_data = None

if "ai_career_advice" not in st.session_state:
    st.session_state.ai_career_advice = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎓 AI Student Placement Predictor"
)

st.write(
    "Predict placement readiness using machine learning and receive personalized, field-specific career guidance."
)

st.info(
    "⚠️ This system provides decision support based on a trained machine-learning model. "
    "It does not guarantee placement."
)


# ============================================================
# STUDENT EDUCATION
# ============================================================

st.header("🎓 Education Profile")

col1, col2 = st.columns(2)

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

    ug_branch = st.selectbox(
        "UG Branch / Major",
        [
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
    )

with col2:

    has_pg = st.checkbox(
        "I have postgraduate education"
    )

    if has_pg:

        pg_degree = st.selectbox(
            "PG Degree",
            [
                "MTech",
                "ME",
                "MSc",
                "MCA",
                "MBA",
                "MCom",
                "MA",
                "Other"
            ]
        )

        pg_branch = st.text_input(
            "PG Specialization"
        )

        pg_cgpa = st.number_input(
            "PG CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    else:

        pg_degree = "None"
        pg_branch = ""
        pg_cgpa = 0.0


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("👤 Student Profile")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=22
    )

with col2:

    cgpa = st.number_input(
        "UG CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    backlogs = st.number_input(
        "Number of Backlogs",
        min_value=0,
        max_value=20,
        value=0
    )

with col3:

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=2
    )


# ============================================================
# ADDITIONAL PROFILE
# ============================================================

st.header("📚 Placement Profile")

col1, col2, col3 = st.columns(3)

with col1:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=20,
        value=2
    )

with col2:

    aptitude_score = st.slider(
        "Aptitude Score",
        min_value=0,
        max_value=100,
        value=60
    )

with col3:

    communication_skills = st.slider(
        "Communication Skills",
        min_value=1,
        max_value=10,
        value=6
    )

coding_skills = st.slider(
    "General Programming / Computational Skills",
    min_value=1,
    max_value=10,
    value=5,
    help="For non-computing fields, this can represent computational or digital skills where applicable."
)


# ============================================================
# CAREER DIRECTION
# ============================================================

st.header("🎯 Career Direction")

career_interest = st.selectbox(
    "Career Interest",
    get_career_interests(ug_branch)
)

target_career_goal = st.text_input(
    "Target Career Goal",
    placeholder="Example: Data Analyst, Clinical Nutritionist, Financial Analyst, Civil Site Engineer..."
)


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

st.header("🧠 Branch-Specific Skills")

st.write(
    f"Rate your current ability in skills relevant to **{ug_branch}**."
)

skills = get_branch_skills(ug_branch)

branch_skill_scores = {}

skill_columns = st.columns(len(skills))

for index, skill in enumerate(skills):

    with skill_columns[index]:

        branch_skill_scores[skill] = st.slider(
            skill,
            min_value=1,
            max_value=10,
            value=5
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Placement Readiness",
    width="stretch"
):

    student = {

        "gender": gender,

        "age": age,

        "ug_degree": ug_degree,

        "ug_branch": ug_branch,

        "pg_degree": pg_degree,

        "pg_branch": pg_branch,

        "pg_cgpa": pg_cgpa,

        "has_pg": has_pg,

        "cgpa": cgpa,

        "backlogs": backlogs,

        "internships": internships,

        "projects": projects,

        "certifications": certifications,

        "coding_skills": coding_skills,

        "communication_skills": communication_skills,

        "aptitude_score": aptitude_score,

        "career_interest": career_interest,

        "target_career_goal": target_career_goal,

        "branch_skills": branch_skill_scores,

        "domain_skills": round(
            np.mean(
                list(branch_skill_scores.values())
            ),
            2
        )
    }

    try:

        prediction, probability, processed = predict_placement(
            student
        )

        reliability = get_reliability(
            probability
        )

        readiness, readiness_icon = get_readiness(
            probability
        )

        recommendations = generate_recommendations(
            student
        )

        skill_gaps = analyze_skill_gaps(
            branch_skill_scores
        )

        result = {

            "prediction": prediction,

            "probability": probability,

            "reliability": reliability,

            "readiness": readiness,

            "recommendations": recommendations,

            "skill_gaps": skill_gaps

        }

        st.session_state.prediction_result = result

        st.session_state.student_data = student

        st.session_state.ai_career_advice = None

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(str(e))


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result
    student = st.session_state.student_data

    st.divider()

    st.header("🎯 Placement Prediction")

    prediction_text = (
        "LIKELY PLACED"
        if result["prediction"] == 1
        else "NOT LIKELY PLACED"
    )

    probability_percent = (
        result["probability"] * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Prediction",
            prediction_text
        )

    with col2:

        st.metric(
            "Placement Probability",
            f"{probability_percent:.2f}%"
        )

    with col3:

        st.metric(
            "Readiness",
            result["readiness"]
        )

    with col4:

        st.metric(
            "Prediction Reliability",
            result["reliability"]
        )

    st.progress(
        max(
            0.0,
            min(
                1.0,
                result["probability"]
            )
        )
    )

    st.caption(
        "The probability is the model's estimated probability for the positive placement class."
    )


    # ========================================================
    # SKILL GAP ANALYSIS
    # ========================================================

    st.divider()

    st.header("📊 Skill Gap Analysis")

    gaps = result["skill_gaps"]

    if gaps["critical_gaps"]:

        st.subheader("🔴 Critical Skill Gaps")

        for item in gaps["critical_gaps"]:

            st.write(
                f"**{item['skill']}** — {item['score']}/10"
            )

    else:

        st.success(
            "No critical skill gaps detected."
        )


    if gaps["development_needed"]:

        st.subheader("🟡 Skills Needing Development")

        for item in gaps["development_needed"]:

            st.write(
                f"**{item['skill']}** — {item['score']}/10"
            )


    if gaps["strengths"]:

        st.subheader("🟢 Current Strengths")

        for item in gaps["strengths"]:

            st.write(
                f"**{item['skill']}** — {item['score']}/10"
            )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.header("💡 Personalized Improvement Recommendations")

    for recommendation in result["recommendations"]:

        st.write(
            f"• {recommendation}"
        )


    # ========================================================
    # GEMINI CAREER GUIDANCE
    # ========================================================

    st.divider()

    st.header("🤖 Gemini Personalized Career Guidance")

    st.write(
        "Generate field-specific career guidance using Gemini."
    )

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your API key is used only for the current session."
    )


    def extract_gemini_text(interaction):

        try:

            text = getattr(
                interaction,
                "output_text",
                None
            )

            if text:
                return text

        except Exception:
            pass

        try:

            output = getattr(
                interaction,
                "output",
                None
            )

            if output:

                collected = []

                for item in output:

                    text = getattr(
                        item,
                        "text",
                        None
                    )

                    if text:
                        collected.append(text)

                if collected:
                    return "\n".join(collected)

        except Exception:
            pass

        return None


    if st.button(
        "🤖 Generate AI Career Guidance",
        width="stretch"
    ):

        if not gemini_api_key.strip():

            st.warning(
                "Please enter your Gemini API key."
            )

        else:

            gaps = result["skill_gaps"]

            critical_text = ", ".join(
                item["skill"]
                for item in gaps["critical_gaps"]
            )

            development_text = ", ".join(
                item["skill"]
                for item in gaps["development_needed"]
            )

            strengths_text = ", ".join(
                item["skill"]
                for item in gaps["strengths"]
            )

            skills_text = "\n".join(
                f"- {skill}: {score}/10"
                for skill, score
                in student["branch_skills"].items()
            )

            pg_information = (

                f"PG Degree: {student['pg_degree']}\n"
                f"PG Specialization: {student['pg_branch']}\n"
                f"PG CGPA: {student['pg_cgpa']}"

                if student["has_pg"]

                else

                "No postgraduate education provided."
            )

            prediction_text_for_ai = (

                "Likely Placed"

                if result["prediction"] == 1

                else

                "Not Likely Placed"
            )

            prompt = f"""
You are an expert career counselor and placement advisor.

Provide highly personalized, realistic and practical career guidance for this student.

The student's ACTUAL academic branch is the PRIMARY academic context.

The student's Career Interest and Target Career Goal must strongly influence the recommendations.

Do NOT automatically recommend Computer Science skills to non-computing students.

For non-computing fields, prioritize their actual domain knowledge, practical field skills, industry tools, laboratory techniques where relevant, professional skills, field-specific projects, internships and certifications.

Never guarantee placement.

Do not mention resumes.

Do not discuss internal model encoding.

Do not discuss hypothetical what-if scenarios.

STUDENT EDUCATION

UG Degree:
{student["ug_degree"]}

UG Branch:
{student["ug_branch"]}

UG CGPA:
{student["cgpa"]}

Postgraduate Education:
{pg_information}

CAREER DIRECTION

Career Interest:
{student["career_interest"]}

Target Career Goal:
{student["target_career_goal"] or "Not specifically provided"}

PLACEMENT PROFILE

Prediction:
{prediction_text_for_ai}

Placement Probability:
{result["probability"] * 100:.2f}%

Prediction Reliability:
{result["reliability"]}

Age:
{student["age"]}

Backlogs:
{student["backlogs"]}

Internships:
{student["internships"]}

Projects:
{student["projects"]}

Certifications:
{student["certifications"]}

Aptitude Score:
{student["aptitude_score"]}/100

Communication Skills:
{student["communication_skills"]}/10

General Programming / Computational Skills:
{student["coding_skills"]}/10

Branch-Specific Skills:
{skills_text}

Overall Domain Skill Score:
{student["domain_skills"]}/10

SKILL GAP ANALYSIS

Critical Skill Gaps:
{critical_text or "None"}

Skills Needing Development:
{development_text or "None"}

Current Strengths:
{strengths_text or "None"}

MODEL-BASED IMPROVEMENT RECOMMENDATIONS:

{chr(10).join("- " + x for x in result["recommendations"])}

IMPORTANT RULES

1. Use the student's actual academic branch.
2. Career Interest and Target Career Goal should strongly influence the advice.
3. Do not automatically recommend software development or programming for non-computing students.
4. Recommend field-specific tools and professional skills.
5. Recommend projects that match the actual academic field and career direction.
6. Consider both UG and PG education when PG is available.
7. Explain the skill gaps clearly.
8. Make the 30-day plan practical.
9. Never guarantee placement.
10. Give exactly 3 project ideas.

Use EXACTLY the following sections:

## Overall Profile Assessment

Assess the student's academic background, practical exposure, skills and placement readiness.

## Career Direction

Explain how the student's academic branch, career interest and target career goal align.

## Recommended Career Paths

Give relevant career paths for this student's actual field.

## Top Strengths

Identify the student's strongest academic, practical and professional areas.

## Skill Gap Analysis

Explain the critical skill gaps and development areas.

## Areas to Improve

Give specific improvements related to the student's actual field.

## 30-Day Improvement Plan

### Week 1
### Week 2
### Week 3
### Week 4

Make each week practical and measurable.

## Technical Topics to Study

Recommend field-specific technical or academic topics.

## Industry Tools and Professional Skills

Recommend appropriate tools, software, laboratory methods, business tools, analytical tools or other professional skills depending on the actual field.

## Project Ideas

Suggest EXACTLY 3 practical projects.

The projects MUST match the student's actual academic field and career direction.

Do NOT automatically suggest software projects.

For example, engineering, science, food, nutrition, commerce, management and humanities students should receive projects appropriate to those fields.

## Interview Preparation

Include:

- Core technical preparation
- HR preparation
- Communication improvement
- Project explanation
- Internship explanation
- Aptitude preparation where relevant

Keep the advice practical and personalized.
"""

            with st.spinner(
                "Generating personalized Gemini career guidance..."
            ):

                try:

                    from google import genai

                    client = genai.Client(
                        api_key=gemini_api_key.strip()
                    )

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=prompt,
                        store=False
                    )

                    generated_text = extract_gemini_text(
                        interaction
                    )

                    if generated_text:

                        st.session_state.ai_career_advice = generated_text

                        st.success(
                            "Gemini Career Guidance Generated Successfully!"
                        )

                        st.markdown(
                            generated_text
                        )

                    else:

                        st.error(
                            "Gemini returned no readable text."
                        )

                except Exception as e:

                    st.error(
                        "Gemini career guidance could not be generated."
                    )

                    st.code(
                        str(e)
                    )


    # ========================================================
    # DISPLAY SAVED GEMINI RESPONSE
    # ========================================================

    if st.session_state.ai_career_advice:

        st.divider()

        st.subheader(
            "🤖 Latest Gemini Career Guidance"
        )

        st.markdown(
            st.session_state.ai_career_advice
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Student Placement Predictor | "
    "Machine Learning + Branch-Specific Skills + "
    "Skill Gap Analysis + Prediction Reliability + "
    "Gemini Personalized Career Guidance"
)
