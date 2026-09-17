
# ============================================================
# AI STUDENT PLACEMENT PREDICTOR
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
SHAP_FILE = "placement_shap_explainer.pkl"
RULES_FILE = "recommendation_rules.pkl"
METADATA_FILE = "placement_model_metadata.pkl"


# ============================================================
# BRANCH OPTIONS
# ============================================================

BRANCH_OPTIONS = [

    "Computer Science",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Machine Learning",
    "Cyber Security",
    "Software Engineering",
    "Computer Applications",

    "Electrical Engineering",
    "Electronics Engineering",
    "Electronics and Communication Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Automobile Engineering",
    "Production Engineering",
    "Industrial Engineering",
    "Aeronautical Engineering",
    "Aerospace Engineering",
    "Biomedical Engineering",

    "Mathematics",
    "Physics",
    "Chemistry",
    "Statistics",
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

    mapping = {

        "Computer Science": [
            "Software Development",
            "Artificial Intelligence / Machine Learning",
            "Data Science",
            "Cyber Security",
            "Cloud Computing",
            "Research and Higher Studies"
        ],

        "Information Technology": [
            "Software Development",
            "Cloud Computing",
            "Cyber Security",
            "IT Infrastructure",
            "Data Analytics"
        ],

        "Data Science": [
            "Data Science",
            "Data Analytics",
            "Machine Learning",
            "Business Analytics",
            "Research"
        ],

        "Artificial Intelligence": [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Data Science",
            "AI Research"
        ],

        "Machine Learning": [
            "Machine Learning Engineering",
            "Artificial Intelligence",
            "Data Science",
            "Deep Learning",
            "AI Research"
        ],

        "Cyber Security": [
            "Cyber Security",
            "Ethical Hacking",
            "Network Security",
            "Security Analysis",
            "Cloud Security"
        ],

        "Mechanical Engineering": [
            "Mechanical Design",
            "Manufacturing",
            "Automobile Industry",
            "Production Engineering",
            "Quality Engineering"
        ],

        "Civil Engineering": [
            "Structural Engineering",
            "Construction Management",
            "Site Engineering",
            "Quantity Surveying",
            "Infrastructure Development"
        ],

        "Electrical Engineering": [
            "Power Systems",
            "Electrical Design",
            "Automation",
            "Control Systems",
            "Embedded Systems"
        ],

        "Electronics Engineering": [
            "Embedded Systems",
            "Electronics Design",
            "Automation",
            "IoT",
            "VLSI"
        ],

        "Electronics and Communication Engineering": [
            "Embedded Systems",
            "Telecommunications",
            "VLSI",
            "IoT",
            "Electronics Design"
        ],

        "Food Science and Nutrition": [
            "Clinical Nutrition",
            "Food Industry",
            "Quality Assurance",
            "Food Research"
        ],

        "Food Technology": [
            "Food Manufacturing",
            "Quality Assurance",
            "Food Safety",
            "Research and Development"
        ],

        "Nutrition and Dietetics": [
            "Clinical Nutrition",
            "Dietetics",
            "Healthcare",
            "Research"
        ],

        "Biotechnology": [
            "Biotechnology Industry",
            "Research",
            "Pharmaceutical Industry",
            "Clinical Research"
        ],

        "Microbiology": [
            "Microbiology Research",
            "Pharmaceutical Industry",
            "Clinical Research",
            "Quality Control"
        ],

        "Chemistry": [
            "Analytical Chemistry",
            "Pharmaceutical Industry",
            "Quality Control",
            "Laboratory Research"
        ],

        "Physics": [
            "Research",
            "Laboratory Science",
            "Instrumentation",
            "Electronics"
        ],

        "Mathematics": [
            "Research",
            "Teaching",
            "Actuarial Science",
            "Analytics",
            "Quantitative Finance"
        ],

        "Statistics": [
            "Data Analytics",
            "Research",
            "Business Analytics",
            "Statistical Research"
        ],

        "Commerce": [
            "Accounting",
            "Banking",
            "Taxation",
            "Auditing",
            "Finance"
        ],

        "Finance": [
            "Financial Analysis",
            "Investment Banking",
            "Corporate Finance",
            "Banking",
            "Investment Analysis"
        ],

        "Management": [
            "Operations Management",
            "Business Development",
            "Human Resources",
            "Marketing",
            "Consulting"
        ],

        "Marketing": [
            "Digital Marketing",
            "Brand Management",
            "Market Research",
            "Business Development"
        ],

        "Human Resources": [
            "Recruitment",
            "HR Operations",
            "Talent Management",
            "Organizational Development"
        ],

        "Psychology": [
            "Counseling",
            "Human Resources",
            "Behavioral Research",
            "Organizational Psychology"
        ],

        "English": [
            "Content Writing",
            "Teaching",
            "Communication",
            "Publishing",
            "Public Relations"
        ]
    }

    return mapping.get(
        branch,
        [
            "Industry Job",
            "Research",
            "Higher Studies",
            "Management",
            "Professional Career"
        ]
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    features = joblib.load(
        FEATURE_FILE
    )

    try:
        explainer = joblib.load(
            SHAP_FILE
        )
    except Exception:
        explainer = None

    try:
        rules = joblib.load(
            RULES_FILE
        )
    except Exception:
        rules = {}

    try:
        metadata = joblib.load(
            METADATA_FILE
        )
    except Exception:
        metadata = {}

    return model, list(features), explainer, rules, metadata


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE
]

missing = [
    x for x in required_files
    if not os.path.exists(x)
]

if missing:

    st.error(
        "Required model files are missing."
    )

    for x in missing:
        st.write(
            f"- {x}"
        )

    st.stop()


model, feature_names, explainer, rules, metadata = load_model()


# ============================================================
# MODEL BRANCH MAPPING
# ============================================================

def map_branch_for_model(branch):

    mapping = {

        "Computer Science": "CS",
        "Software Engineering": "CS",
        "Artificial Intelligence": "CS",
        "Machine Learning": "CS",
        "Cyber Security": "CS",
        "Computer Applications": "CS",

        "Information Technology": "IT",

        "Data Science": "DS",

        "Electrical Engineering": "Electrical",
        "Electronics Engineering": "Electrical",
        "Electronics and Communication Engineering": "Electrical",
        "Biomedical Engineering": "Electrical",

        "Mechanical Engineering": "Mechanical",
        "Automobile Engineering": "Mechanical",
        "Production Engineering": "Mechanical",
        "Industrial Engineering": "Mechanical",
        "Aeronautical Engineering": "Mechanical",
        "Aerospace Engineering": "Mechanical",

        "Civil Engineering": "Mechanical",
        "Chemical Engineering": "Mechanical",

        "Mathematics": "DS",
        "Statistics": "DS",
        "Physics": "Electrical",
        "Chemistry": "DS",
        "Environmental Science": "DS",

        "Biotechnology": "DS",
        "Microbiology": "DS",
        "Biochemistry": "DS",
        "Biological Sciences": "DS",
        "Life Sciences": "DS",
        "Genetics": "DS",
        "Botany": "DS",
        "Zoology": "DS",

        "Food Science and Nutrition": "DS",
        "Food Technology": "DS",
        "Nutrition and Dietetics": "DS",

        "Economics": "DS",
        "Commerce": "DS",
        "Business Administration": "DS",
        "Finance": "DS",
        "Accounting": "DS",
        "Management": "DS",
        "Marketing": "DS",
        "Human Resources": "DS",

        "Psychology": "DS",
        "English": "DS",
        "Political Science": "DS",
        "Sociology": "DS",
        "History": "DS",
        "Public Administration": "DS"
    }

    return mapping.get(
        branch,
        "DS"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(student):

    data = pd.DataFrame([student])

    data["branch"] = data["branch"].apply(
        map_branch_for_model
    )

    data["cgpa_category"] = data["cgpa"].apply(
        lambda x:
        "Low" if x < 6
        else "Good" if x < 7.5
        else "Excellent"
    )

    categorical = [
        "gender",
        "degree",
        "branch",
        "cgpa_category"
    ]

    data = pd.get_dummies(
        data,
        columns=[
            x for x in categorical
            if x in data.columns
        ],
        dtype=int
    )

    for column in feature_names:

        if column not in data.columns:

            data[column] = 0

    data = data[
        feature_names
    ]

    return data


# ============================================================
# GEMINI
# ============================================================

def extract_gemini_text(interaction):

    try:

        text = getattr(
            interaction,
            "output_text",
            None
        )

        if text:
            return str(text).strip()

    except Exception:
        pass

    try:

        output = getattr(
            interaction,
            "output",
            None
        )

        if output:

            texts = []

            for item in output:

                text = getattr(
                    item,
                    "text",
                    None
                )

                if text:
                    texts.append(
                        str(text)
                    )

            if texts:
                return "\n".join(
                    texts
                )

    except Exception:
        pass

    return None


def generate_gemini_advice(
    api_key,
    student,
    result
):

    try:

        from google import genai

    except ImportError:

        return (
            None,
            "Install google-genai using: pip install -U google-genai"
        )

    prompt = f"""
You are an expert student career counselor.

Analyze this student's COMPLETE profile.

ACADEMIC BACKGROUND

UG Degree:
{student['ug_degree']}

UG Branch:
{student['ug_branch']}

UG CGPA:
{student['ug_cgpa']}

PG Available:
{student['has_pg']}

PG Degree:
{student['pg_degree']}

PG Specialization:
{student['pg_branch']}

PG CGPA:
{student['pg_cgpa']}

CAREER

Career Interest:
{student['career_interest']}

Target Career Goal:
{student['target_career_goal']}

BRANCH SKILLS

{student['branch_skills']}

Overall Domain Skill Score:
{student['domain_skills']}/10

PLACEMENT PROFILE

CGPA:
{student['ug_cgpa']}

Backlogs:
{student['backlogs']}

Projects:
{student['projects']}

Internships:
{student['internships']}

Certifications:
{student['certifications']}

Communication:
{student['communication_skills']}/10

Technical/Coding Aptitude:
{student['coding_skills']}/10

Aptitude:
{student['aptitude_score']}/100

MODEL RESULT

Prediction:
{result['prediction']}

Placement Probability:
{result['probability'] * 100:.2f}%

IMPORTANT RULES

- Actual academic branch is the primary factor.
- Career interest and target career goal strongly influence advice.
- Consider UG and PG together.
- Do not automatically recommend CS/software skills to non-CS students.
- Give field-specific technical topics.
- Give field-specific tools.
- Give field-specific certifications.
- Give practical career paths.
- Never guarantee placement.
- Do not discuss internal model encoding.
- Do not mention resume.
- Do not ask for resume.
- Project ideas must match the student's actual field.
- Give EXACTLY 3 project ideas.
- Do not automatically give software projects.

OUTPUT EXACTLY THESE SECTIONS:

## Overall Profile Assessment

## Career Direction

## Recommended Career Paths

## Top Strengths

## Skill Gap Analysis

## Areas to Improve

## 30-Day Improvement Plan

### Week 1
### Week 2
### Week 3
### Week 4

## Technical Topics to Study

## Industry Tools and Professional Skills

## Project Ideas

Project 1:
Project 2:
Project 3:

Exactly 3 projects.

## Interview Preparation

Include:
- Core technical preparation
- HR preparation
- Communication preparation
- Project explanation
- Internship explanation
- Aptitude preparation where relevant

Keep everything personalized to the student's actual field.
"""

    try:

        client = genai.Client(
            api_key=api_key.strip()
        )

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            store=False
        )

        text = extract_gemini_text(
            interaction
        )

        if text:
            return text, None

        return None, "Gemini returned an empty response."

    except Exception as e:

        return None, str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "student" not in st.session_state:
    st.session_state.student = None

if "gemini_advice" not in st.session_state:
    st.session_state.gemini_advice = None


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 AI Student Placement Predictor"
)

st.write(
    """
Predict placement readiness using Machine Learning,
SHAP explainability and personalized Gemini career guidance.
"""
)

st.info(
    "The prediction is a model estimate and is not a guarantee of placement."
)


# ============================================================
# BASIC INFORMATION
# ============================================================

st.header(
    "📋 Student Profile"
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

with col2:

    age = st.number_input(
        "Age",
        16,
        60,
        22
    )

with col3:

    ug_degree = st.selectbox(
        "UG Degree",
        [
            "BTech",
            "BE",
            "BSc",
            "BCA",
            "BCom",
            "BA",
            "BBA",
            "Other"
        ]
    )


# ============================================================
# BRANCH
# ============================================================

ug_branch = st.selectbox(
    "UG Branch / Specialization",
    BRANCH_OPTIONS
)


ug_cgpa = st.number_input(
    "UG CGPA",
    0.0,
    10.0,
    7.0,
    step=0.1
)


# ============================================================
# PG
# ============================================================

st.subheader(
    "🎓 Postgraduate Education"
)

has_pg = st.checkbox(
    "I have a Postgraduate qualification"
)

pg_degree = ""

pg_branch = ""

pg_cgpa = 0.0

if has_pg:

    c1, c2, c3 = st.columns(3)

    with c1:

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

    with c2:

        pg_branch = st.text_input(
            "PG Branch / Specialization"
        )

    with c3:

        pg_cgpa = st.number_input(
            "PG CGPA",
            0.0,
            10.0,
            7.0,
            step=0.1
        )


# ============================================================
# CAREER
# ============================================================

st.subheader(
    "🧭 Career Direction"
)

career_interest = st.selectbox(
    "Career Interest",
    get_career_interests(
        ug_branch
    )
)

target_career_goal = st.text_input(
    "Target Career Goal",
    placeholder="Example: Clinical Nutritionist"
)


# ============================================================
# BRANCH SKILLS
# ============================================================

st.subheader(
    "🧠 Branch-Specific Skills"
)

skills = get_branch_skills(
    ug_branch
)

branch_skills = {}

skill_cols = st.columns(
    len(skills)
)

for i, skill in enumerate(
    skills
):

    with skill_cols[i]:

        branch_skills[skill] = st.slider(
            skill,
            1,
            10,
            5,
            key=f"skill_{i}"
        )


domain_skills = round(
    np.mean(
        list(
            branch_skills.values()
        )
    ),
    1
)

st.metric(
    "Overall Domain Skill Score",
    f"{domain_skills}/10"
)


# ============================================================
# PLACEMENT INPUTS
# ============================================================

st.subheader(
    "📊 Placement Profile"
)

c1, c2, c3 = st.columns(3)

with c1:

    backlogs = st.number_input(
        "Backlogs",
        0,
        30,
        0
    )

    internships = st.number_input(
        "Internships",
        0,
        20,
        1
    )

with c2:

    projects = st.number_input(
        "Projects",
        0,
        30,
        2
    )

    certifications = st.number_input(
        "Certifications",
        0,
        30,
        2
    )

with c3:

    communication_skills = st.slider(
        "Communication Skills",
        1,
        10,
        5
    )

    coding_skills = st.slider(
        "Technical / Coding Aptitude",
        1,
        10,
        5
    )


aptitude_score = st.slider(
    "Aptitude Score",
    0,
    100,
    60
)


# ============================================================
# PREDICT
# ============================================================

if st.button(
    "🔮 Predict Placement",
    type="primary",
    use_container_width=True
):

    student = {

        "gender": gender,
        "age": age,
        "degree": ug_degree,
        "branch": ug_branch,
        "cgpa": ug_cgpa,
        "backlogs": backlogs,
        "internships": internships,
        "certifications": certifications,
        "coding_skills": coding_skills,
        "communication_skills": communication_skills,
        "aptitude_score": aptitude_score,
        "projects": projects,

        "ug_degree": ug_degree,
        "ug_branch": ug_branch,
        "ug_cgpa": ug_cgpa,

        "has_pg": has_pg,
        "pg_degree": pg_degree,
        "pg_branch": pg_branch,
        "pg_cgpa": pg_cgpa,

        "career_interest": career_interest,
        "target_career_goal": target_career_goal,

        "branch_skills": branch_skills,
        "domain_skills": domain_skills
    }

    try:

        processed = prepare_data(
            student
        )

        prediction = model.predict(
            processed
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                processed
            )[0]

            if len(probabilities) == 2:

                probability = float(
                    probabilities[1]
                )

            else:

                probability = float(
                    np.max(probabilities)
                )

        else:

            probability = float(
                prediction
            )

        reliability_distance = (
            abs(
                probability - 0.5
            ) * 2
        )

        if reliability_distance >= 0.70:

            reliability = "High"

        elif reliability_distance >= 0.40:

            reliability = "Moderate"

        else:

            reliability = "Low"

        st.session_state.result = {
            "prediction": int(prediction),
            "probability": probability,
            "reliability": reliability
        }

        st.session_state.student = student

        st.session_state.gemini_advice = None

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# RESULT
# ============================================================

if st.session_state.result:

    result = st.session_state.result

    student = st.session_state.student

    st.divider()

    st.header(
        "🎯 Placement Prediction"
    )

    prediction_text = (
        "LIKELY PLACED"
        if result["prediction"] == 1
        else
        "NOT LIKELY PLACED"
    )

    probability = (
        result["probability"] * 100
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Prediction",
            prediction_text
        )

    with c2:

        st.metric(
            "Placement Probability",
            f"{probability:.2f}%"
        )

    with c3:

        st.metric(
            "Reliability Indicator",
            result["reliability"]
        )

    st.caption(
        "Reliability is a heuristic indicator, not a calibrated confidence measure."
    )


    # ========================================================
    # PROFILE
    # ========================================================

    st.subheader(
        "🎓 Academic Profile"
    )

    st.write(
        f"**UG:** {student['ug_degree']} — "
        f"{student['ug_branch']} — "
        f"CGPA {student['ug_cgpa']}"
    )

    if student["has_pg"]:

        st.write(
            f"**PG:** {student['pg_degree']} — "
            f"{student['pg_branch']} — "
            f"CGPA {student['pg_cgpa']}"
        )


    st.write(
        f"**Career Interest:** "
        f"{student['career_interest']}"
    )

    st.write(
        f"**Target Career Goal:** "
        f"{
            student['target_career_goal']
            if student['target_career_goal']
            else 'Not specified'
        }"
    )


    # ========================================================
    # SKILLS
    # ========================================================

    st.subheader(
        "🧠 Your Branch Skills"
    )

    skill_df = pd.DataFrame(
        [
            {
                "Skill": k,
                "Score": v
            }

            for k, v
            in student[
                "branch_skills"
            ].items()
        ]
    )

    st.dataframe(
        skill_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SKILL GAPS
    # ========================================================

    st.subheader(
        "📉 Skill Gap Analysis"
    )

    critical = [
        (k, v)
        for k, v
        in student[
            "branch_skills"
        ].items()
        if v <= 3
    ]

    development = [
        (k, v)
        for k, v
        in student[
            "branch_skills"
        ].items()
        if 4 <= v <= 6
    ]

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            "### 🔴 Critical Gaps"
        )

        if critical:

            for skill, score in critical:

                st.error(
                    f"{skill}: {score}/10"
                )

        else:

            st.success(
                "No critical gaps."
            )

    with c2:

        st.markdown(
            "### 🟡 Development Needed"
        )

        if development:

            for skill, score in development:

                st.warning(
                    f"{skill}: {score}/10"
                )

        else:

            st.success(
                "No major development gaps."
            )


    # ========================================================
    # GEMINI
    # ========================================================

    st.divider()

    st.header(
        "🤖 Gemini Personalized Career Guidance"
    )

    st.write(
        """
        Gemini uses your academic branch, UG/PG education,
        career interest, target goal, domain skills and
        placement profile to generate personalized guidance.
        """
    )

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    if st.button(
        "🤖 Generate Gemini Career Advice",
        use_container_width=True
    ):

        if not api_key.strip():

            st.warning(
                "Please enter your Gemini API key."
            )

        else:

            with st.spinner(
                "Gemini is analyzing your profile..."
            ):

                advice, error = (
                    generate_gemini_advice(
                        api_key,
                        student,
                        result
                    )
                )

            if advice:

                st.session_state.gemini_advice = advice

            else:

                st.error(
                    "Gemini request failed."
                )

                st.code(
                    error
                    if error
                    else
                    "Unknown error"
                )


    if st.session_state.gemini_advice:

        st.markdown(
            st.session_state.gemini_advice
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Student Placement Predictor | Machine Learning | "
    "SHAP | Branch-Specific Skills | Gemini Career Guidance"
)
