import os
import json
import urllib.request
import urllib.error

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
# CONSTANTS
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULES_FILE = "recommendation_rules.pkl"


# ============================================================
# BASIC STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    .prediction-box {
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    .small-note {
        color: #777;
        font-size: 0.9rem;
    }

    .career-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Student Placement Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predict placement probability and receive AI-powered career,
    skill, project, and placement guidance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)

    feature_names = None
    metadata = {}
    recommendation_rules = {}

    if os.path.exists(FEATURE_FILE):
        feature_names = joblib.load(FEATURE_FILE)

    if os.path.exists(METADATA_FILE):
        metadata = joblib.load(METADATA_FILE)

    if os.path.exists(RULES_FILE):
        recommendation_rules = joblib.load(RULES_FILE)

    return model, feature_names, metadata, recommendation_rules


try:
    model, feature_names, metadata, recommendation_rules = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model = None
    feature_names = None
    metadata = {}
    recommendation_rules = {}

    st.error(f"Could not load the placement model: {e}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System Status")

    if model_loaded:
        st.success("Placement model loaded")
    else:
        st.error("Placement model unavailable")

    gemini_key = st.secrets.get("GEMINI_API_KEY", None)

    if not gemini_key:
        gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key:
        st.success("Gemini AI configured")
    else:
        st.warning("Gemini AI key not configured")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        """
        This application uses a trained machine-learning model to
        estimate placement probability from student academic,
        technical, internship, project and skill information.

        Gemini AI is used separately to provide personalized career
        and placement guidance.
        """
    )

    st.markdown("---")

    st.caption(
        "⚠️ Placement probability is a model estimate and should not "
        "be interpreted as a guaranteed employment probability."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def normalize_degree(degree):
    mapping = {
        "BE": "BE",
        "BTech": "BTech",
        "BSc": "BSc",
        "BCA": "BCA",
        "BBA": "BBA",
        "BCom": "BCom",
        "BA": "BA",
        "Other": "Other",
    }

    return mapping.get(degree, degree)


def get_model_expected_features():
    """
    Find the features expected by the trained model.
    """

    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    if feature_names is not None:
        try:
            if isinstance(feature_names, dict):
                if "feature_names" in feature_names:
                    return list(feature_names["feature_names"])

            if isinstance(feature_names, (list, tuple, np.ndarray)):
                return list(feature_names)
        except Exception:
            pass

    return [
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
        "gender",
        "ug_degree",
        "ug_branch",
    ]


def build_model_input(student):
    """
    Creates a dataframe matching the trained model's expected
    feature names.
    """

    expected = get_model_expected_features()

    row = {}

    for feature in expected:

        if feature in student:
            row[feature] = student[feature]

        else:
            # Safe defaults for unexpected/missing features.
            if feature in [
                "gender",
                "ug_degree",
                "ug_branch",
            ]:
                row[feature] = "Other"
            else:
                row[feature] = 0

    return pd.DataFrame([row], columns=expected)


def get_positive_probability(prediction_model, input_df):
    """
    Extract probability for the positive/placed class.
    """

    probabilities = prediction_model.predict_proba(input_df)[0]

    classes = getattr(prediction_model, "classes_", None)

    if classes is None:
        return float(probabilities[-1])

    positive_indexes = []

    for i, cls in enumerate(classes):

        if cls in [1, True, "1", "Placed", "placed", "Yes", "yes"]:
            positive_indexes.append(i)

    if positive_indexes:
        return float(probabilities[positive_indexes[-1]])

    return float(probabilities[-1])


def predict_student(input_df):
    """
    Generate placement prediction.
    """

    probability = get_positive_probability(model, input_df)

    prediction = model.predict(input_df)[0]

    if isinstance(prediction, str):
        placed = prediction.lower() in [
            "1",
            "true",
            "yes",
            "placed",
            "selected",
        ]
    else:
        placed = bool(prediction)

    return placed, probability


def calculate_profile_summary(student):
    strengths = []
    improvements = []

    if student["ug_cgpa"] >= 8:
        strengths.append("Strong academic performance")
    elif student["ug_cgpa"] < 7:
        improvements.append("Improve academic performance / CGPA")

    if student["coding_skills"] >= 8:
        strengths.append("Strong coding skills")
    elif student["coding_skills"] < 6:
        improvements.append("Strengthen coding fundamentals")

    if student["aptitude_score"] >= 75:
        strengths.append("Good aptitude performance")
    elif student["aptitude_score"] < 60:
        improvements.append("Practice quantitative and logical aptitude")

    if student["communication_skills"] >= 8:
        strengths.append("Strong communication skills")
    elif student["communication_skills"] < 6:
        improvements.append("Improve communication and interview confidence")

    if student["internships"] >= 2:
        strengths.append("Good internship exposure")
    elif student["internships"] == 0:
        improvements.append("Gain practical internship experience")

    if student["projects"] >= 3:
        strengths.append("Good project experience")
    elif student["projects"] < 2:
        improvements.append("Build more practical projects")

    if student["certifications"] >= 2:
        strengths.append("Good certification profile")

    if student["backlogs"] == 0:
        strengths.append("No academic backlogs")
    else:
        improvements.append("Clear and manage academic backlogs")

    return strengths, improvements


# ============================================================
# GEMINI AI
# ============================================================

def call_gemini(prompt, api_key):
    """
    Calls Gemini directly through the REST API.

    Newer Gemini models are attempted first.
    """

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
    ]

    last_error = None

    for model_name in models_to_try:

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{model_name}:generateContent"
            f"?key={api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.55,
                "maxOutputTokens": 7000,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AI-Student-Placement-Predictor/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:

                response_data = json.loads(
                    response.read().decode("utf-8")
                )

                candidates = response_data.get("candidates", [])

                if not candidates:
                    last_error = "Gemini returned no candidates."
                    continue

                content = candidates[0].get("content", {})

                parts = content.get("parts", [])

                text_parts = []

                for part in parts:
                    if "text" in part:
                        text_parts.append(part["text"])

                result = "\n".join(text_parts).strip()

                if result:
                    return result

                last_error = "Gemini returned an empty response."

        except urllib.error.HTTPError as e:

            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = str(e)

            last_error = (
                f"Gemini HTTP {e.code}: {error_body}"
            )

            continue

        except Exception as e:
            last_error = str(e)
            continue

    raise RuntimeError(
        last_error
        or "Gemini could not generate guidance."
    )


def generate_ai_guidance(
    student,
    probability,
    prediction_label,
    career_interest,
    target_goal,
    pg_degree,
):
    """
    Creates detailed personalized placement and career guidance.
    """

    if not gemini_key:
        return None

    domain_skills = {
        "DSA / Problem Solving": student["domain_skill_1"],
        "Machine Learning": student["domain_skill_2"],
        "System Design": student["domain_skill_3"],
        "Hackathons": student["domain_skill_4"],
        "Open Source": student["domain_skill_5"],
    }

    prompt = f"""
You are an expert student career advisor, placement mentor,
technical recruiter and career-roadmap planner.

You are advising a college student based on the following profile.

STUDENT PROFILE
---------------
Age: {student["age"]}
Gender: {student["gender"]}

UG Degree: {student["ug_degree"]}
UG Branch: {student["ug_branch"]}
UG CGPA: {student["ug_cgpa"]}

Postgraduate Degree:
{pg_degree if pg_degree else "No postgraduate degree selected"}

Backlogs: {student["backlogs"]}
Internships: {student["internships"]}
Projects: {student["projects"]}
Certifications: {student["certifications"]}

Coding Skills: {student["coding_skills"]}/10
Communication Skills: {student["communication_skills"]}/10
Aptitude Score: {student["aptitude_score"]}/100

Domain Skills:
DSA / Problem Solving: {domain_skills["DSA / Problem Solving"]}/10
Machine Learning: {domain_skills["Machine Learning"]}/10
System Design: {domain_skills["System Design"]}/10
Hackathons: {domain_skills["Hackathons"]}/10
Open Source: {domain_skills["Open Source"]}/10

PLACEMENT MODEL
---------------
Placement prediction: {prediction_label}
Placement probability: {probability:.1f}%

CAREER PREFERENCES
------------------
Career Interest: {career_interest}
Target Career Goal: {target_goal}

IMPORTANT INSTRUCTIONS
---------------------
Give practical and personalized advice.

Do NOT give generic motivational paragraphs.

The response must be highly structured and should connect the
student's career interest and target goal to specific job roles,
skills, employers, projects and an actionable 30-day plan.

Use the following exact sections.

1. PREDICTION INTERPRETATION

Explain what the placement prediction means.

Mention that the probability is a machine-learning estimate,
not a guaranteed employment probability.

Explain the important profile factors that may have influenced
the prediction.

2. CAREER PATH

Create a career-path table with these columns:

Path | Typical Role | Core Responsibilities | Typical Employers

Include career paths that are relevant to the student's selected
Career Interest and Target Career Goal.

For example, if the student is interested in Data Analytics,
consider paths such as:

Data Analyst (Entry-Level)
Business Intelligence (BI) Analyst
Analytics Engineer
Product Analyst
Junior Data Scientist
Data Science Intern

Do NOT blindly include irrelevant roles.

Explain which 2-4 paths are most aligned with the student's
current profile, without ranking them as universally better.

3. TARGET ROLE EXPLANATION

For the selected target career goal explain:

- What the role does
- Typical daily work
- Important technical skills
- Important soft skills
- Common tools
- Interview topics
- Entry-level expectations

4. CURRENT STRENGTHS

List the student's strongest areas using their actual scores.

Explain how each strength can help during placements.

5. SKILL GAPS

Identify the most important skills that need improvement.

Use the student's actual scores.

For each skill gap explain:

Current level
Target level
Why it matters
How to improve it

6. TOOLS AND TECHNOLOGIES

Create a practical list of tools and technologies the student
should learn for their selected target career.

Group them under:

Programming
Databases / SQL
Data / Analytics
Cloud
Visualization
Development
Version Control
Interview Preparation

Only include technologies relevant to the chosen career.

7. 30-DAY IMPROVEMENT PLAN

Create a detailed table with:

Day | Goal | Activity | Expected Outcome

Use multiple phases such as:

Day 1-3
Day 4-7
Day 8-12
Day 13-17
Day 18-22
Day 23-26
Day 27-30

The plan must be realistic for a college student.

For example, for a Data Analyst path:

Day 1-3:
Choose primary visualization tool.
Audit tools.
Install Tableau Public / Power BI.
Complete beginner tutorials.

Day 4-7:
SQL deep dive.
Practice CTEs, joins, aggregations and window functions.
Build a small SQLite/PostgreSQL database.

Day 8-12:
Statistics refresher.
Practice descriptive statistics, probability,
hypothesis testing and A/B testing.

Day 13-17:
Business/domain immersion.
Choose an industry and identify KPIs.

Day 18-22:
Start Project #1.
Commit code to GitHub.
Create README.

Day 23-26:
Improve project.
Create dashboard.
Document findings.

Day 27-30:
Resume + GitHub + interview preparation.

Adapt the plan to the student's actual target career.

8. TECHNICAL STUDY PLAN

Give specific topics to study.

For example:

SQL:
- SELECT
- JOIN
- GROUP BY
- CTEs
- Window Functions
- Subqueries
- Query optimization

But customize the list to the target career.

9. INTERVIEW PREPARATION

Give:

Technical questions
Coding topics
Domain questions
HR questions
Resume questions
Project questions

Also provide 5 sample interview questions with short guidance
on how the student should answer.

10. PROJECT IDEAS

Give exactly 3 practical projects.

Create a table:

# | Title | What To Build / Do | Skills Demonstrated | Why It's Relevant

Projects must match the selected career.

For Data Analytics, for example:

Project 1:
E-Commerce Sales Dashboard

What to build:
Use a public e-commerce dataset such as Online Retail,
clean it, create a relational/star-schema model,
use PostgreSQL/SQLite and create a Tableau/Power BI dashboard
showing revenue trends, top products, customer segmentation
and cohort analysis.

Skills:
SQL, CTEs, window functions, data modeling, ETL,
Tableau/Power BI, storytelling.

Why relevant:
Shows an end-to-end analytics workflow.

Project 2:
A/B Testing Analysis

Project 3:
Customer Churn / Business KPI Analytics

For other careers, create equally relevant projects.

11. GITHUB PORTFOLIO PLAN

Tell the student what their GitHub should contain.

Include:

- README
- project screenshots
- clean source code
- requirements.txt
- dataset explanation
- results
- architecture / workflow
- demo link where applicable

12. RESUME PLAN

Give specific resume improvements for the selected target role.

Show examples of strong bullet points using action + technology
+ measurable result where possible.

13. WEEKLY CAREER ROADMAP

Give a concise 3-month roadmap:

Month 1
Month 2
Month 3

14. FINAL ACTION CHECKLIST

Give 10 concrete actions the student should take next.

Keep the advice practical, specific and personalized.

Do not claim that completing these actions guarantees placement.

Formatting:
Use Markdown.
Use tables where requested.
Use bullet points.
Keep the response detailed enough to be genuinely useful.
"""


    return call_gemini(prompt, gemini_key)


# ============================================================
# INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-title">👨‍🎓 Student Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=21,
        step=1,
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other",
            "Prefer not to say",
        ],
    )

with col2:
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
            "Other",
        ],
    )

    ug_branch = st.selectbox(
        "UG Branch",
        [
            "Computer Science",
            "Information Technology",
            "Artificial Intelligence",
            "Artificial Intelligence and Machine Learning",
            "Data Science",
            "Computer Science and Engineering",
            "Electronics and Communication",
            "Electrical Engineering",
            "Mechanical Engineering",
            "Civil Engineering",
            "Information Science",
            "Cyber Security",
            "Other",
        ],
    )


# ============================================================
# PG FLOW
# ============================================================

st.markdown(
    '<div class="section-title">🎓 Postgraduate Education</div>',
    unsafe_allow_html=True,
)

st.info(
    "First select your UG details above. If you have completed or "
    "are pursuing a postgraduate degree, choose Yes below. "
    "The PG degree option will appear only after that."
)

has_pg = st.radio(
    "Have you completed or are you currently pursuing a PG degree?",
    [
        "No",
        "Yes",
    ],
    horizontal=True,
)

pg_degree = None

if has_pg == "Yes":

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
            "MS",
            "Other",
        ],
    )

    st.success(
        f"PG selected: {pg_degree}"
    )

else:

    st.caption(
        "No PG degree selected. You can continue with your UG profile."
    )


# ============================================================
# ACADEMIC / PLACEMENT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📚 Academic & Placement Profile</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:

    ug_cgpa = st.number_input(
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

with col2:

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )

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
# SKILLS
# ============================================================

st.markdown(
    '<div class="section-title">💻 Technical & Professional Skills</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:

    coding_skills = st.slider(
        "Coding Skills",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
    )

    communication_skills = st.slider(
        "Communication Skills",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
    )

with col2:

    aptitude_score = st.slider(
        "Aptitude Score",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
    )


# ============================================================
# DOMAIN SKILLS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Domain Skills</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    dsa_score = st.slider(
        "DSA / Problem Solving",
        0.0,
        10.0,
        7.0,
        0.5,
    )

with col2:
    ml_knowledge = st.slider(
        "Machine Learning",
        0.0,
        10.0,
        6.5,
        0.5,
    )

with col3:
    system_design = st.slider(
        "System Design",
        0.0,
        10.0,
        6.0,
        0.5,
    )

with col4:
    hackathons = st.slider(
        "Hackathons",
        0.0,
        10.0,
        5.0,
        0.5,
    )

with col5:
    open_source = st.slider(
        "Open Source",
        0.0,
        10.0,
        4.0,
        0.5,
    )


# ============================================================
# CAREER PREFERENCES
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Career Preferences</div>',
    unsafe_allow_html=True,
)

career_interest = st.selectbox(
    "Career Interest",
    [
        "Software Development",
        "Data Analytics",
        "Data Science",
        "Artificial Intelligence / Machine Learning",
        "Business Intelligence",
        "Cloud / DevOps",
        "Cyber Security",
        "Web Development",
        "Mobile App Development",
        "Product Management",
        "Business Analysis",
        "Other",
    ],
)

career_goal_options = {
    "Software Development": [
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Java Developer",
        "Python Developer",
        "Software Development Intern",
    ],

    "Data Analytics": [
        "Data Analyst",
        "Business Analyst",
        "Business Intelligence Analyst",
        "Product Analyst",
        "Analytics Engineer",
    ],

    "Data Science": [
        "Junior Data Scientist",
        "Data Science Intern",
        "Machine Learning Analyst",
        "Applied Data Scientist",
    ],

    "Artificial Intelligence / Machine Learning": [
        "ML Engineer",
        "AI Engineer",
        "Machine Learning Intern",
        "Applied AI Engineer",
        "NLP Engineer",
    ],

    "Business Intelligence": [
        "BI Analyst",
        "BI Developer",
        "Reporting Analyst",
        "Business Intelligence Engineer",
    ],

    "Cloud / DevOps": [
        "Cloud Engineer",
        "DevOps Engineer",
        "Cloud Support Engineer",
        "Site Reliability Engineer",
    ],

    "Cyber Security": [
        "Security Analyst",
        "SOC Analyst",
        "Cyber Security Engineer",
        "Security Operations Intern",
    ],

    "Web Development": [
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Web Developer",
    ],

    "Mobile App Development": [
        "Android Developer",
        "iOS Developer",
        "Flutter Developer",
        "React Native Developer",
    ],

    "Product Management": [
        "Associate Product Manager",
        "Product Analyst",
        "Product Management Intern",
    ],

    "Business Analysis": [
        "Business Analyst",
        "Junior Business Analyst",
        "Business Operations Analyst",
        "Business Analysis Intern",
    ],

    "Other": [
        "Entry-Level Technology Role",
        "Graduate Trainee",
        "Internship",
        "Other",
    ],
}

target_goal = st.selectbox(
    "Target Career Goal",
    career_goal_options.get(
        career_interest,
        ["Entry-Level Technology Role"],
    ),
)


# ============================================================
# BUILD STUDENT PROFILE
# ============================================================

student = {
    "age": int(age),
    "gender": gender,
    "ug_degree": normalize_degree(ug_degree),
    "ug_branch": ug_branch,

    "ug_cgpa": float(ug_cgpa),
    "backlogs": int(backlogs),
    "internships": int(internships),
    "projects": int(projects),
    "certifications": int(certifications),

    "coding_skills": float(coding_skills),
    "communication_skills": float(communication_skills),
    "aptitude_score": float(aptitude_score),

    "domain_skill_1": float(dsa_score),
    "domain_skill_2": float(ml_knowledge),
    "domain_skill_3": float(system_design),
    "domain_skill_4": float(hackathons),
    "domain_skill_5": float(open_source),
}


# ============================================================
# PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Placement Prediction</div>',
    unsafe_allow_html=True,
)

predict_button = st.button(
    "🔮 Predict Placement",
    type="primary",
    use_container_width=True,
)


if predict_button:

    if not model_loaded:
        st.error(
            "The placement model could not be loaded. "
            "Please check the model files."
        )
        st.stop()

    try:

        model_input = build_model_input(student)

        placed, probability = predict_student(model_input)

        prediction_label = (
            "Placement Likely"
            if placed
            else "Needs Further Preparation"
        )

        st.session_state["prediction_probability"] = probability
        st.session_state["prediction_label"] = prediction_label
        st.session_state["student"] = student
        st.session_state["career_interest"] = career_interest
        st.session_state["target_goal"] = target_goal
        st.session_state["pg_degree"] = pg_degree

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if "prediction_probability" in st.session_state:

    probability = st.session_state["prediction_probability"]

    prediction_label = st.session_state["prediction_label"]

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        if probability >= 0.5:
            st.success(
                f"### ✅ {prediction_label}"
            )
        else:
            st.warning(
                f"### 📚 {prediction_label}"
            )

    with col2:

        st.metric(
            "Placement Probability",
            f"{probability * 100:.1f}%",
        )

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        "This probability is a machine-learning estimate based on "
        "the trained model and the information entered above."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# PROFILE EXPLANATION
# ============================================================

if "prediction_probability" in st.session_state:

    strengths, improvements = calculate_profile_summary(
        st.session_state["student"]
    )

    st.markdown(
        '<div class="section-title">📊 Prediction Explanation</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("💪 Profile Strengths")

        if strengths:

            for item in strengths:
                st.success(item)

        else:
            st.info(
                "No major strengths were automatically detected. "
                "The AI guidance below will provide a detailed analysis."
            )

    with col2:

        st.subheader("🎯 Areas To Improve")

        if improvements:

            for item in improvements:
                st.warning(item)

        else:
            st.success(
                "No major improvement area was automatically detected."
            )


# ============================================================
# AI GUIDANCE
# ============================================================

if "prediction_probability" in st.session_state:

    st.markdown(
        '<div class="section-title">🤖 AI Career & Placement Guidance</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"""
        Based on your selected career interest **{st.session_state["career_interest"]}**
        and target role **{st.session_state["target_goal"]}**, Gemini can generate
        a personalized career roadmap, role guidance, skill-gap analysis,
        30-day plan and project recommendations.
        """
    )

    if not gemini_key:

        st.warning(
            """
            Gemini AI is not configured.

            Add your Gemini API key in Streamlit Cloud:

            `GEMINI_API_KEY="your_key_here"`
            """
        )

    else:

        generate_button = st.button(
            "🤖 Generate Personalized AI Guidance",
            type="secondary",
            use_container_width=True,
        )

        if generate_button:

            with st.spinner(
                "🤖 Gemini is analyzing your profile and creating your career roadmap..."
            ):

                try:

                    ai_response = generate_ai_guidance(
                        student=st.session_state["student"],
                        probability=st.session_state[
                            "prediction_probability"
                        ],
                        prediction_label=st.session_state[
                            "prediction_label"
                        ],
                        career_interest=st.session_state[
                            "career_interest"
                        ],
                        target_goal=st.session_state[
                            "target_goal"
                        ],
                        pg_degree=st.session_state[
                            "pg_degree"
                        ],
                    )

                    st.session_state["ai_response"] = ai_response

                except Exception as e:

                    st.error(
                        "AI guidance could not be generated."
                    )

                    st.code(
                        str(e),
                        language="text",
                    )


# ============================================================
# DISPLAY AI RESPONSE
# ============================================================

if "ai_response" in st.session_state:

    st.markdown(
        '<div class="section-title">🧭 Personalized Career Roadmap</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        st.session_state["ai_response"]
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("ℹ️ Model Information"):

    if metadata:

        if isinstance(metadata, dict):

            for key, value in metadata.items():

                if isinstance(value, (dict, list)):
                    st.write(f"**{key}:**")
                    st.json(value)

                else:
                    st.write(
                        f"**{key}:** {value}"
                    )

    else:

        st.info(
            "Model metadata is not available."
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    """
    ⚠️ Disclaimer: This application is an educational/project
    demonstration. The placement prediction is generated by a
    machine-learning model and is not a guarantee of employment.
    AI career guidance is advisory and should be combined with
    your own research, college placement guidance and professional
    advice.
    """
)
