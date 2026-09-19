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
    initial_sidebar_state="collapsed",
)


# ============================================================
# FILES
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULES_FILE = "recommendation_rules.pkl"


# ============================================================
# CSS - DASHBOARD STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }

    .dashboard-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    .section-header {
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 1.7rem;
        margin-bottom: 1rem;
    }

    .section-description {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-top: -0.5rem;
        margin-bottom: 1.2rem;
    }

    .result-card {
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .result-number {
        font-size: 2.4rem;
        font-weight: 800;
    }

    .ai-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-top: 1.2rem;
    }

    .small-note {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        padding: 0.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-title">🎓 AI Student Placement Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-subtitle">
    AI-powered placement prediction, career direction, skill analysis,
    project recommendations and personalized placement guidance.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_FILE)

    feature_names = None
    metadata = {}
    rules = {}

    if os.path.exists(FEATURE_FILE):
        feature_names = joblib.load(FEATURE_FILE)

    if os.path.exists(METADATA_FILE):
        metadata = joblib.load(METADATA_FILE)

    if os.path.exists(RULES_FILE):
        rules = joblib.load(RULES_FILE)

    return model, feature_names, metadata, rules


try:

    model, feature_names, metadata, recommendation_rules = (
        load_artifacts()
    )

    model_loaded = True

except Exception as e:

    model = None
    feature_names = None
    metadata = {}
    recommendation_rules = {}
    model_loaded = False

    st.error(f"Model loading error: {e}")


# ============================================================
# GEMINI KEY
# ============================================================

try:
    gemini_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    gemini_key = None

if not gemini_key:
    gemini_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# COMPLETE BRANCH LIST
# ============================================================

UG_BRANCHES = [

    # Computer / IT
    "Computer Science and Engineering",
    "Computer Science",
    "Information Technology",
    "Information Science and Engineering",
    "Computer Engineering",
    "Computer Applications",
    "Artificial Intelligence",
    "Artificial Intelligence and Machine Learning",
    "Data Science",
    "Computer Science and Data Science",
    "Computer Science and Business Systems",
    "Cyber Security",
    "Information Security",
    "Cloud Computing",
    "Internet of Things",
    "Computer Science and IoT",
    "Software Engineering",

    # Electronics
    "Electronics and Communication Engineering",
    "Electronics Engineering",
    "Electronics and Telecommunication Engineering",
    "Electrical and Electronics Engineering",
    "Electronics and Instrumentation Engineering",
    "Instrumentation and Control Engineering",
    "Electronics and Computer Engineering",
    "VLSI Design",

    # Electrical
    "Electrical Engineering",
    "Electrical and Electronics Engineering",

    # Mechanical
    "Mechanical Engineering",
    "Automobile Engineering",
    "Mechatronics Engineering",
    "Robotics and Automation Engineering",
    "Manufacturing Engineering",
    "Production Engineering",
    "Industrial Engineering",

    # Civil
    "Civil Engineering",
    "Environmental Engineering",
    "Construction Engineering",
    "Structural Engineering",
    "Transportation Engineering",

    # Chemical / Materials
    "Chemical Engineering",
    "Petrochemical Engineering",
    "Petroleum Engineering",
    "Polymer Engineering",
    "Metallurgical Engineering",
    "Materials Engineering",

    # Biotechnology / Biomedical
    "Biotechnology",
    "Bioinformatics",
    "Biomedical Engineering",
    "Genetic Engineering",
    "Food Technology",

    # Aerospace
    "Aerospace Engineering",
    "Aeronautical Engineering",

    # Agriculture
    "Agricultural Engineering",
    "Agriculture",
    "Food and Agricultural Technology",

    # Mining / Earth
    "Mining Engineering",
    "Geological Engineering",
    "Geoinformatics",

    # Textile
    "Textile Engineering",
    "Textile Technology",

    # Architecture
    "Architecture",
    "Planning",

    # Science
    "Physics",
    "Chemistry",
    "Mathematics",
    "Statistics",
    "Biology",
    "Computer Science - BSc",

    # Commerce / Management
    "Commerce",
    "Business Administration",
    "Business Management",
    "Economics",

    # Other
    "BCA",
    "BBA",
    "BCom",
    "BA",
    "Other",
]


PG_DEGREES = [
    "MTech",
    "ME",
    "MSc",
    "MCA",
    "MBA",
    "MCom",
    "MA",
    "MS",
    "MPhil",
    "Other",
]


PG_SPECIALIZATIONS = [

    # Computing
    "Computer Science",
    "Artificial Intelligence",
    "Machine Learning",
    "Data Science",
    "Data Analytics",
    "Cyber Security",
    "Cloud Computing",
    "Software Engineering",
    "Information Technology",
    "Information Systems",
    "Computer Networks",
    "Database Systems",
    "Internet of Things",
    "Robotics",
    "Blockchain",

    # Electronics
    "VLSI",
    "Embedded Systems",
    "Communication Systems",
    "Signal Processing",
    "Electronics",

    # Management
    "Finance",
    "Marketing",
    "Human Resources",
    "Business Analytics",
    "Operations",
    "International Business",
    "Supply Chain Management",
    "Product Management",
    "Business Management",

    # Science
    "Mathematics",
    "Statistics",
    "Physics",
    "Chemistry",
    "Biotechnology",
    "Bioinformatics",

    # Engineering
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "Electronics Engineering",
    "Chemical Engineering",
    "Industrial Engineering",
    "Other",
]


# ============================================================
# CAREER OPTIONS
# ============================================================

CAREER_GOALS = {

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
        "Applied Data Scientist",
        "Machine Learning Analyst",
        "Data Scientist",
    ],

    "Artificial Intelligence / Machine Learning": [
        "AI Engineer",
        "Machine Learning Engineer",
        "ML Engineer",
        "Applied AI Engineer",
        "NLP Engineer",
        "Computer Vision Engineer",
    ],

    "Business Intelligence": [
        "BI Analyst",
        "BI Developer",
        "Business Intelligence Engineer",
        "Reporting Analyst",
        "Data Visualization Analyst",
    ],

    "Cloud / DevOps": [
        "Cloud Engineer",
        "DevOps Engineer",
        "Cloud Support Engineer",
        "Site Reliability Engineer",
        "Cloud Administrator",
    ],

    "Cyber Security": [
        "Security Analyst",
        "SOC Analyst",
        "Cyber Security Engineer",
        "Security Operations Analyst",
        "Information Security Analyst",
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
        "Mobile App Developer",
    ],

    "Product Management": [
        "Associate Product Manager",
        "Product Analyst",
        "Product Management Intern",
        "Junior Product Manager",
    ],

    "Business Analysis": [
        "Business Analyst",
        "Junior Business Analyst",
        "Business Operations Analyst",
        "Business Analysis Intern",
    ],

    "Networking": [
        "Network Engineer",
        "Network Administrator",
        "Network Support Engineer",
        "Cloud Network Engineer",
    ],

    "Other": [
        "Graduate Trainee",
        "Technology Associate",
        "Business Associate",
        "Entry-Level Professional",
    ],
}


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

BRANCH_SKILLS = {

    "Computer": [
        "Programming",
        "Data Structures & Algorithms",
        "Databases",
        "Software Development",
        "Problem Solving",
    ],

    "Data": [
        "Python",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Data Visualization",
    ],

    "Electronics": [
        "Digital Electronics",
        "Embedded Systems",
        "Circuit Design",
        "Communication Systems",
        "Microcontrollers",
    ],

    "Electrical": [
        "Electrical Machines",
        "Power Systems",
        "Circuit Analysis",
        "Control Systems",
        "Power Electronics",
    ],

    "Mechanical": [
        "CAD",
        "Manufacturing",
        "Thermodynamics",
        "Mechanical Design",
        "Materials",
    ],

    "Civil": [
        "Structural Analysis",
        "AutoCAD",
        "Construction",
        "Surveying",
        "Project Planning",
    ],

    "Chemical": [
        "Process Engineering",
        "Thermodynamics",
        "Chemical Processes",
        "Process Safety",
        "Materials",
    ],

    "Biotechnology": [
        "Biology",
        "Bioinformatics",
        "Laboratory Skills",
        "Data Analysis",
        "Research",
    ],

    "Management": [
        "Business Analysis",
        "Communication",
        "Finance",
        "Marketing",
        "Leadership",
    ],

    "Science": [
        "Mathematics",
        "Statistics",
        "Research",
        "Data Analysis",
        "Problem Solving",
    ],

    "Other": [
        "Technical Skills",
        "Problem Solving",
        "Communication",
        "Analytical Thinking",
        "Domain Knowledge",
    ],
}


def get_skill_labels(branch, specialization=""):

    text = (
        str(branch).lower()
        + " "
        + str(specialization).lower()
    )

    if any(
        word in text
        for word in [
            "data",
            "analytics",
            "statistics",
            "artificial intelligence",
            "machine learning",
            "computer science",
            "information technology",
            "information science",
            "cyber",
            "software",
            "computer",
            "cloud",
            "iot",
            "blockchain",
        ]
    ):
        return BRANCH_SKILLS["Data"]

    if any(
        word in text
        for word in [
            "electronics",
            "vlsi",
            "communication",
            "embedded",
            "instrumentation",
        ]
    ):
        return BRANCH_SKILLS["Electronics"]

    if any(
        word in text
        for word in [
            "electrical",
            "power",
            "control systems",
        ]
    ):
        return BRANCH_SKILLS["Electrical"]

    if any(
        word in text
        for word in [
            "mechanical",
            "automobile",
            "mechatronics",
            "robotics",
            "manufacturing",
            "production",
            "industrial",
        ]
    ):
        return BRANCH_SKILLS["Mechanical"]

    if any(
        word in text
        for word in [
            "civil",
            "structural",
            "construction",
            "environmental",
            "transportation",
        ]
    ):
        return BRANCH_SKILLS["Civil"]

    if any(
        word in text
        for word in [
            "chemical",
            "petroleum",
            "petrochemical",
            "polymer",
        ]
    ):
        return BRANCH_SKILLS["Chemical"]

    if any(
        word in text
        for word in [
            "biotechnology",
            "bioinformatics",
            "biomedical",
            "biology",
        ]
    ):
        return BRANCH_SKILLS["Biotechnology"]

    if any(
        word in text
        for word in [
            "business",
            "management",
            "commerce",
            "mba",
            "bba",
        ]
    ):
        return BRANCH_SKILLS["Management"]

    if any(
        word in text
        for word in [
            "physics",
            "chemistry",
            "mathematics",
            "statistics",
            "science",
        ]
    ):
        return BRANCH_SKILLS["Science"]

    return BRANCH_SKILLS["Other"]


# ============================================================
# MODEL HELPERS
# ============================================================

def get_model_features():

    if model is None:
        return []

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

            if isinstance(
                feature_names,
                (list, tuple, np.ndarray),
            ):
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

    expected = get_model_features()

    row = {}

    for feature in expected:

        if feature in student:
            row[feature] = student[feature]

        elif feature in [
            "gender",
            "ug_degree",
            "ug_branch",
        ]:
            row[feature] = "Other"

        else:
            row[feature] = 0

    return pd.DataFrame(
        [row],
        columns=expected,
    )


def get_probability(input_df):

    probabilities = model.predict_proba(input_df)[0]

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        return float(probabilities[-1])

    for i, cls in enumerate(classes):

        if cls in [
            1,
            True,
            "1",
            "Placed",
            "placed",
            "Yes",
            "yes",
        ]:
            return float(probabilities[i])

    return float(probabilities[-1])


# ============================================================
# GEMINI
# ============================================================

def call_gemini(prompt, api_key):

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
                "maxOutputTokens": 8000,
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AI-Student-Placement-Predictor/1.0",
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=70,
            ) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

                candidates = result.get(
                    "candidates",
                    [],
                )

                if not candidates:
                    last_error = (
                        "Gemini returned no candidates."
                    )
                    continue

                parts = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [])
                )

                text = "\n".join(
                    part.get("text", "")
                    for part in parts
                    if part.get("text")
                ).strip()

                if text:
                    return text

                last_error = (
                    "Gemini returned an empty response."
                )

        except urllib.error.HTTPError as e:

            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = str(e)

            last_error = (
                f"Gemini HTTP {e.code}: {body}"
            )

        except Exception as e:

            last_error = str(e)

    raise RuntimeError(
        last_error
        or "Gemini AI could not generate guidance."
    )


def create_ai_prompt(
    student,
    probability,
    career_interest,
    target_goal,
    pg_info,
    skill_labels,
    skill_values,
):

    skill_text = "\n".join(
        f"- {label}: {value}/10"
        for label, value in zip(
            skill_labels,
            skill_values,
        )
    )

    return f"""
You are an expert college placement mentor, career advisor,
technical recruiter and professional career-roadmap planner.

Create a highly personalized career report for this student.

==================================================
STUDENT PROFILE
==================================================

Age: {student["age"]}
Gender: {student["gender"]}

UG Degree: {student["ug_degree"]}
UG Branch: {student["ug_branch"]}
UG CGPA: {student["ug_cgpa"]}

Postgraduate:
{pg_info}

Backlogs: {student["backlogs"]}
Internships: {student["internships"]}
Projects: {student["projects"]}
Certifications: {student["certifications"]}

Coding Skills: {student["coding_skills"]}/10
Communication Skills: {student["communication_skills"]}/10
Aptitude Score: {student["aptitude_score"]}/100

Branch Skills:
{skill_text}

==================================================
PLACEMENT MODEL
==================================================

Estimated Placement Probability:
{probability * 100:.1f}%

==================================================
CAREER DIRECTION
==================================================

Career Interest:
{career_interest}

Target Career Goal:
{target_goal}

==================================================
IMPORTANT
==================================================

The placement probability is only a machine-learning estimate.
Do not describe it as a guarantee.

The student's career interest and target career goal are very
important.

Do not provide generic advice.

Everything should be connected to the selected career.

==================================================
1. PREDICTION INTERPRETATION
==================================================

Explain:

- What the placement prediction means
- Which profile factors may have influenced it
- Current strengths
- Areas that need improvement

Do not use a "readiness score".

==================================================
2. CAREER PATH
==================================================

Create a Markdown table:

| Path | Typical Role | Core Responsibilities | Typical Employers |
|---|---|---|---|

Give several realistic career paths connected to the student's
selected career interest.

For example, for Data Analytics, relevant paths may include:

Data Analyst (Entry-Level)
Business Intelligence (BI) Analyst
Analytics Engineer
Product Analyst
Junior Data Scientist

For another career, create appropriate career paths.

==================================================
3. TARGET ROLE
==================================================

Explain the selected target career goal.

Include:

- Typical role
- Daily work
- Core responsibilities
- Technical skills
- Soft skills
- Tools
- Entry-level expectations
- Interview topics
- Typical employers

==================================================
4. SKILL GAP ANALYSIS
==================================================

Create a table:

| Skill | Current Level | Target Level | Why It Matters | How To Improve |
|---|---|---|---|---|

Use the student's actual skill levels.

==================================================
5. TOOLS AND TECHNOLOGIES
==================================================

Give only relevant technologies.

Group them into:

Programming
Databases / SQL
Data / Analytics
Cloud
Visualization
Development
Version Control
Interview Preparation

==================================================
6. 30-DAY IMPROVEMENT PLAN
==================================================

Create this table:

| Day | Goal | Activity | Expected Outcome |
|---|---|---|---|

Use:

Day 1-3
Day 4-7
Day 8-12
Day 13-17
Day 18-22
Day 23-26
Day 27-30

Make it practical.

For example, a Data Analyst plan can contain:

Day 1-3:
Choose primary visualization tool.
Install Tableau Public / Power BI.
Review beginner tutorials.

Day 4-7:
SQL deep dive.
Practice joins, CTEs and window functions.
Create a small database.

Day 8-12:
Statistics.
Descriptive statistics.
Probability.
Hypothesis testing.
A/B testing.

Day 13-17:
Business context.
Choose an industry.
Research important KPIs.

Day 18-22:
Project kickoff.
Start Project #1.
Use GitHub.
Write README.

Day 23-26:
Complete dashboard/project.
Document findings.

Day 27-30:
Resume.
GitHub.
Mock interviews.
Applications.

Adapt everything to the student's selected career.

==================================================
7. TECHNICAL STUDY PLAN
==================================================

Give a detailed topic checklist.

Customize it to the target role.

==================================================
8. INTERVIEW PREPARATION
==================================================

Include:

Technical questions
Coding topics
Domain questions
HR questions
Resume questions
Project questions

Then give 5 sample interview questions and explain how the
student should answer them.

==================================================
9. PROJECT IDEAS
==================================================

Give exactly 3 projects.

Create:

| # | Title | What To Build / Do | Skills Demonstrated | Why It's Relevant |
|---|---|---|---|---|

Projects must be directly related to the target career.

For Data Analytics, for example:

1. E-Commerce Sales Dashboard

Use a public e-commerce dataset such as Online Retail.

Build:

- Data cleaning
- PostgreSQL/SQLite database
- Star schema
- SQL analysis
- CTEs
- Window functions
- Tableau / Power BI dashboard
- Revenue trends
- Product analysis
- Customer segmentation
- Cohort analysis

Skills demonstrated:

SQL, ETL, data modeling, visualization,
business analysis and storytelling.

2. A/B Testing Analysis

3. Customer Churn / Business KPI Analytics

Do NOT blindly use these examples if the target career is different.
Create career-specific projects.

==================================================
10. GITHUB PORTFOLIO
==================================================

Explain what the student should put on GitHub:

- README
- clean source code
- screenshots
- architecture/workflow
- requirements.txt
- results
- dataset explanation
- demo link
- meaningful commits

==================================================
11. RESUME PLAN
==================================================

Give target-role-specific resume guidance.

Give examples of strong bullet points.

Use:

Action + Technology + Work + Result

==================================================
12. 3-MONTH ROADMAP
==================================================

Month 1
Month 2
Month 3

==================================================
13. FINAL ACTION CHECKLIST
==================================================

Give exactly 10 practical next actions.

==================================================

Formatting requirements:

Use Markdown.
Use clear headings.
Use tables where requested.
Use bullet points.
Keep it detailed.
Avoid generic motivational content.
Make the advice personalized to the student.
Do not promise employment.
"""


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "ai_response" not in st.session_state:
    st.session_state.ai_response = None


# ============================================================
# 1. STUDENT PROFILE
# ============================================================

st.markdown(
    '<div class="section-header">📋 Student Profile</div>',
    unsafe_allow_html=True,
)

profile_col1, profile_col2, profile_col3 = st.columns(3)

with profile_col1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=60,
        value=21,
        step=1,
    )

with profile_col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other",
            "Prefer not to say",
        ],
    )

with profile_col3:

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
    )


# ============================================================
# 2. EDUCATION
# ============================================================

st.markdown(
    '<div class="section-header">🎓 Education</div>',
    unsafe_allow_html=True,
)

edu_col1, edu_col2, edu_col3 = st.columns(3)

with edu_col1:

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

with edu_col2:

    ug_branch = st.selectbox(
        "UG Branch / Major",
        UG_BRANCHES,
    )

with edu_col3:

    ug_cgpa = st.number_input(
        "UG CGPA",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1,
    )


# ============================================================
# PG CHECKBOX
# ============================================================

has_pg = st.checkbox(
    "I have postgraduate education",
)


pg_degree = ""
pg_specialization = ""
pg_cgpa = None


if has_pg:

    pg_col1, pg_col2, pg_col3 = st.columns(3)

    with pg_col1:

        pg_degree = st.selectbox(
            "PG Degree",
            PG_DEGREES,
        )

    with pg_col2:

        pg_specialization = st.selectbox(
            "PG Specialization",
            PG_SPECIALIZATIONS,
        )

    with pg_col3:

        pg_cgpa = st.number_input(
            "PG CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1,
        )

else:

    st.caption(
        "PG details are optional. Select the checkbox above only "
        "if you have completed or are pursuing a postgraduate degree."
    )


# ============================================================
# 3. PLACEMENT PROFILE
# ============================================================

st.markdown(
    '<div class="section-header">💼 Placement Profile</div>',
    unsafe_allow_html=True,
)

placement_col1, placement_col2, placement_col3, placement_col4 = (
    st.columns(4)
)

with placement_col1:

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
    )

with placement_col2:

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=30,
        value=2,
        step=1,
    )

with placement_col3:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=30,
        value=2,
        step=1,
    )

with placement_col4:

    communication_skills = st.slider(
        "Communication Skills",
        min_value=0,
        max_value=10,
        value=7,
        step=1,
    )


# ============================================================
# 4. CAREER DIRECTION
# ============================================================

st.markdown(
    '<div class="section-header">🧭 Career Direction</div>',
    unsafe_allow_html=True,
)

career_col1, career_col2 = st.columns(2)

with career_col1:

    career_interest = st.selectbox(
        "Career Interest",
        list(CAREER_GOALS.keys()),
    )

with career_col2:

    target_goal = st.selectbox(
        "Target Career Goal",
        CAREER_GOALS[career_interest],
    )


# ============================================================
# 5. SKILLS
# ============================================================

skill_labels = get_skill_labels(
    ug_branch,
    pg_specialization,
)

st.markdown(
    f'<div class="section-header">🧠 {ug_branch} Skills</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
    These branch-specific skills are used to understand your
    technical profile and to personalize the AI career guidance.
    </div>
    """,
    unsafe_allow_html=True,
)


skill_values = []

skill_columns = st.columns(3)

for index, skill_name in enumerate(skill_labels):

    with skill_columns[index % 3]:

        value = st.slider(
            skill_name,
            min_value=0,
            max_value=10,
            value=7,
            step=1,
            key=f"skill_{index}",
        )

        skill_values.append(value)


# ============================================================
# GENERAL TECHNICAL SCORES
# ============================================================

st.markdown(
    '<div class="section-header">⚙️ General Technical Profile</div>',
    unsafe_allow_html=True,
)

general_col1, general_col2, general_col3 = st.columns(3)

with general_col1:

    coding_skills = st.slider(
        "Coding Skills",
        0,
        10,
        7,
        1,
    )

with general_col2:

    aptitude_score = st.slider(
        "Aptitude Score",
        0,
        100,
        73,
        1,
    )

with general_col3:

    internship_quality = st.slider(
        "Practical Experience",
        0,
        10,
        7,
        1,
    )


# ============================================================
# CONVERT BRANCH SKILLS TO MODEL FEATURES
# ============================================================

# The ML model expects five domain skill features.
# The dashboard's branch-specific skills are mapped to those
# five numerical features.

domain_skill_1 = float(skill_values[0])
domain_skill_2 = float(skill_values[1])
domain_skill_3 = float(skill_values[2])
domain_skill_4 = float(skill_values[3])
domain_skill_5 = float(skill_values[4])


# ============================================================
# STUDENT OBJECT
# ============================================================

student = {

    "age": int(age),

    "gender": gender,

    "ug_degree": ug_degree,

    "ug_branch": ug_branch,

    "ug_cgpa": float(ug_cgpa),

    "backlogs": int(backlogs),

    "internships": int(internships),

    "projects": int(projects),

    "certifications": int(certifications),

    "coding_skills": float(coding_skills),

    "communication_skills": float(
        communication_skills
    ),

    "aptitude_score": float(
        aptitude_score
    ),

    "domain_skill_1": domain_skill_1,

    "domain_skill_2": domain_skill_2,

    "domain_skill_3": domain_skill_3,

    "domain_skill_4": domain_skill_4,

    "domain_skill_5": domain_skill_5,

    # Extra information for AI guidance
    "pg_degree": pg_degree,
    "pg_specialization": pg_specialization,
    "pg_cgpa": pg_cgpa,
}


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Placement",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not model_loaded:

        st.error(
            "Placement model is not available."
        )

    else:

        try:

            model_input = build_model_input(
                student
            )

            probability = get_probability(
                model_input
            )

            raw_prediction = model.predict(
                model_input
            )[0]

            if isinstance(
                raw_prediction,
                str,
            ):

                placed = (
                    raw_prediction.lower()
                    in [
                        "1",
                        "true",
                        "yes",
                        "placed",
                        "selected",
                    ]
                )

            else:

                placed = bool(
                    raw_prediction
                )

            st.session_state.prediction_done = True

            st.session_state.probability = probability

            st.session_state.placed = placed

            st.session_state.student = student.copy()

            st.session_state.career_interest = (
                career_interest
            )

            st.session_state.target_goal = (
                target_goal
            )

            st.session_state.pg_degree = (
                pg_degree
            )

            st.session_state.pg_specialization = (
                pg_specialization
            )

            st.session_state.pg_cgpa = (
                pg_cgpa
            )

            st.session_state.skill_labels = (
                skill_labels
            )

            st.session_state.skill_values = (
                skill_values
            )

            st.session_state.ai_response = None

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.prediction_done:

    probability = (
        st.session_state.probability
    )

    placed = (
        st.session_state.placed
    )

    prediction_text = (
        "Placement Likely"
        if placed
        else "Needs Further Preparation"
    )

    st.markdown(
        '<div class="section-header">📊 Placement Prediction</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if placed:

            st.success(
                f"### ✅ {prediction_text}"
            )

        else:

            st.warning(
                f"### 📚 {prediction_text}"
            )

    with result_col2:

        st.metric(
            "Placement Probability",
            f"{probability * 100:.1f}%",
        )

    st.progress(
        min(
            max(
                probability,
                0.0,
            ),
            1.0,
        )
    )

    st.caption(
        "The percentage shown is a machine-learning estimate "
        "and is not a guarantee of employment."
    )


# ============================================================
# PREDICTION EXPLANATION
# ============================================================

if st.session_state.prediction_done:

    st.markdown(
        '<div class="section-header">🔎 Prediction Explanation</div>',
        unsafe_allow_html=True,
    )

    student_result = (
        st.session_state.student
    )

    strengths = []
    improvements = []

    if student_result["ug_cgpa"] >= 8:
        strengths.append(
            f"Strong UG CGPA ({student_result['ug_cgpa']:.1f}/10)"
        )
    else:
        improvements.append(
            "Improve CGPA where possible."
        )

    if student_result["coding_skills"] >= 8:
        strengths.append(
            "Strong coding profile."
        )
    elif student_result["coding_skills"] < 6:
        improvements.append(
            "Strengthen programming fundamentals."
        )

    if student_result["communication_skills"] >= 8:
        strengths.append(
            "Strong communication skills."
        )
    elif student_result["communication_skills"] < 6:
        improvements.append(
            "Practice communication and interview answers."
        )

    if student_result["aptitude_score"] >= 75:
        strengths.append(
            "Good aptitude performance."
        )
    elif student_result["aptitude_score"] < 60:
        improvements.append(
            "Practice quantitative and logical aptitude."
        )

    if student_result["internships"] >= 2:
        strengths.append(
            "Good internship exposure."
        )
    elif student_result["internships"] == 0:
        improvements.append(
            "Try to gain practical internship experience."
        )

    if student_result["projects"] >= 3:
        strengths.append(
            "Good project experience."
        )
    elif student_result["projects"] < 2:
        improvements.append(
            "Build more practical projects."
        )

    if student_result["backlogs"] == 0:
        strengths.append(
            "No academic backlogs."
        )
    else:
        improvements.append(
            "Work toward clearing academic backlogs."
        )

    explanation_col1, explanation_col2 = (
        st.columns(2)
    )

    with explanation_col1:

        st.subheader("💪 Current Strengths")

        for item in strengths:

            st.success(item)

    with explanation_col2:

        st.subheader("🎯 Areas To Improve")

        for item in improvements:

            st.warning(item)


# ============================================================
# AI GUIDANCE
# ============================================================

if st.session_state.prediction_done:

    st.markdown(
        '<div class="section-header">🤖 AI Career Guidance</div>',
        unsafe_allow_html=True,
    )

    st.write(
        f"""
        Your selected career direction is:

        **{st.session_state.career_interest} → {st.session_state.target_goal}**

        The AI will use your UG/PG education, specialization,
        skills and placement profile to create a personalized
        career roadmap.
        """
    )

    if not gemini_key:

        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )

    else:

        generate_ai = st.button(
            "🤖 Generate Personalized AI Guidance",
            use_container_width=True,
        )

        if generate_ai:

            pg_info = "No postgraduate education"

            if st.session_state.pg_degree:

                pg_info = (
                    f"PG Degree: "
                    f"{st.session_state.pg_degree}\n"
                    f"PG Specialization: "
                    f"{st.session_state.pg_specialization}\n"
                    f"PG CGPA: "
                    f"{st.session_state.pg_cgpa}"
                )

            with st.spinner(
                "🤖 AI is creating your personalized career report..."
            ):

                try:

                    prompt = create_ai_prompt(
                        student=st.session_state.student,
                        probability=st.session_state.probability,
                        career_interest=st.session_state.career_interest,
                        target_goal=st.session_state.target_goal,
                        pg_info=pg_info,
                        skill_labels=st.session_state.skill_labels,
                        skill_values=st.session_state.skill_values,
                    )

                    response = call_gemini(
                        prompt,
                        gemini_key,
                    )

                    st.session_state.ai_response = (
                        response
                    )

                except Exception as e:

                    st.error(
                        "AI guidance could not be generated."
                    )

                    st.code(
                        str(e),
                        language="text",
                    )


# ============================================================
# AI RESULT
# ============================================================

if st.session_state.ai_response:

    st.markdown(
        '<div class="section-header">🧭 Your Personalized Career Roadmap</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        st.session_state.ai_response
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Model Information"
):

    if metadata:

        if isinstance(
            metadata,
            dict,
        ):

            for key, value in metadata.items():

                if isinstance(
                    value,
                    (dict, list),
                ):

                    st.write(
                        f"**{key}:**"
                    )

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
    demonstration. Placement predictions are generated by a
    machine-learning model and are not guaranteed employment
    outcomes. AI career guidance is advisory and should be combined
    with your own research, college placement guidance and career
    planning.
    """
)
