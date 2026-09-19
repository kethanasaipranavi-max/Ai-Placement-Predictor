import os
import json
import urllib.request
import urllib.error

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
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULE_FILE = "recommendation_rules.pkl"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "prediction_result": None,
    "student_profile": None,
    "ai_career_advice": None,
    "ai_provider": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

def get_branch_skills(branch):

    skills = {
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
            "Automobile Systems",
            "CAD / Design",
            "Engine Technology",
            "Manufacturing",
            "Vehicle Diagnostics",
        ],

        "Production Engineering": [
            "Manufacturing",
            "Production Planning",
            "Quality Control",
            "Industrial Processes",
            "CAD",
        ],

        "Industrial Engineering": [
            "Operations Management",
            "Production Systems",
            "Quality Management",
            "Supply Chain",
            "Industrial Analysis",
        ],

        "Aeronautical Engineering": [
            "Aerodynamics",
            "Aircraft Systems",
            "CAD",
            "Propulsion",
            "Manufacturing",
        ],

        "Aerospace Engineering": [
            "Aerodynamics",
            "Aircraft Design",
            "Propulsion",
            "CAD",
            "Space Systems",
        ],

        "Electrical Engineering": [
            "Circuit Analysis",
            "Power Systems",
            "Control Systems",
            "PLC / Automation",
            "Electrical Design",
        ],

        "Electronics Engineering": [
            "Electronic Circuits",
            "Embedded Systems",
            "PCB Design",
            "Microcontrollers",
            "Instrumentation",
        ],

        "Electronics and Communication Engineering": [
            "Communication Systems",
            "Embedded Systems",
            "Electronics",
            "Signal Processing",
            "VLSI",
        ],

        "Biomedical Engineering": [
            "Biomedical Instrumentation",
            "Medical Devices",
            "Electronics",
            "Clinical Engineering",
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
            "Process Engineering",
            "Plant Operations",
            "Industrial Safety",
        ],

        "Mathematics": [
            "Mathematical Analysis",
            "Statistics",
            "Problem Solving",
            "Quantitative Reasoning",
            "Research Methods",
        ],

        "Statistics": [
            "Statistical Analysis",
            "Probability",
            "Data Interpretation",
            "Research Methods",
            "Quantitative Analysis",
        ],

        "Physics": [
            "Laboratory Techniques",
            "Instrumentation",
            "Electronics",
            "Scientific Analysis",
            "Research Methods",
        ],

        "Chemistry": [
            "Analytical Chemistry",
            "Laboratory Techniques",
            "Chemical Analysis",
            "Instrumentation",
            "Research Methods",
        ],

        "Environmental Science": [
            "Environmental Analysis",
            "Sustainability",
            "Environmental Monitoring",
            "Research Methods",
            "Data Analysis",
        ],

        "Biotechnology": [
            "Laboratory Techniques",
            "Molecular Biology",
            "Biotechnology Methods",
            "Research Skills",
            "Scientific Analysis",
        ],

        "Microbiology": [
            "Microbiology Techniques",
            "Laboratory Skills",
            "Culture Techniques",
            "Research Methods",
            "Scientific Analysis",
        ],

        "Biochemistry": [
            "Biochemical Techniques",
            "Laboratory Skills",
            "Chemical Analysis",
            "Research Methods",
            "Scientific Analysis",
        ],

        "Biological Sciences": [
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis",
            "Biological Techniques",
            "Data Interpretation",
        ],

        "Life Sciences": [
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis",
            "Biological Techniques",
            "Data Interpretation",
        ],

        "Genetics": [
            "Genetics",
            "Molecular Biology",
            "Laboratory Skills",
            "Research Methods",
            "Scientific Analysis",
        ],

        "Botany": [
            "Plant Biology",
            "Laboratory Skills",
            "Research Methods",
            "Field Research",
            "Scientific Analysis",
        ],

        "Zoology": [
            "Animal Biology",
            "Laboratory Skills",
            "Research Methods",
            "Field Research",
            "Scientific Analysis",
        ],

        "Food Science and Nutrition": [
            "Nutrition Science",
            "Food Analysis",
            "Laboratory Skills",
            "Diet Planning",
            "Food Safety",
        ],

        "Food Technology": [
            "Food Processing",
            "Food Safety",
            "Quality Control",
            "Laboratory Analysis",
            "Manufacturing",
        ],

        "Nutrition and Dietetics": [
            "Clinical Nutrition",
            "Diet Planning",
            "Nutrition Assessment",
            "Food Science",
            "Communication",
        ],

        "Economics": [
            "Economic Analysis",
            "Statistics",
            "Financial Analysis",
            "Research Methods",
            "Quantitative Analysis",
        ],

        "Commerce": [
            "Accounting",
            "Taxation",
            "Financial Analysis",
            "Auditing",
            "Business Knowledge",
        ],

        "Business Administration": [
            "Business Strategy",
            "Marketing",
            "Operations",
            "Management",
            "Business Analysis",
        ],

        "Finance": [
            "Financial Analysis",
            "Accounting",
            "Investment Analysis",
            "Financial Modeling",
            "Banking Knowledge",
        ],

        "Accounting": [
            "Accounting",
            "Taxation",
            "Auditing",
            "Financial Reporting",
            "Financial Analysis",
        ],

        "Management": [
            "Leadership",
            "Operations",
            "Business Strategy",
            "Project Management",
            "Decision Making",
        ],

        "Marketing": [
            "Marketing Strategy",
            "Digital Marketing",
            "Market Research",
            "Brand Management",
            "Consumer Analysis",
        ],

        "Human Resources": [
            "Recruitment",
            "HR Operations",
            "Employee Relations",
            "Talent Management",
            "Organizational Skills",
        ],

        "Psychology": [
            "Psychological Assessment",
            "Research Methods",
            "Counseling Skills",
            "Behavioral Analysis",
            "Data Interpretation",
        ],

        "English": [
            "Writing",
            "Communication",
            "Editing",
            "Research",
            "Presentation Skills",
        ],

        "Political Science": [
            "Political Analysis",
            "Research Methods",
            "Public Policy",
            "International Relations",
            "Communication",
        ],

        "Sociology": [
            "Social Research",
            "Research Methods",
            "Data Analysis",
            "Community Studies",
            "Communication",
        ],

        "History": [
            "Historical Research",
            "Research Methods",
            "Writing",
            "Analysis",
            "Documentation",
        ],

        "Public Administration": [
            "Public Policy",
            "Administration",
            "Governance",
            "Research",
            "Management",
        ],
    }

    return skills.get(
        branch,
        [
            "Core Domain Knowledge",
            "Practical Skills",
            "Research Skills",
            "Industry Knowledge",
            "Problem Solving",
        ],
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
        "Computer Applications",
    ]:
        return [
            "Software Development",
            "Data Science",
            "Artificial Intelligence",
            "Cyber Security",
            "Cloud Computing",
            "Data Analytics",
            "Other",
        ]

    if branch in [
        "Mechanical Engineering",
        "Automobile Engineering",
        "Production Engineering",
        "Industrial Engineering",
    ]:
        return [
            "Design Engineering",
            "Manufacturing",
            "Automotive Engineering",
            "Production Engineering",
            "Operations",
            "Other",
        ]

    if branch in [
        "Electrical Engineering",
        "Electronics Engineering",
        "Electronics and Communication Engineering",
        "Biomedical Engineering",
    ]:
        return [
            "Embedded Systems",
            "Electronics Design",
            "Power Systems",
            "Automation",
            "VLSI",
            "Instrumentation",
            "Other",
        ]

    if branch == "Civil Engineering":
        return [
            "Structural Engineering",
            "Construction Management",
            "Site Engineering",
            "Infrastructure",
            "Surveying",
            "Other",
        ]

    if branch == "Chemical Engineering":
        return [
            "Process Engineering",
            "Plant Operations",
            "Chemical Analysis",
            "Industrial Safety",
            "Research",
            "Other",
        ]

    if branch in [
        "Biotechnology",
        "Microbiology",
        "Biochemistry",
        "Biological Sciences",
        "Life Sciences",
        "Genetics",
        "Botany",
        "Zoology",
    ]:
        return [
            "Research",
            "Laboratory Work",
            "Biotechnology",
            "Pharmaceutical Industry",
            "Quality Control",
            "Other",
        ]

    if branch in [
        "Food Science and Nutrition",
        "Food Technology",
        "Nutrition and Dietetics",
    ]:
        return [
            "Food Industry",
            "Nutrition",
            "Quality Control",
            "Food Safety",
            "Research",
            "Clinical Nutrition",
            "Other",
        ]

    if branch in [
        "Commerce",
        "Finance",
        "Accounting",
        "Economics",
    ]:
        return [
            "Accounting",
            "Finance",
            "Banking",
            "Financial Analysis",
            "Auditing",
            "Business Analytics",
            "Other",
        ]

    if branch in [
        "Business Administration",
        "Management",
        "Marketing",
        "Human Resources",
    ]:
        return [
            "Management",
            "Marketing",
            "Human Resources",
            "Operations",
            "Business Analytics",
            "Finance",
            "Other",
        ]

    return [
        "Research",
        "Teaching",
        "Government Sector",
        "Industry",
        "Higher Studies",
        "Other",
    ]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_components():

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    feature_names = None

    if os.path.exists(FEATURE_FILE):
        feature_names = joblib.load(FEATURE_FILE)

        if isinstance(feature_names, pd.DataFrame):
            feature_names = feature_names.columns.tolist()

        elif isinstance(feature_names, pd.Series):
            feature_names = feature_names.tolist()

        elif isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()

        elif isinstance(feature_names, dict):
            feature_names = feature_names.get(
                "feature_names",
                feature_names.get(
                    "features",
                    list(feature_names.keys()),
                ),
            )

        feature_names = [str(x) for x in feature_names]

    rules = {}

    if os.path.exists(RULE_FILE):
        try:
            rules = joblib.load(RULE_FILE)
        except Exception:
            rules = {}

    metadata = {}

    if os.path.exists(METADATA_FILE):
        try:
            metadata = joblib.load(METADATA_FILE)

            if not isinstance(metadata, dict):
                metadata = {}

        except Exception:
            metadata = {}

    return model, feature_names or [], rules, metadata


# ============================================================
# MODEL SCHEMA
# ============================================================

def get_model_input_features(model, artifact_features):

    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        names = [str(x) for x in names]

        if names:
            return names

    named_steps = getattr(model, "named_steps", None)

    if named_steps:
        for _, step in reversed(list(named_steps.items())):

            names = getattr(step, "feature_names_in_", None)

            if names is not None:
                names = [str(x) for x in names]

                if names:
                    return names

    if artifact_features:
        return artifact_features

    raise RuntimeError(
        "Unable to determine the input features expected by the trained model."
    )


# ============================================================
# MODEL INPUT BUILDER
# ============================================================

def build_model_input(student, model, feature_names):

    expected = get_model_input_features(
        model,
        feature_names,
    )

    # --------------------------------------------------------
    # This is the schema used by the new training pipeline.
    # --------------------------------------------------------

    values = {

        "age": float(student["age"]),

        "ug_cgpa": float(student["ug_cgpa"]),

        "backlogs": float(student["backlogs"]),

        "internships": float(student["internships"]),

        "projects": float(student["projects"]),

        "certifications": float(student["certifications"]),

        "coding_skills": float(student["coding_skills"]),

        "communication_skills": float(
            student["communication_skills"]
        ),

        "aptitude_score": float(
            student["aptitude_score"]
        ),

        "domain_skill_1": float(
            student["domain_scores"][0]
        ),

        "domain_skill_2": float(
            student["domain_scores"][1]
        ),

        "domain_skill_3": float(
            student["domain_scores"][2]
        ),

        "domain_skill_4": float(
            student["domain_scores"][3]
        ),

        "domain_skill_5": float(
            student["domain_scores"][4]
        ),

        "gender": student["gender"],

        "ug_degree": student["ug_degree"],

        "ug_branch": student["ug_branch"],
    }

    missing = [
        feature
        for feature in expected
        if feature not in values
    ]

    if missing:
        raise RuntimeError(
            "The trained model expects these features, "
            "but the app cannot construct them:\n\n"
            + "\n".join(
                f"- {feature}"
                for feature in missing
            )
        )

    return pd.DataFrame(
        [
            {
                feature: values[feature]
                for feature in expected
            }
        ],
        columns=expected,
    )


# ============================================================
# MODEL PROBABILITY
# ============================================================

def get_model_classes(model):

    classes = getattr(model, "classes_", None)

    if classes is not None:
        return list(classes)

    named_steps = getattr(model, "named_steps", None)

    if named_steps:
        for _, step in reversed(list(named_steps.items())):

            classes = getattr(step, "classes_", None)

            if classes is not None:
                return list(classes)

    return None


def get_positive_probability(model, model_input):

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = np.asarray(
        model.predict_proba(model_input),
        dtype=float,
    )

    if probabilities.ndim != 2:
        return None

    classes = get_model_classes(model)

    if classes is not None:

        # Prefer class 1 because the target was created as
        # 0 = not placed, 1 = placed.
        for target in [1, True, "1", "Placed", "PLACED", "Yes", "YES"]:

            if target in classes:

                index = classes.index(target)

                return float(
                    probabilities[0, index]
                )

        # Binary fallback
        if probabilities.shape[1] == 2:
            return float(probabilities[0, 1])

    if probabilities.shape[1] == 2:
        return float(probabilities[0, 1])

    return float(
        np.max(probabilities[0])
    )


# ============================================================
# PREDICTION EXPLANATION
# ============================================================

def explain_prediction(student):

    strengths = []
    improvements = []

    if student["ug_cgpa"] >= 8:
        strengths.append(
            f"Strong CGPA of {student['ug_cgpa']:.1f}/10."
        )
    elif student["ug_cgpa"] < 6.5:
        improvements.append(
            f"CGPA is {student['ug_cgpa']:.1f}/10; improving academic performance can strengthen the profile."
        )

    if student["backlogs"] == 0:
        strengths.append(
            "No active backlogs."
        )
    else:
        improvements.append(
            f"{student['backlogs']} backlog(s) are present."
        )

    if student["internships"] >= 2:
        strengths.append(
            f"{student['internships']} internships provide useful practical exposure."
        )
    elif student["internships"] == 0:
        improvements.append(
            "No internships are listed. Practical industry exposure would strengthen the profile."
        )

    if student["projects"] >= 3:
        strengths.append(
            f"{student['projects']} projects demonstrate practical work."
        )
    elif student["projects"] < 2:
        improvements.append(
            "Add more substantial projects with measurable outcomes."
        )

    if student["certifications"] >= 2:
        strengths.append(
            f"{student['certifications']} certifications show additional learning."
        )

    if student["coding_skills"] >= 8:
        strengths.append(
            f"Strong coding/computational skill level: {student['coding_skills']}/10."
        )
    elif student["coding_skills"] < 6:
        improvements.append(
            f"Coding/computational skills are {student['coding_skills']}/10."
        )

    if student["communication_skills"] >= 8:
        strengths.append(
            f"Strong communication score: {student['communication_skills']}/10."
        )
    elif student["communication_skills"] < 6:
        improvements.append(
            f"Communication score is {student['communication_skills']}/10."
        )

    if student["aptitude_score"] >= 80:
        strengths.append(
            f"Strong aptitude score: {student['aptitude_score']}/100."
        )
    elif student["aptitude_score"] < 60:
        improvements.append(
            f"Aptitude score is {student['aptitude_score']}/100."
        )

    for skill, score in zip(
        student["branch_skill_names"],
        student["domain_scores"],
    ):

        if score >= 8:
            strengths.append(
                f"{skill}: {score}/10."
            )

        elif score <= 4:
            improvements.append(
                f"{skill}: {score}/10 needs improvement."
            )

    return strengths[:6], improvements[:6]


# ============================================================
# PREDICTION
# ============================================================

def run_prediction(student):

    model, feature_names, rules, metadata = load_components()

    model_input = build_model_input(
        student,
        model,
        feature_names,
    )

    prediction = model.predict(model_input)[0]

    probability = get_positive_probability(
        model,
        model_input,
    )

    try:
        placed = int(prediction) == 1
    except Exception:
        placed = str(prediction).strip().lower() in {
            "1",
            "true",
            "placed",
            "yes",
        }

    strengths, improvements = explain_prediction(
        student
    )

    return {
        "prediction": prediction,
        "placed": placed,
        "probability": probability,
        "model_input": model_input,
        "feature_names": feature_names,
        "metadata": metadata,
        "strengths": strengths,
        "improvements": improvements,
    }


# ============================================================
# AI CONFIGURATION
# ============================================================

def get_secret(name):

    try:

        value = st.secrets.get(
            name,
            "",
        )

        if value:
            return str(value).strip()

    except Exception:
        pass

    value = os.getenv(
        name,
        "",
    )

    return str(value).strip()


def get_ai_status():

    return {
        "Groq": bool(
            get_secret("GROQ_API_KEY")
        ),

        "Gemini": bool(
            get_secret("GEMINI_API_KEY")
        ),
    }


# ============================================================
# AI PROMPT
# ============================================================

def build_ai_prompt(
    student,
    prediction_context=None,
):

    skills_text = "\n".join(
        f"- {skill}: {score}/10"
        for skill, score in zip(
            student["branch_skill_names"],
            student["domain_scores"],
        )
    )

    prediction_context = (
        prediction_context
        or "No ML prediction was requested."
    )

    return f"""
You are an expert student placement advisor and career mentor.

Give practical, personalized and realistic guidance.

Do NOT guarantee employment.

Do NOT invent achievements.

Use the student's actual branch and actual scores.

The student wants to improve placement readiness and career skills.

STUDENT PROFILE

Age: {student["age"]}

Gender: {student["gender"]}

UG Degree: {student["ug_degree"]}

UG Branch: {student["ug_branch"]}

CGPA: {student["ug_cgpa"]:.1f}/10

Backlogs: {student["backlogs"]}

Internships: {student["internships"]}

Projects: {student["projects"]}

Certifications: {student["certifications"]}

Coding / Computational Skills:
{student["coding_skills"]}/10

Communication:
{student["communication_skills"]}/10

Aptitude:
{student["aptitude_score"]}/100

Career Interest:
{student["career_interest"]}

Target Career:
{student["target_career_goal"] or "Not specified"}

BRANCH-SPECIFIC SKILLS

{skills_text}

PLACEMENT MODEL CONTEXT

{prediction_context}

IMPORTANT:

Explain the prediction as a model estimate, not as a guarantee.

Focus strongly on actionable improvement.

Respond using these headings:

## 1. Prediction Interpretation

Explain what the ML prediction means.

## 2. Why This Prediction

Identify the strongest positive and negative factors visible in the student's profile.

## 3. Career Direction

Explain suitable career directions for this branch and stated interest.

## 4. Top Strengths

List the student's strongest areas.

## 5. Skill Gaps

Identify the most important gaps.

## 6. What To Improve First

Give the highest-priority improvements.

## 7. 30-Day Placement Plan

Give a practical week-by-week plan.

## 8. Technical Topics To Study

Give branch-specific technical topics.

## 9. Tools To Learn

Give relevant industry tools, platforms or technologies.

## 10. Project Ideas

Give EXACTLY 3 realistic projects.

For each project include:

- Project title
- What to build
- Skills demonstrated
- Why it helps for placements

## 11. Interview Preparation

Give technical, aptitude and HR/interview preparation advice.

Be specific rather than generic.
"""


# ============================================================
# GROQ
# ============================================================

def generate_with_groq(prompt):

    key = get_secret(
        "GROQ_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    payload = {
        "model": "openai/gpt-oss-120b",

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert student "
                    "placement advisor and career mentor."
                ),
            },

            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": 0.4,

        "max_tokens": 3000,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",

        data=json.dumps(
            payload
        ).encode("utf-8"),

        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },

        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=45,
        ) as response:

            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

            body = json.loads(raw)

    except urllib.error.HTTPError as exc:

        details = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Groq API error HTTP {exc.code}: {details}"
        ) from exc

    except urllib.error.URLError as exc:

        raise RuntimeError(
            f"Could not connect to Groq: {exc.reason}"
        ) from exc

    text = (
        body
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )

    if not text:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return str(text).strip()


# ============================================================
# GEMINI FALLBACK
# ============================================================

def generate_with_gemini(prompt):

    key = get_secret(
        "GEMINI_API_KEY"
    )

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    from google import genai

    client = genai.Client(
        api_key=key
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return str(text).strip()


# ============================================================
# AI GUIDANCE
# ============================================================

def generate_ai_guidance(
    student,
    prediction_context=None,
):

    prompt = build_ai_prompt(
        student,
        prediction_context,
    )

    errors = []

    # Groq first
    if get_secret("GROQ_API_KEY"):

        try:

            return (
                generate_with_groq(prompt),
                "Groq",
            )

        except Exception as exc:

            errors.append(
                f"Groq: {exc}"
            )

    else:

        errors.append(
            "Groq: API key not configured."
        )

    # Gemini fallback
    if get_secret("GEMINI_API_KEY"):

        try:

            return (
                generate_with_gemini(prompt),
                "Gemini",
            )

        except Exception as exc:

            errors.append(
                f"Gemini: {exc}"
            )

    else:

        errors.append(
            "Gemini: API key not configured."
        )

    raise RuntimeError(
        "AI guidance could not be generated.\n\n"
        + "\n".join(errors)
    )


# ============================================================
# STUDENT PROFILE
# ============================================================

def build_student_profile(
    gender,
    age,
    ug_degree,
    ug_branch,
    ug_cgpa,
    backlogs,
    internships,
    projects,
    certifications,
    coding_skills,
    communication_skills,
    aptitude_score,
    career_interest,
    target_career_goal,
    branch_skill_names,
    domain_scores,
):

    return {

        "gender": gender,

        "age": int(age),

        "ug_degree": ug_degree,

        "ug_branch": ug_branch,

        "ug_cgpa": float(ug_cgpa),

        "backlogs": int(backlogs),

        "internships": int(internships),

        "projects": int(projects),

        "certifications": int(certifications),

        "coding_skills": int(coding_skills),

        "communication_skills": int(
            communication_skills
        ),

        "aptitude_score": int(
            aptitude_score
        ),

        "career_interest": career_interest,

        "target_career_goal":
            target_career_goal.strip(),

        "branch_skill_names":
            branch_skill_names,

        "domain_scores":
            domain_scores,
    }


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🎓 AI Student Placement Predictor"
)

st.write(
    "Predict placement outcomes using the trained ML model "
    "and receive personalized AI career guidance."
)

st.info(
    "⚠️ The placement percentage is a model-estimated probability "
    "based on the training data. It is not a guaranteed employment probability."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔧 System Status")

    try:

        model, feature_names, rules, metadata = (
            load_components()
        )

        st.success(
            "✅ ML model loaded"
        )

        model_features = get_model_input_features(
            model,
            feature_names,
        )

        st.write(
            f"Model input features: **{len(model_features)}**"
        )

        if metadata:

            st.write(
                "Model:",
                metadata.get(
                    "model_name",
                    "Trained placement model",
                ),
            )

            st.write(
                "Dataset rows:",
                metadata.get(
                    "dataset_size",
                    "Available in metadata",
                ),
            )

    except Exception as exc:

        st.error(
            "❌ ML model could not be loaded."
        )

        st.code(
            str(exc)
        )

    st.divider()

    st.write("**AI providers**")

    ai_status = get_ai_status()

    if ai_status["Groq"]:
        st.success(
            "Groq: Configured"
        )
    else:
        st.warning(
            "Groq: Not configured"
        )

    if ai_status["Gemini"]:
        st.success(
            "Gemini: Configured"
        )
    else:
        st.info(
            "Gemini: Not configured"
        )

    st.caption(
        "API keys are stored in Streamlit Secrets "
        "and are never displayed."
    )


# ============================================================
# STUDENT INPUT
# ============================================================

st.header(
    "📋 Student Profile"
)


# ------------------------------------------------------------
# BASIC DETAILS
# ------------------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=60,
        value=21,
    )

with c2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other",
        ],
    )

with c3:

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=20,
        value=0,
    )


# ------------------------------------------------------------
# EDUCATION
# ------------------------------------------------------------

st.subheader(
    "🎓 Education"
)

ug_degrees = [
    "BE",
    "BTech",
    "BSc",
    "BCA",
    "BBA",
    "BCom",
    "BA",
    "Other",
]

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

    "Other",
]

c1, c2, c3 = st.columns(3)

with c1:

    ug_degree = st.selectbox(
        "UG Degree",
        ug_degrees,
    )

with c2:

    ug_branch = st.selectbox(
        "UG Branch / Major",
        branches,
    )

with c3:

    ug_cgpa = st.number_input(
        "UG CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1,
    )


# ------------------------------------------------------------
# PLACEMENT PROFILE
# ------------------------------------------------------------

st.subheader(
    "💼 Placement Profile"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=20,
        value=1,
    )

with c2:

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=30,
        value=2,
    )

with c3:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=30,
        value=1,
    )

with c4:

    communication_skills = st.slider(
        "Communication Skills",
        min_value=1,
        max_value=10,
        value=7,
    )


# ------------------------------------------------------------
# GENERAL SKILLS
# ------------------------------------------------------------

c1, c2 = st.columns(2)

with c1:

    coding_skills = st.slider(
        "Coding / Computational Skills",
        min_value=1,
        max_value=10,
        value=7,
    )

with c2:

    aptitude_score = st.slider(
        "Aptitude Score",
        min_value=0,
        max_value=100,
        value=70,
    )


# ------------------------------------------------------------
# CAREER DIRECTION
# ------------------------------------------------------------

st.subheader(
    "🧭 Career Direction"
)

c1, c2 = st.columns(2)

with c1:

    career_interest = st.selectbox(
        "Career Interest",
        get_career_interests(
            ug_branch
        ),
    )

with c2:

    target_career_goal = st.text_input(
        "Target Career Goal",
        placeholder=(
            "Example: Software Developer, "
            "Data Analyst, ML Engineer"
        ),
    )


# ------------------------------------------------------------
# BRANCH-SPECIFIC SKILLS
# ------------------------------------------------------------

st.subheader(
    f"🧠 {ug_branch} — Core Domain Skills"
)

st.caption(
    "Rate your current knowledge from 1 to 10."
)

branch_skill_names = get_branch_skills(
    ug_branch
)

domain_scores = []

skill_cols = st.columns(
    min(3, len(branch_skill_names))
)

for index, skill in enumerate(
    branch_skill_names
):

    with skill_cols[
        index % len(skill_cols)
    ]:

        score = st.slider(
            skill,
            min_value=1,
            max_value=10,
            value=5,
            key=f"skill_{ug_branch}_{skill}",
        )

        domain_scores.append(
            int(score)
        )


# ------------------------------------------------------------
# BUILD STUDENT
# ------------------------------------------------------------

student = build_student_profile(
    gender=gender,
    age=age,
    ug_degree=ug_degree,
    ug_branch=ug_branch,
    ug_cgpa=ug_cgpa,
    backlogs=backlogs,
    internships=internships,
    projects=projects,
    certifications=certifications,
    coding_skills=coding_skills,
    communication_skills=communication_skills,
    aptitude_score=aptitude_score,
    career_interest=career_interest,
    target_career_goal=target_career_goal,
    branch_skill_names=branch_skill_names,
    domain_scores=domain_scores,
)


# ============================================================
# ACTION BUTTONS
# ============================================================

st.divider()

c1, c2 = st.columns(2)

with c1:

    predict_button = st.button(
        "🔮 Predict Placement",
        use_container_width=True,
        type="primary",
    )

with c2:

    ai_button = st.button(
        "🤖 Generate AI Guidance",
        use_container_width=True,
    )


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    try:

        with st.spinner(
            "Running the trained placement model..."
        ):

            result = run_prediction(
                student
            )

        st.session_state.prediction_result = result
        st.session_state.student_profile = student

        st.session_state.ai_career_advice = None
        st.session_state.ai_provider = None

    except Exception as exc:

        st.session_state.prediction_result = None

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(exc)
        )


# ============================================================
# PREDICTION RESULT
# ============================================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()

    st.header(
        "🎯 Placement Prediction"
    )

    probability = result[
        "probability"
    ]

    if probability is not None:

        probability_pct = (
            probability * 100
        )

    else:

        probability_pct = None


    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        if result["placed"]:

            st.success(
                "🟢 MODEL CLASSIFICATION: PLACED"
            )

        else:

            st.warning(
                "🟡 MODEL CLASSIFICATION: NOT PLACED"
            )

    with c2:

        if probability_pct is not None:

            st.metric(
                "Placement Probability",
                f"{probability_pct:.2f}%",
            )

        else:

            st.metric(
                "Placement Probability",
                "Unavailable",
            )


    if probability is not None:

        st.progress(
            float(
                np.clip(
                    probability,
                    0,
                    1,
                )
            )
        )

        st.caption(
            "This is the calibrated probability estimated by the trained ML model."
        )


    # --------------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "🔍 What This Prediction Means"
    )

    if probability_pct is not None:

        if probability_pct >= 75:

            st.success(
                f"The model estimates a relatively high positive-class probability of {probability_pct:.2f}% for this profile."
            )

        elif probability_pct >= 50:

            st.info(
                f"The model estimates a moderate positive-class probability of {probability_pct:.2f}% for this profile."
            )

        else:

            st.warning(
                f"The model estimates a lower positive-class probability of {probability_pct:.2f}% for this profile."
            )

    st.caption(
        "The prediction reflects patterns learned from the training dataset. "
        "It should not be treated as a guaranteed employment outcome."
    )


    # --------------------------------------------------------
    # STRENGTHS
    # --------------------------------------------------------

    if result["strengths"]:

        st.subheader(
            "💪 Profile Strengths"
        )

        for item in result["strengths"]:

            st.success(
                f"✓ {item}"
            )


    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    if result["improvements"]:

        st.subheader(
            "⚠️ Factors To Improve"
        )

        for item in result["improvements"]:

            st.warning(
                f"• {item}"
            )


    # --------------------------------------------------------
    # AI BUTTON AFTER PREDICTION
    # --------------------------------------------------------

    st.divider()

    st.header(
        "🤖 Personalized AI Career Guidance"
    )

    st.write(
        "Use the student's profile and ML prediction to generate personalized career guidance."
    )

    if st.button(
        "🧠 Generate Guidance From This Prediction",
        use_container_width=True,
    ):

        try:

            prediction_context = (
                f"Model classification: "
                f"{'PLACED' if result['placed'] else 'NOT PLACED'}\n"
            )

            if probability_pct is not None:

                prediction_context += (
                    f"Model probability: "
                    f"{probability_pct:.2f}%\n"
                )

            with st.spinner(
                "Generating personalized AI guidance..."
            ):

                advice, provider = (
                    generate_ai_guidance(
                        student,
                        prediction_context,
                    )
                )

            st.session_state.ai_career_advice = advice
            st.session_state.ai_provider = provider

        except Exception as exc:

            st.error(
                "❌ AI guidance failed."
            )

            st.code(
                str(exc)
            )


# ============================================================
# DIRECT AI GUIDANCE BUTTON
# ============================================================

if ai_button:

    try:

        st.session_state.student_profile = student

        with st.spinner(
            "Generating personalized AI guidance..."
        ):

            advice, provider = (
                generate_ai_guidance(
                    student
                )
            )

        st.session_state.ai_career_advice = advice
        st.session_state.ai_provider = provider

    except Exception as exc:

        st.error(
            "❌ AI guidance failed."
        )

        st.code(
            str(exc)
        )


# ============================================================
# AI RESULT
# ============================================================

if st.session_state.ai_career_advice:

    st.divider()

    provider = (
        st.session_state.ai_provider
        or "AI"
    )

    st.header(
        f"🧠 Personalized AI Career Guidance — {provider}"
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
    "ML Placement Prediction + Personalized AI Career Guidance"
)
