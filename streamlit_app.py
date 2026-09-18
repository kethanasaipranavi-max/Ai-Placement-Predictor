import os
import json
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
METADATA_FILE = "placement_model_metadata.pkl"
RULE_FILE = "recommendation_rules.pkl"

for key, default in {
    "prediction_result": None,
    "student_profile": None,
    "ai_career_advice": None,
    "ai_provider": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# BRANCH SKILLS
# ============================================================

def get_branch_skills(branch):
    skill_mapping = {
        "Computer Science": ["Programming", "Data Structures & Algorithms", "Databases", "Software Development", "Problem Solving"],
        "Information Technology": ["Programming", "Networking", "Databases", "Cloud Computing", "System Administration"],
        "Data Science": ["Python / Programming", "Statistics", "Data Analysis", "Machine Learning", "Data Visualization"],
        "Artificial Intelligence": ["Programming", "Machine Learning", "Deep Learning", "Mathematics", "Data Analysis"],
        "Machine Learning": ["Python", "Machine Learning", "Statistics", "Deep Learning", "Data Processing"],
        "Cyber Security": ["Networking", "Cyber Security Concepts", "Linux", "Security Tools", "Ethical Hacking"],
        "Software Engineering": ["Programming", "Software Design", "Databases", "Web Development", "Problem Solving"],
        "Computer Applications": ["Programming", "Databases", "Web Development", "Software Applications", "Problem Solving"],
        "Mechanical Engineering": ["CAD / Design", "Thermodynamics", "Manufacturing", "Machine Design", "Production Processes"],
        "Automobile Engineering": ["Automobile Systems", "CAD / Design", "Engine Technology", "Manufacturing", "Vehicle Diagnostics"],
        "Production Engineering": ["Manufacturing", "Production Planning", "Quality Control", "Industrial Processes", "CAD"],
        "Industrial Engineering": ["Operations Management", "Production Systems", "Quality Management", "Supply Chain", "Industrial Analysis"],
        "Aeronautical Engineering": ["Aerodynamics", "Aircraft Systems", "CAD", "Propulsion", "Manufacturing"],
        "Aerospace Engineering": ["Aerodynamics", "Aircraft Design", "Propulsion", "CAD", "Space Systems"],
        "Electrical Engineering": ["Circuit Analysis", "Power Systems", "Control Systems", "PLC / Automation", "Electrical Design"],
        "Electronics Engineering": ["Electronic Circuits", "Embedded Systems", "PCB Design", "Microcontrollers", "Instrumentation"],
        "Electronics and Communication Engineering": ["Communication Systems", "Embedded Systems", "Electronics", "Signal Processing", "VLSI"],
        "Biomedical Engineering": ["Biomedical Instrumentation", "Medical Devices", "Electronics", "Clinical Engineering", "Signal Processing"],
        "Civil Engineering": ["Structural Engineering", "AutoCAD", "Surveying", "Construction Management", "Quantity Estimation"],
        "Chemical Engineering": ["Chemical Processes", "Thermodynamics", "Process Engineering", "Plant Operations", "Industrial Safety"],
        "Mathematics": ["Mathematical Analysis", "Statistics", "Problem Solving", "Quantitative Reasoning", "Research Methods"],
        "Statistics": ["Statistical Analysis", "Probability", "Data Interpretation", "Research Methods", "Quantitative Analysis"],
        "Physics": ["Laboratory Techniques", "Instrumentation", "Electronics", "Scientific Analysis", "Research Methods"],
        "Chemistry": ["Analytical Chemistry", "Laboratory Techniques", "Chemical Analysis", "Instrumentation", "Research Methods"],
        "Environmental Science": ["Environmental Analysis", "Sustainability", "Environmental Monitoring", "Research Methods", "Data Analysis"],
        "Biotechnology": ["Laboratory Techniques", "Molecular Biology", "Biotechnology Methods", "Research Skills", "Scientific Analysis"],
        "Microbiology": ["Microbiology Techniques", "Laboratory Skills", "Culture Techniques", "Research Methods", "Scientific Analysis"],
        "Biochemistry": ["Biochemical Techniques", "Laboratory Skills", "Chemical Analysis", "Research Methods", "Scientific Analysis"],
        "Biological Sciences": ["Laboratory Skills", "Research Methods", "Scientific Analysis", "Biological Techniques", "Data Interpretation"],
        "Life Sciences": ["Laboratory Skills", "Research Methods", "Scientific Analysis", "Biological Techniques", "Data Interpretation"],
        "Genetics": ["Genetics", "Molecular Biology", "Laboratory Skills", "Research Methods", "Scientific Analysis"],
        "Botany": ["Plant Biology", "Laboratory Skills", "Research Methods", "Field Research", "Scientific Analysis"],
        "Zoology": ["Animal Biology", "Laboratory Skills", "Research Methods", "Field Research", "Scientific Analysis"],
        "Food Science and Nutrition": ["Nutrition Science", "Food Analysis", "Laboratory Skills", "Diet Planning", "Food Safety"],
        "Food Technology": ["Food Processing", "Food Safety", "Quality Control", "Laboratory Analysis", "Manufacturing"],
        "Nutrition and Dietetics": ["Clinical Nutrition", "Diet Planning", "Nutrition Assessment", "Food Science", "Communication"],
        "Economics": ["Economic Analysis", "Statistics", "Financial Analysis", "Research Methods", "Quantitative Analysis"],
        "Commerce": ["Accounting", "Taxation", "Financial Analysis", "Auditing", "Business Knowledge"],
        "Business Administration": ["Business Strategy", "Marketing", "Operations", "Management", "Business Analysis"],
        "Finance": ["Financial Analysis", "Accounting", "Investment Analysis", "Financial Modeling", "Banking Knowledge"],
        "Accounting": ["Accounting", "Taxation", "Auditing", "Financial Reporting", "Financial Analysis"],
        "Management": ["Leadership", "Operations", "Business Strategy", "Project Management", "Decision Making"],
        "Marketing": ["Marketing Strategy", "Digital Marketing", "Market Research", "Brand Management", "Consumer Analysis"],
        "Human Resources": ["Recruitment", "HR Operations", "Employee Relations", "Talent Management", "Organizational Skills"],
        "Psychology": ["Psychological Assessment", "Research Methods", "Counseling Skills", "Behavioral Analysis", "Data Interpretation"],
        "English": ["Writing", "Communication", "Editing", "Research", "Presentation Skills"],
        "Political Science": ["Political Analysis", "Research Methods", "Public Policy", "International Relations", "Communication"],
        "Sociology": ["Social Research", "Research Methods", "Data Analysis", "Community Studies", "Communication"],
        "History": ["Historical Research", "Research Methods", "Writing", "Analysis", "Documentation"],
        "Public Administration": ["Public Policy", "Administration", "Governance", "Research", "Management"],
    }
    return skill_mapping.get(
        branch,
        ["Core Domain Knowledge", "Practical Skills", "Research Skills", "Industry Knowledge", "Problem Solving"],
    )

# ============================================================
# CAREER INTERESTS
# ============================================================

def get_career_interests(branch):
    if branch in [
        "Computer Science", "Information Technology", "Data Science",
        "Artificial Intelligence", "Machine Learning", "Cyber Security",
        "Software Engineering", "Computer Applications",
    ]:
        return ["Software Development", "Data Science", "Artificial Intelligence", "Cyber Security", "Cloud Computing", "Data Analytics", "Other"]

    if branch in ["Mechanical Engineering", "Automobile Engineering", "Production Engineering", "Industrial Engineering"]:
        return ["Design Engineering", "Manufacturing", "Automotive Engineering", "Production Engineering", "Operations", "Other"]

    if branch in ["Electrical Engineering", "Electronics Engineering", "Electronics and Communication Engineering", "Biomedical Engineering"]:
        return ["Embedded Systems", "Electronics Design", "Power Systems", "Automation", "VLSI", "Instrumentation", "Other"]

    if branch == "Civil Engineering":
        return ["Structural Engineering", "Construction Management", "Site Engineering", "Infrastructure", "Surveying", "Other"]

    if branch == "Chemical Engineering":
        return ["Process Engineering", "Plant Operations", "Chemical Analysis", "Industrial Safety", "Research", "Other"]

    if branch in ["Biotechnology", "Microbiology", "Biochemistry", "Biological Sciences", "Life Sciences", "Genetics", "Botany", "Zoology"]:
        return ["Research", "Laboratory Work", "Biotechnology", "Pharmaceutical Industry", "Quality Control", "Other"]

    if branch in ["Food Science and Nutrition", "Food Technology", "Nutrition and Dietetics"]:
        return ["Food Industry", "Nutrition", "Quality Control", "Food Safety", "Research", "Clinical Nutrition", "Other"]

    if branch in ["Commerce", "Finance", "Accounting", "Economics"]:
        return ["Accounting", "Finance", "Banking", "Financial Analysis", "Auditing", "Business Analytics", "Other"]

    if branch in ["Business Administration", "Management", "Marketing", "Human Resources"]:
        return ["Management", "Marketing", "Human Resources", "Operations", "Business Analytics", "Finance", "Other"]

    return ["Research", "Teaching", "Government Sector", "Industry", "Higher Studies", "Other"]

# ============================================================
# MODEL COMPATIBILITY
# ============================================================

def map_branch_for_model(branch):
    if branch in [
        "Computer Science", "Computer Applications", "Software Engineering",
        "Artificial Intelligence", "Machine Learning", "Cyber Security",
    ]:
        return "CS"
    if branch == "Information Technology":
        return "IT"
    if branch == "Data Science":
        return "DS"
    if branch in [
        "Mechanical Engineering", "Automobile Engineering", "Production Engineering",
        "Industrial Engineering", "Aeronautical Engineering", "Aerospace Engineering",
    ]:
        return "Mechanical"
    if branch in [
        "Electrical Engineering", "Electronics Engineering",
        "Electronics and Communication Engineering", "Biomedical Engineering",
    ]:
        return "Electrical"
    if branch == "Civil Engineering":
        return "Civil"
    if branch == "Chemical Engineering":
        return "Chemical"
    return "Other"


def get_cgpa_category(cgpa):
    if cgpa < 6:
        return "Low"
    if cgpa < 7.5:
        return "Good"
    return "Excellent"


@st.cache_resource

def load_components():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing {MODEL_FILE}")
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Missing {FEATURE_FILE}")

    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_FILE)

    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns.tolist()
    elif isinstance(feature_names, pd.Series):
        feature_names = feature_names.tolist()
    elif isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    elif isinstance(feature_names, dict):
        feature_names = feature_names.get("feature_names", feature_names.get("features", list(feature_names.keys())))

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

    return model, feature_names, rules, metadata


def get_model_feature_names(model, artifact_features):
    """Use the features the fitted model actually expects.

    The feature-name artifact in this project is stale and contains fields such as
    gender_Female, degree_BCA and branch_AI that are NOT present in the trained
    placement model. The fitted estimator is therefore the source of truth.
    """
    model_names = getattr(model, "feature_names_in_", None)

    if model_names is not None:
        names = [str(x) for x in model_names]
        if names:
            return names

    # For a Pipeline, inspect the fitted steps from the end backwards.
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for _, step in reversed(list(named_steps.items())):
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                names = [str(x) for x in names]
                if names:
                    return names

    steps = getattr(model, "steps", None)
    if steps:
        for _, step in reversed(steps):
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                names = [str(x) for x in names]
                if names:
                    return names

    # This is the exact feature schema used to train the supplied model.
    return [
        "age",
        "cgpa",
        "backlogs",
        "internships",
        "certifications",
        "coding_skills",
        "communication_skills",
        "aptitude_score",
        "projects",
        "gender_Male",
        "degree_BE",
        "degree_BSc",
        "degree_BTech",
        "branch_CS",
        "branch_DS",
        "branch_Electrical",
        "branch_IT",
        "branch_Mechanical",
        "cgpa_category_Excellent",
        "cgpa_category_Good",
        "cgpa_category_Low",
    ]


def build_exact_model_input(student, feature_names, model=None):
    """Build input using the fitted model's real schema.

    IMPORTANT: do not use unsupported/stale feature-artifact columns. The supplied
    artifact may contain gender_Female, degree_BCA and branch_AI, while the trained
    model was fitted with only the 21 columns below.
    """
    if model is not None:
        expected_features = get_model_feature_names(model, feature_names)
    else:
        expected_features = list(feature_names)

    model_branch = map_branch_for_model(student["ug_branch"])
    degree = student["ug_degree"]
    gender = student["gender"]
    category = get_cgpa_category(student["ug_cgpa"])

    values = {
        "age": float(student["age"]),
        "cgpa": float(student["ug_cgpa"]),
        "backlogs": float(student["backlogs"]),
        "internships": float(student["internships"]),
        "certifications": float(student["certifications"]),
        "coding_skills": float(student["coding_skills"]),
        "communication_skills": float(student["communication_skills"]),
        "aptitude_score": float(student["aptitude_score"]),
        "projects": float(student["projects"]),
        "gender_Male": 1.0 if gender == "Male" else 0.0,
        "degree_BE": 1.0 if degree == "BE" else 0.0,
        "degree_BSc": 1.0 if degree == "BSc" else 0.0,
        "degree_BTech": 1.0 if degree == "BTech" else 0.0,
        "branch_CS": 1.0 if model_branch == "CS" else 0.0,
        "branch_DS": 1.0 if model_branch == "DS" else 0.0,
        "branch_Electrical": 1.0 if model_branch == "Electrical" else 0.0,
        "branch_IT": 1.0 if model_branch == "IT" else 0.0,
        "branch_Mechanical": 1.0 if model_branch == "Mechanical" else 0.0,
        "cgpa_category_Excellent": 1.0 if category == "Excellent" else 0.0,
        "cgpa_category_Good": 1.0 if category == "Good" else 0.0,
        "cgpa_category_Low": 1.0 if category == "Low" else 0.0,
    }

    unsupported = [f for f in expected_features if f not in values]
    if unsupported:
        raise RuntimeError(
            "The trained model expects feature(s) this app cannot construct: "
            + ", ".join(unsupported)
            + ". Please use the matching model artifact."
        )

    # Exact order expected by the fitted estimator.
    return pd.DataFrame(
        [{f: values[f] for f in expected_features}],
        columns=expected_features,
    )

def get_classes(model):
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return list(classes)

    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for _, step in reversed(list(named_steps.items())):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                return list(classes)

    steps = getattr(model, "steps", None)
    if steps:
        for _, step in reversed(steps):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                return list(classes)

    return None


def get_positive_probability(model, model_input):
    if not hasattr(model, "predict_proba"):
        return None

    probs = np.asarray(model.predict_proba(model_input), dtype=float)
    if probs.ndim != 2:
        return None

    classes = get_classes(model)
    if classes is not None:
        for target in [1, True, "1", "Placed", "PLACED", "Yes", "YES"]:
            if target in classes:
                return float(probs[0][classes.index(target)])

    if probs.shape[1] == 2:
        return float(probs[0][1])
    return float(probs[0][0])

# ============================================================
# PROFILE READINESS — NOT THE ML PROBABILITY
# ============================================================

def calculate_profile_readiness(student):
    """Transparent heuristic index that reacts to all placement-relevant inputs.
    This is deliberately NOT called a placement probability.
    """
    cgpa_score = np.clip(student["ug_cgpa"] / 10.0, 0, 1) * 100
    backlog_score = max(0.0, 100.0 - min(student["backlogs"], 10) * 12.0)
    internship_score = min(student["internships"], 3) / 3 * 100
    project_score = min(student["projects"], 5) / 5 * 100
    certification_score = min(student["certifications"], 5) / 5 * 100
    coding_score = student["coding_skills"] / 10 * 100
    communication_score = student["communication_skills"] / 10 * 100
    aptitude_score = student["aptitude_score"]
    branch_skill_score = np.mean(list(student["branch_skills"].values())) / 10 * 100

    score = (
        cgpa_score * 0.18
        + backlog_score * 0.12
        + internship_score * 0.12
        + project_score * 0.10
        + certification_score * 0.08
        + coding_score * 0.08
        + communication_score * 0.10
        + aptitude_score * 0.10
        + branch_skill_score * 0.12
    )
    return round(float(np.clip(score, 0, 100)), 1)

# ============================================================
# PROFILE FEEDBACK
# ============================================================

def get_profile_strengths(student):
    strengths = []
    if student["ug_cgpa"] >= 8:
        strengths.append(f"Strong academic performance with a CGPA of {student['ug_cgpa']:.2f}.")
    if student["backlogs"] == 0:
        strengths.append("No current academic backlogs.")
    if student["internships"] >= 1:
        strengths.append(f"{student['internships']} internship(s) provide practical exposure.")
    if student["projects"] >= 2:
        strengths.append(f"{student['projects']} projects demonstrate practical experience.")
    if student["certifications"] >= 2:
        strengths.append(f"{student['certifications']} certifications show continued learning.")
    if student["communication_skills"] >= 7:
        strengths.append("Good communication skill level.")
    if student["aptitude_score"] >= 75:
        strengths.append("Strong aptitude performance.")
    strongest = sorted(student["branch_skills"].items(), key=lambda x: x[1], reverse=True)[:2]
    strengths.extend([f"Strong branch skill: {name} ({score}/10)." for name, score in strongest if score >= 7])
    return strengths[:6] or ["The profile provides a foundation that can be strengthened through focused preparation."]


def get_profile_gaps(student):
    gaps = []
    if student["ug_cgpa"] < 7:
        gaps.append("Academic performance")
    if student["backlogs"] > 0:
        gaps.append("Backlog clearance")
    if student["internships"] == 0:
        gaps.append("Industry/internship exposure")
    if student["projects"] < 2:
        gaps.append("Practical project experience")
    if student["certifications"] == 0:
        gaps.append("Relevant certifications")
    if student["communication_skills"] < 6:
        gaps.append("Communication and interview skills")
    if student["aptitude_score"] < 60:
        gaps.append("Aptitude preparation")
    for skill, score in sorted(student["branch_skills"].items(), key=lambda x: x[1]):
        if score <= 4:
            gaps.append(f"{skill}")
    return gaps[:8]


def generate_recommendations(student, rules):
    fallback = {
        "Academic performance": "Focus on improving academic performance and maintaining a consistent CGPA.",
        "Backlog clearance": "Prioritize clearing academic backlogs because they can affect eligibility for some opportunities.",
        "Industry/internship exposure": "Seek a relevant internship, industry project, research project or supervised practical experience.",
        "Practical project experience": "Build branch-specific projects that demonstrate practical application of your knowledge.",
        "Relevant certifications": "Consider certifications that directly support your chosen career direction.",
        "Communication and interview skills": "Practice structured answers, presentations, group discussions and mock interviews.",
        "Aptitude preparation": "Practice quantitative aptitude, logical reasoning and verbal reasoning regularly.",
    }
    rule_key = {
        "Academic performance": "cgpa",
        "Backlog clearance": "backlogs",
        "Industry/internship exposure": "internships",
        "Practical project experience": "projects",
        "Relevant certifications": "certifications",
        "Communication and interview skills": "communication_skills",
        "Aptitude preparation": "aptitude_score",
    }
    output = []
    for gap in get_profile_gaps(student):
        key = rule_key.get(gap)
        if isinstance(rules, dict) and key in rules:
            item = rules[key]
            message = item.get("message") if isinstance(item, dict) else str(item)
            if message and message not in output:
                output.append(message)
        elif gap in fallback:
            output.append(fallback[gap])
        elif gap not in output:
            output.append(f"Improve {gap} through structured practice and branch-relevant projects.")
    return output[:6]

# ============================================================
# PREDICTION
# ============================================================

def run_prediction(student):
    model, feature_names, rules, metadata = load_components()
    model_input = build_exact_model_input(student, feature_names, model)

    prediction = model.predict(model_input)[0]
    probability = get_positive_probability(model, model_input)
    placed = str(prediction).strip().lower() in {"1", "true", "placed", "yes"} or prediction == 1

    return {
        "prediction": prediction,
        "prediction_text": "LIKELY PLACED" if placed else "NOT PLACED",
        "probability": probability,
        "profile_readiness": calculate_profile_readiness(student),
        "model_input": model_input,
        "feature_names": feature_names,
        "metadata": metadata,
        "strengths": get_profile_strengths(student),
        "gaps": get_profile_gaps(student),
        "recommendations": generate_recommendations(student, rules),
    }

# ============================================================
# AI
# ============================================================

def get_secret(name):
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.getenv(name, "")
    return str(value).strip() if value else ""


def get_ai_configuration_status():
    return {
        "Groq": bool(get_secret("GROQ_API_KEY")),
        "Gemini": bool(get_secret("GEMINI_API_KEY")),
    }


def build_ai_prompt(student, prediction_context=None):
    skills = "\n".join(f"- {k}: {v}/10" for k, v in student["branch_skills"].items())
    critical = ", ".join(f"{k} ({v}/10)" for k, v in student["branch_skills"].items() if v <= 4) or "None identified"
    prediction_context = prediction_context or "Placement prediction was not requested."

    return f"""
You are an expert student career counselor and employability advisor.
Give practical, realistic, personalized guidance for the student's actual academic domain.
Do not guarantee employment. Do not invent achievements. If a target career differs from the branch, explain the bridge skills.
Give exactly 3 branch-appropriate project ideas.

STUDENT PROFILE
Age: {student['age']}
Gender: {student['gender']}
UG Degree: {student['ug_degree']}
UG Branch: {student['ug_branch']}
UG CGPA: {student['ug_cgpa']}
PG: {student['pg_degree'] if student['has_pg'] else 'Not Applicable'}
PG Specialization: {student['pg_branch'] if student['has_pg'] else 'Not Applicable'}
PG CGPA: {student['pg_cgpa'] if student['has_pg'] else 'Not Applicable'}
Backlogs: {student['backlogs']}
Internships: {student['internships']}
Projects: {student['projects']}
Certifications: {student['certifications']}
Communication: {student['communication_skills']}/10
Aptitude: {student['aptitude_score']}/100
Coding/Computational Skills: {student['coding_skills']}/10
Career Interest: {student['career_interest']}
Target Career: {student['target_career_goal'] or 'Not specified'}

BRANCH-SPECIFIC SKILLS
{skills}

CRITICAL BRANCH SKILL GAPS
{critical}

PLACEMENT CONTEXT
{prediction_context}

Respond using exactly these headings:
## 1. Overall Profile Assessment
## 2. Career Direction
## 3. Recommended Career Paths
## 4. Top Strengths
## 5. Skill Gap Analysis
## 6. Areas to Improve
## 7. 30-Day Improvement Plan
## 8. Technical Topics to Study
## 9. Industry Tools and Professional Skills
## 10. Project Ideas
## 11. Interview Preparation

Under Project Ideas provide EXACTLY 3 projects. For each include title, what to build/do, skills demonstrated, and why it is relevant.
"""


def generate_with_groq(prompt):
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are an expert student career counselor and employability advisor."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 3000,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; AI-Student-Placement-Predictor/1.0)",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            raise RuntimeError("Groq authentication failed (HTTP 401). Check that the complete current gsk_... key is in Streamlit Secrets.") from exc
        if exc.code == 403:
            raise RuntimeError(f"Groq access denied (HTTP 403). The provider rejected the request. Details: {details}") from exc
        if exc.code == 429:
            raise RuntimeError(f"Groq rate limit/quota reached (HTTP 429). Details: {details}") from exc
        raise RuntimeError(f"Groq API error (HTTP {exc.code}): {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to Groq: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq returned an invalid response.") from exc

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Groq returned an unexpected response: {body}") from exc

    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return str(text).strip()


def generate_with_gemini(prompt):
    from google import genai

    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=key)
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        store=False,
    )

    text = getattr(interaction, "output_text", None)
    if text:
        return str(text).strip()

    output = getattr(interaction, "output", None)
    if output:
        parts = output if isinstance(output, list) else [output]
        collected = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(str(part_text))
            content = getattr(part, "content", None)
            if content:
                items = content if isinstance(content, list) else [content]
                for item in items:
                    item_text = getattr(item, "text", None)
                    if item_text:
                        collected.append(str(item_text))
        if collected:
            return "\n".join(collected).strip()

    raise RuntimeError("Gemini returned an empty response.")


def generate_ai_guidance(student, prediction_context=None):
    prompt = build_ai_prompt(student, prediction_context)
    errors = []

    # Groq primary
    if get_secret("GROQ_API_KEY"):
        try:
            return generate_with_groq(prompt), "Groq"
        except Exception as exc:
            errors.append(f"Groq: {exc}")
    else:
        errors.append("Groq: GROQ_API_KEY is not configured.")

    # Gemini fallback
    if get_secret("GEMINI_API_KEY"):
        try:
            return generate_with_gemini(prompt), "Gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
    else:
        errors.append("Gemini: GEMINI_API_KEY is not configured.")

    raise RuntimeError("AI providers could not generate guidance.\n\n" + "\n".join(errors))

# ============================================================
# STUDENT PROFILE
# ============================================================

def build_student_profile(
    gender, age, ug_degree, ug_branch, ug_cgpa, has_pg, pg_degree,
    pg_branch, pg_cgpa, backlogs, internships, projects, certifications,
    coding_skills, communication_skills, aptitude_score, career_interest,
    target_career_goal, domain_scores,
):
    return {
        "gender": gender,
        "age": int(age),
        "ug_degree": ug_degree,
        "degree": ug_degree,
        "ug_branch": ug_branch,
        "branch": map_branch_for_model(ug_branch),
        "ug_cgpa": float(ug_cgpa),
        "cgpa": float(ug_cgpa),
        "has_pg": bool(has_pg),
        "pg_degree": pg_degree,
        "pg_branch": pg_branch,
        "pg_cgpa": float(pg_cgpa) if has_pg else None,
        "backlogs": int(backlogs),
        "internships": int(internships),
        "projects": int(projects),
        "certifications": int(certifications),
        "coding_skills": int(coding_skills),
        "communication_skills": int(communication_skills),
        "aptitude_score": int(aptitude_score),
        "career_interest": career_interest,
        "target_career_goal": target_career_goal.strip(),
        "branch_skills": domain_scores,
        "domain_skills": round(float(np.mean(list(domain_scores.values()))), 2),
    }

# ============================================================
# UI
# ============================================================

st.title("🎓 AI Student Placement Predictor")
st.write("A transparent ML placement assessment plus field-specific AI career guidance.")
st.info("⚠️ The supplied placement model was trained on a synthetic, highly separable dataset. Its probability should not be interpreted as a real-world employment probability or guarantee.")

with st.sidebar:
    st.header("🔧 System Status")
    try:
        _, feature_names, _, metadata = load_components()
        st.success("ML model loaded")
        st.write(f"Model features: **{len(feature_names)}**")
        if metadata:
            st.write(f"Model: **{metadata.get('model_name', 'Unknown')}**")
            st.write(f"Dataset size: **{metadata.get('dataset_size', 'Unknown')}**")
            if metadata.get("dataset_note"):
                st.caption(metadata["dataset_note"])
    except Exception as exc:
        st.error("ML model files could not be loaded.")
        st.code(str(exc))

    ai_status = get_ai_configuration_status()
    st.divider()
    st.write("**AI providers**")
    st.write(f"Groq: **{'Configured' if ai_status['Groq'] else 'Not configured'}**")
    st.write(f"Gemini fallback: **{'Configured' if ai_status['Gemini'] else 'Not configured'}**")
    st.caption("API keys are read from Streamlit Secrets/environment variables and are never displayed.")

st.divider()
mode = st.radio("Choose Analysis Mode", ["🔮 Placement Prediction", "🤖 AI Career Guidance"], horizontal=True)

st.header("📋 Student Profile")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", 16, 60, 22)
with c2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with c3:
    backlogs = st.number_input("Backlogs", 0, 20, 0)

st.subheader("🎓 Education")
ug_degrees = ["BE", "BTech", "BSc", "BCA", "BBA", "BCom", "BA", "Other"]
branches = [
    "Computer Science", "Information Technology", "Data Science", "Artificial Intelligence", "Machine Learning", "Cyber Security", "Software Engineering", "Computer Applications",
    "Mechanical Engineering", "Automobile Engineering", "Production Engineering", "Industrial Engineering", "Aeronautical Engineering", "Aerospace Engineering", "Electrical Engineering", "Electronics Engineering", "Electronics and Communication Engineering", "Biomedical Engineering", "Civil Engineering", "Chemical Engineering",
    "Mathematics", "Statistics", "Physics", "Chemistry", "Environmental Science", "Biotechnology", "Microbiology", "Biochemistry", "Biological Sciences", "Life Sciences", "Genetics", "Botany", "Zoology",
    "Food Science and Nutrition", "Food Technology", "Nutrition and Dietetics", "Economics", "Commerce", "Business Administration", "Finance", "Accounting", "Management", "Marketing", "Human Resources", "Psychology", "English", "Political Science", "Sociology", "History", "Public Administration", "Other",
]

c1, c2, c3 = st.columns(3)
with c1:
    ug_degree = st.selectbox("UG Degree", ug_degrees)
with c2:
    ug_branch = st.selectbox("UG Branch / Major", branches)
with c3:
    ug_cgpa = st.number_input("UG CGPA", 0.0, 10.0, 7.0, 0.1)

has_pg = st.checkbox("I have postgraduate education")
pg_degree = "Not Applicable"
pg_branch = "Not Applicable"
pg_cgpa = 0.0
if has_pg:
    c1, c2, c3 = st.columns(3)
    with c1:
        pg_degree = st.selectbox("PG Degree", ["MTech", "MSc", "MCA", "MBA", "MCom", "MA", "Other"])
    with c2:
        pg_branch = st.text_input("PG Specialization", placeholder="Example: Data Science")
    with c3:
        pg_cgpa = st.number_input("PG CGPA", 0.0, 10.0, 7.0, 0.1)

st.subheader("💼 Placement Profile")
c1, c2, c3, c4 = st.columns(4)
with c1:
    internships = st.number_input("Internships", 0, 20, 1)
with c2:
    projects = st.number_input("Projects", 0, 30, 2)
with c3:
    certifications = st.number_input("Certifications", 0, 30, 2)
with c4:
    communication_skills = st.slider("Communication Skills", 1, 10, 5)

st.subheader("🧭 Career Direction")
c1, c2 = st.columns(2)
with c1:
    career_interest = st.selectbox("Career Interest", get_career_interests(ug_branch))
with c2:
    target_career_goal = st.text_input("Target Career Goal", placeholder="Example: Data Analyst, Food Safety Officer, Financial Analyst")

st.subheader(f"🧠 {ug_branch} Skills")
st.caption("These branch-specific skills are used by the AI career guidance and the transparent readiness index. They are NOT features of the supplied ML model.")
branch_skill_names = get_branch_skills(ug_branch)
domain_scores = {}
skill_cols = st.columns(min(3, len(branch_skill_names)))
for index, skill in enumerate(branch_skill_names):
    with skill_cols[index % len(skill_cols)]:
        domain_scores[skill] = st.slider(skill, 1, 10, 5, key=f"skill_{ug_branch}_{skill}")

if ug_branch in [
    "Computer Science", "Information Technology", "Data Science", "Artificial Intelligence", "Machine Learning", "Cyber Security", "Software Engineering", "Computer Applications",
]:
    relevant_scores = [
        score for skill, score in domain_scores.items()
        if any(keyword in skill for keyword in ["Programming", "Software", "Problem Solving", "Python"])
    ]
    coding_skills = int(round(np.mean(relevant_scores or [5])))
    aptitude_score = st.slider("Aptitude Score", 0, 100, 60)
else:
    coding_skills = st.slider("Computational / Analytical Skills", 1, 10, 5)
    aptitude_score = st.slider("Aptitude Score", 0, 100, 60)

student = build_student_profile(
    gender, age, ug_degree, ug_branch, ug_cgpa, has_pg, pg_degree,
    pg_branch, pg_cgpa, backlogs, internships, projects, certifications,
    coding_skills, communication_skills, aptitude_score, career_interest,
    target_career_goal, domain_scores,
)

st.divider()

# ============================================================
# ACTIONS
# ============================================================

if mode == "🔮 Placement Prediction":
    if st.button("🔮 Predict Placement", use_container_width=True, type="primary"):
        try:
            with st.spinner("Running the trained placement model..."):
                result = run_prediction(student)
            st.session_state.prediction_result = result
            st.session_state.student_profile = student
            st.session_state.ai_career_advice = None
            st.session_state.ai_provider = None
        except Exception as exc:
            st.session_state.prediction_result = None
            st.error("Prediction failed.")
            st.code(str(exc))
else:
    if st.button("🤖 Generate AI Career Guidance", use_container_width=True, type="primary"):
        st.session_state.student_profile = student
        st.session_state.ai_career_advice = None
        try:
            with st.spinner("Generating personalized career guidance..."):
                advice, provider = generate_ai_guidance(student)
            st.session_state.ai_career_advice = advice
            st.session_state.ai_provider = provider
        except Exception as exc:
            st.error("AI guidance failed.")
            st.code(str(exc))

# ============================================================
# PLACEMENT RESULT
# ============================================================

result = st.session_state.prediction_result
if result is not None:
    st.divider()
    st.header("🎯 Placement Prediction Result")

    probability = result["probability"]
    probability_pct = probability * 100 if probability is not None else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model Prediction", result["prediction_text"])
    with c2:
        st.metric("Placement Probability", f"{probability_pct:.2f}%" if probability_pct is not None else "N/A")
    with c3:
        st.metric("Profile Readiness", f"{result['profile_readiness']:.1f}/100")

    if probability_pct is not None:
        st.progress(float(np.clip(probability, 0, 1)))
        st.caption("Estimated probability from the trained placement model.")

    st.subheader("💪 Your Strengths")
    for item in result["strengths"]:
        st.success(item)

    st.subheader("📈 Areas to Improve")
    if result["gaps"]:
        for item in result["gaps"]:
            st.warning(item)
    else:
        st.success("No major gaps were identified from the supplied profile.")

    st.subheader("💡 Personalized Recommendations")
    for i, item in enumerate(result["recommendations"], 1):
        st.write(f"**{i}.** {item}")

    st.divider()
    st.subheader("🤖 Personalized AI Career Guidance")
    st.caption("Groq is tried first. Gemini is used automatically only if Groq is unavailable or fails.")

    if st.button("🧠 Generate Personalized AI Guidance", use_container_width=True):
        try:
            context = (
                f"ML model prediction: {result['prediction_text']}\n"
                f"ML estimated probability: {probability_pct:.2f}%\n"
                f"Profile readiness index: {result['profile_readiness']:.1f}/100"
                if probability_pct is not None
                else f"ML model prediction: {result['prediction_text']}\nProbability unavailable.\nProfile readiness index: {result['profile_readiness']:.1f}/100"
            )
            with st.spinner("Generating personalized AI guidance..."):
                advice, provider = generate_ai_guidance(student, context)
            st.session_state.ai_career_advice = advice
            st.session_state.ai_provider = provider
        except Exception as exc:
            st.error("AI guidance failed.")
            st.code(str(exc))

# ============================================================
# AI RESULT
# ============================================================

if st.session_state.ai_career_advice:
    st.divider()
    provider = st.session_state.ai_provider or "AI"
    st.header(f"🧠 Personalized AI Career Guidance ({provider})")
    st.markdown(st.session_state.ai_career_advice)

st.divider()
st.caption("AI Student Placement Predictor | Transparent ML Assessment + Branch-Specific AI Career Guidance")
