
import os
import json
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
)

MODEL_FILE = "placement_prediction_final.pkl"
FEATURE_FILE = "placement_feature_names_final.pkl"
RULE_FILE = "recommendation_rules.pkl"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "student_profile" not in st.session_state:
    st.session_state.student_profile = None
if "ai_career_advice" not in st.session_state:
    st.session_state.ai_career_advice = None
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = None


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


def get_career_interests(branch):
    if branch in [
        "Computer Science", "Information Technology", "Data Science",
        "Artificial Intelligence", "Machine Learning", "Cyber Security",
        "Software Engineering", "Computer Applications",
    ]:
        return [
            "Software Development", "Data Science", "Artificial Intelligence",
            "Cyber Security", "Cloud Computing", "Data Analytics", "Other",
        ]

    if branch in [
        "Mechanical Engineering", "Automobile Engineering",
        "Production Engineering", "Industrial Engineering",
    ]:
        return [
            "Design Engineering", "Manufacturing", "Automotive Engineering",
            "Production Engineering", "Operations", "Other",
        ]

    if branch in [
        "Electrical Engineering", "Electronics Engineering",
        "Electronics and Communication Engineering", "Biomedical Engineering",
    ]:
        return [
            "Embedded Systems", "Electronics Design", "Power Systems",
            "Automation", "VLSI", "Instrumentation", "Other",
        ]

    if branch == "Civil Engineering":
        return [
            "Structural Engineering", "Construction Management",
            "Site Engineering", "Infrastructure", "Surveying", "Other",
        ]

    if branch == "Chemical Engineering":
        return [
            "Process Engineering", "Plant Operations", "Chemical Analysis",
            "Industrial Safety", "Research", "Other",
        ]

    if branch in [
        "Biotechnology", "Microbiology", "Biochemistry",
        "Biological Sciences", "Life Sciences", "Genetics", "Botany", "Zoology",
    ]:
        return [
            "Research", "Laboratory Work", "Biotechnology",
            "Pharmaceutical Industry", "Quality Control", "Other",
        ]

    if branch in [
        "Food Science and Nutrition", "Food Technology",
        "Nutrition and Dietetics",
    ]:
        return [
            "Food Industry", "Nutrition", "Quality Control",
            "Food Safety", "Research", "Clinical Nutrition", "Other",
        ]

    if branch in ["Commerce", "Finance", "Accounting", "Economics"]:
        return [
            "Accounting", "Finance", "Banking", "Financial Analysis",
            "Auditing", "Business Analytics", "Other",
        ]

    if branch in [
        "Business Administration", "Management",
        "Marketing", "Human Resources",
    ]:
        return [
            "Management", "Marketing", "Human Resources",
            "Operations", "Business Analytics", "Finance", "Other",
        ]

    return [
        "Research", "Teaching", "Government Sector",
        "Industry", "Higher Studies", "Other",
    ]


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
        "Mechanical Engineering", "Automobile Engineering",
        "Production Engineering", "Industrial Engineering",
        "Aeronautical Engineering", "Aerospace Engineering",
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
    model = joblib.load(MODEL_FILE)
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
            feature_names.get("features", list(feature_names.keys())),
        )

    feature_names = [str(x) for x in feature_names]

    rules = {}

    if os.path.exists(RULE_FILE):
        try:
            rules = joblib.load(RULE_FILE)
        except Exception:
            rules = {}

    return model, feature_names, rules


RAW_FEATURE_NAMES = {
    "gender", "age", "degree", "branch", "cgpa",
    "backlogs", "internships", "certifications",
    "coding_skills", "communication_skills",
    "aptitude_score", "projects", "cgpa_category",
}


def get_model_expected_features(model):
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return [str(x) for x in names]

    named_steps = getattr(model, "named_steps", None)

    if named_steps:
        for _, step in named_steps.items():
            names = getattr(step, "feature_names_in_", None)

            if names is not None:
                return [str(x) for x in names]

    steps = getattr(model, "steps", None)

    if steps:
        for _, step in steps:
            names = getattr(step, "feature_names_in_", None)

            if names is not None:
                return [str(x) for x in names]

    return None


def build_raw_model_dataframe(student):
    raw = {
        "gender": student["gender"],
        "age": student["age"],
        "degree": student["ug_degree"],
        "branch": map_branch_for_model(student["ug_branch"]),
        "cgpa": student["ug_cgpa"],
        "backlogs": student["backlogs"],
        "internships": student["internships"],
        "certifications": student["certifications"],
        "coding_skills": student["coding_skills"],
        "communication_skills": student["communication_skills"],
        "aptitude_score": student["aptitude_score"],
        "projects": student["projects"],
    }

    df = pd.DataFrame([raw])
    df["cgpa_category"] = df["cgpa"].apply(get_cgpa_category)

    return df


def build_encoded_candidates(student):
    base = build_raw_model_dataframe(student)
    variants = [base.copy()]

    original = base.copy()
    original["branch"] = student["ug_branch"]
    variants.append(original)

    aliases = {
        "Artificial Intelligence": "AI",
        "Computer Science": "CS",
        "Information Technology": "IT",
        "Data Science": "DS",
        "Machine Learning": "ML",
        "Computer Applications": "BCA",
        "Electronics and Communication Engineering": "ECE",
        "Mechanical Engineering": "Mechanical",
        "Electrical Engineering": "Electrical",
        "Civil Engineering": "Civil",
        "Chemical Engineering": "Chemical",
    }

    abbreviated = base.copy()
    abbreviated["branch"] = aliases.get(
        student["ug_branch"],
        student["ug_branch"],
    )
    variants.append(abbreviated)

    degree_aliases = {
        "BTech": ["BTech", "B.Tech", "B.Tech."],
        "BE": ["BE", "B.E", "B.E."],
        "BSc": ["BSc", "B.Sc", "B.Sc."],
        "BCA": ["BCA"],
        "BBA": ["BBA"],
        "BCom": ["BCom", "B.Com", "B.Com."],
        "BA": ["BA", "B.A", "B.A."],
    }

    for degree in degree_aliases.get(
        student["ug_degree"],
        [student["ug_degree"]],
    ):
        degree_variant = base.copy()
        degree_variant["degree"] = degree
        variants.append(degree_variant)

    return [
        pd.get_dummies(
            variant,
            columns=["gender", "degree", "branch", "cgpa_category"],
            dtype=int,
        )
        for variant in variants
    ]


def align_encoded_dataframe(encoded_df, expected_features):
    aligned = pd.DataFrame(
        0.0,
        index=[0],
        columns=expected_features,
    )

    for column in encoded_df.columns:
        if str(column) in aligned.columns:
            aligned.loc[0, str(column)] = encoded_df.iloc[0][column]

    return aligned


def prepare_model_input(model, student, artifact_features):
    raw_df = build_raw_model_dataframe(student)
    expected = get_model_expected_features(model)

    if expected:
        overlap = set(expected).intersection(RAW_FEATURE_NAMES)

        if len(overlap) >= 3 and all(
            column in raw_df.columns
            for column in expected
        ):
            return raw_df[expected], "raw"

        candidates = build_encoded_candidates(student)

        best = max(
            candidates,
            key=lambda x: len(
                set(x.columns).intersection(expected)
            ),
        )

        return (
            align_encoded_dataframe(
                best,
                expected,
            ),
            "encoded_model_features",
        )

    candidates = build_encoded_candidates(student)

    return (
        align_encoded_dataframe(
            candidates[0],
            artifact_features,
        ),
        "encoded_artifact_features",
    )


def get_positive_probability(model, model_input):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = np.asarray(
        model.predict_proba(model_input)
    )

    if probabilities.ndim != 2:
        return None

    classes = getattr(model, "classes_", None)

    if classes is None:
        named_steps = getattr(
            model,
            "named_steps",
            None,
        )

        if named_steps:
            for _, step in reversed(
                list(named_steps.items())
            ):
                classes = getattr(
                    step,
                    "classes_",
                    None,
                )

                if classes is not None:
                    break

    if classes is not None:
        classes = list(classes)

        if 1 in classes:
            return float(
                probabilities[0][classes.index(1)]
            )

        for target in [
            "1",
            "Placed",
            "PLACED",
            "Yes",
            "YES",
            True,
        ]:
            if target in classes:
                return float(
                    probabilities[0][classes.index(target)]
                )

    if probabilities.shape[1] == 2:
        return float(probabilities[0][1])

    return float(probabilities[0][0])


def get_profile_strengths(student):
    strengths = []

    if student["ug_cgpa"] >= 8:
        strengths.append(
            f"Strong academic performance with a CGPA of {student['ug_cgpa']:.2f}."
        )

    if student["backlogs"] == 0:
        strengths.append("No current academic backlogs.")

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

    return strengths or [
        "The profile provides a foundation that can be strengthened through focused preparation."
    ]


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

    if student["coding_skills"] < 6:
        gaps.append("Technical/computational skills")

    if student["communication_skills"] < 6:
        gaps.append("Communication and interview skills")

    if student["aptitude_score"] < 60:
        gaps.append("Aptitude preparation")

    return gaps


def generate_recommendations(student, rules):
    fallback = {
        "Technical/computational skills":
            "Strengthen technical skills most relevant to your branch and target career.",
        "Communication and interview skills":
            "Practice structured answers, presentations, group discussions and mock interviews.",
        "Aptitude preparation":
            "Practice quantitative aptitude, logical reasoning and verbal reasoning regularly.",
        "Academic performance":
            "Focus on improving academic performance and maintaining a consistent CGPA.",
        "Backlog clearance":
            "Prioritize clearing academic backlogs because they can affect eligibility for some opportunities.",
        "Industry/internship exposure":
            "Seek a relevant internship, industry project, research project or supervised practical experience.",
        "Practical project experience":
            "Build branch-specific projects that demonstrate practical application of your knowledge.",
        "Relevant certifications":
            "Consider certifications that directly support your chosen career direction.",
    }

    rule_key = {
        "Technical/computational skills": "coding_skills",
        "Communication and interview skills": "communication_skills",
        "Aptitude preparation": "aptitude_score",
        "Academic performance": "cgpa",
        "Backlog clearance": "backlogs",
        "Industry/internship exposure": "internships",
        "Practical project experience": "projects",
        "Relevant certifications": "certifications",
    }

    output = []

    for gap in get_profile_gaps(student):
        key = rule_key.get(gap)

        if isinstance(rules, dict) and key in rules:
            item = rules[key]

            if isinstance(item, dict):
                message = item.get("message")
            else:
                message = str(item)

            if message:
                output.append(message)

        if gap not in output and fallback.get(gap):
            output.append(fallback[gap])

    return output[:5]


def run_prediction(student):
    model, feature_names, rules = load_components()

    model_input, input_type = prepare_model_input(
        model,
        student,
        feature_names,
    )

    prediction = model.predict(model_input)[0]

    probability = get_positive_probability(
        model,
        model_input,
    )

    prediction_text = (
        "LIKELY PLACED"
        if (
            str(prediction) in {
                "1",
                "True",
                "Placed",
                "PLACED",
            }
            or prediction == 1
        )
        else "NOT PLACED"
    )

    return {
        "prediction": prediction,
        "prediction_text": prediction_text,
        "probability": probability,
        "model_input_type": input_type,
        "strengths": get_profile_strengths(student),
        "gaps": get_profile_gaps(student),
        "recommendations": generate_recommendations(
            student,
            rules,
        ),
    }


def build_ai_prompt(student, prediction_context=None):
    skill_lines = "\n".join(
        f"- {skill}: {score}/10"
        for skill, score
        in student["branch_skills"].items()
    )

    critical_gaps = [
        f"{skill} ({score}/10)"
        for skill, score
        in student["branch_skills"].items()
        if score <= 4
    ]

    developing = [
        f"{skill} ({score}/10)"
        for skill, score
        in student["branch_skills"].items()
        if 5 <= score <= 6
    ]

    prediction_context = (
        prediction_context
        or
        "Placement prediction was not requested. Give career guidance from the profile."
    )

    return f"""
You are an expert student career counselor and employability advisor.

Give practical, realistic, personalized guidance.

Rules:
- The student's actual academic branch is the primary domain.
- Career interest and target career goal are important.
- Consider UG and PG education.
- Do not assume every student is a Computer Science student.
- Do not recommend software-development projects to non-software students unless their goal specifically requires it.
- Give exactly 3 project ideas, appropriate to the student's branch.
- Do not mention internal ML encoding, SHAP internals, resume analysis, or implementation details.
- Do not guarantee employment.
- If the target career differs from the branch, explain the bridge skills.

STUDENT
Age: {student["age"]}
Gender: {student["gender"]}
UG Degree: {student["ug_degree"]}
UG Branch: {student["ug_branch"]}
UG CGPA: {student["ug_cgpa"]}
PG Degree: {student["pg_degree"] if student["has_pg"] else "Not Applicable"}
PG Specialization: {student["pg_branch"] if student["has_pg"] else "Not Applicable"}
PG CGPA: {student["pg_cgpa"] if student["has_pg"] else "Not Applicable"}

CAREER
Career Interest: {student["career_interest"]}
Target Career Goal: {student["target_career_goal"] or "Not specified"}

PLACEMENT PROFILE
{prediction_context}
Projects: {student["projects"]}
Internships: {student["internships"]}
Certifications: {student["certifications"]}
Backlogs: {student["backlogs"]}
Coding/Computational Skills: {student["coding_skills"]}/10
Communication Skills: {student["communication_skills"]}/10
Aptitude: {student["aptitude_score"]}/100

BRANCH-SPECIFIC SKILLS
{skill_lines}

CRITICAL SKILL GAPS
{", ".join(critical_gaps) if critical_gaps else "No critical gaps based on the self-assessment."}

SKILLS NEEDING DEVELOPMENT
{", ".join(developing) if developing else "Continue strengthening existing skills."}

Respond with exactly these headings:

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

Under Project Ideas give EXACTLY 3 projects. For each include:
- Project title
- What to build/do
- Skills demonstrated
- Why it is relevant to the career goal
"""


def get_secret(name):
    """
    Read an API key from Streamlit Secrets first, then environment variables.
    This never displays the actual secret value in the UI.
    """
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass

    value = os.getenv(name, "")
    return str(value).strip() if value else ""


def get_ai_configuration_status():
    """
    Check whether the AI provider keys are available.
    Only returns True/False; secret values are never exposed.
    """
    return {
        "Groq": bool(get_secret("GROQ_API_KEY")),
        "Gemini": bool(get_secret("GEMINI_API_KEY")),
    }


def generate_with_gemini(prompt):
    from google import genai

    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in Streamlit Secrets."
        )

    client = genai.Client(
        api_key=key
    )

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        store=False,
    )

    text = getattr(
        interaction,
        "output_text",
        None,
    )

    if text:
        return str(text).strip()

    output = getattr(
        interaction,
        "output",
        None,
    )

    if output:
        parts = (
            output
            if isinstance(output, list)
            else [output]
        )

        collected = []

        for part in parts:
            part_text = getattr(
                part,
                "text",
                None,
            )

            if part_text:
                collected.append(
                    str(part_text)
                )

            content = getattr(
                part,
                "content",
                None,
            )

            if content:
                items = (
                    content
                    if isinstance(content, list)
                    else [content]
                )

                for item in items:
                    item_text = getattr(
                        item,
                        "text",
                        None,
                    )

                    if item_text:
                        collected.append(
                            str(item_text)
                        )

        if collected:
            return "\n".join(
                collected
            ).strip()

    raise RuntimeError(
        "Gemini returned an empty response."
    )


def generate_with_groq(prompt):
    """
    Generate guidance using Groq.

    The Groq key is expected to be stored as GROQ_API_KEY.
    Groq API keys normally begin with gsk_, and the complete key
    is sent unchanged as a Bearer token.
    """
    key = get_secret("GROQ_API_KEY")

    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured in Streamlit Secrets. "
            "Expected a Groq API key such as gsk_..."
        )

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert student career counselor "
                    "and employability advisor."
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

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            body_text = response.read().decode(
                "utf-8",
                errors="replace",
            )

            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Groq returned a non-JSON response (HTTP {response.status})."
                )

    except urllib.error.HTTPError as exc:
        details = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        # Give a useful diagnosis instead of hiding the provider error.
        try:
            error_json = json.loads(details)
            api_message = (
                error_json.get("error", {}).get("message")
                or error_json.get("message")
                or details
            )
        except Exception:
            api_message = details

        if exc.code == 401:
            raise RuntimeError(
                "Groq authentication failed (HTTP 401). "
                "GROQ_API_KEY is present, but Groq rejected the key. "
                "Check that the complete gsk_... key is copied into "
                "Streamlit Secrets and has not been revoked."
            ) from exc

        if exc.code == 403:
            raise RuntimeError(
                f"Groq access was denied (HTTP 403): {api_message}"
            ) from exc

        if exc.code == 429:
            raise RuntimeError(
                f"Groq rate limit/quota reached (HTTP 429): {api_message}"
            ) from exc

        raise RuntimeError(
            f"Groq API error (HTTP {exc.code}): {api_message}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Groq: {exc.reason}"
        ) from exc

    text = (
        body
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    if not text:
        api_error = body.get("error", {}).get("message")
        if api_error:
            raise RuntimeError(
                f"Groq returned an error: {api_error}"
            )

        raise RuntimeError(
            "Groq returned an empty response."
        )

    return text.strip()


def generate_ai_guidance(
    student,
    prediction_context=None,
):
    """
    Groq is PRIMARY.
    Gemini is FALLBACK only when Groq is unavailable or fails.

    The returned provider tells the UI which service actually
    generated the response.
    """
    prompt = build_ai_prompt(
        student,
        prediction_context,
    )

    errors = []

    groq_key = get_secret("GROQ_API_KEY")

    # 1. GROQ PRIMARY
    if groq_key:
        try:
            return generate_with_groq(prompt), "Groq"
        except Exception as exc:
            errors.append(f"Groq: {exc}")
    else:
        errors.append(
            "Groq: GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    # 2. GEMINI FALLBACK
    gemini_key = get_secret("GEMINI_API_KEY")

    if gemini_key:
        try:
            return generate_with_gemini(prompt), "Gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
    else:
        errors.append(
            "Gemini: GEMINI_API_KEY is not configured in Streamlit Secrets."
        )

    # IMPORTANT:
    # Do not hide the real provider errors behind a generic message.
    raise RuntimeError(
        "AI providers could not generate guidance.\n\n"
        + "\n".join(errors)
    )


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
    domain_scores,
):
    return {
        "gender": gender,
        "age": int(age),
        "ug_degree": ug_degree,
        "degree": ug_degree,
        "ug_branch": ug_branch,
        "branch": map_branch_for_model(
            ug_branch
        ),
        "ug_cgpa": float(ug_cgpa),
        "cgpa": float(ug_cgpa),
        "has_pg": has_pg,
        "pg_degree": pg_degree,
        "pg_branch": pg_branch,
        "pg_cgpa": (
            float(pg_cgpa)
            if has_pg
            else None
        ),
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
        "target_career_goal": (
            target_career_goal.strip()
        ),
        "branch_skills": domain_scores,
        "domain_skills": round(
            float(
                np.mean(
                    list(
                        domain_scores.values()
                    )
                )
            ),
            2,
        ),
    }


# ============================================================
# APP UI
# ============================================================

st.title(
    "🎓 AI Student Placement Predictor"
)

st.write(
    "Predict placement readiness with the trained ML model, "
    "or get independent AI career guidance."
)

st.info(
    "⚠️ Placement predictions are decision-support information, "
    "not a guarantee of employment."
)

st.divider()

st.subheader(
    "Choose Analysis Mode"
)

mode = st.radio(
    "What would you like to use?",
    [
        "🔮 Placement Prediction",
        "🤖 AI Career Guidance",
    ],
    horizontal=True,
)

st.header(
    "📋 Student Profile"
)

# ============================================================
# PERSONAL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        16,
        60,
        22,
    )

with col2:
    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other",
        ],
    )

with col3:
    backlogs = st.number_input(
        "Backlogs",
        0,
        20,
        0,
    )

# ============================================================
# EDUCATION
# ============================================================

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
        0.0,
        10.0,
        7.0,
        0.1,
    )

has_pg = st.checkbox(
    "I have postgraduate education"
)

pg_degree = "Not Applicable"
pg_branch = "Not Applicable"
pg_cgpa = 0.0

if has_pg:

    p1, p2, p3 = st.columns(3)

    with p1:
        pg_degree = st.selectbox(
            "PG Degree",
            [
                "MTech",
                "MSc",
                "MCA",
                "MBA",
                "MCom",
                "MA",
                "Other",
            ],
        )

    with p2:
        pg_branch = st.text_input(
            "PG Specialization",
            placeholder="Example: Data Science",
        )

    with p3:
        pg_cgpa = st.number_input(
            "PG CGPA",
            0.0,
            10.0,
            7.0,
            0.1,
        )

# ============================================================
# PLACEMENT PROFILE
# ============================================================

st.subheader(
    "💼 Placement Profile"
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    internships = st.number_input(
        "Internships",
        0,
        20,
        1,
    )

with c2:
    projects = st.number_input(
        "Projects",
        0,
        30,
        2,
    )

with c3:
    certifications = st.number_input(
        "Certifications",
        0,
        30,
        2,
    )

with c4:
    communication_skills = st.slider(
        "Communication Skills",
        1,
        10,
        5,
    )

# ============================================================
# CAREER DIRECTION
# ============================================================

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
            "Example: Data Analyst, Food Safety Officer, "
            "Financial Analyst"
        ),
    )

# ============================================================
# BRANCH SKILLS
# ============================================================

st.subheader(
    f"🧠 {ug_branch} Skills"
)

st.caption(
    "Rate your current confidence in the skills relevant to your actual branch."
)

branch_skill_names = get_branch_skills(
    ug_branch
)

domain_scores = {}

skill_cols = st.columns(
    min(3, len(branch_skill_names))
)

for index, skill in enumerate(
    branch_skill_names
):

    with skill_cols[
        index % len(skill_cols)
    ]:

        domain_scores[skill] = st.slider(
            skill,
            1,
            10,
            5,
            key=f"skill_{ug_branch}_{skill}",
        )

if ug_branch in [
    "Computer Science",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Machine Learning",
    "Cyber Security",
    "Software Engineering",
    "Computer Applications",
]:

    relevant_scores = [
        score
        for skill, score
        in domain_scores.items()
        if any(
            keyword in skill
            for keyword in [
                "Programming",
                "Software",
                "Problem Solving",
                "Python",
            ]
        )
    ]

    coding_skills = int(
        round(
            np.mean(
                relevant_scores or [5]
            )
        )
    )

    aptitude_score = st.slider(
        "Aptitude Score",
        0,
        100,
        60,
    )

else:

    coding_skills = 5
    aptitude_score = 60

student = build_student_profile(
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
    domain_scores,
)

st.divider()

# ============================================================
# MAIN BUTTONS
# ============================================================

if mode == "🔮 Placement Prediction":

    if st.button(
        "🔮 Predict Placement",
        use_container_width=True,
        type="primary",
    ):

        try:

            with st.spinner(
                "Running placement prediction..."
            ):

                result = run_prediction(
                    student
                )

            st.session_state.prediction_result = result
            st.session_state.student_profile = student
            st.session_state.ai_career_advice = None
            st.session_state.ai_provider = None

        except Exception as exc:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(exc)
            )

else:

    ai_status = get_ai_configuration_status()
    with st.expander("🔐 AI Provider Configuration Status"):
        st.write(
            f"Groq API key: **{'Configured' if ai_status['Groq'] else 'Not configured'}**"
        )
        st.write(
            f"Gemini API key: **{'Configured' if ai_status['Gemini'] else 'Not configured'}**"
        )
        st.caption(
            "Groq is the primary provider. Gemini is used only as a fallback. "
            "Your API keys are never displayed."
        )

    if st.button(
        "🤖 Generate AI Career Guidance",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.student_profile = student
        st.session_state.ai_career_advice = None

        try:

            with st.spinner(
                "Preparing personalized AI career guidance..."
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
                "AI guidance failed."
            )

            st.code(
                str(exc)
            )

# ============================================================
# PLACEMENT RESULT
# ============================================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()

    st.header(
        "🎯 Placement Prediction Result"
    )

    probability = result[
        "probability"
    ]

    probability_pct = (
        probability * 100
        if probability is not None
        else None
    )

    prediction_text = result[
        "prediction_text"
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction",
            prediction_text,
        )

    with c2:
        st.metric(
            "Placement Probability",
            (
                f"{probability_pct:.2f}%"
                if probability_pct is not None
                else "N/A"
            ),
        )

    with c3:

        if (
            probability is not None
            and probability >= 0.75
        ):
            readiness = (
                "High Placement Readiness"
            )

        elif (
            probability is not None
            and probability >= 0.50
        ):
            readiness = (
                "Moderate Placement Readiness"
            )

        elif probability is not None:
            readiness = (
                "Needs Improvement"
            )

        else:
            readiness = (
                "Prediction Available"
            )

        st.metric(
            "Placement Readiness",
            readiness,
        )

    st.subheader(
        "💪 Your Strengths"
    )

    for item in result[
        "strengths"
    ]:
        st.success(item)

    st.subheader(
        "📈 Areas to Improve"
    )

    if result["gaps"]:

        for item in result[
            "gaps"
        ]:
            st.warning(item)

    else:

        st.success(
            "No major profile gaps were identified from the supplied inputs."
        )

    st.subheader(
        "💡 Personalized Recommendations"
    )

    if result[
        "recommendations"
    ]:

        for index, item in enumerate(
            result["recommendations"],
            1,
        ):
            st.write(
                f"**{index}.** {item}"
            )

    else:

        st.info(
            "No additional rule-based recommendations are available."
        )

    with st.expander(
        "Model compatibility information"
    ):

        st.write(
            "Input preparation mode:",
            result[
                "model_input_type"
            ],
        )

    # ========================================================
    # AI AFTER PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Personalized AI Career Guidance"
    )

    st.write(
        "Groq is tried first. If Groq is unavailable or rate-limited, "
        "Gemini is used automatically as a fallback."
    )

    if st.button(
        "🧠 Generate Personalized AI Guidance",
        use_container_width=True,
    ):

        try:

            if probability_pct is not None:

                context = (
                    f"Placement Prediction: "
                    f"{prediction_text}\n"
                    f"Placement Probability: "
                    f"{probability_pct:.2f}%"
                )

            else:

                context = (
                    f"Placement Prediction: "
                    f"{prediction_text}\n"
                    "Probability unavailable."
                )

            with st.spinner(
                "Generating personalized AI guidance..."
            ):

                advice, provider = (
                    generate_ai_guidance(
                        student,
                        context,
                    )
                )

            st.session_state.ai_career_advice = advice
            st.session_state.ai_provider = provider

        except Exception as exc:

            st.error(
                "AI guidance failed."
            )

            st.code(
                str(exc)
            )

# ============================================================
# DISPLAY AI GUIDANCE
# ============================================================

if st.session_state.ai_career_advice:

    st.divider()

    provider = (
        st.session_state.ai_provider
        or "AI"
    )

    st.header(
        f"🧠 Personalized AI Career Guidance ({provider})"
    )

    st.markdown(
        st.session_state.ai_career_advice
    )

st.divider()

st.caption(
    "AI Student Placement Predictor | "
    "Machine Learning + AI Career Guidance"
)
