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
# EDUCATION / SPECIALIZATION OPTIONS
# ============================================================

# UG degree list intentionally excludes BCA/BBA/BCom/BA from the UG
# specialization/major list. Those are degree choices, not branch choices.
UG_DEGREES = ["BE", "BTech", "BSc", "Other"]

UG_SPECIALIZATIONS = [
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

# The user asked for the complete final UG specialization list to also be
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
        st.caption("Estimated probability from the trained placement model; not a guarantee of employment.")

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
    st.caption("The app tries Groq first, then Gemini with retry/backoff and model fallback. If both providers are temporarily unavailable, a built-in career report is shown instead of an error page.")

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
            st.error("AI guidance failed unexpectedly.")
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
st.caption("AI Student Placement Predictor | Dashboard + ML Placement Assessment + UG/PG-Aware Career Guidance")
