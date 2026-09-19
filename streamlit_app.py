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
# available in PG Specialization. BCA/BBA/BCom/BA are included here because
# they were explicitly requested for the PG specialization dropdown, while
# they are not present in the UG specialization dropdown.
PG_SPECIALIZATIONS = UG_SPECIALIZATIONS + ["BCA", "BBA", "BCom", "BA"]

PG_DEGREES = ["MTech", "ME", "MSc", "MCA", "MBA", "MCom", "MA", "MS", "MPhil", "Other"]
    "Computer": ["Software Development", "Data Analytics", "Cyber Security", "Cloud / DevOps"],
    "Information Technology": ["Software Development", "Cloud / DevOps", "Cyber Security", "Networking"],
    "Cyber": ["Cyber Security", "Networking", "Cloud / DevOps"],
    "Finance": ["Finance / Accounting", "Data Analytics", "Business / Management"],
    "Accounting": ["Finance / Accounting", "Business / Management", "Teaching / Education"],
    "Commerce": ["Finance / Accounting", "Business / Management", "Teaching / Education"],
    "Business": ["Business / Management", "Data Analytics", "Marketing"],
    "Management": ["Business / Management", "Marketing", "Human Resources", "Data Analytics"],
    "Marketing": ["Marketing", "Business / Management", "Data Analytics"],
    "Human Resources": ["Human Resources", "Business / Management", "Research / Academia"],
    "Psychology": ["Psychology / Counseling", "Research / Academia", "Human Resources", "Teaching / Education"],
    "Physics": ["Research / Academia", "Teaching / Education", "Instrumentation"],
    "Chemistry": ["Research / Academia", "Teaching / Education", "Chemical / Process", "Quality Engineering"],
    "Biotechnology": ["Biotechnology / Life Sciences", "Research / Academia", "Quality Engineering"],
    "Microbiology": ["Biotechnology / Life Sciences", "Research / Academia", "Healthcare / Diagnostics"],
    "Food": ["Food / Nutrition", "Quality Engineering", "Research / Academia"],
    "Nutrition": ["Food / Nutrition", "Healthcare / Nutrition", "Research / Academia", "Teaching / Education"],
    "Electrical": ["Electrical / Power", "Automation", "Embedded / VLSI", "Research / Academia"],
    "Electronics": ["Electronics / Instrumentation", "Embedded / VLSI", "Research / Academia"],
    "Communication": ["Networking", "Telecommunications", "Embedded / VLSI", "Research / Academia"],
    "Mechanical": ["Mechanical / Design", "Manufacturing", "Operations", "Research / Academia"],
    "Automobile": ["Automotive", "Mechanical / Design", "Manufacturing", "Research / Academia"],
    "Civil": ["Civil / Construction", "Infrastructure", "Government / Public Sector", "Research / Academia"],
    "Chemical": ["Chemical / Process", "Quality Engineering", "Research / Academia"],
    "Environmental": ["Environmental Science", "Sustainability", "Research / Academia", "Government / Public Sector"],
    "English": ["Content / Communication", "Teaching / Education", "Research / Academia", "Publishing / Editing"],
    "Political": ["Government / Public Sector", "Policy Research", "Research / Academia", "Teaching / Education"],
    "Sociology": ["Social Research", "Research / Academia", "Government / Public Sector", "Teaching / Education"],
    "History": ["Research / Academia", "Teaching / Education", "Museum / Heritage", "Government / Public Sector"],
}


def _dedupe(items):
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _career_groups_for_profile(ug_branch, pg_branch="Not Applicable"):
    groups = list(BRANCH_CAREER_GROUP.get(ug_branch, COMMON_CAREERS))
    combined = f"{ug_branch} {pg_branch}".lower()
    for keyword, extra in SPECIALIZATION_KEYWORDS.items():
        if keyword.lower() in combined:
            groups.extend(extra)
    groups.extend(COMMON_CAREERS)
    return _dedupe(groups)


def get_career_interests(ug_branch, pg_branch="Not Applicable"):
    return _career_groups_for_profile(ug_branch, pg_branch)


def get_target_careers(ug_branch, career_interest, pg_branch="Not Applicable"):
    groups = _career_groups_for_profile(ug_branch, pg_branch)
    if career_interest in CAREER_TARGETS:
        targets = list(CAREER_TARGETS[career_interest])
    else:
        # If a branch has a career group not explicitly mapped, derive sensible
        # targets from its domain rather than showing unrelated careers.
        targets = []
        if career_interest in groups:
            targets = CAREER_TARGETS.get(career_interest, [])

    if career_interest == "Teaching / Education":
        targets = [
            "School Teacher", "Subject Teacher", "College Lecturer", "Assistant Professor Track",
            "Online Instructor", "Academic Coordinator", "Private Tutor",
        ]
    elif career_interest == "Research / Academia":
        targets = [
            "Research Assistant", "Research Associate", "Project Assistant", "Junior Research Fellow",
            "Academic Researcher", "PhD / Doctoral Research Track",
        ]
    elif career_interest == "Higher Studies":
        targets = ["Master's Degree", "Specialized Higher Studies", "PhD / Doctoral Track", "Professional Certification Track"]
    elif career_interest == "Government / Public Sector":
        targets = ["Government Technical Officer", "Public Sector Analyst", "Administrative Officer", "Government Exam Candidate"]

    if not targets:
        targets = ["Domain-specific Entry-Level Role", "Research Assistant", "Teaching / Education"]

    return _dedupe(targets)
        "Cyber Security": [
            "Security Log Monitoring Dashboard",
            "Network Security Assessment Lab",
            "Phishing Detection and Awareness Tool",
        ],
        "Cloud / DevOps": [
            "Containerized Web Application Deployment",
            "CI/CD Pipeline for a Student Project",
            "Cloud Monitoring and Deployment Dashboard",
        ],
        "Teaching / Education": [
            "Interactive Subject Learning Portal",
            "Practice-Test and Progress Tracking System",
            "Digital Lesson and Assessment Resource",
        ],
        "Research / Academia": [
            "Literature Review and Research Gap Study",
            "Reproducible Domain Experiment",
            "Research Dataset Analysis and Report",
        ],
    }.get(interest, [
        f"{ug} Practical Portfolio Project",
        f"{ug} Data / Process Analysis Project",
        f"{ug} Research or Industry Case Study",
    ])

    gaps = [name for name, score in weakest if score <= 6]
    gap_text = ", ".join(gaps) if gaps else "No major branch-skill gap was identified from the selected ratings."
    strength_text = ", ".join(f"{name} ({score}/10)" for name, score in strongest)
    pg_text = f"PG specialization: {pg}" if pg else "No PG specialization selected."

    return f"""### Built-in Career Guidance\n\nGemini/Groq was temporarily unavailable, so this report was generated by the app's built-in career guidance rules. It is based on the profile entered in the dashboard.\n\n## 1. Overall Profile Assessment\n- UG domain: **{ug}**\n- {pg_text}\n- Career interest: **{interest}**\n- Target role: **{target}**\n- Strongest selected skills: **{strength_text}**\n\n## 2. Career Direction\nYour selected career direction is **{interest}**. Build the portfolio around **{target}**, while using your UG/PG specialization as the domain foundation.\n\n## 3. Career Path to the Target Role\n1. Strengthen the core concepts required for {interest}.\n2. Learn the main tools listed below and use them in practical work.\n3. Complete the three portfolio projects below and publish documented work on GitHub.\n4. Add internship, research, teaching, volunteering, freelance, or supervised practical experience where appropriate.\n5. Prepare role-specific interview questions and a focused resume.\n\n## 4. Top Strengths\n- {strength_text}\n- Projects: {student['projects']}\n- Internships: {student['internships']}\n- Certifications: {student['certifications']}\n- Communication: {student['communication_skills']}/10\n\n## 5. Skill Gap Analysis\nMain areas to improve from the selected skill ratings: **{gap_text}**.\n\n## 6. Areas to Improve\n- Build more role-specific practical evidence.\n- Practice communication and interview explanations.\n- Improve the lowest-rated technical/domain skills first.\n- Keep GitHub projects documented with README files, screenshots, setup steps and results.\n\n## 7. 30-Day Improvement Plan\n- **Days 1-7:** Revise fundamentals for {interest}; identify the exact skills needed for {target}.\n- **Days 8-14:** Build Project 1 and document the work.\n- **Days 15-21:** Build Project 2 and complete targeted practice/interview questions.\n- **Days 22-30:** Finish Project 3, improve resume/GitHub, and conduct mock interviews.\n\n## 8. Technical Topics to Study\nFocus on the core concepts of **{interest}**, then the tools below.\n\n## 9. Industry Tools and Professional Skills\n**Suggested tools:** {tools}\n\n## 10. Project Ideas\n### 1. {projects[0]}\nBuild a complete, documented version relevant to **{target}**. Demonstrate problem solving, domain knowledge, implementation and measurable results.\n\n### 2. {projects[1]}\nCreate a second project that solves a different practical problem in the same career direction. Include data/process/design decisions and a clear README.\n\n### 3. {projects[2]}\nCreate a third project that shows depth, testing/evaluation and professional presentation.\n\n## 11. Teaching and Research Options\n- **Teaching:** consider tutoring, subject-content creation, lab assistance, workshops or a future lecturer/teacher path if it matches your qualifications.\n- **Research:** consider literature reviews, research projects, faculty-guided work, research internships or a postgraduate/PhD path.\n\n## 12. Interview Preparation\nPrepare a 60-second introduction, explain each project clearly, revise core domain concepts, and practice behavioral questions using real examples from your experience.\n\n## 13. GitHub and Resume Plan\nKeep 3-5 strong projects pinned, add clear README files, include technologies and outcomes, and tailor the resume to **{target}**.\n\n## 14. 3-Month Roadmap\n- **Month 1:** fundamentals + Project 1.\n- **Month 2:** Project 2 + internship/research/teaching applications.\n- **Month 3:** Project 3 + resume + GitHub + mock interviews + targeted applications.\n\n## 15. Final Action Checklist\n- [ ] Strengthen the weakest skills.\n- [ ] Finish exactly 3 strong portfolio projects.\n- [ ] Publish and document projects on GitHub.\n- [ ] Improve resume for {target}.\n- [ ] Practice technical and behavioral interviews.\n- [ ] Apply for relevant internships, jobs, research or teaching opportunities.\n"""


def generate_ai_guidance(student, prediction_context=None):
    prompt = build_ai_prompt(student, prediction_context)
    errors = []

    if get_secret("GROQ_API_KEY"):
        try:
            return generate_with_groq(prompt), "Groq"
        except Exception as exc:
            errors.append(f"Groq: {exc}")
    else:
        errors.append("Groq: GROQ_API_KEY is not configured.")

    if get_secret("GEMINI_API_KEY"):
        try:
            return generate_with_gemini(prompt), "Gemini"
        except Exception as exc:
            errors.append(f"Gemini: {exc}")
    else:
        errors.append("Gemini: GEMINI_API_KEY is not configured.")

    # Never leave the user with a blank AI panel because a provider is temporarily down.
    fallback = build_builtin_career_report(student, prediction_context)
    return fallback, "Built-in Career Guidance"
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
