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
UG_DEGREES = ["BE", "BTech", "BSc", "BCA", "BBA", "BCom", "BA", "Other"]

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
PG_SPECIALIZATIONS = UG_SPECIALIZATIONS.copy()

PG_DEGREES = ["MTech", "ME", "MSc", "MCA", "MBA", "MCom", "MA", "MS", "MPhil", "Other"]
# ============================================================
# CAREER DIRECTION — BASED ON UG + PG, NOT RANDOM
# ============================================================

COMMON_CAREERS = [
    "Teaching / Education",
    "Research / Academia",
    "Government / Public Sector",
    "Higher Studies",
]

CAREER_TARGETS = {
    "Software Development": [
        "Software Developer", "Backend Developer", "Frontend Developer", "Full Stack Developer",
        "Mobile App Developer", "QA / Test Engineer", "Software Engineer",
    ],
    "Data Analytics": [
        "Data Analyst", "Business Intelligence Analyst", "Product Analyst", "Reporting Analyst",
        "Business Analyst", "Junior Data Scientist",
    ],
    "Data Science": [
        "Data Scientist", "Junior Data Scientist", "Data Analyst", "ML Data Analyst", "Research Data Analyst",
    ],
    "Artificial Intelligence / Machine Learning": [
        "AI Engineer", "Machine Learning Engineer", "Applied AI Developer", "NLP Engineer",
        "Computer Vision Engineer", "AI Research Assistant",
    ],
    "Cyber Security": [
        "Cyber Security Analyst", "SOC Analyst", "Security Engineer", "Security Tester",
        "Digital Forensics Analyst", "Information Security Analyst",
    ],
    "Cloud / DevOps": [
        "Cloud Engineer", "DevOps Engineer", "Cloud Support Engineer", "Site Reliability Engineer",
    ],
    "Networking": [
        "Network Engineer", "Network Administrator", "Network Support Engineer", "Infrastructure Engineer",
    ],
    "Embedded / VLSI": [
        "Embedded Systems Engineer", "Firmware Engineer", "VLSI Engineer", "Verification Engineer",
        "Hardware Design Engineer",
    ],
    "Electronics / Instrumentation": [
        "Electronics Engineer", "Instrumentation Engineer", "Control Engineer", "Electronics Design Engineer",
    ],
    "Electrical / Power": [
        "Electrical Engineer", "Power Systems Engineer", "Control Engineer", "Electrical Design Engineer",
        "Automation Engineer",
    ],
    "Mechanical / Design": [
        "Mechanical Design Engineer", "CAD Engineer", "Manufacturing Engineer", "Production Engineer",
        "Quality Engineer",
    ],
    "Automotive": [
        "Automotive Engineer", "Vehicle Design Engineer", "Automotive Testing Engineer", "Service Engineer",
    ],
    "Civil / Construction": [
        "Civil Engineer", "Structural Engineer", "Site Engineer", "Construction Engineer",
        "Quantity Surveyor", "Planning Engineer",
    ],
    "Chemical / Process": [
        "Process Engineer", "Chemical Engineer", "Plant Engineer", "Quality Engineer", "Safety Engineer",
    ],
    "Biotechnology / Life Sciences": [
        "Biotechnologist", "Laboratory Analyst", "Research Assistant", "Clinical Research Assistant",
        "Quality Control Analyst",
    ],
    "Food / Nutrition": [
        "Food Technologist", "Food Safety Analyst", "Quality Control Analyst", "Nutritionist",
        "Clinical Nutrition Assistant", "Food Research Assistant",
    ],
    "Finance / Accounting": [
        "Financial Analyst", "Accountant", "Audit Associate", "Tax Associate", "Banking Analyst",
        "Investment Operations Analyst",
    ],
    "Business / Management": [
        "Business Analyst", "Operations Analyst", "Management Trainee", "Project Coordinator",
        "Marketing Analyst", "HR Executive",
    ],
    "Economics": [
        "Economic Analyst", "Research Analyst", "Policy Research Assistant", "Data Analyst",
        "Financial Analyst",
    ],
    "Social Sciences / Humanities": [
        "Content Writer", "Research Assistant", "Policy Research Assistant", "Program Coordinator",
        "Public Relations Executive", "Social Research Assistant",
    ],
    "Teaching / Education": [
        "School Teacher", "Subject Faculty", "Academic Coordinator", "Online Instructor", "Tutor",
    ],
    "Research / Academia": [
        "Research Assistant", "Research Associate", "Project Assistant", "Junior Research Fellow", "Academic Researcher",
    ],
    "Government / Public Sector": [
        "Government Technical Officer", "Public Sector Analyst", "Administrative Officer", "Government Exam Candidate",
    ],
    "Higher Studies": [
        "Master's Degree", "Doctoral / PhD Track", "Professional Certification Track", "Specialized Higher Studies",
    ],
    "Manufacturing": ["Manufacturing Engineer", "Production Engineer", "Quality Engineer", "Process Engineer"],
    "Operations": ["Operations Analyst", "Operations Engineer", "Process Improvement Analyst", "Supply Chain Analyst"],
    "Quality Engineering": ["Quality Engineer", "Quality Analyst", "Quality Control Officer", "Process Quality Engineer"],
    "Automation": ["Automation Engineer", "Control Systems Engineer", "PLC Engineer", "Industrial Automation Engineer"],
    "Aerospace / Aeronautical": ["Aerospace Engineer", "Aeronautical Engineer", "Aircraft Design Engineer", "Aerospace Research Assistant"],
    "Telecommunications": ["Telecom Engineer", "Network Engineer", "RF Engineer", "Communication Systems Engineer"],
    "Biomedical Engineering": ["Biomedical Engineer", "Medical Device Engineer", "Clinical Engineering Assistant", "Biomedical Research Assistant"],
    "Healthcare Technology": ["Healthcare Technology Analyst", "Medical Device Engineer", "Clinical Systems Assistant", "Health Technology Research Assistant"],
    "Infrastructure": ["Infrastructure Engineer", "Urban Infrastructure Analyst", "Project Engineer", "Planning Engineer"],
    "Environmental Science": ["Environmental Analyst", "Environmental Consultant", "Sustainability Analyst", "Environmental Research Assistant"],
    "Sustainability": ["Sustainability Analyst", "ESG Analyst", "Environmental Consultant", "Sustainability Research Assistant"],
    "Pharmaceutical Research": ["Research Assistant", "Pharmaceutical Research Associate", "Quality Control Analyst", "Clinical Research Assistant"],
    "Healthcare / Diagnostics": ["Clinical Research Assistant", "Laboratory Analyst", "Diagnostics Analyst", "Healthcare Research Assistant"],
    "Bioinformatics": ["Bioinformatics Analyst", "Computational Biology Assistant", "Research Assistant", "Data Analyst"],
    "Healthcare / Nutrition": ["Nutritionist", "Clinical Nutrition Assistant", "Public Health Nutrition Assistant", "Nutrition Research Assistant"],
    "Instrumentation": ["Instrumentation Engineer", "Test Engineer", "Lab Instrumentation Engineer", "Instrumentation Research Assistant"],
    "Marketing": ["Marketing Analyst", "Digital Marketing Executive", "Market Research Analyst", "Brand Associate"],
    "Human Resources": ["HR Executive", "Recruitment Coordinator", "Talent Acquisition Associate", "People Operations Analyst"],
    "Psychology / Counseling": ["Psychology Assistant", "Counseling Assistant", "Behavioral Research Assistant", "HR / People Analyst"],
    "Content / Communication": ["Content Writer", "Technical Writer", "Communications Executive", "Content Strategist"],
    "Publishing / Editing": ["Editor", "Publishing Assistant", "Content Editor", "Technical Editor"],
    "Policy Research": ["Policy Research Assistant", "Policy Analyst", "Program Research Assistant", "Public Policy Associate"],
    "Social Research": ["Social Research Assistant", "Research Associate", "Program Coordinator", "Community Research Assistant"],
    "Museum / Heritage": ["Museum Assistant", "Heritage Research Assistant", "Archivist Assistant", "Cultural Program Coordinator"],
    "Other": ["Custom Career Goal"],
}
BRANCH_CAREER_GROUP = {
    "Computer Science": ["Software Development", "Data Analytics", "Data Science", "Artificial Intelligence / Machine Learning", "Cyber Security", "Cloud / DevOps", "Networking"],
    "Information Technology": ["Software Development", "Data Analytics", "Cloud / DevOps", "Cyber Security", "Networking"],
    "Data Science": ["Data Analytics", "Data Science", "Artificial Intelligence / Machine Learning", "Research / Academia"],
    "Artificial Intelligence": ["Artificial Intelligence / Machine Learning", "Data Science", "Data Analytics", "Research / Academia"],
    "Machine Learning": ["Artificial Intelligence / Machine Learning", "Data Science", "Data Analytics", "Research / Academia"],
    "Cyber Security": ["Cyber Security", "Networking", "Cloud / DevOps", "Research / Academia"],
    "Software Engineering": ["Software Development", "Cloud / DevOps", "Data Analytics", "Cyber Security"],
    "Computer Applications": ["Software Development", "Data Analytics", "Cloud / DevOps", "Cyber Security"],
    "Mechanical Engineering": ["Mechanical / Design", "Manufacturing", "Operations", "Research / Academia"],
    "Automobile Engineering": ["Automotive", "Mechanical / Design", "Manufacturing", "Research / Academia"],
    "Production Engineering": ["Manufacturing", "Operations", "Mechanical / Design", "Quality Engineering"],
    "Industrial Engineering": ["Operations", "Manufacturing", "Business / Management", "Research / Academia"],
    "Aeronautical Engineering": ["Aerospace / Aeronautical", "Mechanical / Design", "Manufacturing", "Research / Academia"],
    "Aerospace Engineering": ["Aerospace / Aeronautical", "Mechanical / Design", "Research / Academia", "Government / Public Sector"],
    "Electrical Engineering": ["Electrical / Power", "Automation", "Embedded / VLSI", "Research / Academia"],
    "Electronics Engineering": ["Electronics / Instrumentation", "Embedded / VLSI", "Automation", "Research / Academia"],
    "Electronics and Communication Engineering": ["Embedded / VLSI", "Electronics / Instrumentation", "Networking", "Telecommunications", "Research / Academia"],
    "Biomedical Engineering": ["Biomedical Engineering", "Electronics / Instrumentation", "Research / Academia", "Healthcare Technology"],
    "Civil Engineering": ["Civil / Construction", "Infrastructure", "Government / Public Sector", "Research / Academia"],
    "Chemical Engineering": ["Chemical / Process", "Quality Engineering", "Research / Academia", "Government / Public Sector"],
    "Mathematics": ["Data Analytics", "Data Science", "Teaching / Education", "Research / Academia"],
    "Statistics": ["Data Analytics", "Data Science", "Research / Academia", "Teaching / Education"],
    "Physics": ["Research / Academia", "Teaching / Education", "Data Analytics", "Instrumentation"],
    "Chemistry": ["Research / Academia", "Teaching / Education", "Chemical / Process", "Quality Engineering"],
    "Environmental Science": ["Environmental Science", "Research / Academia", "Government / Public Sector", "Sustainability"],
    "Biotechnology": ["Biotechnology / Life Sciences", "Research / Academia", "Quality Engineering", "Pharmaceutical Research"],
    "Microbiology": ["Biotechnology / Life Sciences", "Research / Academia", "Quality Engineering", "Healthcare / Diagnostics"],
    "Biochemistry": ["Biotechnology / Life Sciences", "Research / Academia", "Healthcare / Diagnostics", "Quality Engineering"],
    "Biological Sciences": ["Biotechnology / Life Sciences", "Research / Academia", "Teaching / Education", "Healthcare / Diagnostics"],
    "Life Sciences": ["Biotechnology / Life Sciences", "Research / Academia", "Healthcare / Diagnostics", "Teaching / Education"],
    "Genetics": ["Biotechnology / Life Sciences", "Research / Academia", "Healthcare / Diagnostics", "Bioinformatics"],
    "Botany": ["Research / Academia", "Teaching / Education", "Environmental Science", "Biotechnology / Life Sciences"],
    "Zoology": ["Research / Academia", "Teaching / Education", "Environmental Science", "Healthcare / Diagnostics"],
    "Food Science and Nutrition": ["Food / Nutrition", "Research / Academia", "Quality Engineering", "Healthcare / Nutrition"],
    "Food Technology": ["Food / Nutrition", "Manufacturing", "Quality Engineering", "Research / Academia"],
    "Nutrition and Dietetics": ["Food / Nutrition", "Healthcare / Nutrition", "Research / Academia", "Teaching / Education"],
    "Economics": ["Economics", "Finance / Accounting", "Data Analytics", "Research / Academia", "Government / Public Sector"],
    "Commerce": ["Finance / Accounting", "Business / Management", "Data Analytics", "Teaching / Education"],
    "Business Administration": ["Business / Management", "Finance / Accounting", "Data Analytics", "Marketing"],
    "Finance": ["Finance / Accounting", "Data Analytics", "Business / Management", "Research / Academia"],
    "Accounting": ["Finance / Accounting", "Business / Management", "Teaching / Education"],
    "Management": ["Business / Management", "Finance / Accounting", "Data Analytics", "Teaching / Education"],
    "Marketing": ["Marketing", "Business / Management", "Data Analytics", "Content / Communication"],
    "Human Resources": ["Human Resources", "Business / Management", "Research / Academia", "Teaching / Education"],
    "Psychology": ["Psychology / Counseling", "Research / Academia", "Teaching / Education", "Human Resources"],
    "English": ["Content / Communication", "Teaching / Education", "Research / Academia", "Publishing / Editing"],
    "Political Science": ["Government / Public Sector", "Research / Academia", "Policy Research", "Teaching / Education"],
    "Sociology": ["Social Research", "Research / Academia", "Government / Public Sector", "Teaching / Education"],
    "History": ["Research / Academia", "Teaching / Education", "Museum / Heritage", "Government / Public Sector"],
    "Public Administration": ["Government / Public Sector", "Policy Research", "Research / Academia", "Teaching / Education"],
    "Other": COMMON_CAREERS,
}

# Additional career groups that can be activated by PG specialization.
SPECIALIZATION_KEYWORDS = {
    "Data": ["Data Analytics", "Data Science", "Research / Academia"],
    "Statistics": ["Data Analytics", "Data Science", "Research / Academia", "Teaching / Education"],
    "Artificial Intelligence": ["Artificial Intelligence / Machine Learning", "Data Science", "Research / Academia"],
    "Machine Learning": ["Artificial Intelligence / Machine Learning", "Data Science", "Research / Academia"],
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
    """Build the exact 17-feature input used by the current trained model."""
    if model is not None:
        expected_features = get_model_feature_names(model, feature_names)
    else:
        expected_features = list(feature_names)

   # The current trained model uses these raw features:
    # age, ug_cgpa, backlogs, internships, projects, certifications,
    # coding_skills, communication_skills, aptitude_score,
    # domain_skill_1 ... domain_skill_5, gender, ug_degree, ug_branch
    domain_scores = list(student.get("branch_skills", {}).values())[:5]
    while len(domain_scores) < 5:
        domain_scores.append(5.0)
        
    values = {
        "age": float(student["age"]),
        "ug_cgpa": float(student["ug_cgpa"]),
        "backlogs": float(student["backlogs"]),
        "internships": float(student["internships"]),
        "projects": float(student["projects"]),
        "certifications": float(student["certifications"]),
        "coding_skills": float(student["coding_skills"]),
        "communication_skills": float(student["communication_skills"]),
        "aptitude_score": float(student["aptitude_score"]),
        "domain_skill_1": float(domain_scores[0]),
        "domain_skill_2": float(domain_scores[1]),
        "domain_skill_3": float(domain_scores[2]),
        "domain_skill_4": float(domain_scores[3]),
        "domain_skill_5": float(domain_scores[4]),
        "gender": str(student["gender"]),
        "ug_degree": str(student["ug_degree"]),
        "ug_branch": str(student["ug_branch"]),        
    }
    unsupported = [f for f in expected_features if f not in values]
    if unsupported:
        raise RuntimeError(
            "The trained model expects feature(s) this app cannot construct: "
            + ", ".join(unsupported)           
        )

     return pd.DataFrame([{f: values[f] for f in expected_features}])

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
# AI CAREER GUIDANCE
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


    skills = "\n".join(f"- {k}: {v}/10" for k, v in student["branch_skills"].items())
    critical = ", ".join(f"{k} ({v}/10)" for k, v in student["branch_skills"].items() if v <= 4) or "None identified"
    prediction_context = prediction_context or "Placement prediction was not requested."
    target_options = ", ".join(student.get("available_target_careers", []))
    interest_options = ", ".join(student.get("available_career_interests", []))

    return f"""
You are an expert student career counselor and employability advisor.
Create a practical, personalized career report based ONLY on the student's supplied profile.
Do not invent achievements, internships, skills, marks, employers, or certifications.
Do not guarantee employment.
The UG branch is the student's primary academic domain. If a PG specialization exists, treat it as the latest specialization and prioritize it when relevant, while still respecting the UG foundation.
Career guidance must stay relevant to the student's UG + PG combination.
The student selected one career interest and one target role from the allowed options below. Explain the path to that target.
Include both employment and higher-education possibilities where relevant.
Include teaching/education and research/academia as options when relevant to the student's field.
Give EXACTLY 3 realistic project ideas suited to the student's domain and target career.

STUDENT PROFILE
Age: {student['age']}
Gender: {student['gender']}
UG Degree: {student['ug_degree']}
UG Specialization: {student['ug_branch']}
UG CGPA: {student['ug_cgpa']}
PG Degree: {student['pg_degree'] if student['has_pg'] else 'Not Applicable'}
PG Specialization: {student['pg_branch'] if student['has_pg'] else 'Not Applicable'}
PG CGPA: {student['pg_cgpa'] if student['has_pg'] else 'Not Applicable'}
Backlogs: {student['backlogs']}
Internships: {student['internships']}
Projects: {student['projects']}
Certifications: {student['certifications']}
Communication Skills: {student['communication_skills']}/10
Aptitude Score: {student['aptitude_score']}/100
Coding / Computational Skills: {student['coding_skills']}/10
Career Interest: {student['career_interest']}
Target Career Goal: {student['target_career_goal']}

ALLOWED CAREER INTERESTS FOR THIS PROFILE
{interest_options}

ALLOWED TARGET ROLES FOR THIS PROFILE AND INTEREST
{target_options}

BRANCH / SPECIALIZATION SKILLS
{skills}

CRITICAL SKILL GAPS
{critical}

PLACEMENT CONTEXT
{prediction_context}

Respond using exactly these headings:
## 1. Overall Profile Assessment
## 2. Career Direction
## 3. Career Path to the Target Role
## 4. Top Strengths
## 5. Skill Gap Analysis
## 6. Areas to Improve
## 7. 30-Day Improvement Plan
## 8. Technical Topics to Study
## 9. Industry Tools and Professional Skills
## 10. Project Ideas
## 11. Teaching and Research Options
## 12. Interview Preparation
## 13. GitHub and Resume Plan
## 14. 3-Month Roadmap
## 15. Final Action Checklist

For Career Path to the Target Role, explain the sequence of skills, projects, experience, and interview preparation needed for the selected target.
For Project Ideas provide EXACTLY 3 projects. For each include: title, what to build/do, skills demonstrated, tools/technologies, and why it is relevant.
For the 30-Day Improvement Plan use a practical week-by-week or day-range plan.
Keep the report detailed but focused. Do not output unrelated career paths just to fill space.
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
        "temperature": 0.35,
        "max_tokens": 4200,
    }

    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AI-Student-Placement-Predictor/2.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            raise RuntimeError("Groq authentication failed (HTTP 401). Check GROQ_API_KEY in Streamlit Secrets.") from exc
        if exc.code == 403:
            raise RuntimeError(f"Groq access denied (HTTP 403). Details: {details}") from exc
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


def _is_retryable_provider_error(exc):
    message = str(exc).lower()
    retry_markers = [
        "503", "unavailable", "high demand", "overloaded", "temporarily",
        "429", "rate limit", "too many requests", "500", "502", "504", "timeout",
    ]
    return any(marker in message for marker in retry_markers)


def generate_with_gemini(prompt):
    """Gemini generation with model fallback and retry/backoff for transient 5xx/429 errors."""
    from google import genai
    from google.genai import types
    import time
    import random

    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=key)
    models = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
    errors = []

    for model_name in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=5000,
                        temperature=1.0,
                    ),
                )
                text = getattr(response, "text", None)
                if text:
                    return str(text).strip()
                raise RuntimeError(f"{model_name} returned an empty response.")
            except Exception as exc:
                errors.append(f"{model_name} attempt {attempt + 1}: {exc}")
                if _is_retryable_provider_error(exc) and attempt < 2:
                    # Exponential backoff with a small jitter. This is specifically
                    # for transient provider overload/rate-limit conditions.
                    time.sleep((1.5 * (2 ** attempt)) + random.uniform(0.0, 0.7))
                    continue
                break

    raise RuntimeError("Gemini models were unavailable after retries.\n" + "\n".join(errors[-8:]))


def build_builtin_career_report(student, prediction_context=None):
    """Deterministic fallback so the dashboard remains useful when an AI provider is down."""
    interest = student["career_interest"]
    target = student["target_career_goal"]
    ug = student["ug_branch"]
    pg = student["pg_branch"] if student["has_pg"] else None
    skills = student["branch_skills"]
    strongest = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:3]
    weakest = sorted(skills.items(), key=lambda x: x[1])[:3]
    tools = {
        "Data Analytics": "SQL, Excel, Power BI/Tableau, Python, pandas",
        "Data Science": "Python, pandas, NumPy, scikit-learn, SQL, Jupyter",
        "Artificial Intelligence / Machine Learning": "Python, NumPy, pandas, scikit-learn, PyTorch/TensorFlow",
        "Software Development": "Git, GitHub, one programming language, SQL, APIs, testing",
        "Cyber Security": "Linux, networking, Python, security fundamentals, SIEM concepts",
        "Cloud / DevOps": "Linux, Git, Docker, CI/CD, one cloud platform",
        "Finance / Accounting": "Excel, accounting software, SQL, financial analysis tools",
        "Business / Management": "Excel, PowerPoint, SQL basics, project-management tools",
        "Teaching / Education": "Subject mastery, lesson planning, presentation tools, digital learning platforms",
        "Research / Academia": "Literature review, research methods, statistics, citation/reference tools",
    }.get(interest, "Domain-specific software, Excel/SQL where useful, Git/GitHub, presentation and documentation tools")

    projects = {
        "Data Analytics": [
            "E-Commerce Sales and Customer Analytics Dashboard",
            "Customer Churn and Retention Analysis",
            "A/B Testing and Business KPI Analysis",
        ],
        "Data Science": [
            "End-to-End Student Performance Prediction",
            "Customer Churn Prediction with Explainable ML",
            "Demand Forecasting and Model Evaluation",
        ],
        "Artificial Intelligence / Machine Learning": [
            "Domain-Specific AI Assistant",
            "Image or Text Classification System",
            "Explainable Machine Learning Prediction App",
        ],
        "Software Development": [
            "Full-Stack Career or Student Management Portal",
            "REST API with Authentication and Database",
            "Production-Style Task or Placement Tracking App",
        ],
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
st.write("A dashboard-style placement assessment with field-specific career guidance.")
st.info("⚠️ Placement probability is a model estimate based on the training data. It is not a guarantee of employment.")

with st.sidebar:
    st.header("🔧 System Status")
    try:
        load_components()
        st.success("ML model loaded")
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

c1, c2, c3 = st.columns(3)
with c1:
    ug_degree = st.selectbox("UG Degree", UG_DEGREES)
with c2:
    ug_branch = st.selectbox("UG Specialization / Major", UG_SPECIALIZATIONS)
with c3:
    ug_cgpa = st.number_input("UG CGPA", 0.0, 10.0, 7.0, 0.1)

has_pg = st.checkbox("I have postgraduate education")
pg_degree = "Not Applicable"
pg_branch = "Not Applicable"
pg_cgpa = 0.0

if has_pg:
    c1, c2, c3 = st.columns(3)
    with c1:
        pg_degree = st.selectbox("PG Degree", PG_DEGREES)
    with c2:
        pg_branch = st.selectbox("PG Specialization", PG_SPECIALIZATIONS)
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
career_interest_options = get_career_interests(ug_branch, pg_branch if has_pg else "Not Applicable")
c1, c2 = st.columns(2)
with c1:
    career_interest = st.selectbox("Career Interest", career_interest_options)
with c2:
    target_options = get_target_careers(ug_branch, career_interest, pg_branch if has_pg else "Not Applicable")
    target_career_goal = st.selectbox("Target Career Goal", target_options)

st.caption("Career interests and target roles are generated from your UG specialization and, when selected, your PG specialization.")

st.subheader(f"🧠 {ug_branch} Skills")
st.caption("These field-specific skills support the career guidance and readiness view. They are not extra columns added to the trained ML model.")
branch_skill_names = get_branch_skills(ug_branch)
domain_scores = {}
skill_cols = st.columns(min(3, len(branch_skill_names)))
for index, skill in enumerate(branch_skill_names):
    with skill_cols[index % len(skill_cols)]:
        domain_scores[skill] = st.slider(skill, 1, 10, 5, key=f"skill_{ug_branch}_{skill}")

technical_branches = {
    "Computer Science", "Information Technology", "Data Science", "Artificial Intelligence",
    "Machine Learning", "Cyber Security", "Software Engineering", "Computer Applications",
}

if ug_branch in technical_branches:
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
