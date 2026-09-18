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
        "Industrial Engineering",
        "Aeronautical Engineering",
        "Aerospace Engineering"
    ]:
        return [
            "Design Engineering",
            "Manufacturing",
            "Automotive Engineering",
            "Production Engineering",
            "Operations",
            "Aerospace / Aeronautical",
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

    if branch == "Civil Engineering":
        return [
            "Structural Engineering",
            "Construction Management",
            "Site Engineering",
            "Infrastructure",
            "Surveying",
            "Other"
        ]

    if branch == "Chemical Engineering":
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


# ============================================================
# DEGREE OPTIONS
# ============================================================

UG_DEGREES = [
    "BE",
    "BTech",
    "BSc",
    "BCA",
    "BBA",
    "BCom",
    "BA",
    "Other"
]

PG_DEGREES = [
    "MTech",
    "ME",
    "MSc",
    "MCA",
    "MBA",
    "MCom",
    "MA",
    "Other"
]


# ============================================================
# MODEL BRANCH MAPPING
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
# MODEL COMPATIBILITY HELPERS
# ============================================================

def normalize_degree_for_model(degree, feature_columns):

    possible_values = set()

    for column in feature_columns:

        if column.startswith("degree_"):

            possible_values.add(
                column.replace("degree_", "", 1)
            )

    if not possible_values:
        return degree

    if degree in possible_values:
        return degree

    degree_aliases = {

        "BTech": ["BTech", "BE", "B.E", "B.E.", "Bachelor of Engineering"],
        "BE": ["BE", "BTech", "B.E", "B.E.", "Bachelor of Engineering"],
        "BSc": ["BSc", "B.Sc", "B.Sc.", "Bachelor of Science"],
        "BCA": ["BCA", "Bachelor of Computer Applications"],
        "BBA": ["BBA", "Bachelor of Business Administration"],
        "BCom": ["BCom", "B.Com", "B.Com.", "Bachelor of Commerce"],
        "BA": ["BA", "B.A", "B.A.", "Bachelor of Arts"]
    }

    candidates = degree_aliases.get(
        degree,
        [degree]
    )

    for candidate in candidates:

        if candidate in possible_values:
            return candidate

    return degree


def normalize_gender_for_model(gender, feature_columns):

    possible_values = set()

    for column in feature_columns:

        if column.startswith("gender_"):

            possible_values.add(
                column.replace("gender_", "", 1)
            )

    if gender in possible_values:
        return gender

    gender_aliases = {
        "Male": ["Male", "M", "male", "m"],
        "Female": ["Female", "F", "female", "f"]
    }

    for candidate in gender_aliases.get(
        gender,
        [gender]
    ):

        if candidate in possible_values:
            return candidate

    return gender


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

    return (
        model,
        list(feature_names),
        recommendation_rules
    )


try:

    (
        model,
        feature_names,
        recommendation_rules
    ) = load_components()

except Exception as e:

    st.error(
        "Unable to load the placement prediction model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_student_data(
    student,
    feature_columns
):

    model_degree = normalize_degree_for_model(
        student["degree"],
        feature_columns
    )

    model_gender = normalize_gender_for_model(
        student["gender"],
        feature_columns
    )

    model_branch = student["branch"]

    df = pd.DataFrame([{

        "gender": model_gender,

        "age": student["age"],

        "degree": model_degree,

        "branch": model_branch,

        "cgpa": student["cgpa"],

        "backlogs": student["backlogs"],

        "internships": student["internships"],

        "certifications": student["certifications"],

        "coding_skills": student["coding_skills"],

        "communication_skills": student["communication_skills"],

        "aptitude_score": student["aptitude_score"],

        "projects": student["projects"]

    }])

    def get_cgpa_category(cgpa):

        if cgpa < 6:
            return "Low"

        elif cgpa < 7.5:
            return "Good"

        return "Excellent"

    df["cgpa_category"] = df["cgpa"].apply(
        get_cgpa_category
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

    df = df[
        feature_columns
    ]

    return df


# ============================================================
# PREDICTION
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

        probabilities = model.predict_proba(
            processed
        )[0]

        if len(probabilities) >= 2:

            probability = float(
                probabilities[1]
            )

        else:

            probability = float(
                probabilities[0]
            )

    except Exception:

        probability = float(
            prediction
        )

    return (
        prediction,
        probability,
        processed
    )


# ============================================================
# RELIABILITY
# ============================================================

def get_reliability(probability):

    distance = abs(
        probability - 0.5
    ) * 2

    if distance >= 0.70:
        return "High"

    elif distance >= 0.40:
        return "Moderate"

    return "Low"


# ============================================================
# READINESS
# ============================================================

def get_readiness(probability):

    if probability >= 0.75:

        return (
            "High Placement Readiness",
            "🟢"
        )

    elif probability >= 0.50:

        return (
            "Moderate Placement Readiness",
            "🟡"
        )

    return (
        "Needs Improvement",
        "🔴"
    )


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

        if student["ug_branch"] in [
            "Computer Science",
            "Information Technology",
            "Data Science",
            "Artificial Intelligence",
            "Machine Learning",
            "Cyber Security",
            "Software Engineering",
            "Computer Applications"
        ]:

            recommendations.append(
                "Develop stronger programming and computational skills relevant to your career direction."
            )

    if not recommendations:

        recommendations.append(
            "Continue strengthening practical experience, advanced skills, projects, and interview preparation."
        )

    return recommendations


# ============================================================
# GEMINI TEXT EXTRACTION
# ============================================================

def extract_gemini_text(interaction):

    try:

        text = getattr(
            interaction,
            "output_text",
            None
        )

        if text:

            return str(text)

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

                    collected.append(
                        str(text)
                    )

                else:

                    content = getattr(
                        item,
                        "content",
                        None
                    )

                    if content:

                        for content_item in content:

                            content_text = getattr(
                                content_item,
                                "text",
                                None
                            )

                            if content_text:

                                collected.append(
                                    str(content_text)
                                )

            if collected:

                return "\n".join(
                    collected
                )

    except Exception:
        pass

    return None


# ============================================================
# GEMINI CAREER GUIDANCE
# ============================================================

def generate_gemini_guidance(
    student,
    skill_gaps,
    prediction_result=None
):

    gemini_api_key = student.get(
        "gemini_api_key",
        ""
    )

    if not gemini_api_key.strip():

        raise ValueError(
            "Please enter your Gemini API key."
        )

    try:

        from google import genai

    except ImportError:

        raise ImportError(
            "Google GenAI package is not installed. "
            "Make sure google-genai is present in requirements.txt."
        )

    branch_skills_text = "\n".join(
        [
            f"- {skill}: {score}/10"
            for skill, score
            in student["branch_skills"].items()
        ]
    )

    critical_text = ", ".join(
        [
            item["skill"]
            for item in skill_gaps["critical_gaps"]
        ]
    )

    development_text = ", ".join(
        [
            item["skill"]
            for item in skill_gaps["development_needed"]
        ]
    )

    strengths_text = ", ".join(
        [
            item["skill"]
            for item in skill_gaps["strengths"]
        ]
    )

    recommendations = generate_recommendations(
        student
    )

    recommendations_text = "\n".join(
        [
            f"- {item}"
            for item in recommendations
        ]
    )

    if student["has_pg"]:

        pg_information = f"""
PG Degree:
{student["pg_degree"]}

PG Specialization:
{student["pg_branch"]}

PG CGPA:
{student["pg_cgpa"]}
"""

    else:

        pg_information = (
            "No postgraduate education provided."
        )

    if prediction_result is not None:

        prediction_label = (

            "Likely Placed"

            if prediction_result["prediction"] == 1

            else

            "Not Likely Placed"
        )

        prediction_information = f"""
Machine Learning Placement Prediction:
{prediction_label}

Estimated Placement Probability:
{prediction_result["probability"] * 100:.2f}%

Prediction Reliability:
{prediction_result["reliability"]}

Placement Readiness:
{prediction_result["readiness"]}
"""

    else:

        prediction_information = """
No machine-learning placement prediction was requested.

Provide career guidance independently using the student's
education, branch, skills and career direction.
"""

    prompt = f"""
You are an expert career counselor, academic advisor,
and placement preparation mentor.

Provide highly personalized, realistic, practical and
actionable career guidance for this student.

The student's ACTUAL academic branch is the PRIMARY
academic context.

The student's Career Interest and Target Career Goal
must strongly influence your recommendations.

Do NOT automatically recommend Computer Science,
programming, software development or coding to
students from non-computing fields.

For non-computing students, prioritize:

- Their actual academic domain
- Domain-specific professional skills
- Practical field skills
- Industry tools
- Laboratory methods where relevant
- Analytical methods
- Business tools where relevant
- Communication and professional skills
- Field-specific projects
- Internships
- Certifications
- Interview preparation

Never guarantee placement or employment.

Do not mention resumes.

Do not discuss internal model encoding.

Do not discuss hypothetical what-if scenarios.

============================================================
STUDENT EDUCATION
============================================================

UG Degree:
{student["ug_degree"]}

UG Branch:
{student["ug_branch"]}

UG CGPA:
{student["cgpa"]}

{pg_information}

============================================================
CAREER DIRECTION
============================================================

Career Interest:
{student["career_interest"]}

Target Career Goal:
{student["target_career_goal"] or "Not specifically provided"}

============================================================
STUDENT PROFILE
============================================================

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

Overall Domain Skill Score:
{student["domain_skills"]}/10

============================================================
BRANCH-SPECIFIC SKILLS
============================================================

{branch_skills_text}

============================================================
SKILL GAP ANALYSIS
============================================================

Critical Skill Gaps:
{critical_text or "None"}

Skills Needing Development:
{development_text or "None"}

Current Strengths:
{strengths_text or "None"}

============================================================
PLACEMENT INFORMATION
============================================================

{prediction_information}

============================================================
CURRENT IMPROVEMENT RECOMMENDATIONS
============================================================

{recommendations_text}

============================================================
IMPORTANT RULES
============================================================

1. Use the student's actual academic branch as the
   primary academic context.

2. Career Interest and Target Career Goal should strongly
   influence the guidance.

3. Do not automatically recommend software development
   or programming to non-computing students.

4. Recommend field-specific tools and professional skills.

5. Recommend projects that match the actual academic field.

6. Consider both UG and PG education when PG is available.

7. Explain skill gaps clearly.

8. Make the 30-day plan practical and measurable.

9. Never guarantee placement or employment.

10. Give EXACTLY 3 project ideas.

11. The 3 projects must be genuinely relevant to the
    student's academic branch and career direction.

12. Do not give generic projects when a field-specific
    project is possible.

============================================================
REQUIRED OUTPUT FORMAT
============================================================

Use EXACTLY these sections:

## Overall Profile Assessment

Assess the student's academic background, practical
exposure, current skills and career readiness.

## Career Direction

Explain how the student's actual academic branch,
career interest and target career goal connect.

## Recommended Career Paths

Give relevant career paths for the student's actual
academic field and career interest.

## Top Strengths

Identify the student's strongest academic, practical,
technical and professional areas.

## Skill Gap Analysis

Explain the student's critical skill gaps and
development areas.

## Areas to Improve

Give specific and actionable improvements related to
the student's actual academic field.

## 30-Day Improvement Plan

### Week 1

Give practical measurable tasks.

### Week 2

Give practical measurable tasks.

### Week 3

Give practical measurable tasks.

### Week 4

Give practical measurable tasks.

## Technical Topics to Study

Recommend field-specific technical, academic or
professional topics.

## Industry Tools and Professional Skills

Recommend appropriate industry tools, software,
laboratory methods, business tools, analytical tools,
professional methods or domain-specific skills.

## Project Ideas

Give EXACTLY 3 practical project ideas.

Project 1:
Include title, objective, skills/tools and expected outcome.

Project 2:
Include title, objective, skills/tools and expected outcome.

Project 3:
Include title, objective, skills/tools and expected outcome.

The three projects MUST match the student's actual
academic field and career direction.

Do NOT automatically suggest software projects.

Engineering students should receive engineering projects.

Science students should receive scientific, laboratory,
research or analytical projects where appropriate.

Food and nutrition students should receive food,
nutrition, dietetics, food safety or food analysis projects.

Commerce and finance students should receive accounting,
finance, auditing, banking or business projects.

Management, marketing and HR students should receive
business, management, marketing or HR projects.

Humanities students should receive research,
communication, public-policy, documentation or other
appropriate humanities projects.

## Interview Preparation

Include:

- Core technical preparation
- HR preparation
- Communication improvement
- Project explanation
- Internship explanation
- Aptitude preparation where relevant

Keep the advice practical, realistic and personalized.

Do not guarantee placement.
"""

    client = genai.Client(
        api_key=gemini_api_key.strip()
    )

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        store=False
    )

    return extract_gemini_text(
        interaction
    )


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
    "Predict placement readiness using machine learning "
    "and receive personalized, field-specific AI career guidance."
)

st.info(
    "⚠️ This system provides decision support based on a "
    "trained machine-learning model. It does not guarantee placement."
)


# ============================================================
# MODE SELECTION
# ============================================================

st.header(
    "🚀 Choose a Service"
)

service_mode = st.radio(
    "What would you like to use?",
    [
        "🔮 Placement Prediction",
        "🤖 AI Career Guidance"
    ],
    horizontal=True
)


# ============================================================
# EDUCATION PROFILE
# ============================================================

st.header(
    "🎓 Education Profile"
)

col1, col2 = st.columns(2)


with col1:

    ug_degree = st.selectbox(
        "UG Degree",
        UG_DEGREES
    )

    ug_branch = st.selectbox(
        "UG Branch / Major",
        BRANCH_OPTIONS
    )


with col2:

    has_pg = st.checkbox(
        "I have postgraduate education"
    )

    if has_pg:

        pg_degree = st.selectbox(
            "PG Degree",
            PG_DEGREES
        )

        pg_branch = st.text_input(
            "PG Specialization",
            placeholder="Example: Data Analytics"
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

st.header(
    "👤 Student Profile"
)

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
        max_value=60,
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
        max_value=20,
        value=1
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=30,
        value=2
    )


# ============================================================
# PLACEMENT PROFILE
# ============================================================

st.header(
    "📚 Placement Profile"
)

col1, col2, col3 = st.columns(3)


with col1:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=30,
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


# ============================================================
# GENERAL COMPUTATIONAL SKILLS
# ============================================================

st.subheader(
    "💻 General Programming / Computational Skills"
)

coding_skills = st.slider(
    "General Programming / Computational Skills",
    min_value=1,
    max_value=10,
    value=5,
    help=(
        "For computing branches this represents programming "
        "ability. For non-computing branches it can represent "
        "computational or digital skills where applicable."
    )
)


# ============================================================
# CAREER DIRECTION
# ============================================================

st.header(
    "🎯 Career Direction"
)

career_options = get_career_interests(
    ug_branch
)

col1, col2 = st.columns(2)


with col1:

    career_interest = st.selectbox(
        "Career Interest",
        career_options
    )


with col2:

    target_career_goal = st.text_input(
        "Target Career Goal",
        placeholder=(
            "Example: Data Analyst, Clinical Nutritionist, "
            "Financial Analyst, Civil Site Engineer..."
        )
    )


# ============================================================
# BRANCH-SPECIFIC SKILLS
# ============================================================

st.header(
    "🧠 Branch-Specific Skills"
)

st.write(
    f"Rate your current ability in skills relevant to "
    f"**{ug_branch}**."
)

skills = get_branch_skills(
    ug_branch
)

branch_skill_scores = {}

skill_columns = st.columns(
    len(skills)
)

for index, skill in enumerate(skills):

    with skill_columns[index]:

        branch_skill_scores[skill] = st.slider(
            skill,
            min_value=1,
            max_value=10,
            value=5,
            key=f"skill_{skill}"
        )


# ============================================================
# BUILD STUDENT DATA
# ============================================================

domain_skill_average = round(
    np.mean(
        list(
            branch_skill_scores.values()
        )
    ),
    2
)

student = {

    "gender": gender,

    "age": age,

    # IMPORTANT:
    # These are the model-compatible names.
    "degree": ug_degree,

    "branch": map_branch_for_model(
        ug_branch
    ),

    # User-facing fields
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

    "domain_skills": domain_skill_average
}


# ============================================================
# PLACEMENT PREDICTION MODE
# ============================================================

if service_mode == "🔮 Placement Prediction":

    st.divider()

    st.header(
        "🔮 Placement Prediction"
    )

    st.write(
        "Use the trained machine-learning model to estimate "
        "placement readiness from the academic and placement profile."
    )

    if st.button(
        "🔮 Predict Placement Readiness",
        width="stretch"
    ):

        try:

            prediction, probability, processed = (
                predict_placement(
                    student
                )
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

                "readiness_icon": readiness_icon,

                "recommendations": recommendations,

                "skill_gaps": skill_gaps

            }

            st.session_state.prediction_result = result

            st.session_state.student_data = student

            st.session_state.ai_career_advice = None

            st.success(
                "Placement prediction completed successfully."
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if (
    service_mode == "🔮 Placement Prediction"
    and
    st.session_state.prediction_result is not None
):

    result = (
        st.session_state.prediction_result
    )

    st.divider()

    st.header(
        "🎯 Placement Prediction Result"
    )

    prediction_text = (

        "LIKELY PLACED"

        if result["prediction"] == 1

        else

        "NOT LIKELY PLACED"
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
            f"{result['readiness_icon']} "
            f"{result['readiness']}"
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
        "The probability is the model's estimated probability "
        "for the positive placement class."
    )


    # ========================================================
    # SKILL GAP ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "📊 Skill Gap Analysis"
    )

    gaps = result["skill_gaps"]


    if gaps["critical_gaps"]:

        st.subheader(
            "🔴 Critical Skill Gaps"
        )

        for item in gaps["critical_gaps"]:

            st.write(
                f"**{item['skill']}** — "
                f"{item['score']}/10"
            )

    else:

        st.success(
            "No critical skill gaps detected."
        )


    if gaps["development_needed"]:

        st.subheader(
            "🟡 Skills Needing Development"
        )

        for item in gaps["development_needed"]:

            st.write(
                f"**{item['skill']}** — "
                f"{item['score']}/10"
            )


    if gaps["strengths"]:

        st.subheader(
            "🟢 Current Strengths"
        )

        for item in gaps["strengths"]:

            st.write(
                f"**{item['skill']}** — "
                f"{item['score']}/10"
            )


    # ========================================================
    # RULE-BASED RECOMMENDATIONS
    # ========================================================

    st.divider()

    st.header(
        "💡 Personalized Improvement Recommendations"
    )

    for recommendation in result["recommendations"]:

        st.write(
            f"• {recommendation}"
        )


# ============================================================
# AI CAREER GUIDANCE MODE
# ============================================================

if service_mode == "🤖 AI Career Guidance":

    st.divider()

    st.header(
        "🤖 AI Career Guidance"
    )

    st.write(
        "Generate personalized career guidance using Gemini. "
        "This option works independently and does not require "
        "a placement prediction first."
    )

    st.info(
        "Your actual academic branch is used as the primary "
        "academic context. Gemini will tailor the guidance "
        "to your branch, career interest and target career goal."
    )

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        key="gemini_ai_only_key",
        help=(
            "Your API key is used only for the current "
            "session to generate career guidance."
        )
    )

    student["gemini_api_key"] = (
        gemini_api_key
    )

    existing_prediction = (
        st.session_state.prediction_result
    )

    if existing_prediction is not None:

        st.success(
            "A placement prediction is available. "
            "Gemini can use it as additional context."
        )

    else:

        st.caption(
            "No placement prediction is required for AI guidance."
        )


    if st.button(
        "🤖 Generate AI Career Guidance",
        width="stretch",
        key="generate_ai_guidance"
    ):

        if not gemini_api_key.strip():

            st.warning(
                "Please enter your Gemini API key."
            )

        else:

            try:

                skill_gaps = analyze_skill_gaps(
                    branch_skill_scores
                )

                with st.spinner(
                    "Generating personalized Gemini career guidance..."
                ):

                    advice = generate_gemini_guidance(
                        student,
                        skill_gaps,
                        existing_prediction
                    )

                if advice:

                    st.session_state.ai_career_advice = (
                        advice
                    )

                    st.session_state.student_data = (
                        student
                    )

                    st.success(
                        "AI Career Guidance Generated Successfully!"
                    )

                else:

                    st.error(
                        "Gemini returned no readable text."
                    )

            except ImportError as e:

                st.error(
                    "Google GenAI package is missing."
                )

                st.code(
                    str(e)
                )

            except Exception as e:

                st.error(
                    "Gemini career guidance could not be generated."
                )

                st.code(
                    str(e)
                )


# ============================================================
# DISPLAY AI GUIDANCE
# ============================================================

if (
    st.session_state.ai_career_advice
    and
    service_mode == "🤖 AI Career Guidance"
):

    st.divider()

    st.header(
        "🤖 Personalized AI Career Guidance"
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
