# 🎓 AI Student Placement Predictor

An AI-powered Streamlit application that predicts student placement outcomes and provides personalized career guidance using Machine Learning and Generative AI.

## 🚀 Live Demo

👉 https://student-placement-predictor-1625.streamlit.app/

## 📌 About the Project

The **AI Student Placement Predictor** is designed to help students understand their placement readiness based on their academic profile, skills, internships, projects, certifications, aptitude, communication skills, and career interests.

The application combines a Machine Learning placement prediction model with Generative AI-powered career guidance.

## ✨ Features

- 🔮 Placement prediction
- 📊 Placement probability estimation
- 📈 Profile readiness analysis
- 🎓 Support for multiple academic branches and disciplines
- 💼 Internship assessment
- 🛠️ Project experience assessment
- 🏆 Certification assessment
- 🧠 Aptitude and communication assessment
- 🎯 Career interest selection
- 🤖 Personalized AI career guidance
- 💡 Skill-based career recommendations
- 🔄 Groq + Gemini AI provider fallback
- 🌐 Streamlit deployment

## 🤖 AI Career Guidance

The application provides personalized career guidance based on the student's:

- Academic background
- Skills
- Internships
- Projects
- Certifications
- Communication skills
- Aptitude
- Career interests
- Target career

AI guidance is generated using **Groq**, with **Google Gemini** available as a fallback provider.

## 🧠 Machine Learning

The placement prediction component uses a **Random Forest** machine-learning model.

The application provides a placement probability estimate based on the trained model and a separate profile readiness analysis to help students understand areas that may need improvement.

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Imbalanced-learn
- Groq API
- Google Gemini API
- Joblib

## 📂 Project Structure

Ai-Placement-Predictor/
│
├── streamlit_app.py
├── requirements.txt
│
├── placement_prediction_final.pkl
├── placement_feature_names_final.pkl
├── placement_model_metadata.pkl
├── recommendation_rules.pkl
│
├── README.md
├── LICENSE
└── .gitignore
