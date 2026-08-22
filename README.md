# 🩺 AI-Powered Multi-Disease Health Prediction System

An end-to-end **Machine Learning–based healthcare prediction platform** that predicts the likelihood of multiple chronic diseases through a web-based interface. The system combines disease-specific datasets, exploratory analysis, trained machine learning models, and a Flask-powered prediction backend to provide an integrated health-risk assessment experience.

> **Project Type:** Major Internship Project
> **Domain:** Machine Learning • Healthcare AI • Predictive Analytics • Web Application

---

## 🚀 Overview

The **AI-Powered Multi-Disease Health Prediction System** is designed to provide an accessible interface for preliminary health-risk prediction across multiple chronic diseases.

Instead of building separate applications for each disease, the project integrates multiple disease-specific machine learning pipelines into a **single web application**.

The current implementation covers:

* 🩸 **Diabetes Prediction**
* ❤️ **Heart Disease Prediction**
* 🫘 **Chronic Kidney Disease Prediction**
* 🫀 **Liver Disease Prediction**

The repository contains trained models, disease-specific datasets, exploratory analysis notebooks, a Flask backend, and the frontend components required to run the application.

---

## ✨ Key Features

### 🔬 Multi-Disease Prediction

A unified platform capable of performing predictions for multiple diseases using disease-specific machine learning models.

### 🧠 Disease-Specific ML Pipelines

Each prediction task has its own dataset and analysis workflow, allowing preprocessing and modeling to be adapted to the characteristics of the respective disease.

### 📊 Exploratory Data Analysis

Dedicated Jupyter notebooks are included for lifestyle and health-data analysis for diabetes, heart disease, kidney disease, and liver disease.

### 🌐 Web-Based Prediction Interface

Users can enter relevant health parameters through an interactive web interface and obtain predictions without directly interacting with the underlying ML models.

### ⚙️ Flask Backend

The backend is implemented using **Flask**, which handles prediction requests and connects the web interface with the trained machine learning models.

### 🧩 Modular Model Architecture

Disease-specific models are maintained separately under the `models/` directory, making the system easier to extend with additional prediction modules.

---

## 🏗️ System Architecture

```text
                  ┌─────────────────────────┐
                  │       User Input        │
                  │   Health / Lifestyle    │
                  │       Parameters        │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      Web Frontend       │
                  │     HTML / CSS / JS     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      Flask Backend      │
                  │       app.py            │
                  └────────────┬────────────┘
                               │
               ┌───────────────┼────────────────┐
               │               │                │
               ▼               ▼                ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │  Diabetes  │  │   Heart    │  │   Kidney   │
        │   Model    │  │   Model    │  │   Model    │
        └────────────┘  └────────────┘  └────────────┘
                               │
                               ▼
                        ┌────────────┐
                        │   Liver    │
                        │   Model    │
                        └────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Prediction / Result     │
                  │      Display            │
                  └─────────────────────────┘
```

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

### 1. Data Collection

Disease-specific datasets are used for model development, including datasets for diabetes, heart disease, chronic kidney disease, and liver disease. The repository stores these datasets directly alongside the application.

### 2. Exploratory Data Analysis

Disease-specific notebooks are used to investigate the underlying data, understand feature distributions, examine relationships between health variables, and prepare the datasets for modeling.

Examples include:

```text
diabetes_lifestyle.ipynb
heart_lifestyle.ipynb
kidney_lifestyle.ipynb
liver_lifestyle.ipynb
```

### 3. Data Preprocessing

The datasets are transformed into a format suitable for machine learning, including handling the disease-specific feature structure and preparing input variables for prediction.

### 4. Model Training

Separate machine learning models are trained for each disease instead of applying a single generic model across all health conditions.

### 5. Model Persistence

The trained models are stored under the `models/` directory and loaded by the application at prediction time.

### 6. Application Integration

The Flask backend receives user inputs, processes the corresponding features, invokes the appropriate disease model, and returns the prediction to the frontend.

---

## 🛠️ Tech Stack

### Machine Learning & Data Science

* Python
* Scikit-learn
* Pandas
* NumPy
* Jupyter Notebook
* Machine Learning Classification
* Exploratory Data Analysis
* Feature Engineering
* Data Preprocessing

### Backend

* Flask
* Python

### Frontend

* HTML
* JavaScript
* CSS

### Development & Version Control

* Git
* GitHub

The repository is primarily implemented in Python with supporting HTML and JavaScript components.

---

## 📂 Repository Structure

```text
AI-Powered-Multi-Disease-Health-Prediction-System/
│
├── models/
│   └── Trained ML Models
│
├── templates/
│   └── Web Templates
│
├── app.py
├── predict.js
├── index.html
├── result.html
├── book.html
│
├── Chronic_Kidney_Dsease_data.csv
├── diabetes_binary_health_indicators_BRFSS2015.csv
├── heart_2020_cleaned.csv
├── indian_liver_patient.csv
│
├── diabetes_lifestyle.ipynb
├── heart_lifestyle.ipynb
├── kidney_lifestyle.ipynb
├── liver_lifestyle.ipynb
│
├── appointments.csv
├── requirements.txt
└── README.md
```

This structure reflects the files currently present in the repository.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Anish1708Kumar/AI-Powered-Multi-Disease-Health-Prediction-System.git
```

### 2. Navigate to the Project

```bash
cd AI-Powered-Multi-Disease-Health-Prediction-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

Open the local Flask URL displayed in the terminal to access the application.

---

## 🔍 Disease Prediction Modules

### 🩸 Diabetes

The diabetes module uses health and lifestyle indicators to estimate the likelihood of diabetes-related outcomes.

Dataset:

```text
diabetes_binary_health_indicators_BRFSS2015.csv
```

### ❤️ Heart Disease

The heart disease module analyzes cardiovascular health-related features and predicts the likelihood of heart disease.

Dataset:

```text
heart_2020_cleaned.csv
```

### 🫘 Chronic Kidney Disease

The kidney disease module uses clinical attributes to generate a prediction for chronic kidney disease.

Dataset:

```text
Chronic_Kidney_Dsease_data.csv
```

### 🫀 Liver Disease

The liver disease module processes patient health characteristics to predict liver disease.

Dataset:

```text
indian_liver_patient.csv
```

These datasets are included directly in the repository.

---

## 📈 End-to-End Workflow

```text
Data
  ↓
Exploratory Data Analysis
  ↓
Data Cleaning & Preprocessing
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Persistence
  ↓
Flask Integration
  ↓
Web-Based User Input
  ↓
Disease Prediction
  ↓
Result Display
```

---

## 🎯 Project Objectives

The major objectives of the project are:

* Develop a unified **multi-disease prediction platform**.
* Apply machine learning to real-world healthcare datasets.
* Build disease-specific predictive pipelines.
* Integrate ML models into a functional web application.
* Provide a simple interface for health-risk assessment.
* Demonstrate the complete **ML-to-application deployment workflow**.

---

## 💡 Why This Project?

Healthcare datasets often contain large numbers of clinical and lifestyle variables that can be leveraged for predictive analytics.

This project explores how machine learning can transform such structured health data into an interactive application capable of providing **preliminary risk predictions across multiple disease categories**.

The emphasis is not only on model training but on the complete journey from **data analysis to a usable software application**.

---

## 🔮 Future Enhancements

* Add more disease prediction modules.
* Improve model performance through hyperparameter optimization.
* Add model comparison and evaluation dashboards.
* Introduce prediction probability / confidence scores.
* Integrate explainable AI techniques such as SHAP.
* Improve security and user authentication.
* Deploy the application to a cloud platform.
* Add personalized health recommendations.
* Develop a centralized patient dashboard.
* Introduce automated model monitoring and retraining.

---

## ⚠️ Disclaimer

This project is intended **strictly for educational and research purposes**.

The predictions generated by the system should not be treated as a medical diagnosis, treatment recommendation, or substitute for professional medical advice. Users should consult qualified healthcare professionals for actual medical decisions.

---

## 👨‍💻 Author

**Anish Kumar Mandal**

IIT Kanpur

GitHub:
https://github.com/Anish1708Kumar

---

## ⭐ Project Highlights

**Multi-Disease Prediction • Healthcare AI • Machine Learning • Scikit-learn • Python • Flask • Predictive Analytics • Data Preprocessing • Feature Engineering • Exploratory Data Analysis • Model Training • Web Application • End-to-End ML Pipeline**
