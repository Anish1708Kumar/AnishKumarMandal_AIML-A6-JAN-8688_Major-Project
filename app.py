from flask import Flask, render_template, request, send_file, session
import pickle
import numpy as np
import pandas as pd
import os
import io
import csv
import datetime
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "healthai2024"

# loading all saved models from the models folder
BASE = os.path.join(os.path.dirname(__file__), "models")

def load_model(name):
    path = os.path.join(BASE, name)
    file = open(path, "rb")
    model = pickle.load(file)
    return model

diabetes_model = load_model("diabetes_model.pkl")
diabetes_scaler = load_model("diabetes_scaler.pkl")
diabetes_features = load_model("diabetes_features.pkl")

heart_model = load_model("heart_model.pkl")
heart_scaler = load_model("heart_scaler.pkl")
heart_features = load_model("heart_features.pkl")
heart_age_map = load_model("heart_age_map.pkl")
heart_gen_map = load_model("heart_gen_map.pkl")

liver_model = load_model("liver_model.pkl")
liver_scaler = load_model("liver_scaler.pkl")
liver_features = load_model("liver_features.pkl")

kidney_model = load_model("kidney_model.pkl")
kidney_scaler = load_model("kidney_scaler.pkl")
kidney_features = load_model("kidney_features.pkl")

print("All 4 lifestyle models loaded.")


# helper function to convert yes/no from form to 1/0
def yes_no(form, key):
    if form.get(key) == "yes":
        return 1
    else:
        return 0


# this function creates the shap chart image and returns base64 string
def shap_chart(model, scaler, X, feature_names, title, color):
    feature_names = list(feature_names)

    # Background = all-zeros array (the mean of any StandardScaler-transformed dataset
    # is exactly 0, so this represents the "average" patient and gives meaningful SHAP diffs)
    background = np.zeros((1, X.shape[1]))

    explainer = shap.LinearExplainer(model, background)
    shap_vals = explainer.shap_values(X)

    # LinearExplainer returns a 2D array (n_samples, n_features).
    # For binary LogisticRegression it returns a single array for the positive class.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    sv = shap_vals[0]

    # get top 10 features by importance
    idx = np.argsort(np.abs(sv))[-10:]

    labels = []
    vals = []
    colors = []
    for i in idx:
        labels.append(feature_names[int(i)])
        vals.append(sv[int(i)])
        if sv[int(i)] >= 0:
            colors.append(color)
        else:
            colors.append("#555577")

    # create the bar chart
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="white", lw=0.6, alpha=0.5)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel("SHAP value", fontsize=8, color="#aaa")
    ax.tick_params(labelsize=7.5, colors="#ccc")
    ax.set_facecolor("#1c2230")
    fig.patch.set_facecolor("#161b22")
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")

    # save chart to buffer and convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return img_base64


# returns risk label and color based on probability
def risk_label(prob):
    if prob >= 0.65:
        return "High Risk", "#f85149"
    elif prob >= 0.38:
        return "Moderate Risk", "#fb8f44"
    else:
        return "Low Risk", "#3fb950"


# generates health tips based on prediction results
def get_tips(results, form):
    out = []

    smoker = False
    if form.get("smoker") == "yes":
        smoker = True

    alcohol = False
    if form.get("alcohol") == "yes":
        alcohol = True

    active = False
    if form.get("phys_active") == "yes":
        active = True

    bmi = float(form.get("bmi", 24))

    # diabetes tips
    if results["diabetes"]["prob"] >= 0.38:
        out.append("Cut down on sugar, white rice, and processed carbohydrates")
        out.append("A 30-minute brisk walk daily significantly lowers blood sugar")
        if bmi >= 30:
            out.append("Losing 5-7% of body weight cuts diabetes risk by over 50%")
        if not active:
            out.append("Even light exercise after meals helps regulate blood sugar")

    # heart tips
    if results["heart"]["prob"] >= 0.38:
        if smoker:
            out.append("Quitting smoking is the single biggest thing you can do for your heart")
        out.append("Reduce salt and fried food - these raise cholesterol and blood pressure")
        out.append("Get blood pressure and cholesterol checked at least once a year")

    # liver tips
    if results["liver"]["prob"] >= 0.38:
        if alcohol:
            out.append("Cutting alcohol is critical - the liver cannot heal while processing it")
        out.append("Avoid self-medicating with painkillers - many damage the liver over time")
        out.append("More vegetables and less fried food directly supports liver recovery")

    # kidney tips
    if results["kidney"]["prob"] >= 0.38:
        out.append("Drink 8-10 glasses of water daily - the simplest kidney protection")
        out.append("Avoid regular ibuprofen/diclofenac use - they damage kidneys over time")
        out.append("Controlling blood pressure and blood sugar prevents most kidney disease")

    # if no risk is high enough, give general tips
    if len(out) == 0:
        out.append("Maintain a balanced diet - more vegetables, less processed food")
        out.append("Exercise at least 30 minutes, 5 days a week")
        out.append("Get a full health checkup once a year, even if you feel fine")
        out.append("Stay hydrated - most people drink less water than they need")

    # remove duplicates
    unique = []
    for t in out:
        if t not in unique:
            unique.append(t)
    return unique


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.form

    # getting all the form inputs
    age = float(f.get("age", 40))
    bmi = float(f.get("bmi", 25))

    if f.get("sex") == "Male":
        sex_male = 1
    else:
        sex_male = 0

    smoker = yes_no(f, "smoker")
    alcohol = yes_no(f, "alcohol")
    active = yes_no(f, "phys_active")
    diabetic = yes_no(f, "diabetic")
    high_bp = yes_no(f, "high_bp")
    high_chol = yes_no(f, "high_chol")
    stroke = yes_no(f, "stroke")
    diff_walk = yes_no(f, "diff_walk")
    gen_health = int(f.get("gen_health", 3))
    sleep_hrs = float(f.get("sleep_hrs", 7))
    phys_days = float(f.get("phys_days", 0))
    age_cat = f.get("age_cat", "40-44")
    family_kid = yes_no(f, "family_kidney")
    family_diab = yes_no(f, "family_diabetes")
    family_htn = yes_no(f, "family_hypertension")
    family_hrt = yes_no(f, "family_heart")
    sys_bp = float(f.get("systolic_bp", 120))
    dia_bp = float(f.get("diastolic_bp", 80))
    diet_score = float(f.get("diet_score", 5))

    # symptom inputs
    jaundice = yes_no(f, "jaundice")
    fatigue = yes_no(f, "fatigue")
    nausea = yes_no(f, "nausea")
    abd_pain = yes_no(f, "abdominal_pain")
    loss_app = yes_no(f, "loss_appetite")
    dark_urine = yes_no(f, "dark_urine")
    swelling = yes_no(f, "swelling")
    muscle_cramps = yes_no(f, "muscle_cramps")
    itching = yes_no(f, "itching")
    uti = yes_no(f, "uti")
    prev_aki = yes_no(f, "prev_aki")
    fruits = yes_no(f, "fruits")
    veggies = yes_no(f, "veggies")
    kidney_hx = yes_no(f, "kidney_hx")

    # DIABETES prediction
    d_map = {
        "HighBP": high_bp, "HighChol": high_chol, "BMI": bmi,
        "Smoker": smoker, "PhysActivity": active,
        "Fruits": fruits, "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "GenHlth": gen_health, "PhysHlth": phys_days,
        "DiffWalk": diff_walk, "Sex": sex_male, "Age": age
    }

    d_row = []
    for feat in diabetes_features:
        d_row.append(d_map[feat])
    d_arr = pd.DataFrame([d_row], columns=diabetes_features)
    d_sc = diabetes_scaler.transform(d_arr)
    d_prob = float(diabetes_model.predict_proba(d_sc)[0][1])
    d_chart = shap_chart(diabetes_model, diabetes_scaler, d_sc,
                         diabetes_features, "Diabetes — Risk Factors", "#f85149")

    # HEART prediction
    age_enc = heart_age_map.get(age_cat, 6)

    # flipping the gen_health scale for heart model
    if gen_health == 1:
        gen_enc = 5
    elif gen_health == 2:
        gen_enc = 4
    elif gen_health == 3:
        gen_enc = 3
    elif gen_health == 4:
        gen_enc = 2
    elif gen_health == 5:
        gen_enc = 1
    else:
        gen_enc = 3

    h_map = {
        "BMI": bmi, "Smoking": smoker, "AlcoholDrinking": alcohol,
        "Stroke": stroke, "PhysicalHealth": phys_days,
        "DiffWalking": diff_walk, "Sex": sex_male,
        "AgeCategory": age_enc, "Diabetic": diabetic,
        "PhysicalActivity": active, "GenHealth": gen_enc,
        "SleepTime": sleep_hrs, "KidneyDisease": kidney_hx
    }

    h_row = []
    for feat in heart_features:
        h_row.append(h_map[feat])
    h_arr = pd.DataFrame([h_row], columns=heart_features)
    h_sc = heart_scaler.transform(h_arr)
    h_prob = float(heart_model.predict_proba(h_sc)[0][1])
    h_chart = shap_chart(heart_model, heart_scaler, h_sc,
                         heart_features, "Heart Disease — Risk Factors", "#fb8f44")

    # LIVER prediction
    l_map = {
        "Age": age, "Sex": sex_male, "BMI": bmi,
        "Alcohol": alcohol, "Smoker": smoker, "PhysActive": active,
        "DietScore": diet_score,
        "Diabetes": diabetic, "Hypertension": high_bp,
        "FamilyHistory": family_hrt,
        "Jaundice": jaundice, "Fatigue": fatigue, "Nausea": nausea,
        "AbdominalPain": abd_pain, "LossOfAppetite": loss_app,
        "DarkUrine": dark_urine
    }

    l_row = []
    for feat in liver_features:
        l_row.append(l_map[feat])
    l_arr = pd.DataFrame([l_row], columns=liver_features)
    l_sc = liver_scaler.transform(l_arr)
    l_prob = float(liver_model.predict_proba(l_sc)[0][1])
    l_chart = shap_chart(liver_model, liver_scaler, l_sc,
                         liver_features, "Liver Disease — Risk Factors", "#9b59b6")

    # KIDNEY prediction
    sleep_q = min(sleep_hrs, 10)  # keep it in 0-10 range
    diet_q = diet_score

    k_map = {
        "Age": age, "Gender": sex_male, "BMI": bmi,
        "Smoking": smoker, "AlcoholConsumption": alcohol * 5,
        "PhysicalActivity": active * 5,
        "DietQuality": diet_q, "SleepQuality": sleep_q,
        "FamilyHistoryKidneyDisease": family_kid,
        "FamilyHistoryHypertension": family_htn,
        "FamilyHistoryDiabetes": family_diab,
        "PreviousAcuteKidneyInjury": prev_aki,
        "UrinaryTractInfections": uti,
        "SystolicBP": sys_bp, "DiastolicBP": dia_bp,
        "Edema": swelling,
        "FatigueLevels": fatigue * 5,
        "NauseaVomiting": nausea * 5,
        "MuscleCramps": muscle_cramps * 5,
        "Itching": itching * 5
    }

    k_row = []
    for feat in kidney_features:
        k_row.append(k_map[feat])
    k_arr = pd.DataFrame([k_row], columns=kidney_features)
    k_sc = kidney_scaler.transform(k_arr)
    k_prob = float(kidney_model.predict_proba(k_sc)[0][1])
    k_chart = shap_chart(kidney_model, kidney_scaler, k_sc,
                         kidney_features, "Kidney Disease — Risk Factors", "#4f8ef7")

    # putting all results together
    results = {
        "diabetes": {"prob": d_prob, "chart": d_chart},
        "heart": {"prob": h_prob, "chart": h_chart},
        "liver": {"prob": l_prob, "chart": l_chart},
        "kidney": {"prob": k_prob, "chart": k_chart},
    }

    for key in results:
        lbl, clr = risk_label(results[key]["prob"])
        results[key]["label"] = lbl
        results[key]["color"] = clr
        results[key]["pct"] = round(results[key]["prob"] * 100, 1)

    tip_list = get_tips(results, f)
    patient_name = f.get("patient_name", "Patient")

    # saving to session so pdf route can use it
    session_results = {}
    for k in results:
        session_results[k] = {
            "prob": results[k]["prob"],
            "pct": results[k]["pct"],
            "label": results[k]["label"]
        }
    session["results"] = session_results
    session["patient_name"] = patient_name
    session["patient_age"] = str(int(age))
    session["tips"] = tip_list

    return render_template("result.html", results=results, tips=tip_list,
                           patient_name=patient_name, patient_age=int(age))


@app.route("/download_pdf")
def download_pdf():
    results = session.get("results", {})
    patient_name = session.get("patient_name", "Patient")
    patient_age = session.get("patient_age", "N/A")
    tip_list = session.get("tips", [])

    pdf = FPDF()
    pdf.add_page()

    # header background
    pdf.set_fill_color(20, 40, 80)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "HealthAI - Health Risk Assessment Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 20)
    now = datetime.datetime.now().strftime('%d %B %Y, %H:%M')
    pdf.cell(0, 6, "Generated: " + now, ln=True)

    # patient info
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 36)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient: " + patient_name + "   |   Age: " + patient_age, ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Screening only - consult a qualified doctor for diagnosis.", ln=True)
    pdf.ln(4)

    # risk summary
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Risk Summary", ln=True)
    pdf.ln(2)

    # disease names and their colors for the pdf
    diseases = ["diabetes", "heart", "liver", "kidney"]
    disease_names = {
        "diabetes": "Diabetes",
        "heart": "Heart Disease",
        "liver": "Liver Disease",
        "kidney": "Kidney Disease"
    }
    disease_colors = {
        "diabetes": (248, 81, 73),
        "heart": (251, 143, 68),
        "liver": (155, 89, 182),
        "kidney": (79, 142, 247)
    }

    for disease in diseases:
        r = results.get(disease, {})
        pct = r.get("pct", 0)
        lbl = r.get("label", "N/A")
        clr = disease_colors[disease]
        name = disease_names[disease]
        y = pdf.get_y()

        pdf.set_fill_color(245, 245, 250)
        pdf.set_draw_color(220, 220, 230)
        pdf.rect(10, y, 190, 20, "FD")
        pdf.set_fill_color(clr[0], clr[1], clr[2])
        pdf.ellipse(16, y + 6, 8, 8, "F")

        pdf.set_xy(28, y + 3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(80, 7, name, ln=False)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(50, 7, "Score: " + str(pct) + "%", ln=False)

        if lbl == "High Risk":
            pdf.set_text_color(200, 50, 50)
        elif lbl == "Moderate Risk":
            pdf.set_text_color(200, 100, 20)
        else:
            pdf.set_text_color(30, 140, 60)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, lbl, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

    # bar chart visualisation
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Risk Score Visualisation", ln=True)
    pdf.ln(2)

    short_names = {
        "diabetes": "Diabetes",
        "heart": "Heart",
        "liver": "Liver",
        "kidney": "Kidney"
    }

    for disease in diseases:
        r = results.get(disease, {})
        pct = r.get("pct", 0)
        clr = disease_colors[disease]

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(35, 7, short_names[disease], ln=False)

        # grey background bar
        pdf.set_fill_color(220, 220, 230)
        pdf.rect(47, pdf.get_y() + 1.5, 120, 5, "F")

        # colored fill bar
        bar_width = int(pct / 100 * 120)
        if bar_width < 1:
            bar_width = 1
        pdf.set_fill_color(clr[0], clr[1], clr[2])
        pdf.rect(47, pdf.get_y() + 1.5, bar_width, 5, "F")

        pdf.set_xy(170, pdf.get_y())
        pdf.cell(0, 7, str(pct) + "%", ln=True)
        pdf.ln(2)

    # recommendations
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Personalised Recommendations", ln=True)
    pdf.ln(2)

    for tip in tip_list:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(240, 248, 255)
        pdf.set_draw_color(200, 220, 240)
        pdf.rect(10, pdf.get_y(), 190, 11, "FD")
        pdf.set_xy(16, pdf.get_y() + 2)
        pdf.set_text_color(30, 58, 95)
        # replacing special dashes that fpdf cant handle
        safe_tip = tip.replace('\u2014', '-').replace('\u2013', '-')
        pdf.cell(0, 7, "-  " + safe_tip, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    # footer
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6, "AI-generated report - not a substitute for professional medical advice.",
             ln=True, align="C")

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)

    filename = "health_report_" + patient_name.replace(" ", "_") + ".pdf"
    return send_file(buf, as_attachment=True,
                     download_name=filename,
                     mimetype="application/pdf")


@app.route("/book", methods=["GET", "POST"])
def book():
    if request.method == "POST":
        f = request.form

        entry = {}
        fields = ["name", "email", "phone", "age", "specialty",
                   "doctor", "date", "time_slot", "notes"]
        for field in fields:
            entry[field] = f.get(field, "")

        entry["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        csv_path = os.path.join(os.path.dirname(__file__), "appointments.csv")
        write_hdr = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=entry.keys())
            if write_hdr:
                w.writeheader()
            w.writerow(entry)

        return render_template("book.html", success=True, **entry)

    # if GET request, suggest a specialist based on highest risk
    suggested = ""
    r = session.get("results", {})
    if r:
        # find which disease has the highest probability
        highest = None
        highest_prob = -1
        for key in r:
            if r[key]["prob"] > highest_prob:
                highest_prob = r[key]["prob"]
                highest = key

        if highest == "diabetes":
            suggested = "Endocrinologist"
        elif highest == "heart":
            suggested = "Cardiologist"
        elif highest == "liver":
            suggested = "Hepatologist"
        elif highest == "kidney":
            suggested = "Nephrologist"

    return render_template("book.html", success=False, suggested=suggested)


if __name__ == "__main__":
    app.run(debug=True)
