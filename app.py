from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

app = Flask(__name__)

print("📂 Loading models...")

try:
    diabetes_model  = tf.keras.models.load_model("models/diabetes_model.h5")
    diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
    diabetes_cols   = joblib.load("models/diabetes_feature_columns.pkl")

    ckd_model          = tf.keras.models.load_model("models/ckd_model.h5")
    ckd_scaler         = joblib.load("models/ckd_scaler.pkl")
    ckd_cols           = joblib.load("models/ckd_feature_columns.pkl")
    ckd_label_encoders = joblib.load("models/ckd_label_encoders.pkl")

    print("✅ All models loaded!")

except FileNotFoundError as e:
    print(f"❌ Model file missing: {e}")
    print("👉 Run train_ckd.py and train_diabetes.py first!")
    raise

def encode_cat(col_name, value):
    le = ckd_label_encoders[col_name]
    value = str(value).strip().lower()
    if value in le.classes_:
        return int(le.transform([value])[0])
    else:
        return int(le.transform([le.classes_[0]])[0])

def safe_float(form, field, min_val=None, max_val=None):
    try:
        val = float(form[field])
    except (ValueError, KeyError):
        raise ValueError(f"Invalid or missing value for '{field}'.")
    if min_val is not None and val < min_val:
        raise ValueError(f"'{field}' = {val} is below minimum {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"'{field}' = {val} is above maximum {max_val}.")
    return val

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        glucose     = safe_float(request.form, 'glucose',     0, 300)
        bmi         = safe_float(request.form, 'bmi',         0, 70)
        age         = safe_float(request.form, 'age',         1, 120)
        pedigree    = safe_float(request.form, 'pedigree',    0, 3)
        pregnancies = safe_float(request.form, 'pregnancies', 0, 20)

        diabetes_input = pd.DataFrame([[glucose, bmi, age, pedigree, pregnancies]], columns=diabetes_cols)
        diabetes_input_scaled = diabetes_scaler.transform(diabetes_input)
        diabetes_prob = float(diabetes_model.predict(diabetes_input_scaled)[0][0])
        diabetes_has  = diabetes_prob > 0.5
        diabetes_result     = "Diabetes Detected 🔴" if diabetes_has else "No Diabetes 🟢"
        diabetes_confidence = f"{diabetes_prob * 100:.1f}%" if diabetes_has else f"{(1 - diabetes_prob) * 100:.1f}%"

        ckd_age = safe_float(request.form, 'ckd_age', 1, 120)
        bp      = safe_float(request.form, 'bp',      0, 200)
        al      = safe_float(request.form, 'al',      0, 5)
        hemo    = safe_float(request.form, 'hemo',    0, 20)
        sc      = safe_float(request.form, 'sc',      0, 30)

        htn   = encode_cat('htn',   request.form.get('htn', 'no'))
        dm    = encode_cat('dm',    request.form.get('dm', 'no'))
        appet = encode_cat('appet', request.form.get('appet', 'good'))

        ckd_data = {'age': ckd_age, 'bp': bp, 'al': al, 'hemo': hemo, 'sc': sc, 'htn': htn, 'dm': dm, 'appet': appet}
        ckd_input = pd.DataFrame([ckd_data])[ckd_cols]
        ckd_input_scaled = ckd_scaler.transform(ckd_input)
        ckd_prob = float(ckd_model.predict(ckd_input_scaled)[0][0])
        ckd_has  = ckd_prob > 0.5
        ckd_result     = "CKD Detected 🔴" if ckd_has else "No CKD 🟢"
        ckd_confidence = f"{ckd_prob * 100:.1f}%" if ckd_has else f"{(1 - ckd_prob) * 100:.1f}%"

        return render_template('result.html',
            diabetes_result=diabetes_result, diabetes_confidence=diabetes_confidence, diabetes_has=diabetes_has,
            ckd_result=ckd_result, ckd_confidence=ckd_confidence, ckd_has=ckd_has)

    except ValueError as e:
        return render_template('error.html', error=str(e)), 400
    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template('error.html', error="Something went wrong. Please check your inputs and try again."), 500

if __name__ == '__main__':
    app.run(debug=True)