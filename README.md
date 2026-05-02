# 🏥 Dual Disease Early Risk Screener
### *AI-Powered Clinical Screening System for Diabetes & Chronic Kidney Disease*

---

> ⚠️ **Disclaimer:** This application is for **educational and screening purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified physician before making any medical decisions.

---

## 📌 Project Overview

The **Dual Disease Early Risk Screener** is a full-stack medical web application that uses **Deep Learning (Neural Networks)** to predict the risk of two critical diseases:

- 🩸 **Diabetes Mellitus**
- 🫘 **Chronic Kidney Disease (CKD)**

Users enter basic health metrics through a clean browser-based form. The app processes the inputs through two independently trained neural networks and returns a **personalized clinical report** — complete with risk predictions, a diet plan, foods to avoid, and lifestyle recommendations.

No external APIs. No database. Just health inputs in → intelligent predictions out.

---

## ✨ Features

- 🤖 **Two Deep Learning Models** — one for Diabetes, one for CKD, each trained on real medical datasets
- 📋 **Personalized Clinical Report** — tailored diet plans, lifestyle advice, and food restrictions based on your specific results
- 🎨 **Professional Medical UI** — clinical-grade design with Playfair Display, Lato, and DM Mono fonts
- 📱 **Fully Responsive** — side-by-side cards on desktop, stacked on mobile (≤640px)
- 🖨️ **Print-Ready Report** — one-click browser print button
- 🛡️ **Server-Side Validation** — all inputs range-validated on the backend
- 🐳 **Dockerized** — models train at build time, app runs instantly
- ⚡ **No JavaScript Framework** — pure HTML/CSS + Jinja2 server-side rendering

---

## 🗂️ Project Structure

```
project/
│
├── app.py                          ← Flask backend (routes, validation, prediction)
├── train_diabetes.py               ← Diabetes model training script
├── train_ckd.py                    ← CKD model training script
├── requirements.txt                ← Python dependencies (pinned versions)
├── Dockerfile                      ← Container definition
│
├── datasets/
│   ├── diabetes.csv                ← Pima Indians Diabetes Dataset
│   └── kidney_disease.csv          ← UCI CKD Dataset
│
├── templates/
│   ├── index.html                  ← Input form (GET /)
│   ├── result.html                 ← Clinical report (POST /predict)
│   └── error.html                  ← Validation/model error page
│
└── models/                         ← Auto-generated after training
    ├── diabetes_model.h5
    ├── diabetes_scaler.pkl
    ├── diabetes_feature_columns.pkl
    ├── ckd_model.h5
    ├── ckd_scaler.pkl
    ├── ckd_feature_columns.pkl
    ├── ckd_label_encoders.pkl
    ├── diabetes_training_plot.png
    └── ckd_training_plot.png
```

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.0.3 |
| **Deep Learning** | TensorFlow 2.15.0 / Keras |
| **Data Processing** | Pandas 2.2.2, NumPy 1.26.4 |
| **ML Utilities** | scikit-learn 1.5.0, joblib 1.4.2 |
| **Visualization** | Matplotlib 3.9.0 |
| **Frontend** | HTML5, CSS3, Jinja2 |
| **Fonts** | Google Fonts (Playfair Display, Lato, DM Mono, Inter) |
| **Containerization** | Docker (python:3.10-slim) |

---

## 🧠 Model Architecture

Both models share the same neural network design:

```
Input Layer  →  Dense(64, ReLU)  →  BatchNorm  →  Dropout(0.3)
             →  Dense(32, ReLU)  →  BatchNorm  →  Dropout(0.2)
             →  Dense(16, ReLU)  →  Dropout(0.2)
             →  Dense(1, Sigmoid)   ← Binary probability output
```

| Setting | Diabetes Model | CKD Model |
|---|---|---|
| **Features** | 5 | 8 |
| **Batch Size** | 32 | 16 |
| **Optimizer** | Adam | Adam |
| **Loss** | Binary Crossentropy | Binary Crossentropy |
| **Max Epochs** | 150 | 150 |
| **Early Stopping** | Patience = 15 | Patience = 15 |
| **Class Imbalance** | Class Weights | — |
| **Train/Test Split** | 80/20 Stratified | 80/20 Stratified |

---

## 📥 Input Features

### 🩸 Diabetes Card
| Feature | Range | Description |
|---|---|---|
| Glucose | 0 – 300 mg/dL | Fasting blood sugar level |
| BMI | 0 – 70 kg/m² | Body Mass Index |
| Age | 1 – 120 years | Patient age |
| Diabetes Pedigree Function | 0 – 3 | Family history score |
| Pregnancies | 0 – 20 | Number of pregnancies (0 if male) |

### 🫘 Kidney Disease Card
| Feature | Range | Description |
|---|---|---|
| Age | 1 – 120 years | Patient age |
| Blood Pressure | 0 – 200 mm Hg | Diastolic blood pressure |
| Albumin | 0 – 5 scale | Urine albumin level |
| Hemoglobin | 0 – 20 g/dL | Blood hemoglobin |
| Serum Creatinine | 0 – 30 mg/dL | Kidney waste clearance indicator |
| Hypertension | yes / no | Dropdown |
| Diabetes Mellitus | yes / no | Dropdown |
| Appetite | good / poor | Dropdown |

---

## 🚀 Getting Started

### Option 1 — Run Locally (Python)

**1. Clone the repository**
```bash
git clone https://github.com/your-username/dual-disease-screener.git
cd dual-disease-screener
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Place datasets** in the `datasets/` folder:
- `datasets/diabetes.csv`
- `datasets/kidney_disease.csv`

**4. Train the models** *(run once before starting the app)*
```bash
python train_diabetes.py
python train_ckd.py
```

**5. Start the Flask server**
```bash
python app.py
```

**6. Open your browser** and go to:
```
http://localhost:5000
```

---

### Option 2 — Run with Docker 🐳

```bash
# Build the image (trains models automatically during build)
docker build -t dual-disease-screener .

# Run the container
docker run -p 5000:5000 dual-disease-screener
```

Then open `http://localhost:5000` in your browser.

> 💡 The Dockerfile trains both models during `docker build`, so no manual training step is needed.

---

## 🔄 Application Flow

```
User fills form (index.html)
        ↓
Browser sends POST /predict
        ↓
app.py validates all inputs (server-side range checks)
        ↓  ← ValueError → error.html (HTTP 400)
app.py encodes categorical inputs (LabelEncoder)
        ↓
app.py scales numeric inputs (StandardScaler)
        ↓
Two neural networks predict independently
        ↓
result.html renders personalized clinical report
(predictions + diet + lifestyle + disclaimer)
```

---

## 📊 Report Sections

The results page is structured like a formal clinical report with **7 numbered sections**:

| # | Section | Content |
|---|---|---|
| 01 | Prediction Results | Color-coded risk cards (red = detected, green = clear) with confidence % |
| 02 | Clinical Summary | Condition-specific prose narrative with specialist recommendations |
| 03 | Personalised Diet Plan | 4 meal cards: Morning, Lunch, Dinner, Snacks — tailored to your results |
| 04 | Foods to Avoid | Pill-tag list of restricted foods based on detected conditions |
| 05 | Lifestyle Recommendations | 6-item grid: Exercise, Hydration, Sleep, Monitor, Stress, No Smoking |
| 06 | Important Notice | Medical disclaimer |
| 07 | Actions | Back to form + Print Report |

---

## 🎨 Design System

### Colour Palette

| Name | Hex | Usage |
|---|---|---|
| Ink | `#1c1917` | Dark header, primary text |
| Cream | `#faf8f4` | Results page background |
| Light Blue | `#f0f4f8` | Form page background |
| Risk Red | `#dc2626` | Detected condition cards |
| Clear Green | `#16a34a` | No-risk cards |
| Muted | `#718096` | Hints, secondary text |
| Border | `#e2e8f0` | Card and input borders |

### Typography
- **Playfair Display** — Headings on results page (serif, clinical authority)
- **Lato** — Body text (clean, highly readable)
- **DM Mono** — Labels, tags, metadata (monospace, technical precision)
- **Inter** — Form page (modern, neutral)

---

## 🛡️ Data Preprocessing

### Diabetes
- Zeros in `Glucose` and `BMI` replaced with column **medians** (biologically impossible values treated as missing)
- Features scaled using `StandardScaler` (mean=0, std=1)

### CKD
- `id` column dropped (not a medical feature)
- All column names and string values stripped of whitespace
- `?` characters replaced with `NaN`
- Target labels normalized: `'ckd'`, `'ckd\t'` → `1`; `'notckd'`, `'not ckd'` → `0`
- Numeric NaNs filled with column **median**
- Categorical NaNs filled with column **mode**
- Categorical columns encoded with `LabelEncoder` (saved for inference)

---

## 📁 Saved Model Artifacts

| File | Purpose |
|---|---|
| `diabetes_model.h5` | Full Keras model (architecture + weights) |
| `diabetes_scaler.pkl` | Fitted StandardScaler for diabetes features |
| `diabetes_feature_columns.pkl` | Ordered list of 5 feature names |
| `ckd_model.h5` | Full Keras model for CKD |
| `ckd_scaler.pkl` | Fitted StandardScaler for CKD features |
| `ckd_feature_columns.pkl` | Ordered list of 8 feature names |
| `ckd_label_encoders.pkl` | Dict of LabelEncoders for htn, dm, appet |
| `*_training_plot.png` | Accuracy & loss curves per model |

---

## ⚠️ Error Handling

| Scenario | Response |
|---|---|
| Missing or non-numeric field | HTTP 400 + descriptive error message |
| Value out of valid range | HTTP 400 + field name + actual value + allowed range |
| Unknown categorical value | Silent fallback to first encoded class (no crash) |
| Model file not found at startup | App refuses to start with clear instructions |
| Any unexpected server error | HTTP 500 + generic user-facing message + full stack trace in server logs |

---

## 📦 Requirements

```
flask==3.0.3
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.0
tensorflow==2.15.0
joblib==1.4.2
matplotlib==3.9.0
```

---

## 🔮 Potential Improvements

- [ ] SHAP / LIME explainability to show which features drove each prediction
- [ ] Optimal classification threshold tuning (precision-recall curve)
- [ ] SMOTE oversampling as alternative to class weights
- [ ] Patient history tracking with a lightweight SQLite database
- [ ] HTTPS / authentication for real deployment
- [ ] Export report as PDF directly from the app
- [ ] Multi-language support for broader accessibility

---

## 📜 Datasets Used

| Dataset | Source | Rows | Features |
|---|---|---|---|
| Pima Indians Diabetes | UCI Machine Learning Repository | 768 | 9 |
| Chronic Kidney Disease | UCI Machine Learning Repository | 400 | 26 |

---

## 📄 License

This project is built for **educational purposes**. Please credit appropriately if you use or adapt this work.

---

<br><br>

```
██████╗ ██╗   ██╗    ███████╗██████╗  █████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗
██╔══██╗╚██╗ ██╔╝    ██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║██║
██████╔╝ ╚████╔╝     ███████╗██████╔╝███████║██║   ██║███████║██╔██╗ ██║██║
██╔══██╗  ╚██╔╝      ╚════██║██╔══██╗██╔══██║╚██╗ ██╔╝██╔══██║██║╚██╗██║██║
██████╔╝   ██║       ███████║██║  ██║██║  ██║ ╚████╔╝ ██║  ██║██║ ╚████║██║
╚═════╝    ╚═╝       ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝
```

### 👩‍💻 Designed & Developed with ❤️ by **SRAVANI**

---

