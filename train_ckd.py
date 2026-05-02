import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import os

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
print("📂 Loading CKD Dataset...")
df = pd.read_csv("datasets/kidney_disease.csv")
print(f"✅ Dataset loaded! Shape: {df.shape}")

# ─────────────────────────────────────────
# 2. CLEAN THE DATA
# ─────────────────────────────────────────
df.drop('id', axis=1, inplace=True)
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
df = df.replace('?', np.nan)

# Fix classification column
df['classification'] = df['classification'].astype(str).str.strip().str.lower()
df['classification'] = df['classification'].replace({
    'ckd': 1, 'ckd\t': 1,
    'notckd': 0, 'not ckd': 0, 'notckd\t': 0
})
df['classification'] = pd.to_numeric(df['classification'], errors='coerce')
df = df.dropna(subset=['classification'])
df['classification'] = df['classification'].astype(int)

print("\n✅ Classification distribution:")
print(df['classification'].value_counts())

# ─────────────────────────────────────────
# 3. SELECT ONLY THE 8 KEY FEATURES
# These are clinically meaningful and easy
# to collect without full lab panel
# ─────────────────────────────────────────
# Numeric features
num_cols = ['age', 'bp', 'al', 'hemo', 'sc']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Categorical features
cat_cols = ['htn', 'dm', 'appet']
label_encoders = {}

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace('nan', np.nan)
    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ─────────────────────────────────────────
# 4. SPLIT FEATURES AND TARGET
# ─────────────────────────────────────────
FEATURES = ['age', 'bp', 'al', 'hemo', 'sc', 'htn', 'dm', 'appet']

X = df[FEATURES]
y = df['classification']

print(f"\n✅ Using {len(FEATURES)} features: {FEATURES}")
print(f"✅ Dataset shape: {X.shape}")

os.makedirs("models", exist_ok=True)
joblib.dump(FEATURES, "models/ckd_feature_columns.pkl")
joblib.dump(label_encoders, "models/ckd_label_encoders.pkl")
print("✅ Feature columns and label encoders saved!")

# ─────────────────────────────────────────
# 5. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 6. SCALE THE DATA
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "models/ckd_scaler.pkl")
print("✅ CKD Scaler saved!")

# ─────────────────────────────────────────
# 7. BUILD MODEL
# Simpler model suits the smaller feature set
# ─────────────────────────────────────────
print("\n🧠 Building CKD Neural Network...")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ─────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────
print("\n🚀 Training CKD Model...")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ─────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────
print("\n📈 Evaluating CKD Model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['No CKD', 'CKD']))

# ─────────────────────────────────────────
# 10. SAVE
# ─────────────────────────────────────────
model.save("models/ckd_model.h5")
print("\n✅ CKD model saved!")

# ─────────────────────────────────────────
# 11. PLOT
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='teal')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
axes[0].set_title('CKD Model - Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train Loss', color='red')
axes[1].plot(history.history['val_loss'], label='Val Loss', color='purple')
axes[1].set_title('CKD Model - Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("models/ckd_training_plot.png")
plt.show()
print("✅ Training plot saved!")
print("\n🎉 CKD Model Training Complete!")