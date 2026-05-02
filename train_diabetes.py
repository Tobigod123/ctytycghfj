import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
print("📂 Loading Diabetes Dataset...")
df = pd.read_csv("datasets/diabetes.csv")
print(f"✅ Dataset loaded! Shape: {df.shape}")

# ─────────────────────────────────────────
# 2. HANDLE ZERO VALUES
# ─────────────────────────────────────────
cols_to_fix = ['Glucose', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

print("✅ Zero values handled!")

# ─────────────────────────────────────────
# 3. SELECT ONLY THE 5 KEY FEATURES
# Easy to collect at a basic health checkup
# ─────────────────────────────────────────
FEATURES = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Pregnancies']

X = df[FEATURES]
y = df['Outcome']

print(f"\n✅ Using {len(FEATURES)} features: {FEATURES}")
print(f"📊 Dataset shape: {X.shape}")
print(f"📊 Target distribution:\n{y.value_counts()}")

os.makedirs("models", exist_ok=True)
joblib.dump(FEATURES, "models/diabetes_feature_columns.pkl")
print("✅ Diabetes feature columns saved!")

# ─────────────────────────────────────────
# 4. TRAIN/TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 5. SCALE
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "models/diabetes_scaler.pkl")
print("✅ Diabetes scaler saved!")

# ─────────────────────────────────────────
# 6. BUILD MODEL
# ─────────────────────────────────────────
print("\n🧠 Building Neural Network...")

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
# 7. TRAIN
# ─────────────────────────────────────────
print("\n🚀 Training Diabetes Model...")

# Handle class imbalance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
total = len(y_train)
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
print(f"📊 Class weights: {class_weight}")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# ─────────────────────────────────────────
# 8. EVALUATE
# ─────────────────────────────────────────
print("\n📈 Evaluating Model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# ─────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────
model.save("models/diabetes_model.h5")
print("\n✅ Diabetes model saved!")

# ─────────────────────────────────────────
# 10. PLOT
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
axes[0].set_title('Diabetes Model - Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train Loss', color='red')
axes[1].plot(history.history['val_loss'], label='Val Loss', color='green')
axes[1].set_title('Diabetes Model - Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("models/diabetes_training_plot.png")
plt.show()
print("✅ Training plot saved!")
print("\n🎉 Diabetes Model Training Complete!")