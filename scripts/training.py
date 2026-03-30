import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
 
from features import X_scaled, y
# ─────────────────────────────────────────
#DIVIDING IN TRAINING AND TESTING SETS

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[SPLIT] Training set: {X_train.shape[0]} rows, Testing set: {X_test.shape[0]} rows.")

# ─────────────────────────────────────────
#TRAINING THE MODEL 
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

model.fit(X_train, y_train)
print(f"[TRAINING] Model trained successfully.")

# ─────────────────────────────────────────
#EVALUATION
y_pred = model.predict_proba(X_test)[:,1]
y_pred = (y_pred >= 0.3).astype(int)
print(f"[EVALUATION] Predictions completed.")
print(f"[CLASSIFICATION REPORT]\n{classification_report(y_test, y_pred)}")

print(f"[METRICS] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ─────────────────────────────────────────
#SAVING THE MODEL
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"[SAVING] Model saved successfully.")