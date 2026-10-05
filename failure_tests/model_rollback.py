import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODEL = PROJECT_ROOT / "models" / "production_model.pkl"
CANDIDATE_MODEL = PROJECT_ROOT / "models" / "candidate_model.pkl"

print("==========================================")
print("NEW MODEL PERFORMANCE & ROLLBACK TEST")
print("==========================================")

# Load dataset
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target
)

# Load current production model
production_model = joblib.load(PRODUCTION_MODEL)

# Evaluate production model
production_predictions = production_model.predict(X_test)
production_f1 = f1_score(y_test, production_predictions)

print(f"Production Model F1 Score: {production_f1:.4f}")

# Create intentionally poor candidate model
candidate_model = DummyClassifier(
    strategy="most_frequent"
)

candidate_model.fit(X_train, y_train)

# Save candidate model
joblib.dump(candidate_model, CANDIDATE_MODEL)

# Evaluate candidate
candidate_predictions = candidate_model.predict(X_test)
candidate_f1 = f1_score(y_test, candidate_predictions)

print(f"Candidate Model F1 Score : {candidate_f1:.4f}")

print("\n==========================================")
print("MODEL DEPLOYMENT GATE")
print("==========================================")

if candidate_f1 >= production_f1:
    print("Candidate model passed performance gate.")
    print("Candidate model can be promoted.")
else:
    print("ALERT: NEW MODEL PERFORMANCE IS POOR")
    print("Candidate model rejected.")
    print("ROLLBACK ACTIVATED")
    print("Stable production model remains active.")

print("==========================================")