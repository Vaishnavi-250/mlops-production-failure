import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

print("==========================================")
print("DATASET INFORMATION")
print("==========================================")
print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================
# 4. MLFLOW EXPERIMENT
# ==========================================

mlflow.set_experiment(
    "Breast_Cancer_MLOps"
)


# Store results
results = []


# ==========================================
# 5. TRAIN MODELS
# ==========================================

for name, model in models.items():

    print("\nTraining:", name)

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]


        # Metrics
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )


        # Print metrics
        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("ROC-AUC  :", round(roc_auc, 4))


        # ==================================
        # MLFLOW LOGGING
        # ==================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # Save model in MLflow
        if name == "Decision Tree":

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

        elif name == "Random Forest":

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )


        # Store results
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "Model Object": model
        })


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ].to_string(index=False)
)


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

best_index = results_df[
    "F1 Score"
].idxmax()

best_model = results_df.loc[
    best_index,
    "Model Object"
]

best_model_name = results_df.loc[
    best_index,
    "Model"
]

best_f1 = results_df.loc[
    best_index,
    "F1 Score"
]


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Best F1 Score:",
    round(best_f1, 4)
)


# ==========================================
# 8. SAVE PRODUCTION + BACKUP MODEL
# ==========================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


production_path = (
    MODEL_DIR /
    "production_model.pkl"
)

backup_path = (
    MODEL_DIR /
    "backup_model.pkl"
)


joblib.dump(
    best_model,
    production_path
)

joblib.dump(
    best_model,
    backup_path
)


print("\n==========================================")
print("MODEL STORAGE")
print("==========================================")

print(
    "Production model:",
    production_path
)

print(
    "Backup model:",
    backup_path
)


# ==========================================
# 9. FINAL STATUS
# ==========================================

print("\n==========================================")
print("MLOPS TRAINING PIPELINE COMPLETED!")
print("==========================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Best F1 Score:",
    round(best_f1, 4)
)

print(
    "Production Model:",
    production_path
)

print(
    "Backup Model:",
    backup_path
)