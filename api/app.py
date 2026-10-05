from flask import Flask, request, jsonify
import joblib
import logging
from pathlib import Path

app = Flask(__name__)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "production_model.pkl"
)

BACKUP_MODEL_PATH = (
    PROJECT_ROOT / "models" / "backup_model.pkl"
)

model = None
active_model = None


# ==========================================
# LOAD MODEL WITH AUTOMATIC FAILOVER
# ==========================================

def load_model():

    global model
    global active_model

    try:

        model = joblib.load(
            PRODUCTION_MODEL_PATH
        )

        active_model = "production"

        logger.info(
            "Production model loaded successfully."
        )

        return True

    except Exception as production_error:

        logger.error(
            "Production model failure: %s",
            production_error
        )

        try:

            model = joblib.load(
                BACKUP_MODEL_PATH
            )

            active_model = "backup"

            logger.warning(
                "FAILOVER ACTIVATED: Backup model loaded."
            )

            return True

        except Exception as backup_error:

            logger.critical(
                "Backup model also failed: %s",
                backup_error
            )

            model = None
            active_model = None

            return False


# Load model when API starts
load_model()


# ==========================================
# HOME
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "MLOps Breast Cancer Prediction API",
        "status": "running",
        "active_model": active_model
    })


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    if model is not None:

        return jsonify({
            "status": "healthy",
            "active_model": active_model
        }), 200

    return jsonify({
        "status": "unhealthy",
        "active_model": None
    }), 503


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        logger.error(
            "Prediction failed: No model available."
        )

        return jsonify({
            "error": "No model available"
        }), 503

    try:

        data = request.get_json()

        if data is None:

            return jsonify({
                "error": "Request body must contain JSON"
            }), 400

        if "features" not in data:

            return jsonify({
                "error": "Missing 'features' field"
            }), 400

        features = data["features"]

        if not isinstance(features, list):

            return jsonify({
                "error": "'features' must be a list"
            }), 400

        if len(features) != 30:

            logger.warning(
                "Invalid input: received %d features",
                len(features)
            )

            return jsonify({
                "error": "Exactly 30 features are required",
                "received": len(features)
            }), 400

        # Prediction
        prediction = model.predict(
            [features]
        )[0]

        probabilities = model.predict_proba(
            [features]
        )[0]

        confidence = float(
            max(probabilities)
        )

        result = (
            "malignant"
            if prediction == 0
            else "benign"
        )

        logger.info(
            "Prediction successful using %s model.",
            active_model
        )

        return jsonify({

            "prediction": int(prediction),

            "result": result,

            "confidence": round(
                confidence,
                4
            ),

            "model_used": active_model

        }), 200

    except Exception as error:

        logger.error(
            "Prediction failed: %s",
            error
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ==========================================
# START API
# ==========================================

if __name__ == "__main__":

    logger.info(
        "Starting MLOps Flask API..."
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )