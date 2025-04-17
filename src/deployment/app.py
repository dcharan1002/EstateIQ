import os
import joblib
import logging
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime
from flask_cors import CORS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model loading
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/models'))
MODEL_PATH = MODEL_DIR / "current" / "model.joblib"
METRICS_PATH = MODEL_DIR / "current" / "metrics.json"

def load_model():
    """Load the model and return it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

try:
    model = load_model()
    # Get expected features
    if hasattr(model, 'feature_names_in_'):
        EXPECTED_FEATURES = model.feature_names_in_
    elif hasattr(model, 'get_booster'):
        EXPECTED_FEATURES = model.get_booster().feature_names
    else:
        EXPECTED_FEATURES = []
except Exception as e:
    logger.error(f"Failed to load model on startup: {e}")
    raise

def prepare_features(input_data):
    """Prepare features matching the exact column names from training data"""
    features = {}
    
    # Base numeric features (exact names from CSV)
    numeric_features = [
        'GROSS_AREA', 'LIVING_AREA', 'LAND_SF', 'YR_BUILT',
        'BED_RMS', 'FULL_BTH', 'HLF_BTH', 'NUM_PARKING',
        'FIREPLACES', 'KITCHENS', 'TT_RMS', 'ZIP_CODE',
        'YR_REMODEL'
    ]
    for feature in numeric_features:
        # Convert ZIP_CODE from string to float if it's a string
        if feature == 'ZIP_CODE' and isinstance(input_data.get(feature), str):
            features[feature] = float(input_data.get(feature).lstrip('0'))
        else:
            features[feature] = float(input_data.get(feature, 0))
    
    # Binary features (exact names from CSV)
    binary_features = [
        # Structure and material features
        'STRUCTURE_CLASS_C - BRICK/CONCR',
        'STRUCTURE_CLASS_D - WOOD/FRAME',
        'STRUCTURE_CLASS_B - REINF CONCR',
        
        # Interior condition features
        'INT_COND_E - EXCELLENT',
        'INT_COND_G - GOOD',
        'INT_COND_A - AVERAGE',
        'INT_COND_F - FAIR',
        'INT_COND_P - POOR',
        
        # Overall condition features
        'OVERALL_COND_E - EXCELLENT',
        'OVERALL_COND_VG - VERY GOOD',
        'OVERALL_COND_G - GOOD',
        'OVERALL_COND_A - AVERAGE',
        'OVERALL_COND_F - FAIR',
        'OVERALL_COND_P - POOR',
        
        # Kitchen features
        'KITCHEN_STYLE2_M - MODERN',
        'KITCHEN_STYLE2_L - LUXURY',
        'KITCHEN_STYLE2_S - SEMI-MODERN',
        'KITCHEN_TYPE_F - FULL EAT IN',
        
        # Amenities and systems
        'AC_TYPE_C - CENTRAL AC',
        'AC_TYPE_D - DUCTLESS AC',
        'HEAT_TYPE_F - FORCED HOT AIR',
        'HEAT_TYPE_W - HT WATER/STEAM',
        
        # Property characteristics
        'PROP_VIEW_E - EXCELLENT',
        'PROP_VIEW_G - GOOD',
        'CORNER_UNIT_Y - YES',
        'ORIENTATION_E - END',
        'ORIENTATION_F - FRONT/STREET',
        
        # Exterior features
        'EXT_COND_E - EXCELLENT',
        'EXT_COND_G - GOOD',
        'ROOF_COVER_S - SLATE',
        'ROOF_COVER_A - ASPHALT SHINGL'
    ]
    
    # Handle binary features
    for feature in binary_features:
        features[feature] = int(input_data.get(feature, 0))
    
    # Create initial DataFrame
    X = pd.DataFrame([features])
    
    # Add derived features exactly as in train.py
    X['property_age'] = 2025 - X['YR_BUILT'].astype(int)
    X['total_bathrooms'] = X['FULL_BTH'] + 0.5 * X['HLF_BTH']
    X['living_area_ratio'] = np.where(
        X['GROSS_AREA'] > 0,
        X['LIVING_AREA'] / X['GROSS_AREA'],
        0
    )
    
    # Handle renovation features
    X['years_since_renovation'] = 2025 - X['YR_REMODEL'].fillna(X['YR_BUILT']).astype(int)
    X['has_renovation'] = (X['YR_REMODEL'] > X['YR_BUILT']).astype(int)
    
    # Calculate area ratios
    X['floor_area_ratio'] = np.where(
        X['LAND_SF'] > 0,
        X['GROSS_AREA'] / X['LAND_SF'],
        0
    )
    X['non_living_area'] = np.maximum(0, X['GROSS_AREA'] - X['LIVING_AREA'])
    X['rooms_per_area'] = np.where(
        X['LIVING_AREA'] > 0,
        X['TT_RMS'] / X['LIVING_AREA'],
        0
    )
    
    # Create condition scores using helper function
    condition_map = {'E': 5, 'VG': 4.5, 'G': 4, 'A': 3, 'F': 2, 'P': 1}
    
    def calculate_condition_score(row, condition_cols):
        max_score = 3  # default score
        for col in condition_cols:
            if row[col]:  # if this condition is true
                cond_code = col.split(' - ')[0].split('_')[-1]
                score = condition_map.get(cond_code, 0)
                max_score = max(max_score, score)
        return max_score
    
    # Calculate interior score
    int_cond_cols = [col for col in X.columns if col.startswith('INT_COND_')]
    if int_cond_cols:
        X['interior_score'] = X.apply(
            lambda row: calculate_condition_score(row, int_cond_cols), axis=1
        )
    
    # Calculate exterior score
    ext_cond_cols = [col for col in X.columns if col.startswith('EXT_COND_')]
    if ext_cond_cols:
        X['exterior_score'] = X.apply(
            lambda row: calculate_condition_score(row, ext_cond_cols), axis=1
        )
    
    # Calculate overall score
    overall_cond_cols = [col for col in X.columns if col.startswith('OVERALL_COND_')]
    if overall_cond_cols:
        X['overall_score'] = X.apply(
            lambda row: calculate_condition_score(row, overall_cond_cols), axis=1
        )
    
    return X
    
    return df
@app.route('/')
def index():
    return "Welcome to EstateIQ API!"


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model."""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create DataFrame with defaults for missing features
        try:
            features = prepare_features(data)
            # Replace NaN and inf values with 0
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            return jsonify({"error": f"Invalid data format: {str(e)}"}), 400

        # Make prediction
        prediction = model.predict(features)

        # Log prediction details
        logger.info(f"Made prediction: {prediction[0]} for features: {data}")

        # Return prediction with timestamp and model info
        response = {
            "prediction": float(prediction[0]),
            "timestamp": datetime.now().isoformat(),
            "model_version": os.getenv('BUILD_ID', 'unknown')
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also verifies model is loaded."""
    try:
        # Verify model is loaded
        if model is None:
            return jsonify({
                "status": "unhealthy",
                "error": "Model not loaded"
            }), 503

        # Return healthy status with model info
        return jsonify({
            "status": "healthy",
            "model": {
                "path": str(MODEL_PATH),
                "version": os.getenv('BUILD_ID', 'unknown'),
                "loaded": True
            }
        })

    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route('/metadata', methods=['GET'])
def metadata():
    """Return model metadata and performance metrics."""
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        else:
            metrics = {"warning": "No metrics file found"}

        return jsonify({
            "model_version": os.getenv('BUILD_ID', 'unknown'),
            "metrics": metrics,
            "deployment_timestamp": os.getenv('DEPLOYMENT_TIMESTAMP', 'unknown')
        })

    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    app.run(host='0.0.0.0', port=port)
