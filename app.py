import os
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask application
app = Flask(__name__)

# ========================================================================
# Configuration and Model Loading with Error Handling
# ========================================================================

def load_configuration(file_path):
    """Load and validate configuration file"""
    try:
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
            
        # Validate required configuration structure
        required_keys = ['finishing_status', 'areas', 'area_compounds', 'compound_avg', 'global_avg']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
                
        return config
    except Exception as e:
        raise RuntimeError(f"Config loading failed: {str(e)}")

def load_ml_model(file_path):
    """Load and validate ML model"""
    try:
        with open(file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            
        # Basic model validation
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model doesn't have predict method")
            
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Load critical files with explicit error handling
try:
    # Load ML model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model = load_ml_model(MODEL_PATH)
    
    # Load configuration
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
    cfg = load_configuration(CONFIG_PATH)
except RuntimeError as e:
    print(f"CRITICAL STARTUP ERROR: {str(e)}")
    exit(1)  # Prevent deployment if files are missing

# ========================================================================
# API Endpoints
# ========================================================================

@app.route('/config', methods=['GET'])
def get_config():
    """Return complete configuration"""
    return jsonify({
        "status": "success",
        "config": cfg
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handle price prediction requests"""
    # Input validation
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    
    # Required fields check
    required_fields = ["finishing_status", "area", "compound", 
                      "size", "bedrooms", "bathrooms"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Numeric validation
    try:
        numeric_data = {
            "size": float(data["size"]),
            "bedrooms": int(data["bedrooms"]),
            "bathrooms": int(data["bathrooms"])
        }
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid numeric format"}), 400

    # Categorical validation
    validation_errors = []
    
    # Finishing status check
    if data["finishing_status"] not in cfg["finishing_status"]:
        validation_errors.append(
            f"Invalid finishing_status. Valid options: {cfg['finishing_status']}"
        )
        
    # Area check
    if data["area"] not in cfg["areas"]:
        validation_errors.append(
            f"Invalid area. Valid options: {cfg['areas']}"
        )
    else:
        # Compound validation per area
        allowed_compounds = cfg["area_compounds"].get(data["area"], [])
        if data["compound"] not in allowed_compounds:
            validation_errors.append(
                f"Invalid compound for {data['area']}. Valid options: {allowed_compounds}"
            )

    if validation_errors:
        return jsonify({"error": " | ".join(validation_errors)}), 400

    # Prepare features DataFrame
    try:
        compound_avg = cfg["compound_avg"].get(
            data["compound"], 
            cfg["global_avg"]
        )
        
        features = pd.DataFrame([{
            "size": numeric_data["size"],
            "bedrooms": numeric_data["bedrooms"],
            "bathrooms": numeric_data["bathrooms"],
            "finishing_status": data["finishing_status"],
            "area": data["area"],
            "compound": data["compound"],
            "compound_avg": compound_avg
        }])
        
        # Make prediction
        prediction = model.predict(features)[0]
        return jsonify({
            "predicted_price": float(prediction),
            "currency": "EGP",
            "model_version": "1.0"
        }), 200
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500

# ========================================================================
# Health Check and Server Configuration
# ========================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for service health verification"""
    return jsonify({
        "status": "healthy",
        "version": "1.0",
        "dependencies": {
            "model_loaded": True,
            "config_loaded": True
        }
    }), 200

if __name__ == '__main__':
    # Render-specific configuration
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment
    app.run(
        host='0.0.0.0',
        port=port,
        # Never enable debug in production!
        debug=False  
    )
