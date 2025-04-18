import os
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 1. Load model.pkl (trained on raw price)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# 2. Load config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

@app.route('/config', methods=['GET'])
def get_config():
    """
    Return the categorical configuration and compound averages.
    """
    return jsonify(cfg)  # wraps Python dict into JSON response :contentReference[oaicite:0]{index=0}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept JSON with keys:
      - finishing_status, area, compound
      - size, bedrooms, bathrooms
    Validate inputs, compute compound_avg from cfg,
    build a DataFrame, and return predicted price.
    """
    data = request.get_json()  # parses application/json body :contentReference[oaicite:1]{index=1}
    # Required fields
    required = ["finishing_status","area","compound","size","bedrooms","bathrooms"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error":f"Missing fields: {missing}"}), 400

    # Cast numeric fields
    try:
        size = float(data["size"])
        bedrooms = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
    except (ValueError, TypeError):
        return jsonify({"error":"Size, bedrooms, and bathrooms must be numbers."}), 400

    fs = data["finishing_status"]
    area = data["area"]
    comp = data["compound"]

    # Validate categorical
    errors = []
    if fs not in cfg["finishing_status"]:
        errors.append(f"Invalid finishing_status; options: {cfg['finishing_status']}")
    if area not in cfg["areas"]:
        errors.append(f"Invalid area; options: {cfg['areas']}")
    allowed = cfg["area_compounds"].get(area, [])
    if comp not in allowed:
        errors.append(f"Invalid compound for {area}; options: {allowed}")
    if errors:
        return jsonify({"error": " ; ".join(errors)}), 400

    # Compute compound_avg internally
    cavg = cfg["compound_avg"].get(comp, cfg["global_avg"])

    # Build DataFrame exactly as training pipeline expected
    df = pd.DataFrame([{
        "size": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "finishing_status": fs,
        "area": area,
        "compound": comp,
        "compound_avg": cavg
    }])

    # Predict using loaded model
    try:
        raw_price = model.predict(df)[0]
        # Convert NumPy scalar (e.g., float32) to native Python float
        price = float(raw_price)
    except Exception as e:
        return jsonify({"error":f"Prediction failed: {str(e)}"}), 500

    return jsonify({"predicted_price": price})

if __name__ == '__main__':
    # Run on localhost:5000 in debug mode
    app.run(host='0.0.0.0', port=5000, debug=True)
