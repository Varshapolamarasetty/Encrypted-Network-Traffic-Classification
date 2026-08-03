from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import os

# ----------------------------
# 1. Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# 2. Global simulation flag
# ----------------------------
simulation_running = False

# ----------------------------
# 3. Load model, encoder, and dataset
# ----------------------------
# Get the base directory of the app
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and encoder
xgb_model = joblib.load(os.path.join(base_dir, 'models', 'xgb_vpn_model.pkl'))
le = joblib.load(os.path.join(base_dir, 'models', 'label_encoder.pkl'))

# Load dataset
df = pd.read_csv(os.path.join(base_dir, 'data', 'vpn_nonvpn_merged_final.csv'))

# Get feature names (all columns except 'class1')
feature_names = [col for col in df.columns if col != 'class1']

# ----------------------------
# 4. Prediction function
# ----------------------------
def predict_random_sample():
    """
    Selects a random row from the dataset and returns its features,
    true class, and predicted class.
    """
    # Pick a random row
    sample = df.sample(1, random_state=np.random.randint(0, 10000))
    X_input = sample[feature_names]
    
    # Predict
    y_pred_enc = xgb_model.predict(X_input)
    y_pred_label = le.inverse_transform(y_pred_enc)
    
    # Get true class
    true_class = sample['class1'].values[0]
    
    # Create feature dictionary for easier display
    features_dict = {name: float(value) for name, value in zip(feature_names, X_input.values[0])}
    
    # Return as dictionary
    return {
        "features": features_dict,
        "feature_names": feature_names,
        "feature_values": X_input.values[0].tolist(),
        "true_class": str(true_class),
        "pred_class": str(y_pred_label[0])
    }

# ----------------------------
# 5. Routes
# ----------------------------
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['GET'])
def predict():
    """
    Returns a random row prediction as JSON.
    This endpoint is called by JavaScript every second when simulation is running.
    """
    try:
        result = predict_random_sample()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# 6. Run app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)

