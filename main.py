from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import joblib
import numpy as np
import os

# ----------------------------
# 1. Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# 2. Load model, encoder, and dataset
# ----------------------------
xgb_model = joblib.load(r'models/xgboost_vpn_modell.pkl')
le = joblib.load(r'models/label_encoderr.pkl')
df = pd.read_csv(r"C:\Users\ASHISH\Downloads\Vignan\Encrypted traffic flow\Encrypted traffic flow final\data\vpn_nonvpn_reconstructed.csv")  # Update path if needed

feature_names = [col for col in df.columns if col != 'class1']

# ----------------------------
# 3. Prediction function
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
# 4. Routes
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

@app.route('/sounds/<filename>')
def sounds(filename):
    """Serve sound files from the sounds directory."""
    return send_from_directory('sounds', filename)

# ----------------------------
# 5. Run app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
