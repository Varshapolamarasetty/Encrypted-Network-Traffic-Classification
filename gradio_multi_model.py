import gradio as gr
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_dir, "vpn_nonvpn_reconstructed.csv"))

feature_order = [
    "duration",
    "min_fiat","mean_fiat","max_fiat",
    "min_biat","mean_biat","max_biat",
    "min_flowiat","mean_flowiat","max_flowiat","std_flowiat",
    "flowPktsPerSecond","flowBytesPerSecond",
    "min_active","mean_active","max_active","std_active",
    "min_idle","mean_idle","max_idle","std_idle",
    "total_fiat","total_biat"
]

X_data = df[feature_order].values
y_raw  = df["class1"].values   # string labels

NUM_FEATURES = len(feature_order)

# --------------------------------------------------
# Load Label Encoder (COMMON)
# --------------------------------------------------
label_encoder = joblib.load(os.path.join(base_dir, 'models', 'label_encoderr.pkl'))

# --------------------------------------------------
# Load Scalers (ONLY for DL)
# --------------------------------------------------
cnn_scaler  = joblib.load(os.path.join(base_dir, 'models', 'scalercnn.pkl'))
lstm_scaler = joblib.load(os.path.join(base_dir, 'models', 'lstm_scaler.pkl'))

# --------------------------------------------------
# Load Models
# --------------------------------------------------
models = {
    "Random Forest": joblib.load(os.path.join(base_dir, 'models', 'vpn_nonvpn_rf_model.pkl')),
    "XGBoost": joblib.load(os.path.join(base_dir, 'models', 'xgboost_vpn_modell.pkl')),
    "CNN": load_model(os.path.join(base_dir, 'models', 'cnn_vpnnnn_model.h5')),
    "LSTM": load_model(os.path.join(base_dir, 'models', 'vpn_nonvpn_lstm_model.h5'))
}

# --------------------------------------------------
# Prediction function (Random sample)
# --------------------------------------------------
def predict_random(model_name):
    idx = np.random.randint(0, len(X_data))

    X_raw = X_data[idx].reshape(1, NUM_FEATURES)
    actual_label = y_raw[idx]

    model = models[model_name]

    # ---------- ML MODELS ----------
    if model_name in ["Random Forest", "XGBoost"]:
        pred_enc = model.predict(X_raw)[0]

    # ---------- CNN ----------
    elif model_name == "CNN":
        X_scaled = cnn_scaler.transform(X_raw)
        X_cnn = X_scaled.reshape(1, NUM_FEATURES, 1)
        pred_enc = np.argmax(model.predict(X_cnn, verbose=0), axis=1)[0]

    # ---------- LSTM ----------
    elif model_name == "LSTM":
        X_scaled = lstm_scaler.transform(X_raw)
        X_lstm = X_scaled.reshape(1, NUM_FEATURES, 1)
        pred_enc = np.argmax(model.predict(X_lstm, verbose=0), axis=1)[0]

    predicted_label = label_encoder.inverse_transform([pred_enc])[0]

    result = "✅ MATCH" if predicted_label == actual_label else "❌ MISMATCH"

    feature_df = pd.DataFrame({
        "Feature": feature_order,
        "Value": X_raw.flatten()
    })

    return actual_label, predicted_label, result, feature_df

# --------------------------------------------------
# Gradio UI
# --------------------------------------------------
iface = gr.Interface(
    fn=predict_random,
    inputs=gr.Dropdown(
        choices=list(models.keys()),
        value="Random Forest",
        label="Select Model"
    ),
    outputs=[
        gr.Textbox(label="Actual Label"),
        gr.Textbox(label="Predicted Label"),
        gr.Textbox(label="Match Result"),
        gr.Dataframe(label="Input Features (Random Sample)", interactive=False)
    ],
    title="VPN vs Non-VPN Classification (ML + DL)",
    description="Randomly selects a flow, applies correct preprocessing, and compares Actual vs Predicted"
)

iface.launch()
