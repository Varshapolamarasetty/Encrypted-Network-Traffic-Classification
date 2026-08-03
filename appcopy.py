from flask import Flask, render_template, jsonify, request

import pandas as pd

import joblib

import numpy as np

import os

import glob

from werkzeug.utils import secure_filename

from pcap_processor import extract_flow_features, process_pcap_file
from optimized_suspicious_processor import extract_model_features_from_pcap, MODEL_FEATURES

from tensorflow.keras.models import load_model



app = Flask(__name__)



base_dir = os.path.dirname(os.path.abspath(__file__))



# Load trained models and label encoder

models = {

    "Random Forest": joblib.load(os.path.join(base_dir, 'models', 'vpn_nonvpn_rf_model.pkl')),

    "XGBoost": joblib.load(os.path.join(base_dir, 'models', 'xgboost_vpn_modell.pkl')),

    "CNN": load_model(os.path.join(base_dir, 'models', 'cnn_vpnnnn_model.h5')),

    "LSTM": load_model(os.path.join(base_dir, 'models', 'vpn_nonvpn_lstm_model.h5'))

}

le = joblib.load(os.path.join(base_dir, 'models', 'label_encoderr.pkl'))



# Load suspicious detection model (no feature selector needed)
suspicious_model = joblib.load(os.path.join(base_dir, 'models', 'random_forest_modelsuspiciouses.joblib'))

# The 39 features that the model was trained on
suspicious_feature_names = MODEL_FEATURES

print(f"Suspicious model loaded successfully")
print(f"Model uses {len(suspicious_feature_names)} features")
print("Model features:", suspicious_feature_names[:5], "...", suspicious_feature_names[-3:])

# Suspicious detection variables
current_suspicious_file = None
current_suspicious_flow_index = 0
cached_suspicious_flows = None
cached_suspicious_file_path = None



# Load Scalers (ONLY for DL)fo

cnn_scaler = joblib.load(os.path.join(base_dir, 'models', 'scalercnn.pkl'))

lstm_scaler = joblib.load(os.path.join(base_dir, 'models', 'lstm_scaler.pkl'))



# Load training dataset ONLY to get feature names and order

# NOTE: This CSV is NOT used as input data - it's only for reference!

# All actual predictions come from PCAP files (see predict_from_pcap function below)

# Using the actual training dataset: vpn_nonvpn_reconstructed.csv

training_file = os.path.join(base_dir, 'vpn_nonvpn_reconstructed.csv')

if not os.path.exists(training_file):

    training_file = os.path.join(base_dir, 'data', 'vpn_nonvpn_merged_final.csv')



training_df = pd.read_csv(training_file)



# CRITICAL: Use the EXACT same feature order as used during model training

# This must match the feature_order list from your training notebook

feature_order = [

    "duration",

    "min_fiat", "mean_fiat", "max_fiat",

    "min_biat", "mean_biat", "max_biat",

    "min_flowiat", "mean_flowiat", "max_flowiat", "std_flowiat",

    "flowPktsPerSecond", "flowBytesPerSecond",

    "min_active", "mean_active", "max_active", "std_active",

    "min_idle", "mean_idle", "max_idle", "std_idle",

    "total_fiat", "total_biat"

]



# Use the hardcoded feature_order (same as training), not CSV column order

feature_names = feature_order



print(f"Using training dataset for feature reference: {os.path.basename(training_file)}")
print(f"   Using hardcoded feature_order (matches training notebook): {len(feature_names)} features")
print("   Note: CSV is only for validation stats. Feature order comes from training code!")



# Track the current input file used for simulation (user-provided CSV or PCAP)

current_input_file = None    # Absolute path of last uploaded/selected file

current_input_type = None    # 'pcap' or 'csv'

current_csv_row_index = 0    # Sequential index for CSV rows (cycles through all rows)

current_pcap_flow_index = 0  # Sequential index for PCAP flows (cycles through all flows)



# Cache loaded data to avoid re-reading files repeatedly (performance optimization)

cached_csv_data = None       # Cached DataFrame for current CSV file

cached_pcap_flows = None     # Cached flow_features list for current PCAP file

cached_file_path = None      # Path of the file that's currently cached





def resolve_pcap_path(pcap_file_name):

    """

    Resolve a PCAP file name to an absolute path.

    - Checks uploads directory first (for user-uploaded files)

    - If an absolute path is provided, returns it directly if it exists

    """

    if not pcap_file_name:

        return None



    # If absolute path, return directly if it exists

    if os.path.isabs(pcap_file_name):

        return pcap_file_name if os.path.exists(pcap_file_name) else None



    # Check uploads directory (user uploads)

    upload_path = os.path.join(UPLOAD_FOLDER, pcap_file_name)

    if os.path.exists(upload_path):

        return upload_path



    # Not found

    return None



print("Model and encoder loaded successfully")

print("Total features:", len(feature_names))

print("Feature names:", feature_names[:5], "...", feature_names[-3:])



# Get feature statistics from training data for validation

feature_stats = training_df[feature_names].describe()

print("\nTraining data feature ranges (for validation):")

print(f"   duration: {feature_stats.loc['min', 'duration']:.2f} to {feature_stats.loc['max', 'duration']:.2f}")

print(f"   flowPktsPerSecond: {feature_stats.loc['min', 'flowPktsPerSecond']:.2f} to {feature_stats.loc['max', 'flowPktsPerSecond']:.2f}")



# Upload configuration

UPLOAD_FOLDER = os.path.join(base_dir, 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pcap', 'csv'}





def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





def predict_from_csv(csv_file_path, row_index: int | None = None, use_cache=True, model_name="XGBoost"):

    """

    Load a CSV file containing flow features and make prediction.

    The CSV is expected to have the same feature columns as the training data.

    

    Args:

        csv_file_path: Path to CSV file

        row_index: Which row to use (0-based). If None, uses row 0.

        use_cache: If True, uses cached DataFrame if available (faster for repeated calls)

    """

    global cached_csv_data, cached_file_path

    

    # Use cached data if available and file hasn't changed

    if use_cache and cached_csv_data is not None and cached_file_path == csv_file_path:

        df = cached_csv_data.copy()

        print(f"Using cached CSV data: {os.path.basename(csv_file_path)}")

    else:

        print(f"Loading CSV file: {os.path.basename(csv_file_path)}")

        df = pd.read_csv(csv_file_path)

        # Cache it for future use

        cached_csv_data = df.copy()

        cached_file_path = csv_file_path

    if df.empty:

        raise ValueError(f"No data rows found in CSV file: {csv_file_path}")



    # If label column exists, keep it for the chosen row and then drop it

    true_class_value = None

    has_label = 'class1' in df.columns



    # Determine which row to use (0-based)

    n_rows = len(df)



    # If no specific row requested, use first row (row 0)

    if row_index is None:

        row_index = 0

    else:

        # Clamp to valid range

        if row_index < 0:

            row_index = 0

        elif row_index >= n_rows:

            row_index = n_rows - 1



    if has_label:

        true_class_value = str(df.iloc[row_index]['class1'])

        df = df.drop(columns=['class1'])



    # Reindex to match feature_names, fill missing with 0

    df_aligned = df.reindex(columns=feature_names, fill_value=0.0)



    # Use the selected row for prediction

    feature_vector = df_aligned.iloc[row_index].to_numpy(dtype=float).tolist()



    # Debug: ranges

    duration_idx = feature_names.index('duration') if 'duration' in feature_names else 0

    flowpkt_idx = feature_names.index('flowPktsPerSecond') if 'flowPktsPerSecond' in feature_names else -1

    print(f"Sample features from CSV:")

    print(f"   duration={feature_vector[duration_idx]:.6f} (training range: {feature_stats.loc['min', 'duration']:.2f} to {feature_stats.loc['max', 'duration']:.2f})")

    if flowpkt_idx >= 0:

        print(f"   flowPktsPerSecond={feature_vector[flowpkt_idx]:.6f} (training range: {feature_stats.loc['min', 'flowPktsPerSecond']:.2f} to {feature_stats.loc['max', 'flowPktsPerSecond']:.2f})")



    # Select the appropriate model

    model = models[model_name]

    

    # ---------- ML MODELS ----------

    if model_name in ["Random Forest", "XGBoost"]:

        X_input = np.array(feature_vector).reshape(1, -1)

        y_pred_enc = model.predict(X_input)

        y_pred_label = le.inverse_transform(y_pred_enc)

        prediction = str(y_pred_label[0])

    

    # ---------- CNN ----------

    elif model_name == "CNN":

        X_scaled = cnn_scaler.transform(np.array(feature_vector).reshape(1, -1))

        X_cnn = X_scaled.reshape(1, len(feature_names), 1)

        pred_enc = np.argmax(model.predict(X_cnn, verbose=0), axis=1)[0]

        prediction = le.inverse_transform([pred_enc])[0]

    

    # ---------- LSTM ----------

    elif model_name == "LSTM":

        X_scaled = lstm_scaler.transform(np.array(feature_vector).reshape(1, -1))

        X_lstm = X_scaled.reshape(1, len(feature_names), 1)

        pred_enc = np.argmax(model.predict(X_lstm, verbose=0), axis=1)[0]

        prediction = le.inverse_transform([pred_enc])[0]



    is_vpn = 'vpn' in prediction.lower()



    features_dict = {name: float(value) for name, value in zip(feature_names, feature_vector)}



    file_name = os.path.basename(csv_file_path)



    result = {

        "features": features_dict,

        "pred_class": prediction,

        # Generic name used by frontend

        "input_file": file_name,

        # Kept for backward compatibility

        "pcap_file": file_name,

        "is_vpn": is_vpn,

        # For CSV, "flows" are rows; we can step through them

        "flow_count": int(n_rows),

        "selected_flow_index": int(row_index + 1),

    }



    # Attach true class if it was present so UI can compare

    if true_class_value is not None:

        result["true_class"] = true_class_value



    return result





def predict_from_pcap(pcap_file_path=None, flow_index: int | None = None, use_cache=True, model_name="XGBoost"):

    """

    Extract features from a pcap file and make prediction.

    A PCAP path **must** be provided (no more random built‑in files).

    

    Args:

        pcap_file_path: Path to PCAP file

        flow_index: Which flow to use (0-based). If None, uses flow 0.

        use_cache: If True, uses cached flow_features if available (faster for repeated calls)

    """

    global cached_pcap_flows, cached_file_path

    

    if pcap_file_path is None:

        raise ValueError("No PCAP file specified. Please upload a PCAP file first.")



    # Use cached flows if available and file hasn't changed

    if use_cache and cached_pcap_flows is not None and cached_file_path == pcap_file_path:

        flow_features = cached_pcap_flows

        print(f"Using cached PCAP flows: {os.path.basename(pcap_file_path)} ({len(flow_features)} flows)")

    else:

        # Extract features from pcap (this is slow - only do once per file)

        print(f"Extracting features from PCAP file: {os.path.basename(pcap_file_path)}")

        flow_features = extract_flow_features(pcap_file_path)

        # Cache it for future use

        cached_pcap_flows = flow_features

        cached_file_path = pcap_file_path



    if not flow_features:

        raise ValueError(f"No valid flows found in pcap file: {pcap_file_path}")



    n_flows = len(flow_features)

    print(f"Found {n_flows} flow(s) in PCAP file")



    # Choose which flow to use (0-based index)

    if flow_index is None:

        # No specific flow requested - use first flow (flow 0)

        flow_index = 0

    else:

        # Clamp to valid range

        if flow_index < 0:

            flow_index = 0

        elif flow_index >= n_flows:

            flow_index = n_flows - 1



    selected_flow = flow_features[flow_index]

    selected_idx = flow_index

    print(f"Using flow {selected_idx + 1} of {n_flows}")



    # Create feature array in the EXACT order expected by the model

    # This is critical - feature order must match training data

    feature_vector = []

    missing_features = []

    for feat_name in feature_names:

        if feat_name in selected_flow:

            feature_vector.append(selected_flow[feat_name])

        else:

            feature_vector.append(0.0)  # Default value if feature not found

            missing_features.append(feat_name)



    if missing_features:

        print(f"Warning: {len(missing_features)} features missing, using 0.0: {missing_features[:5]}")



    # Debug: Print first few features and validate ranges

    duration_idx = feature_names.index('duration') if 'duration' in feature_names else 0

    flowpkt_idx = feature_names.index('flowPktsPerSecond') if 'flowPktsPerSecond' in feature_names else -1



    print(f"Sample features:")

    print(f"   duration={feature_vector[duration_idx]:.6f} (training range: {feature_stats.loc['min', 'duration']:.2f} to {feature_stats.loc['max', 'duration']:.2f})")

    if flowpkt_idx >= 0:

        print(f"   flowPktsPerSecond={feature_vector[flowpkt_idx]:.6f} (training range: {feature_stats.loc['min', 'flowPktsPerSecond']:.2f} to {feature_stats.loc['max', 'flowPktsPerSecond']:.2f})")

    print(f"   First 5 features: {[f'{x:.4f}' for x in feature_vector[:5]]}")

    print(f"   Feature order check: {feature_names[:5]}")



    # Validate feature vector length

    if len(feature_vector) != len(feature_names):

        print(f"CRITICAL: Feature vector length mismatch! Expected {len(feature_names)}, got {len(feature_vector)}")



    # Convert to numpy array and reshape for prediction

    X_input = np.array(feature_vector).reshape(1, -1)



    # Select the appropriate model

    model = models[model_name]

    

    # ---------- ML MODELS ----------

    if model_name in ["Random Forest", "XGBoost"]:

        y_pred_enc = model.predict(X_input)

        y_pred_label = le.inverse_transform(y_pred_enc)

        prediction = str(y_pred_label[0])

    

    # ---------- CNN ----------

    elif model_name == "CNN":

        X_scaled = cnn_scaler.transform(X_input)

        X_cnn = X_scaled.reshape(1, len(feature_names), 1)

        pred_enc = np.argmax(model.predict(X_cnn, verbose=0), axis=1)[0]

        prediction = le.inverse_transform([pred_enc])[0]

    

    # ---------- LSTM ----------

    elif model_name == "LSTM":

        X_scaled = lstm_scaler.transform(X_input)

        X_lstm = X_scaled.reshape(1, len(feature_names), 1)

        pred_enc = np.argmax(model.predict(X_lstm, verbose=0), axis=1)[0]

        prediction = le.inverse_transform([pred_enc])[0]



    print(f"Prediction: {prediction}")



    # Debug: Check if prediction contains 'vpn'

    is_vpn_pred = 'vpn' in prediction.lower()

    print(f"   Is VPN: {is_vpn_pred}")



    # Create features dictionary for display

    features_dict = {name: float(value) for name, value in zip(feature_names, feature_vector)}



    # Determine if prediction is VPN

    is_vpn = 'vpn' in prediction.lower()



    return {

        "features": features_dict,

        "pred_class": prediction,

        # Generic name used by frontend

        "input_file": os.path.basename(pcap_file_path) if pcap_file_path else "unknown",

        # Kept for backward compatibility

        "pcap_file": os.path.basename(pcap_file_path) if pcap_file_path else "unknown",

        "is_vpn": is_vpn,

        "flow_count": int(n_flows),

        "selected_flow_index": int(selected_idx + 1)

    }





def predict_suspicious_from_pcap(pcap_file_path=None, flow_index: int | None = None, use_cache=True):
    """
    Extract ONLY the 38 features that the model was trained on from PCAP and make prediction.
    No feature selection needed - extracts exactly what the model needs.
    
    Args:
        pcap_file_path: Path to PCAP file
        flow_index: Which flow to use (0-based). If None, uses flow 0.
        use_cache: If True, uses cached flow_features if available (faster for repeated calls)
    """
    global cached_suspicious_flows, cached_suspicious_file_path
    
    if pcap_file_path is None:
        raise ValueError("No PCAP file specified. Please upload a PCAP file first.")

    # Use cached flows if available and file hasn't changed
    if use_cache and cached_suspicious_flows is not None and cached_suspicious_file_path == pcap_file_path:
        flow_features = cached_suspicious_flows
        print(f"Using cached suspicious PCAP flows: {os.path.basename(pcap_file_path)} ({len(flow_features)} flows)")
    else:
        # Extract ONLY 39 features that model needs (much faster!)
        print(f"Extracting 39 model features from PCAP: {os.path.basename(pcap_file_path)}")
        flow_features = extract_model_features_from_pcap(pcap_file_path)
        # Cache it for future use
        cached_suspicious_flows = flow_features
        cached_suspicious_file_path = pcap_file_path

    if not flow_features:
        raise ValueError(f"No valid flows found in pcap file: {pcap_file_path}")

    n_flows = len(flow_features)
    print(f"Found {n_flows} flow(s) for suspicious analysis")

    # Choose which flow to use (0-based index)
    if flow_index is None:
        flow_index = 0
    else:
        if flow_index < 0:
            flow_index = 0
        elif flow_index >= n_flows:
            flow_index = n_flows - 1

    selected_flow = flow_features[flow_index]
    selected_idx = flow_index
    print(f"Using flow {selected_idx + 1} of {n_flows} for suspicious analysis")

    # Create feature vector in the EXACT order expected by the model
    # No feature selection needed - we already have exactly 39 features
    feature_vector = []
    missing_features = []
    
    for feat_name in suspicious_feature_names:
        if feat_name in selected_flow:
            feature_vector.append(selected_flow[feat_name])
        else:
            feature_vector.append(0.0)  # Default value if feature not found
            missing_features.append(feat_name)

    if missing_features:
        print(f"Warning: {len(missing_features)} features missing, using 0.0: {missing_features[:5]}")

    # Convert to numpy array and reshape
    X_input = np.array(feature_vector).reshape(1, -1)
    
    print(f"Feature vector shape: {X_input.shape} (no feature selection needed!)")
    print(f"Sample features: {[f'{x:.4f}' for x in X_input[0][:5]]}")
    
    # Make prediction directly (no feature selection needed!)
    y_pred = suspicious_model.predict(X_input)
    y_pred_proba = suspicious_model.predict_proba(X_input)
    
    # Get prediction and confidence
    prediction = str(y_pred[0])
    confidence = float(np.max(y_pred_proba[0]))
    
    # Apply business logic rules for better classification
    is_suspicious = prediction == 'Suspicious'
    pred_label = prediction
    
    # Rule 1: If inter-arrival time and flow duration within normal ranges -> likely non-suspicious
    flow_duration = selected_flow.get('Flow Duration', 0)
    flow_iat_mean = selected_flow.get('Flow IAT Mean', 0)
    
    # Normal ranges (adjust based on your data)
    normal_duration_min = 0.001  # 1ms minimum
    normal_duration_max = 300.0  # 5 minutes maximum
    normal_iat_min = 0.0001   # 0.1ms minimum
    normal_iat_max = 1.0      # 1 second maximum
    
    is_normal_timing = (normal_duration_min <= flow_duration <= normal_duration_max and 
                     normal_iat_min <= flow_iat_mean <= normal_iat_max)
    
    # Rule 2: If confidence < 90%, mark as suspicious regardless of model prediction
    if confidence < 0.70:
        pred_label = 'Suspicious'
        is_suspicious = True
        print(f"Business Rule Applied: Low confidence ({confidence:.3f} < 90%) -> Marked as Suspicious")
    
    # Rule 3: If timing is normal AND model says non-suspicious AND confidence >= 90%, keep non-suspicious
    elif is_normal_timing and not is_suspicious and confidence >= 0.70:
        pred_label = 'Non-suspicious'
        is_suspicious = False
        print(f"Business Rule Applied: Normal timing (duration={flow_duration:.6f}, iat={flow_iat_mean:.6f}) + High confidence -> Kept as Non-suspicious")
    
    # Rule 4: If timing is abnormal OR model says suspicious -> suspicious
    else:
        pred_label = 'Suspicious'
        is_suspicious = True
        print(f"Business Rule Applied: Abnormal timing (duration={flow_duration:.6f}, iat={flow_iat_mean:.6f}) OR Model prediction -> Marked as Suspicious")
    
    # Convert to readable labels

    # Create features dictionary for display
    features_dict = {name: float(value) for name, value in zip(suspicious_feature_names, feature_vector)}

    return {
        "features": features_dict,
        "pred_class": pred_label,
        "is_suspicious": is_suspicious,
        "confidence": confidence,
        "input_file": os.path.basename(pcap_file_path) if pcap_file_path else "unknown",
        "flow_count": int(n_flows),
        "selected_flow_index": int(selected_idx + 1),
        "features_used": len(suspicious_feature_names)
    }





@app.route('/')

def index():

    return render_template('index.html', feature_names=feature_names, models=list(models.keys()))





@app.route('/predict', methods=['GET'])

def predict():

    try:

        global current_input_file, current_input_type

        global current_csv_row_index, current_pcap_flow_index



        # Optional: allow overriding with explicit pcap_file query

        pcap_file = request.args.get('pcap_file', None)

        model_name = request.args.get('model', 'XGBoost')  # Get model from query param, default XGBoost

        

        if pcap_file:

            pcap_path = resolve_pcap_path(pcap_file)

            if not pcap_path:

                return jsonify({"error": f"PCAP file not found: {pcap_file}"}), 404

            result = predict_from_pcap(pcap_path, model_name=model_name)

            return jsonify(result)



        # Otherwise, use the last uploaded input file (CSV or PCAP)

        if not current_input_file or not current_input_type:

            return jsonify({

                "error": "No input file available. Please upload a CSV or PCAP file first using 'Upload & Predict'."

            }), 400



        if not os.path.exists(current_input_file):

            # File was removed from disk

            current_input_file = None

            current_input_type = None

            current_csv_row_index = 0

            current_pcap_flow_index = 0

            return jsonify({

                "error": "Previously uploaded file is no longer available. Please upload again."

            }), 400



        # For simulation, cycle through ALL flows/rows sequentially
        
        if current_input_type == 'pcap':
            
            result = predict_from_pcap(current_input_file, flow_index=current_pcap_flow_index, model_name=model_name)
            
            # Advance to next flow, but stop when reaching the end (no wrap around)
            
            flow_count = result.get("flow_count", 1) or 1
            
            current_pcap_flow_index += 1
            
            # Check if we've processed all flows
            
            if current_pcap_flow_index >= flow_count:
                
                # All flows processed - reset for next upload but indicate completion
                
                result["all_flows_completed"] = True
                
                current_pcap_flow_index = 0  # Reset for next upload
                
                current_input_file = None  # Clear file to force re-upload
                
                current_input_type = None
                
            else:
                
                result["all_flows_completed"] = False

        else:
            
            result = predict_from_csv(current_input_file, row_index=current_csv_row_index, model_name=model_name)
            
            # Advance to next row, but stop when reaching the end (no wrap around)
            
            row_count = result.get("flow_count", 1) or 1
            
            current_csv_row_index += 1
            
            # Check if we've processed all rows
            
            if current_csv_row_index >= row_count:
                
                # All rows processed - reset for next upload but indicate completion
                
                result["all_flows_completed"] = True
                
                current_csv_row_index = 0  # Reset for next upload
                
                current_input_file = None  # Clear file to force re-upload
                
                current_input_type = None
                
            else:
                
                result["all_flows_completed"] = False



        return jsonify(result)

    except Exception as e:

        print("Prediction error:", e)

        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500





@app.route('/pcap_files', methods=['GET'])

def list_pcap_files():

    """List available PCAP files (only user uploads)."""

    upload_pcaps = glob.glob(os.path.join(UPLOAD_FOLDER, '*.pcap'))

    pcap_list = [os.path.basename(f) for f in upload_pcaps]

    return jsonify({"pcap_files": pcap_list})





@app.route('/upload_predict', methods=['POST'])

def upload_predict():

    """

    Handle user-uploaded file (PCAP or CSV) and return prediction.

    Always uses row 1 / flow 1 on upload (not random).

    """

    try:

        global current_input_file, current_input_type

        global current_csv_row_index, current_pcap_flow_index



        if 'file' not in request.files:

            return jsonify({"error": "No file part in the request"}), 400



        file = request.files['file']



        if file.filename == '':

            return jsonify({"error": "No selected file"}), 400



        if not allowed_file(file.filename):

            return jsonify({"error": "Unsupported file type. Allowed: .pcap, .csv"}), 400



        filename = secure_filename(file.filename)

        save_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(save_path)



        ext = filename.rsplit('.', 1)[1].lower()

        

        # Get model from form data or default to XGBoost

        model_name = request.form.get('model', 'XGBoost')

        

        # Check if this is a new file (clear cache) or same file (use cache for speed)

        global cached_csv_data, cached_pcap_flows, cached_file_path

        is_new_file = (cached_file_path != save_path)

        if is_new_file:

            cached_csv_data = None

            cached_pcap_flows = None

            cached_file_path = None

        

        if ext == 'pcap':

            # For PCAP uploads, remember this file for subsequent simulations

            current_input_file = save_path

            current_input_type = 'pcap'

            current_pcap_flow_index = 0  # Reset to first flow

            # Use cache if same file (faster), otherwise process fresh

            result = predict_from_pcap(save_path, flow_index=0, model_name=model_name, use_cache=not is_new_file)

        else:

            # For CSV uploads, remember this file so Start/Stop simulation

            # can repeatedly predict from the same CSV as well.

            current_input_file = save_path

            current_input_type = 'csv'

            current_csv_row_index = 0  # Reset to first row

            # Use cache if same file (faster), otherwise process fresh

            result = predict_from_csv(save_path, row_index=0, model_name=model_name, use_cache=not is_new_file)



        return jsonify(result)

    except Exception as e:

        print("Upload prediction error:", e)

        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500





@app.route('/suspicious_upload', methods=['POST'])

def suspicious_upload():

    """

    Handle PCAP file upload for suspicious detection and return prediction.

    Only accepts PCAP files for suspicious analysis.

    """

    try:

        global current_suspicious_file, current_suspicious_flow_index

        global cached_suspicious_flows, cached_suspicious_file_path

        if 'file' not in request.files:

            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':

            return jsonify({"error": "No selected file"}), 400

        # Only allow PCAP files for suspicious detection

        if not file.filename.lower().endswith('.pcap'):

            return jsonify({"error": "Only PCAP files are supported for suspicious detection"}), 400

        filename = secure_filename(file.filename)

        save_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(save_path)

        # Check if this is a new file (clear cache) or same file (use cache for speed)

        is_new_file = (cached_suspicious_file_path != save_path)

        if is_new_file:

            cached_suspicious_flows = None

            cached_suspicious_file_path = None

        # Remember this file for subsequent simulations

        current_suspicious_file = save_path

        current_suspicious_flow_index = 0  # Reset to first flow

        # Use cache if same file (faster), otherwise process fresh

        result = predict_suspicious_from_pcap(save_path, flow_index=0, use_cache=not is_new_file)

        return jsonify(result)

    except Exception as e:

        print("Suspicious upload error:", e)

        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500



@app.route('/suspicious_predict', methods=['GET'])

def suspicious_predict():

    """

    Get suspicious prediction from current uploaded PCAP file.

    Cycles through flows sequentially for simulation.

    """

    try:

        global current_suspicious_file, current_suspicious_flow_index

        if not current_suspicious_file:

            return jsonify({

                "error": "No PCAP file available for suspicious analysis. Please upload a PCAP file first."

            }), 400

        if not os.path.exists(current_suspicious_file):

            # File was removed from disk

            current_suspicious_file = None

            current_suspicious_flow_index = 0

            return jsonify({
                "error": "Previously uploaded PCAP file is no longer available. Please upload again."
            }), 400

        # Get prediction for current flow
        result = predict_suspicious_from_pcap(current_suspicious_file, flow_index=current_suspicious_flow_index)
        
        # Advance to next flow, but stop when reaching the end (no wrap around)
        flow_count = result.get("flow_count", 1) or 1
        current_suspicious_flow_index += 1
        
        # Check if we've processed all flows
        if current_suspicious_flow_index >= flow_count:
            # All flows processed - reset for next upload but indicate completion
            result["all_flows_completed"] = True
            current_suspicious_flow_index = 0  # Reset for next upload
            current_suspicious_file = None  # Clear file to force re-upload
        else:
            result["all_flows_completed"] = False

        return jsonify(result)

    except Exception as e:
        print("Suspicious prediction error:", e)

        import traceback

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':

    app.run(debug=True)