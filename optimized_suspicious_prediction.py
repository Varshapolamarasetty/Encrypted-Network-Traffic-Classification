"""
Optimized Suspicious Traffic Prediction
Uses only the 38 features that the model was trained on
No feature selection needed - extracts exactly what the model needs
"""

import joblib
import numpy as np
import pandas as pd
import os
from optimized_suspicious_processor import extract_model_features_from_pcap, MODEL_FEATURES

class OptimizedSuspiciousPredictor:
    def __init__(self, base_dir=None):
        """Initialize the optimized suspicious predictor"""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load the suspicious model (trained on 38 features)
        self.model = joblib.load(os.path.join(base_dir, 'models', 'random_forest_modelsuspiciouses.joblib'))
        
        # No need to load feature selector - we extract exactly what the model needs
        self.model_features = MODEL_FEATURES
        
        print(f"Optimized Suspicious Predictor loaded")
        print(f"Model expects {len(self.model_features)} features")
        print(f"Features: {self.model_features[:5]} ... {self.model_features[-3:]}")
    
    def predict_from_pcap(self, pcap_file_path: str, flow_index: int = 0):
        """
        Extract exactly 38 features from PCAP and make prediction
        
        Args:
            pcap_file_path: Path to PCAP file
            flow_index: Which flow to use (0-based)
            
        Returns:
            Dictionary with prediction results
        """
        # Extract exactly the 38 features the model needs
        flows = extract_model_features_from_pcap(pcap_file_path)
        
        if not flows:
            raise ValueError(f"No valid flows found in pcap file: {pcap_file_path}")
        
        n_flows = len(flows)
        print(f"Found {n_flows} flow(s) in PCAP file")
        
        # Select the requested flow
        if flow_index >= n_flows:
            flow_index = n_flows - 1
        
        selected_flow = flows[flow_index]
        print(f"Using flow {flow_index + 1} of {n_flows}")
        
        # Create feature vector in the exact order expected by the model
        feature_vector = []
        for feature_name in self.model_features:
            if feature_name in selected_flow:
                feature_vector.append(selected_flow[feature_name])
            else:
                feature_vector.append(0.0)  # Default if missing
                print(f"Warning: Feature '{feature_name}' not found, using 0.0")
        
        # Convert to numpy array and reshape
        X_input = np.array(feature_vector).reshape(1, -1)
        
        print(f"Feature vector shape: {X_input.shape}")
        print(f"Sample features: {[f'{x:.4f}' for x in X_input[0][:5]]}")
        
        # Make prediction directly (no feature selection needed!)
        y_pred = self.model.predict(X_input)
        y_pred_proba = self.model.predict_proba(X_input)
        
        # Get prediction and confidence
        prediction = str(y_pred[0])
        confidence = float(np.max(y_pred_proba[0]))
        
        # Convert to readable labels
        is_suspicious = prediction == 'Suspicious'
        pred_label = prediction
        
        print(f"Prediction: {pred_label} (Confidence: {confidence:.4f})")
        print(f"Is Suspicious: {is_suspicious}")
        
        # Create features dictionary for display
        features_dict = {name: float(value) for name, value in zip(self.model_features, feature_vector)}
        
        return {
            "features": features_dict,
            "pred_class": pred_label,
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "input_file": os.path.basename(pcap_file_path),
            "flow_count": int(n_flows),
            "selected_flow_index": int(flow_index + 1),
            "features_used": len(self.model_features)
        }

def test_optimized_prediction(pcap_file_path: str):
    """Test the optimized suspicious prediction"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize predictor
    predictor = OptimizedSuspiciousPredictor(base_dir)
    
    # Make prediction
    try:
        result = predictor.predict_from_pcap(pcap_file_path)
        print("\n" + "="*50)
        print("OPTIMIZED SUSPICIOUS PREDICTION RESULTS")
        print("="*50)
        print(f"File: {result['input_file']}")
        print(f"Flow: {result['selected_flow_index']}/{result['flow_count']}")
        print(f"Prediction: {result['pred_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Is Suspicious: {result['is_suspicious']}")
        print(f"Features Used: {result['features_used']}")
        print("="*50)
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    pcap_file = "test.pcap"  # Replace with actual PCAP file
    test_optimized_prediction(pcap_file)
