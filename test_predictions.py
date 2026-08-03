import os
import pandas as pd
import numpy as np

# Check what classes your models should predict
print("🔍 Checking Model Prediction Classes...")

base_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(base_dir, 'vpn_nonvpn_reconstructed.csv')

if os.path.exists(training_file):
    df = pd.read_csv(training_file)
    
    print(f"📊 Dataset: {os.path.basename(training_file)}")
    print(f"   Total samples: {len(df)}")
    
    # Check class distribution
    if 'class1' in df.columns:
        class_counts = df['class1'].value_counts()
        print("\n📈 Class Distribution:")
        for cls, count in class_counts.items():
            print(f"   {cls}: {count} samples ({count/len(df)*100:.1f}%)")
        
        print(f"\n🎯 Unique Classes ({len(class_counts)}):")
        for i, cls in enumerate(sorted(class_counts.keys())):
            print(f"   {i+1}. {cls}")
    
    # Show sample features for each class
    print("\n🔬 Sample Features by Class:")
    for cls in df['class1'].unique()[:5]:  # Show first 5 classes
        cls_data = df[df['class1'] == cls].iloc[0]
        print(f"\n   {cls}:")
        print(f"     duration: {cls_data.get('duration', 'N/A')}")
        print(f"     flowPktsPerSecond: {cls_data.get('flowPktsPerSecond', 'N/A')}")
        print(f"     flowBytesPerSecond: {cls_data.get('flowBytesPerSecond', 'N/A')}")
        print(f"     std_flowiat: {cls_data.get('std_flowiat', 'N/A')}")

else:
    print(f"❌ Training file not found: {training_file}")

print("\n" + "="*50)
print("🚀 TO TEST YOUR MODELS:")
print("1. Install missing dependencies: pip install flask tensorflow scikit-learn xgboost")
print("2. Run: python appcopy.py") 
print("3. Open: http://localhost:5000")
print("4. Upload your PCAP files and check predictions!")
print("="*50)
