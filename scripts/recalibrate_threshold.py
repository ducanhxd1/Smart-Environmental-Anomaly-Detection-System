# scripts/recalibrate_threshold.py
import numpy as np
import pandas as pd
import json
import tensorflow as tf

# Load data và scaler params
df = pd.read_csv('/home/da/edge_anomaly/data/raw.csv').dropna()
X = df[['temperature', 'humidity']].values
y = df['label'].values

with open('model/scaler_params.json') as f:
    params = json.load(f)

mean  = np.array(params['mean'])
scale = np.array(params['scale'])

# Normalize
X_scaled = (X - mean) / scale
X_normal_scaled = X_scaled[y == 0].astype(np.float32)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model/autoencoder.tflite')
interpreter.allocate_tensors()
input_idx  = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

# Tính reconstruction error trên TOÀN BỘ normal data
mse_list = []
for sample in X_normal_scaled:
    inp = sample.reshape(1, 2)
    interpreter.set_tensor(input_idx, inp)
    interpreter.invoke()
    recon = interpreter.get_tensor(output_idx)
    mse = float(np.mean((inp - recon) ** 2))
    mse_list.append(mse)

mse_arr = np.array(mse_list)
print(f"Normal MSE — mean: {mse_arr.mean():.4f} | std: {mse_arr.std():.4f}")
print(f"Normal MSE — min:  {mse_arr.min():.4f} | max: {mse_arr.max():.4f}")

# Threshold = 95th percentile của normal data
new_threshold = float(np.percentile(mse_arr, 95))
print(f"\n✅ New threshold (95th percentile): {new_threshold:.6f}")

# Cập nhật scaler_params.json
params['ae_threshold'] = new_threshold
with open('model/scaler_params.json', 'w') as f:
    json.dump(params, f, indent=2)
print("✅ Đã cập nhật scaler_params.json")

# Verify lại
print("\n--- Verify ---")
# Normal sample
s = X_normal_scaled[:1]
interpreter.set_tensor(input_idx, s)
interpreter.invoke()
r = interpreter.get_tensor(output_idx)
mse = float(np.mean((s - r) ** 2))
print(f"Normal sample  → MSE: {mse:.6f} | {'NORMAL ✅' if mse <= new_threshold else 'ANOMALY ❌'}")

# Anomaly sample
X_anomaly = X_scaled[y == 1].astype(np.float32)
if len(X_anomaly) > 0:
    s = X_anomaly[:1]
    interpreter.set_tensor(input_idx, s)
    interpreter.invoke()
    r = interpreter.get_tensor(output_idx)
    mse = float(np.mean((s - r) ** 2))
    print(f"Anomaly sample → MSE: {mse:.6f} | {'ANOMALY ✅' if mse > new_threshold else 'NORMAL ❌'}")
