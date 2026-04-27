import serial
import numpy as np
import json
import csv
import time
import os
from datetime import datetime

# ── Thử dùng LiteRT mới, fallback về TFLite cũ ──
try:
    from ai_edge_litert.interpreter import Interpreter
    print("[INFO] Dùng ai_edge_litert interpreter")
except ImportError:
    import warnings
    warnings.filterwarnings('ignore')
    from tensorflow.lite.python.interpreter import Interpreter
    print("[INFO] Dùng tensorflow.lite interpreter")

# ─────────────────────────────────────────
# 0. Load config
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'model/scaler_params.json')) as f:
    params = json.load(f)

MEAN      = np.array(params['mean'],  dtype=np.float32)
SCALE     = np.array(params['scale'], dtype=np.float32)
THRESHOLD = float(params['ae_threshold'])

print(f"[INFO] Threshold: {THRESHOLD:.6f}")
print(f"[INFO] Mean:      {MEAN}")
print(f"[INFO] Scale:     {SCALE}")

# ─────────────────────────────────────────
# 1. Load TFLite model
# ─────────────────────────────────────────
model_path = os.path.join(BASE_DIR, 'model/autoencoder.tflite')
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_idx  = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']
print(f"[INFO] Model loaded: {model_path}")

# ─────────────────────────────────────────
# 2. Hàm inference
# ─────────────────────────────────────────
def run_inference(temp: float, hum: float):
    """
    Trả về: (mse, is_anomaly, label_str)
    """
    raw = np.array([[temp, hum]], dtype=np.float32)
    scaled = (raw - MEAN) / SCALE

    interpreter.set_tensor(input_idx, scaled)
    interpreter.invoke()
    recon = interpreter.get_tensor(output_idx)

    mse = float(np.mean((scaled - recon) ** 2))
    is_anomaly = mse > THRESHOLD
    label = "ANOMALY" if is_anomaly else "NORMAL"
    return mse, is_anomaly, label

def parse_uart(line: str):
    """Parse 'T:27.43,H:61.20,MS:1023' → (temp, hum, ms)"""
    try:
        parts = {}
        for token in line.strip().split(','):
            k, v = token.split(':')
            parts[k.strip()] = float(v.strip())
        return parts['T'], parts['H'], int(parts['MS'])
    except Exception:
        return None

# ─────────────────────────────────────────
# 3. Setup UART & log file
# ─────────────────────────────────────────
UART_PORT = '/dev/ttyS0'
UART_BAUD = 115200
LOG_FILE  = os.path.join(BASE_DIR, 'data/inference_log.csv')

ser = serial.Serial(UART_PORT, UART_BAUD, timeout=2)
print(f"[INFO] UART: {UART_PORT} @ {UART_BAUD}")
print(f"[INFO] Log: {LOG_FILE}")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
log_exists = os.path.isfile(LOG_FILE)

# ─────────────────────────────────────────
# 4. Vòng lặp inference chính
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  Edge AI – Anomaly Detection (Ctrl+C để dừng)")
print("="*55)
print(f"{'Time':<10} {'Temp':>6} {'Hum':>6}  {'MSE':>10}  {'Status'}")
print("-"*55)

anomaly_count = 0
total_count   = 0

with open(LOG_FILE, 'a', newline='') as logf:
    writer = csv.writer(logf)
    if not log_exists:
        writer.writerow(['timestamp', 'temperature', 'humidity',
                         'mse', 'threshold', 'is_anomaly', 'label'])

    try:
        while True:
            raw_line = ser.readline().decode('utf-8', errors='ignore')

            if not raw_line.strip() or raw_line.startswith('ERROR'):
                continue

            result = parse_uart(raw_line)
            if result is None:
                continue

            temp, hum, ms = result
            mse, is_anomaly, label = run_inference(temp, hum)

            total_count += 1
            if is_anomaly:
                anomaly_count += 1

            ts = datetime.now().strftime('%H:%M:%S')

            # ── In ra terminal ──
            status_icon = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
            print(f"{ts:<10} {temp:>5.1f}° {hum:>5.1f}%  {mse:>10.4f}  {status_icon}")

            # ── Ghi log ──
            writer.writerow([
                datetime.now().isoformat(),
                round(temp, 2), round(hum, 2),
                round(mse, 6), round(THRESHOLD, 6),
                int(is_anomaly), label
            ])
            logf.flush()

            # ── In thống kê mỗi 30 samples ──
            if total_count % 30 == 0:
                rate = anomaly_count / total_count * 100
                print(f"\n{'─'*55}")
                print(f"  📊 Stats: {total_count} samples | "
                      f"Anomaly: {anomaly_count} ({rate:.1f}%)")
                print(f"{'─'*55}\n")

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Dừng. Tổng: {total_count} samples | "
              f"Anomaly: {anomaly_count} ({anomaly_count/max(total_count,1)*100:.1f}%)")
        print(f"[INFO] Log đã lưu tại: {LOG_FILE}")

    finally:
        ser.close()
