import serial
import csv
import time
import os
import argparse
from datetime import datetime

def parse_line(line: str):
    """Parse 'T:27.43,H:61.20,MS:1023' → (temp, hum, ms)"""
    try:
        parts = {}
        for token in line.strip().split(','):
            k, v = token.split(':')
            parts[k] = float(v)
        return parts['T'], parts['H'], int(parts['MS'])
    except Exception:
        return None

def collect(port='/dev/serial0', baud=115200, duration=300, label=0, output='data/raw.csv'):
    """
    label=0 → NORMAL
    label=1 → ANOMALY
    duration: số giây thu thập
    """
    os.makedirs(os.path.dirname(output), exist_ok=True)
    file_exists = os.path.isfile(output)

    print(f"[INFO] Thu thập dữ liệu trong {duration}s | Label: {'ANOMALY' if label else 'NORMAL'}")
    print(f"[INFO] Lưu vào: {output}")
    print("[INFO] Nhấn Ctrl+C để dừng sớm\n")

    ser = serial.Serial(port, baud, timeout=2)
    count = 0
    start = time.time()

    with open(output, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'temperature', 'humidity', 'ms_device', 'label'])

        try:
            while (time.time() - start) < duration:
                raw = ser.readline().decode('utf-8', errors='ignore')
                if raw.startswith('ERROR'):
                    print(f"[WARN] Sensor error: {raw.strip()}")
                    continue

                result = parse_line(raw)
                if result is None:
                    continue

                temp, hum, ms = result
                ts = datetime.now().isoformat()
                writer.writerow([ts, temp, hum, ms, label])
                f.flush()
                count += 1

                # In progress mỗi 10 samples
                if count % 10 == 0:
                    elapsed = time.time() - start
                    print(f"[{elapsed:5.0f}s] Samples: {count:4d} | T={temp:.2f}°C | H={hum:.2f}%")

        except KeyboardInterrupt:
            print("\n[INFO] Dừng thu thập sớm.")

    ser.close()
    print(f"\n✅ Hoàn tất! Tổng cộng {count} samples → {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',     default='/dev/serial0')
    parser.add_argument('--baud',     type=int, default=115200)
    parser.add_argument('--duration', type=int, default=300,
                        help='Số giây thu thập (default: 300)')
    parser.add_argument('--label',    type=int, default=0,
                        choices=[0, 1], help='0=NORMAL, 1=ANOMALY')
    parser.add_argument('--output',   default='data/raw.csv')
    args = parser.parse_args()

    collect(args.port, args.baud, args.duration, args.label, args.output)
