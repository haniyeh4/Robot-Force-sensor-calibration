import socket
import struct
import time
import threading
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

ROBOT_IP = 'localhost'
ROBOT_PORT = 18871
FEEDBACK_BYTES = 37 * 8
BUFFER_SIZE = 200
CALIBRATION_SAMPLES = 100

# === Load trained model ===
model_path = 'models/1_TheRightWay.pkl'
model = joblib.load(model_path)
print("Loaded model from:", model_path)

# === Shared data ===
latest_input = None
latest_ff_ee = None
lock = threading.Lock()

# === Calibration ===
calibration_offsets = np.zeros(3)
calibrated = False
calibration_inputs = []
calibration_actuals = []

# === Thread: receive live data ===
def receive_data():
    global latest_input, latest_ff_ee
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ROBOT_IP, ROBOT_PORT))
    print("Connected to robot.")

    try:
        while True:
            data = b''
            while len(data) < FEEDBACK_BYTES:
                packet = sock.recv(FEEDBACK_BYTES - len(data))
                if not packet:
                    raise ConnectionError("Robot closed connection.")
                data += packet

            feedback = struct.unpack('<37d', data)
            x_pos = feedback[0:6]
            ff_ee = feedback[12:18]

            features = [x_pos[3], x_pos[4], x_pos[5]]

            with lock:
                latest_input = np.array(features).reshape(1, -1)
                latest_ff_ee = np.array(ff_ee)
    except Exception as e:
        print(f"[Receiver Error] {e}")
    finally:
        sock.close()

# === Start data thread ===
threading.Thread(target=receive_data, daemon=True).start()

# === Plotting setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

actual_data = [[] for _ in range(3)]
actual_lines = []
for i in range(3):
    line, = ax1.plot([], [], label=f"FF_EE[{i}]")
    actual_lines.append(line)

ax1.set_ylim(-20, 20)
ax1.set_xlim(0, BUFFER_SIZE)
ax1.set_title("Live Actual FF_EE[0:3]")
ax1.set_ylabel("Force (N)")
ax1.legend()

error_data = [[] for _ in range(3)]
error_lines = []
for i in range(3):
    line, = ax2.plot([], [], label=f"Corrected Error FF_EE[{i}]")
    error_lines.append(line)

ax2.set_ylim(-10, 10)
ax2.set_xlim(0, BUFFER_SIZE)
ax2.set_title("Prediction Error After Calibration")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Error (N)")
ax2.legend()

x_data = []
index = 0

# === Main loop ===
try:
    while True:
        with lock:
            x_input = latest_input
            y_true = latest_ff_ee

        if x_input is not None and y_true is not None:
            y_pred = model.predict(x_input)[0]
            ff_ee_actual = y_true[:3]

            # === Calibration phase ===
            if not calibrated:
                calibration_inputs.append(x_input.copy())
                calibration_actuals.append(ff_ee_actual.copy())

                if len(calibration_inputs) >= CALIBRATION_SAMPLES:
                    calibration_preds = np.array([model.predict(x)[0] for x in calibration_inputs])
                    calibration_actuals_np = np.array(calibration_actuals)
                    calibration_offsets = np.mean(calibration_actuals_np - calibration_preds, axis=0)
                    print(f"Calibration complete. Offset: {calibration_offsets}")
                    calibrated = True
                continue  # skip plotting until calibration is done

            # === Apply calibration bias ===
            y_pred_corrected = y_pred + calibration_offsets
            error = ff_ee_actual - y_pred_corrected

            x_data.append(index)

            for i in range(3):
                actual_data[i].append(ff_ee_actual[i])
                actual_lines[i].set_data(x_data[-BUFFER_SIZE:], actual_data[i][-BUFFER_SIZE:])
                error_data[i].append(error[i])
                error_lines[i].set_data(x_data[-BUFFER_SIZE:], error_data[i][-BUFFER_SIZE:])

            for ax in [ax1, ax2]:
                ax.set_xlim(max(0, index - BUFFER_SIZE), index)

            index += 1

        plt.pause(0.05)
except KeyboardInterrupt:
    print("Stopped.")
