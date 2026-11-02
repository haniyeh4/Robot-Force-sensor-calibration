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

# === Load trained model ===
model_path = 'models/Aug29/1_Quasi_Static_Poly.pkl'
model = joblib.load(model_path)
print("Loaded model from:", model_path)

# === Shared data ===
latest_input = None
latest_ff_ee = None
lock = threading.Lock()

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

            # Extract input features and true FF_EE
            x_pos = feedback[0:6]
            ff_ee = feedback[12:18]  # True FF_EE

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

# ... all imports and setup above unchanged ...

# === Plotting setup ===
plt.ion()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

# --- Force (0:3) ---
actual_force_data = [[] for _ in range(3)]
force_lines = []
for i in range(3):
    line, = ax1.plot([], [], label=f"FF_EE[{i}]")
    force_lines.append(line)

ax1.set_ylim(-20, 20)
ax1.set_xlim(0, BUFFER_SIZE)
ax1.set_title("Live Force FF_EE[0:3]")
ax1.set_ylabel("Force (N)")
ax1.legend()

# --- Force Error (0:3) ---
force_error_data = [[] for _ in range(3)]
force_error_lines = []
for i in range(3):
    line, = ax2.plot([], [], label=f"Error_FF_EE[{i}]")
    force_error_lines.append(line)

ax2.set_ylim(-10, 10)
ax2.set_xlim(0, BUFFER_SIZE)
ax2.set_title("Force Prediction Error")
ax2.set_ylabel("Error (N)")
ax2.legend()

# --- Torque (3:6) ---
actual_torque_data = [[] for _ in range(3)]
torque_lines = []
for i in range(3):
    line, = ax3.plot([], [], label=f"FF_EE[{i+3}]")
    torque_lines.append(line)

ax3.set_ylim(-1, 1)
ax3.set_xlim(0, BUFFER_SIZE)
ax3.set_title("Live Torque FF_EE[3:6]")
ax3.set_ylabel("Torque (Nm)")
ax3.legend()

# --- Torque Error (3:6) ---
torque_error_data = [[] for _ in range(3)]
torque_error_lines = []
for i in range(3):
    line, = ax4.plot([], [], label=f"Error_FF_EE[{i+3}]")
    torque_error_lines.append(line)

ax4.set_ylim(-1, 1)
ax4.set_xlim(0, BUFFER_SIZE)
ax4.set_title("Torque Prediction Error")
ax4.set_xlabel("Frame")
ax4.set_ylabel("Error (Nm)")
ax4.legend()

x_data = []
index = 0

# === Main plotting loop ===
try:
    while True:
        with lock:
            x_input = latest_input
            y_true = latest_ff_ee

        if x_input is not None and y_true is not None:
            y_pred = model.predict(x_input)[0]
            x_data.append(index)

            # Force components
            for i in range(3):
                actual = y_true[i]
                pred = y_pred[i]
                actual_force_data[i].append(actual)
                force_error_data[i].append(actual - pred)
                force_lines[i].set_data(x_data[-BUFFER_SIZE:], actual_force_data[i][-BUFFER_SIZE:])
                force_error_lines[i].set_data(x_data[-BUFFER_SIZE:], force_error_data[i][-BUFFER_SIZE:])

            # Torque components
            for i in range(3, 6):
                actual = y_true[i]
                pred = y_pred[i]
                actual_torque_data[i - 3].append(actual)
                torque_error_data[i - 3].append(actual - pred)
                torque_lines[i - 3].set_data(x_data[-BUFFER_SIZE:], actual_torque_data[i - 3][-BUFFER_SIZE:])
                torque_error_lines[i - 3].set_data(x_data[-BUFFER_SIZE:], torque_error_data[i - 3][-BUFFER_SIZE:])

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(max(0, index - BUFFER_SIZE), index)

            index += 1

        plt.pause(0.05)
except KeyboardInterrupt:
    print("Stopped.")
