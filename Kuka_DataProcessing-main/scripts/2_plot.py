import socket
import struct
import time
import threading
import matplotlib.pyplot as plt
import joblib
import numpy as np

# === Config ===
ROBOT_IP = 'localhost'
ROBOT_PORT = 18871
FEEDBACK_BYTES = 37 * 8
BUFFER_SIZE = 200

# === Load models ===
qs_model = joblib.load('models/1_TheRightWay.pkl')
xdd_model = joblib.load('models/2_dynamic_model_xdd_only.pkl')
print("Models loaded.")

# === Shared state ===
latest_pos = None
latest_ff_ee = None
latest_xdd = None
lock = threading.Lock()

def receive_data():
    global latest_pos, latest_ff_ee, latest_xdd
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
            x_dd = feedback[31:37]

            with lock:
                latest_pos = x_pos
                latest_ff_ee = ff_ee
                latest_xdd = x_dd
    except Exception as e:
        print(f"[Receiver Error] {e}")
    finally:
        sock.close()

# === Start data thread ===
threading.Thread(target=receive_data, daemon=True).start()

# === Plotting setup ===
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

labels = [f"FF_EE[{i}]" for i in range(6)]

# QS Error Plot (Top)
qs_error_data = [[] for _ in range(6)]
qs_error_lines = []
for i in range(6):
    line, = ax1.plot([], [], label=f"QS Error {labels[i]}")
    qs_error_lines.append(line)

ax1.set_ylim(-10, 10)
ax1.set_xlim(0, BUFFER_SIZE)
ax1.set_title("Quasi-Static Prediction Error")
ax1.set_ylabel("Error (N / Nm)")
ax1.legend()

# Final Error Plot (Bottom)
final_error_data = [[] for _ in range(6)]
final_error_lines = []
for i in range(6):
    line, = ax2.plot([], [], label=f"Final Error {labels[i]}")
    final_error_lines.append(line)

ax2.set_ylim(-10, 10)
ax2.set_xlim(0, BUFFER_SIZE)
ax2.set_title("Final Error (QS + XDD Prediction)")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Error (N / Nm)")
ax2.legend()

x_data = []
index = 0

# === Main plotting loop ===
try:
    while True:
        with lock:
            pos = latest_pos
            ff_ee = latest_ff_ee
            xdd = latest_xdd

        if pos is not None and ff_ee is not None and xdd is not None:
            orientation = np.array(pos[3:6]).reshape(1, -1)
            acceleration = np.array(xdd).reshape(1, -1)

            qs_pred = qs_model.predict(orientation)[0]      # (6,)
            xdd_pred = xdd_model.predict(acceleration)[0]   # (6,)
            total_pred = np.copy(qs_pred)
            total_pred[:3] += xdd_pred  # Only apply dynamic correction to force

            actual_ff_ee = np.array(ff_ee[:6])
            qs_error = actual_ff_ee - qs_pred
            final_error = actual_ff_ee - total_pred

            x_data.append(index)

            for i in range(6):
                qs_error_data[i].append(qs_error[i])
                final_error_data[i].append(final_error[i])

                qs_error_lines[i].set_data(x_data[-BUFFER_SIZE:], qs_error_data[i][-BUFFER_SIZE:])
                final_error_lines[i].set_data(x_data[-BUFFER_SIZE:], final_error_data[i][-BUFFER_SIZE:])

            for ax in [ax1, ax2]:
                ax.set_xlim(max(0, index - BUFFER_SIZE), index)

            index += 1

        plt.pause(0.05)
except KeyboardInterrupt:
    print("Stopped.")
