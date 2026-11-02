import socket
import struct
import json
import time
from pathlib import Path

# === Config ===
HOST = 'localhost'          # Replace with your KUKA IP if needed
PORT = 18871
OUTPUT_JSONL = 'paper/data/manual_fresh_log.jsonl'
FEEDBACK_BYTES = 37 * 8     # 37 doubles

# === Ensure output path exists ===
Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

# === Helper function ===
def recv_all(sock, nbytes):
    """Receive exactly nbytes from socket."""
    data = b''
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            return None
        data += packet
    return data

# === Connect to KUKA ===
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
print(f"Connected to {HOST}:{PORT}")
print("Press ENTER to grab the latest snapshot. Type 'q' to quit.\n")

with open(OUTPUT_JSONL, 'w') as jsonl_file:
    while True:
        user_input = input("[ENTER] Grab snapshot | [q + ENTER] Quit: ")
        if user_input.lower() == 'q':
            break

        # === Flush stale data from buffer ===
        sock.setblocking(0)
        try:
            while True:
                sock.recv(1024)
        except BlockingIOError:
            pass
        sock.setblocking(1)

        # === Read the latest data ===
        data = recv_all(sock, FEEDBACK_BYTES)
        if data is None:
            print("Connection closed by robot.")
            break

        feedback = struct.unpack('<37d', data)

        entry = {
            "X_Pos": list(feedback[0:6]),
            "FRaw": list(feedback[6:12]),
            "FF_EE": list(feedback[12:18]),
            "FF_GB": list(feedback[18:24]),
            "Foot_Pedal": feedback[24],
            "X_Dot_Cmds": list(feedback[25:31]),
            "X_DoubleDot_Cmds": list(feedback[31:37])
        }

        jsonl_file.write(json.dumps(entry) + '\n')
        print("✅ Fresh snapshot logged.")

sock.close()
print(f"\nConnection closed. Log saved to {OUTPUT_JSONL}")
