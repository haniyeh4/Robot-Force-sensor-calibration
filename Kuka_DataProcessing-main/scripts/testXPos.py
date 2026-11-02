import socket
import json
import time

def send_kuka_command(position, orientation=None):
    """Send command to KUKA command server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 5000))
    
    command = {
        'type': 'move',
        'position': position,
        'speed': 0.1
    }
    
    if orientation is not None:
        command['orientation'] = orientation
    
    sock.send(json.dumps(command).encode('utf-8'))
    response = sock.recv(1024)
    sock.close()
    
    return json.loads(response.decode('utf-8'))

# Example usage in your loop
positions = [
    [0.4, 0.0, 0.4],
    [0.45, 0.05, 0.4],
    [0.35, -0.05, 0.4]
]

for pos in positions:
    result = send_kuka_command(pos)
    print(f"Command result: {result}")
    time.sleep(10)  # Wait for movement