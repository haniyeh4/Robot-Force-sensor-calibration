#!/usr/bin/env python3
"""
Kuka Data Collection Script
Connects to Kuka controller to send commands and to actual Kuka robot to read sensor data.
Sends commands every 2 seconds from the generated path file and saves sensor data.
"""

import socket
import json
import time
import struct
import csv
from tqdm import tqdm
import os

class KukaDataCollector:
    def __init__(self, controller_host='localhost', controller_port=5000, 
                 robot_host='localhost', robot_port=18871):
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.robot_host = robot_host
        self.robot_port = robot_port
        
        # Controller connection (for sending commands)
        self.controller_socket = None
        self.controller_connected = False
        
        # Robot connection (for reading data)
        self.robot_socket = None
        self.robot_connected = False
        
        # Data collection parameters
        self.feedback_bytes = 37 * 8  # 37 doubles
        self.commands = []
        self.collected_data = []
        
    def connect_to_controller(self):
        """Connect to Kuka controller for sending commands."""
        try:
            self.controller_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.controller_socket.connect((self.controller_host, self.controller_port))
            self.controller_connected = True
            print(f"✅ Connected to Kuka controller at {self.controller_host}:{self.controller_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to controller: {e}")
            return False
    
    def connect_to_robot(self):
        """Connect to actual Kuka robot for reading sensor data."""
        try:
            self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.robot_socket.connect((self.robot_host, self.robot_port))
            self.robot_connected = True
            print(f"✅ Connected to Kuka robot at {self.robot_host}:{self.robot_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to robot: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from both connections."""
        if self.controller_socket:
            self.controller_socket.close()
            self.controller_connected = False
        if self.robot_socket:
            self.robot_socket.close()
            self.robot_connected = False
        print("🔌 Disconnected from all connections")
    
    def load_commands_from_file(self, filename="../data/paths/kuka_path_commands.txt"):
        """Load commands from the generated path file."""
        try:
            if not os.path.exists(filename):
                print(f"❌ Command file '{filename}' not found")
                return False
            
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('move ') and not line.startswith('#'):
                        # Parse the move command
                        parts = line.split()
                        if len(parts) == 7:  # move x y z rx ry rz
                            try:
                                command = [float(parts[1]), float(parts[2]), float(parts[3]), 
                                         float(parts[4]), float(parts[5]), float(parts[6])]
                                self.commands.append(command)
                            except ValueError:
                                print(f"⚠️ Skipping invalid command: {line}")
                                continue
            
            print(f"📋 Loaded {len(self.commands)} commands from '{filename}'")
            return True
            
        except Exception as e:
            print(f"❌ Error loading commands: {e}")
            return False
    
    def send_command_to_controller(self, command):
        """Send a command to the Kuka controller."""
        if not self.controller_connected:
            return False
        
        try:
            # Format command as JSON for the controller
            command_dict = {
                "type": "move",
                "position": command[:3],  # x, y, z
                "orientation": command[3:]  # rx, ry, rz
            }
            
            command_json = json.dumps(command_dict)
            self.controller_socket.send(command_json.encode('utf-8'))
            
            # Wait for response (non-blocking)
            self.controller_socket.settimeout(0.1)
            try:
                response = self.controller_socket.recv(1024)
                response_data = json.loads(response.decode('utf-8'))
                if response_data.get('status') == 'error':
                    print(f"⚠️ Controller error: {response_data.get('message', 'Unknown error')}")
                    return False
            except socket.timeout:
                # No response received, assume command was sent
                pass
            except Exception as e:
                print(f"⚠️ Controller response error: {e}")
            
            self.controller_socket.settimeout(None)
            return True
            
        except Exception as e:
            print(f"❌ Error sending command to controller: {e}")
            return False
    
    def read_robot_data(self):
        """Read sensor data from the actual Kuka robot."""
        if not self.robot_connected:
            return None
        
        try:
            # Clear any previous data in the buffer
            self.robot_socket.setblocking(0)
            try:
                while True:
                    self.robot_socket.recv(1024)
            except BlockingIOError:
                pass
            self.robot_socket.setblocking(1)
            
            # Read the current data (37 doubles)
            data = self.recv_all(self.robot_socket, self.feedback_bytes)
            if data is None:
                print("❌ Connection closed by robot")
                return None
            
            # Unpack the data
            feedback = struct.unpack('<37d', data)
            
            # Extract the relevant data
            robot_data = {
                "X_Pos": list(feedback[0:6]),      # Position and orientation
                "FRaw": list(feedback[6:12]),      # Raw force readings
                "FF_EE": list(feedback[12:18]),    # End-effector force feedback
                "FF_GB": list(feedback[18:24]),    # Gravity bias force feedback
                "Foot_Pedal": feedback[24],         # Foot pedal reading
                "X_Dot_Cmds": list(feedback[25:31]), # Velocity commands
                "X_DoubleDot_Cmds": list(feedback[31:37]) # Acceleration commands
            }
            
            return robot_data
            
        except Exception as e:
            print(f"❌ Error reading robot data: {e}")
            return None
    
    def recv_all(self, sock, nbytes):
        """Receive exactly nbytes from socket."""
        data = b''
        while len(data) < nbytes:
            packet = sock.recv(nbytes - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def collect_data(self, output_file="kuka_collected_data.jsonl"):
        """Main data collection loop."""
        if not self.commands:
            print("❌ No commands loaded. Please load commands first.")
            return False
        
        print(f"\n🚀 Starting data collection...")
        print(f"📊 Total commands to execute: {len(self.commands)}")
        print(f"⏱️  Command interval: 2 seconds")
        print(f"💾 Output file: {output_file}")
        
        try:
            with open(output_file, 'w') as f:
                for i, command in enumerate(tqdm(self.commands, desc="Executing commands")):
                    print(f"\n📍 Command {i+1}/{len(self.commands)}: {command}")
                    
                    # Send command to controller
                    if not self.send_command_to_controller(command):
                        print(f"⚠️ Failed to send command {i+1}, skipping...")
                        continue
                    
                    # Wait for robot to move and stabilize
                    if i == 0:
                        time.sleep(10)
                    else:
                        time.sleep(2)
                    
                    # Read sensor data from robot
                    robot_data = self.read_robot_data()
                    if robot_data is None:
                        print(f"⚠️ Failed to read data for command {i+1}, skipping...")
                        continue
                    
                    # Add command information to the data
                    data_point = {
                        "command_index": i,
                        "command": command,
                        "timestamp": time.time(),
                        "robot_data": robot_data
                    }
                    
                    # Save to file
                    f.write(json.dumps(data_point) + '\n')
                    f.flush()  # Ensure data is written immediately
                    
                    # Store in memory
                    self.collected_data.append(data_point)
                    
                    print(f"✅ Data collected and saved for command {i+1}")
                    
                    # Wait before next command (total 2 seconds per command)
                    if i < len(self.commands) - 1:  # Don't wait after last command
                        time.sleep(0.5)  # Additional wait to make total 2 seconds
            
            print(f"\n🎯 Data collection complete!")
            print(f"📁 Data saved to: {output_file}")
            print(f"📊 Total data points collected: {len(self.collected_data)}")
            return True
            
        except Exception as e:
            print(f"❌ Error during data collection: {e}")
            return False
    
    def print_summary(self):
        """Print a summary of the collected data."""
        if not self.collected_data:
            print("📋 No data collected yet.")
            return
        
        print(f"\n📋 Data Collection Summary")
        print("=" * 40)
        print(f"Total data points: {len(self.collected_data)}")
        print(f"Commands executed: {len(self.commands)}")
        
        if self.collected_data:
            first_data = self.collected_data[0]
            last_data = self.collected_data[-1]
            
            print(f"\nFirst command: {first_data['command']}")
            print(f"Last command: {last_data['command']}")
            print(f"Collection duration: {last_data['timestamp'] - first_data['timestamp']:.1f} seconds")

def main():
    """Main function."""
    print("🤖 Kuka Data Collection Script")
    print("=" * 40)
    
    # Configuration
    controller_host = 'localhost'
    controller_port = 5000
    robot_host = 'localhost'
    robot_port = 18871
    command_file = "data/paths/kuka_path_v2_commands.txt"
    output_file = "data/Temp/1_Quasi_Static_Log.jsonl"
    
    # Create collector
    collector = KukaDataCollector(controller_host, controller_port, robot_host, robot_port)
    
    try:
        # Load commands
        print("📋 Loading commands from file...")
        if not collector.load_commands_from_file(command_file):
            print("❌ Failed to load commands. Exiting.")
            return 1
        
        # Connect to controller
        print("🔌 Connecting to Kuka controller...")
        if not collector.connect_to_controller():
            print("❌ Failed to connect to controller. Exiting.")
            return 1
        
        # Connect to robot
        print("🔌 Connecting to Kuka robot...")
        if not collector.connect_to_robot():
            print("❌ Failed to connect to robot. Exiting.")
            return 1
        
        # Start data collection
        print("🚀 Starting data collection...")
        if not collector.collect_data(output_file):
            print("❌ Data collection failed.")
            return 1
        
        # Print summary
        collector.print_summary()
        
    except KeyboardInterrupt:
        print("\n🛑 Data collection interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        collector.disconnect()
    
    return 0

if __name__ == "__main__":
    exit(main())

