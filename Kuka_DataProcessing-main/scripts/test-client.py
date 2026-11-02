#!/usr/bin/env python3
"""
Test client for the Standalone KUKA Motion Controller.
Demonstrates how to connect and send commands to the controller.
"""

import socket
import json
import time
import sys


class KUKAClient:
    """Test client for the KUKA motion controller."""
    
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to the KUKA controller."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✅ Connected to KUKA controller at {self.host}:{self.port}")
            
            # Send immediate status command to prevent timeout
            print("📊 Sending initial status request...")
            initial_status = self.get_status()
            print(f"Initial status: {json.dumps(initial_status, indent=2)}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the controller."""
        if self.socket:
            self.socket.close()
        self.connected = False
        print("🔌 Disconnected from controller")
    
    def send_command(self, command: dict) -> dict:
        """Send a command and receive response."""
        if not self.connected:
            return {"status": "error", "message": "Not connected"}
        
        try:
            # Send command
            command_json = json.dumps(command)
            self.socket.send(command_json.encode('utf-8'))
            
            # Receive response
            response = self.socket.recv(1024)
            return json.loads(response.decode('utf-8'))
            
        except Exception as e:
            return {"status": "error", "message": f"Communication error: {e}"}
    
    def get_status(self) -> dict:
        """Get current status from controller."""
        return self.send_command({"type": "status"})
    
    def move_to_position(self, position: list, orientation: list = None) -> dict:
        """Move to a specific position and optionally orientation."""
        command = {"type": "move", "position": position}
        if orientation:
            command["orientation"] = orientation
        return self.send_command(command)
    
    def stop_movement(self) -> dict:
        """Stop current movement."""
        return self.send_command({"type": "stop"})
    
    def emergency_stop(self) -> dict:
        """Activate emergency stop."""
        return self.send_command({"type": "emergency_stop"})


def demo_movement(client: KUKAClient):
    """Demonstrate basic movement commands."""
    print("\n🎯 Movement Demo")
    print("=" * 30)
    
    # Get initial status
    print("📊 Getting initial status...")
    status = client.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Move to a safe position
    print("\n📍 Moving to position [0.4, 0.0, 0.43]...")
    response = client.move_to_position([0.4, 0.0, 0.43])
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Wait a bit
    time.sleep(3)
    
    # Check status again
    print("\n📊 Checking status after movement...")
    status = client.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Move to another position
    print("\n📍 Moving to position [0.35, 0.05, 0.45]...")
    response = client.move_to_position([0.35, 0.05, 0.45])
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Wait a bit
    time.sleep(3)
    
    # Stop movement
    print("\n🛑 Stopping movement...")
    response = client.stop_movement()
    print(f"Response: {json.dumps(response, indent=2)}")


def demo_safety_checks(client: KUKAClient):
    """Demonstrate safety validation."""
    print("\n🛡️ Safety Validation Demo")
    print("=" * 30)
    
    # Try to move to an unsafe position (outside bounds)
    print("⚠️ Attempting to move to unsafe position [0.6, 0.0, 0.43]...")
    response = client.move_to_position([0.6, 0.0, 0.43])
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Try to move to a very far position
    print("\n⚠️ Attempting to move to very far position [0.4, 0.0, 0.8]...")
    response = client.move_to_position([0.4, 0.0, 0.8])
    print(f"Response: {json.dumps(response, indent=2)}")
    
    # Try to move with excessive distance
    print("\n⚠️ Attempting to move with excessive distance [0.1, 0.0, 0.43]...")
    response = client.move_to_position([0.1, 0.0, 0.43])
    print(f"Response: {json.dumps(response, indent=2)}")


def interactive_mode(client: KUKAClient):
    """Interactive command mode."""
    print("\n🎮 Interactive Mode")
    print("=" * 30)
    print("Commands:")
    print("  move <x> <y> <z> [rx] [ry] [rz] - Move to position/orientation")
    print("  status - Get current status")
    print("  stop - Stop movement")
    print("  emergency - Emergency stop")
    print("  quit - Exit")
    
    while True:
        try:
            command = input("\n> ").strip().lower().split()
            if not command:
                continue
            
            if command[0] == 'quit':
                break
            elif command[0] == 'status':
                response = client.get_status()
                print(json.dumps(response, indent=2))
            elif command[0] == 'stop':
                response = client.stop_movement()
                print(json.dumps(response, indent=2))
            elif command[0] == 'emergency':
                response = client.emergency_stop()
                print(json.dumps(response, indent=2))
            elif command[0] == 'move':
                if len(command) < 4:
                    print("Usage: move <x> <y> <z> [rx] [ry] [rz]")
                    continue
                
                try:
                    position = [float(command[1]), float(command[2]), float(command[3])]
                    orientation = None
                    if len(command) >= 7:
                        orientation = [float(command[4]), float(command[5]), float(command[6])]
                    
                    response = client.move_to_position(position, orientation)
                    print(json.dumps(response, indent=2))
                except ValueError:
                    print("Invalid number format")
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    print("🚀 KUKA Motion Controller Test Client")
    print("=" * 40)
    
    # Parse command line arguments
    host = 'localhost'
    port = 5000
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print("❌ Invalid port number. Using default port 5000.")
    
    # Create client
    client = KUKAClient(host, port)
    
    try:
        # Connect to controller
        if not client.connect():
            print("❌ Failed to connect. Make sure the controller is running.")
            return 1
        
        # Run demos
        demo_movement(client)
        demo_safety_checks(client)
        
        # Interactive mode
        interactive_mode(client)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

