#!/usr/bin/env python3
"""
Kuka Robot Path Generator v3
Generates 50 random orientation points within robot limits.
rx and ry are randomly selected from -20 to +20 degrees.
Position is fixed at [0.4, 0, 0.33] and rz at 3.14.
"""

import math
import os
import random

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0

def generate_random_path(num_points=50, min_angle=-20, max_angle=20):
    """
    Generate random orientation points within the specified limits.
    
    Args:
        num_points: Number of random points to generate (default: 50)
        min_angle: Minimum angle in degrees (default: -20)
        max_angle: Maximum angle in degrees (default: 20)
    
    Returns:
        List of command strings
    """
    
    # Base values
    base_x, base_y, base_z = 0.4, 0.0, 0.33
    base_rx, base_ry, base_rz = 3.14, 0.0, 3.14
    
    commands = []
    
    # Generate random points
    for _ in range(num_points):
        # Generate random rx and ry angles within limits
        rx_angle_deg = random.uniform(min_angle, max_angle)
        ry_angle_deg = random.uniform(min_angle, max_angle)
        
        # Convert to radians and add to base
        rx_rad = base_rx + degrees_to_radians(rx_angle_deg)
        ry_rad = base_ry + degrees_to_radians(ry_angle_deg)
        
        # Create the move command
        command = f"move {base_x} {base_y} {base_z} {rx_rad:.6f} {ry_rad:.6f} {base_rz}"
        commands.append(command)
    
    return commands

def save_commands_to_file(commands, filename="kuka_path_v3_commands.txt"):
    """Save the generated commands to a text file."""
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filename, 'w') as f:
        for command in commands:
            f.write(f"{command}\n")
    
    print(f"✅ Generated {len(commands)} commands and saved to '{filename}'")
    return filename

def print_path_summary(commands, min_angle, max_angle):
    """Print a summary of the generated path."""
    print("\n📋 Path Generation Summary")
    print("=" * 40)
    print(f"Total commands: {len(commands)}")
    print(f"Position: [0.4, 0.0, 0.33] (fixed)")
    print(f"rz: 3.14 (fixed)")
    print(f"rx range: {min_angle}° to {max_angle}° (random)")
    print(f"ry range: {min_angle}° to {max_angle}° (random)")
    
    # Parse and show range of actual values
    rx_values = []
    ry_values = []
    
    for cmd in commands:
        parts = cmd.split()
        rx_rad = float(parts[4])
        ry_rad = float(parts[5])
        
        # Convert back to degrees relative to base
        rx_deg = math.degrees(rx_rad - 3.14)
        ry_deg = math.degrees(ry_rad - 0.0)
        
        rx_values.append(rx_deg)
        ry_values.append(ry_deg)
    
    print(f"\nActual rx range: {min(rx_values):.2f}° to {max(rx_values):.2f}°")
    print(f"Actual ry range: {min(ry_values):.2f}° to {max(ry_values):.2f}°")
    
    # Show first few commands
    print(f"\nFirst 5 commands:")
    for i, cmd in enumerate(commands[:5]):
        print(f"  {i+1}: {cmd}")
    
    # Show last few commands
    print(f"\nLast 5 commands:")
    for i, cmd in enumerate(commands[-5:]):
        print(f"  {len(commands)-4+i}: {cmd}")

def main():
    """Main function to generate and save the Kuka path."""
    print("🤖 Kuka Robot Path Generator v3")
    print("=" * 40)
    
    # Configuration
    num_points = 50
    min_angle = -20  # degrees
    max_angle = 20   # degrees
    
    try:
        # Generate random path
        print(f"🔄 Generating {num_points} random orientation points...")
        commands = generate_random_path(num_points, min_angle, max_angle)
        
        # Print summary
        print_path_summary(commands, min_angle, max_angle)
        
        # Save to file
        filename = "kuka_path_v3_commands.txt"
        save_commands_to_file(commands, filename)
        
        print(f"\n🎯 Path generation complete!")
        print(f"📁 File saved as: {filename}")
        print(f"📊 Total commands: {len(commands)}")
        
        # Show how to use the file
        print(f"\n💡 Usage:")
        print(f"  1. Copy commands from '{filename}'")
        print(f"  2. Paste them into your Kuka controller")
        print(f"  3. Or modify the script to send commands directly via your test-client")
        print(f"\n⚠️  Note: Commands are in random order (no specific sequencing)")
        
    except Exception as e:
        print(f"❌ Error generating path: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


