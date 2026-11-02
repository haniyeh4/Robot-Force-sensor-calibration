#!/usr/bin/env python3
"""
Kuka Robot Path Generator v2
Generates a smooth path for rx and ry orientation changes from -20 to +20 degrees
in 2-degree increments, keeping position at [0.4, 0, 0.33] and rz at 3.14.
"""

import math
import os

def degrees_to_radians(degrees):
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0

def generate_smooth_path():
    """
    Generate a smooth path that covers all rx, ry combinations from -20 to +20 degrees
    in 2-degree increments, ensuring smooth transitions between commands.
    """
    
    # Base values
    base_x, base_y, base_z = 0.4, 0.0, 0.33
    base_rx, base_ry, base_rz = 3.14, 0.0, 3.14
    
    # Range for rx and ry (-20 to +20 degrees)
    min_angle = -20
    max_angle = 20
    step_size = 2
    
    # Generate all angle combinations
    angles = list(range(min_angle, max_angle + 1, step_size))
    
    # Create a smooth path that covers all combinations
    # Strategy: Forward through rx, then backward through rx for each ry
    # This ensures smooth transitions without large jumps
    
    commands = []
    
    for i, ry_angle in enumerate(angles):
        ry_rad = base_ry + degrees_to_radians(ry_angle)
        
        # Forward through rx angles
        if i % 2 == 0:
            rx_angles = angles
        else:
            # Backward through rx angles for smooth transition
            rx_angles = list(reversed(angles))
        
        for rx_angle in rx_angles:
            rx_rad = base_rx + degrees_to_radians(rx_angle)
            
            # Create the move command
            command = f"move {base_x} {base_y} {base_z} {rx_rad:.6f} {ry_rad:.6f} {base_rz}"
            commands.append(command)
    
    return commands

def save_commands_to_file(commands, filename="kuka_path_v2_commands.txt"):
    """Save the generated commands to a text file."""
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(filename, 'w') as f:
        for i, command in enumerate(commands):
            f.write(f"{command}\n")
    
    print(f"✅ Generated {len(commands)} commands and saved to '{filename}'")
    return filename

def print_path_summary(commands):
    """Print a summary of the generated path."""
    print("\n📋 Path Generation Summary")
    print("=" * 40)
    print(f"Total commands: {len(commands)}")
    print(f"Position: [{0.4}, {0.0}, {0.33}] (fixed)")
    print(f"rz: {3.14} (fixed)")
    print(f"rx range: -20° to +20° in 2° increments")
    print(f"ry range: -20° to +20° in 2° increments")
    
    # Show first few and last few commands
    print(f"\nFirst 3 commands:")
    for i, cmd in enumerate(commands[:3]):
        print(f"  {i+1}: {cmd}")
    
    print(f"\nLast 3 commands:")
    for i, cmd in enumerate(commands[-3:]):
        print(f"  {len(commands)-2+i}: {cmd}")
    
    # Calculate total angular distance and smoothness metrics
    total_rx_change = 0
    total_ry_change = 0
    max_rx_jump = 0
    max_ry_jump = 0
    
    for i in range(1, len(commands)):
        # Parse previous and current commands
        prev_parts = commands[i-1].split()
        curr_parts = commands[i].split()
        
        prev_rx = float(prev_parts[4])
        prev_ry = float(prev_parts[5])
        curr_rx = float(curr_parts[4])
        curr_ry = float(curr_parts[5])
        
        # Calculate angular differences
        rx_diff = abs(curr_rx - prev_rx)
        ry_diff = abs(curr_ry - prev_ry)
        
        total_rx_change += rx_diff
        total_ry_change += ry_diff
        
        # Track maximum jumps
        max_rx_jump = max(max_rx_jump, rx_diff)
        max_ry_jump = max(max_ry_jump, ry_diff)
    
    print(f"\nPath smoothness metrics:")
    print(f"  Average rx change per step: {total_rx_change/(len(commands)-1):.4f} rad")
    print(f"  Average ry change per step: {total_ry_change/(len(commands)-1):.4f} rad")
    print(f"  Maximum rx jump: {max_rx_jump:.4f} rad ({math.degrees(max_rx_jump):.2f}°)")
    print(f"  Maximum ry jump: {max_ry_jump:.4f} rad ({math.degrees(max_ry_jump):.2f}°)")

def main():
    """Main function to generate and save the Kuka path."""
    print("🤖 Kuka Robot Path Generator v2")
    print("=" * 40)
    
    try:
        # Generate the smooth path
        print("🔄 Generating smooth path...")
        commands = generate_smooth_path()
        
        # Print summary
        print_path_summary(commands)
        
        # Save to file
        filename = "kuka_path_v2_commands.txt"
        save_commands_to_file(commands, filename)
        
        print(f"\n🎯 Path generation complete!")
        print(f"📁 File saved as: {filename}")
        print(f"📊 Total commands: {len(commands)}")
        
        # Show how to use the file
        print(f"\n💡 Usage:")
        print(f"  1. Copy commands from '{filename}'")
        print(f"  2. Paste them into your Kuka controller")
        print(f"  3. Or modify the script to send commands directly via your test-client")
        
    except Exception as e:
        print(f"❌ Error generating path: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

