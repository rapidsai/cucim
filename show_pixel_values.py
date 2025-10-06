#!/usr/bin/env python3
"""
Show actual pixel values of the original test image to demonstrate the mathematical pattern
"""

import os
import numpy as np

def load_ppm_image(filepath):
    """Load a PPM P6 format image"""
    with open(filepath, 'rb') as f:
        # Read header
        magic = f.readline().strip()
        if magic != b'P6':
            raise ValueError("Not a P6 PPM file")
        
        # Skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        
        # Parse dimensions
        dimensions = line.strip().split()
        width, height = int(dimensions[0]), int(dimensions[1])
        
        # Parse max value
        max_val = int(f.readline().strip())
        
        # Read image data
        image_data = f.read()
        image = np.frombuffer(image_data, dtype=np.uint8)
        image = image.reshape((height, width, 3))
        
        return image

def show_pixel_values():
    """Show actual pixel values to demonstrate the mathematical pattern"""
    print("🔍 Original Image Pixel Values Analysis")
    print("=" * 50)
    
    # Load the original image
    original_ppm = "/tmp/test_image.ppm"
    if not os.path.exists(original_ppm):
        print(f"❌ Original test image not found: {original_ppm}")
        print("💡 Please run test_cuslide2_simple.py first")
        return
    
    try:
        image = load_ppm_image(original_ppm)
        print(f"✅ Loaded image: {image.shape}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # Show a small section of pixel values (top-left 8x8)
    print(f"\n📊 Top-Left 8x8 Pixel Values:")
    print("=" * 40)
    
    section = image[0:8, 0:8]
    
    print(f"Position format: [Row, Col] = (Red, Green, Blue)")
    print(f"Mathematical formulas:")
    print(f"  Red   = (row + col) % 256")
    print(f"  Green = (row * 2) % 256")
    print(f"  Blue  = (col * 2) % 256")
    print()
    
    for row in range(8):
        for col in range(8):
            r, g, b = section[row, col]
            
            # Calculate expected values
            expected_r = (row + col) % 256
            expected_g = (row * 2) % 256
            expected_b = (col * 2) % 256
            
            print(f"[{row},{col}] = ({r:3d},{g:3d},{b:3d})", end="  ")
            
            # Verify the pattern
            if r == expected_r and g == expected_g and b == expected_b:
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status}", end="   ")
            
            if col == 7:  # End of row
                print()
    
    # Show pattern verification for a larger section
    print(f"\n🧮 Pattern Verification (16x16 section):")
    print("=" * 45)
    
    section_16 = image[0:16, 0:16]
    correct_pixels = 0
    total_pixels = 16 * 16
    
    for row in range(16):
        for col in range(16):
            r, g, b = section_16[row, col]
            
            expected_r = (row + col) % 256
            expected_g = (row * 2) % 256
            expected_b = (col * 2) % 256
            
            if r == expected_r and g == expected_g and b == expected_b:
                correct_pixels += 1
    
    print(f"Correct pixels: {correct_pixels}/{total_pixels}")
    print(f"Pattern accuracy: {100 * correct_pixels / total_pixels:.1f}%")
    
    # Show some interesting positions
    print(f"\n🎯 Interesting Pattern Positions:")
    print("=" * 35)
    
    interesting_positions = [
        (0, 0, "Top-left corner"),
        (0, 255, "Top-right corner"),
        (255, 0, "Bottom-left corner"),
        (255, 255, "Bottom-right corner"),
        (128, 128, "Center pixel"),
        (100, 50, "Random position"),
        (200, 150, "Another position")
    ]
    
    for row, col, description in interesting_positions:
        if row < image.shape[0] and col < image.shape[1]:
            r, g, b = image[row, col]
            expected_r = (row + col) % 256
            expected_g = (row * 2) % 256
            expected_b = (col * 2) % 256
            
            print(f"{description}:")
            print(f"  Position: [{row:3d},{col:3d}]")
            print(f"  Actual:   RGB({r:3d},{g:3d},{b:3d})")
            print(f"  Expected: RGB({expected_r:3d},{expected_g:3d},{expected_b:3d})")
            print(f"  Match: {'✅ Yes' if (r,g,b) == (expected_r,expected_g,expected_b) else '❌ No'}")
            print()
    
    # Show color distribution
    print(f"🌈 Color Channel Distributions:")
    print("=" * 32)
    
    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        channel = image[:, :, i]
        unique_values = np.unique(channel)
        
        print(f"{channel_name} Channel:")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Range: {unique_values.min()} to {unique_values.max()}")
        print(f"  First 10 values: {unique_values[:10].tolist()}")
        print(f"  Last 10 values: {unique_values[-10:].tolist()}")
        print()
    
    # Show why this pattern is good for testing
    print(f"💡 Why This Pattern is Perfect for Testing:")
    print("=" * 45)
    print(f"✅ Predictable: Every pixel value can be calculated")
    print(f"✅ Comprehensive: Uses full 0-255 range in red channel")
    print(f"✅ Varied: Contains gradients, stripes, and transitions")
    print(f"✅ Detectable: Compression artifacts are easily visible")
    print(f"✅ Mathematical: Precise quality measurements possible")
    print(f"✅ Colorful: Tests all RGB combinations")
    
    print(f"\n🎨 Visual Pattern Description:")
    print(f"• Red creates diagonal stripes (top-left to bottom-right)")
    print(f"• Green creates horizontal bands (128 different shades)")
    print(f"• Blue creates vertical bands (128 different shades)")
    print(f"• Combined: Creates a complex, colorful test pattern")

if __name__ == "__main__":
    show_pixel_values()
