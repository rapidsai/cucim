#!/usr/bin/env python3
"""
Visualize the original test image pattern in detail
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

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

def create_original_pattern_visualization():
    """Create a detailed visualization of the original test pattern"""
    print("🎨 Creating Original Test Pattern Visualization")
    print("=" * 55)
    
    # Check if original image exists
    original_ppm = "/tmp/test_image.ppm"
    if not os.path.exists(original_ppm):
        print(f"❌ Original test image not found: {original_ppm}")
        print("💡 Please run test_cuslide2_simple.py first to create the test image")
        return
    
    # Load the original image
    try:
        original_image = load_ppm_image(original_ppm)
        print(f"✅ Loaded original image: {original_image.shape}")
        print(f"   Data type: {original_image.dtype}")
        print(f"   Value range: {original_image.min()} - {original_image.max()}")
    except Exception as e:
        print(f"❌ Failed to load original image: {e}")
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Main image (top left)
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Test Pattern\n(Full 256x256 RGB Image)', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Individual color channels
    # Red channel
    axes[0, 1].imshow(original_image[:, :, 0], cmap='Reds')
    axes[0, 1].set_title('Red Channel\n(i + j) % 256', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # Green channel
    axes[0, 2].imshow(original_image[:, :, 1], cmap='Greens')
    axes[0, 2].set_title('Green Channel\n(i * 2) % 256', fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    
    # Blue channel
    axes[1, 0].imshow(original_image[:, :, 2], cmap='Blues')
    axes[1, 0].set_title('Blue Channel\n(j * 2) % 256', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    # Zoomed section (center 64x64 pixels)
    center_y, center_x = 128, 128
    zoom_size = 32
    zoomed_section = original_image[
        center_y-zoom_size:center_y+zoom_size,
        center_x-zoom_size:center_x+zoom_size
    ]
    axes[1, 1].imshow(zoomed_section)
    axes[1, 1].set_title('Zoomed Center Section\n(64x64 pixels)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    # Pattern analysis
    axes[1, 2].axis('off')
    
    # Create pattern analysis text
    analysis_text = """Pattern Analysis:
    
🔴 Red Channel: (i + j) % 256
   • Creates diagonal gradient
   • Values: 0-255 repeating
   • Pattern: Diagonal stripes
    
🟢 Green Channel: (i * 2) % 256  
   • Creates horizontal bands
   • Values: 0-254 (even numbers)
   • Pattern: Horizontal stripes
    
🔵 Blue Channel: (j * 2) % 256
   • Creates vertical bands  
   • Values: 0-254 (even numbers)
   • Pattern: Vertical stripes
    
📊 Combined Result:
   • Complex colorful pattern
   • Tests compression algorithms
   • Reveals encoding artifacts
   • Good for quality assessment"""
    
    axes[1, 2].text(0.05, 0.95, analysis_text, transform=axes[1, 2].transAxes,
                   fontsize=10, fontfamily='monospace', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 2].set_title('Mathematical Pattern Details', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.suptitle('Original Test Image - Mathematical Pattern Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save the visualization
    output_path = "/tmp/original_pattern_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Original pattern visualization saved: {output_path}")
    
    # Create a simple single image view
    create_simple_original_view(original_image)
    
    # Print detailed analysis
    print_pattern_analysis(original_image)

def create_simple_original_view(image):
    """Create a simple, clean view of just the original image"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.imshow(image)
    ax.set_title('Original Test Pattern\n256x256 RGB Mathematical Gradient', 
                fontweight='bold', fontsize=14)
    ax.axis('off')
    
    # Add some information as text
    info_text = f"""Image Properties:
Size: {image.shape[0]}×{image.shape[1]} pixels
Channels: {image.shape[2]} (RGB)
Data Type: {image.dtype}
Value Range: {image.min()}-{image.max()}

Pattern Formula:
Red = (row + col) % 256
Green = (row × 2) % 256  
Blue = (col × 2) % 256"""
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    # Save simple view
    simple_output = "/tmp/original_image_simple.png"
    plt.savefig(simple_output, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Simple original image view saved: {simple_output}")

def print_pattern_analysis(image):
    """Print detailed analysis of the pattern"""
    print(f"\n📊 Detailed Pattern Analysis:")
    print("=" * 40)
    
    print(f"Image Dimensions: {image.shape}")
    print(f"Total Pixels: {image.shape[0] * image.shape[1]:,}")
    print(f"Total Data Size: {image.nbytes:,} bytes")
    
    # Analyze each channel
    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        channel = image[:, :, i]
        print(f"\n{channel_name} Channel Analysis:")
        print(f"  Min value: {channel.min()}")
        print(f"  Max value: {channel.max()}")
        print(f"  Mean value: {channel.mean():.1f}")
        print(f"  Unique values: {len(np.unique(channel))}")
    
    # Show pattern characteristics
    print(f"\n🎨 Pattern Characteristics:")
    print(f"• Red Channel: Diagonal gradient pattern")
    print(f"  Formula: (row + column) % 256")
    print(f"  Creates diagonal stripes from top-left to bottom-right")
    
    print(f"• Green Channel: Horizontal stripe pattern")
    print(f"  Formula: (row × 2) % 256")
    print(f"  Creates horizontal bands, only even values (0,2,4...254)")
    
    print(f"• Blue Channel: Vertical stripe pattern")
    print(f"  Formula: (column × 2) % 256")
    print(f"  Creates vertical bands, only even values (0,2,4...254)")
    
    print(f"\n🔍 Why This Pattern is Good for Testing:")
    print(f"• Contains all possible color combinations")
    print(f"• Has both smooth gradients and sharp transitions")
    print(f"• Tests compression algorithm effectiveness")
    print(f"• Reveals compression artifacts clearly")
    print(f"• Mathematical precision allows quality measurement")

def main():
    """Main function"""
    try:
        create_original_pattern_visualization()
        
        print(f"\n📁 Generated Files:")
        files = [
            "/tmp/original_pattern_analysis.png",
            "/tmp/original_image_simple.png"
        ]
        
        for filepath in files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   {filepath}: {size:,} bytes")
        
        print(f"\n👀 To view the original image:")
        print(f"   firefox /tmp/original_image_simple.png")
        print(f"   eog /tmp/original_pattern_analysis.png")
        
        print(f"\n🎯 This is the base image that nvImageCodec processes!")
        print(f"   All the compression tests start with this mathematical pattern.")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
