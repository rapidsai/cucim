#!/usr/bin/env python3
"""
Visualize test images created by nvImageCodec testing
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def visualize_test_images():
    """Visualize the original test image and encoded/decoded versions"""
    print("🖼️  Visualizing nvImageCodec Test Images")
    print("=" * 50)
    
    # Image paths
    original_ppm = "/tmp/test_image.ppm"
    encoded_files = [
        "/tmp/test_output.jpg",
        "/tmp/test_output.png", 
        "/tmp/test_output.bmp",
        "/tmp/test_lossless.j2k",
        "/tmp/test_psnr30.j2k",
        "/tmp/test_advanced.j2k"
    ]
    
    # Check if original image exists
    if not os.path.exists(original_ppm):
        print(f"❌ Original test image not found: {original_ppm}")
        print("💡 Please run test_cuslide2_simple.py first to create test images")
        return
    
    # Load original image
    try:
        original_image = load_ppm_image(original_ppm)
        print(f"✅ Loaded original image: {original_image.shape}")
    except Exception as e:
        print(f"❌ Failed to load original image: {e}")
        return
    
    # Try to import nvImageCodec for decoding
    try:
        from nvidia import nvimgcodec
        decoder = nvimgcodec.Decoder()
        print("✅ nvImageCodec decoder available")
    except ImportError:
        print("❌ nvImageCodec not available, cannot decode encoded images")
        decoder = None
    
    # Create visualization
    available_files = [f for f in encoded_files if os.path.exists(f)]
    
    if not available_files:
        print("❌ No encoded test images found")
        print("💡 Please run test_cuslide2_simple.py first to create encoded images")
        return
    
    # Calculate grid size
    total_images = 1 + len(available_files)  # original + encoded versions
    cols = min(3, total_images)
    rows = (total_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if total_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Show original image
    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title('Original Test Image\n(Colorful Pattern)', fontweight='bold')
    axes_flat[0].axis('off')
    
    # Show encoded/decoded images
    for i, filepath in enumerate(available_files, 1):
        if i >= len(axes_flat):
            break
            
        try:
            if decoder:
                # Decode using nvImageCodec
                decoded_image = decoder.read(filepath)
                # Convert to CPU if needed
                if hasattr(decoded_image, 'cpu'):
                    decoded_image = decoded_image.cpu()
                # Convert to numpy array
                image_array = np.asarray(decoded_image)
            else:
                # Fallback: try to load with matplotlib/PIL
                import matplotlib.image as mpimg
                image_array = mpimg.imread(filepath)
                if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                    image_array = (image_array * 255).astype(np.uint8)
            
            axes_flat[i].imshow(image_array)
            
            # Get file info
            file_size = os.path.getsize(filepath)
            file_ext = Path(filepath).suffix.upper()
            
            axes_flat[i].set_title(f'{file_ext} Format\n({file_size:,} bytes)', fontweight='bold')
            axes_flat[i].axis('off')
            
            print(f"✅ Visualized {Path(filepath).name}: {image_array.shape}, {file_size:,} bytes")
            
        except Exception as e:
            axes_flat[i].text(0.5, 0.5, f'Error loading\n{Path(filepath).name}\n{str(e)}', 
                            ha='center', va='center', transform=axes_flat[i].transAxes)
            axes_flat[i].set_title(f'{Path(filepath).suffix.upper()} - Error')
            axes_flat[i].axis('off')
            print(f"⚠️  Failed to load {Path(filepath).name}: {e}")
    
    # Hide unused subplots
    for i in range(total_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('nvImageCodec Test Images: Original vs Encoded/Decoded', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save the visualization
    output_path = "/tmp/nvimagecodec_test_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print analysis
    print(f"\n📊 Image Analysis:")
    print(f"Original image shape: {original_image.shape}")
    print(f"Original image data type: {original_image.dtype}")
    print(f"Original image value range: {original_image.min()} - {original_image.max()}")
    
    # Analyze the pattern
    print(f"\n🎨 Pattern Analysis:")
    print(f"The test image is a 256x256 RGB image with a mathematical pattern:")
    print(f"  Red channel:   (i + j) % 256")
    print(f"  Green channel: (i * 2) % 256") 
    print(f"  Blue channel:  (j * 2) % 256")
    print(f"This creates a colorful gradient pattern that's good for testing compression algorithms.")
    
    if available_files:
        print(f"\n💾 File Size Comparison:")
        original_size = len(original_image.tobytes())
        print(f"  Original (uncompressed): {original_size:,} bytes")
        
        for filepath in available_files:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                compression_ratio = original_size / file_size if file_size > 0 else 0
                print(f"  {Path(filepath).name}: {file_size:,} bytes (compression: {compression_ratio:.1f}x)")

def main():
    """Main function"""
    try:
        visualize_test_images()
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
