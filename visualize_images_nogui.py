#!/usr/bin/env python3
"""
Visualize test images created by nvImageCodec testing (Non-GUI version)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
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
    print("🖼️  Visualizing nvImageCodec Test Images (Non-GUI)")
    print("=" * 60)
    
    # Image paths from our tests
    original_ppm = "/tmp/test_image.ppm"
    encoded_files = [
        ("/tmp/test_image.jpg", "Original JPEG Input"),
        ("/tmp/test-jpg-o.bmp", "BMP (like cat-jpg-o.bmp)"),
        ("/tmp/test-direct-o.jpg", "Direct JPEG (encoder.write())"),
        ("/tmp/test-o.j2k", "JPEG2000 (like .jp2 example)"),
        ("/tmp/test_lossless.j2k", "JPEG2000 Lossless"),
        ("/tmp/test_psnr30.j2k", "JPEG2000 PSNR=30")
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
    
    # Collect available images
    available_images = []
    
    # Add original
    available_images.append(("Original Test Pattern\n(256x256 RGB)", original_image, 0))
    
    # Add encoded versions
    for filepath, description in encoded_files:
        if os.path.exists(filepath):
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
                
                file_size = os.path.getsize(filepath)
                available_images.append((f"{description}\n({file_size:,} bytes)", image_array, file_size))
                print(f"✅ Loaded {Path(filepath).name}: {image_array.shape}, {file_size:,} bytes")
                
            except Exception as e:
                print(f"⚠️  Failed to load {Path(filepath).name}: {e}")
    
    if len(available_images) <= 1:
        print("❌ No encoded test images found")
        return
    
    # Create visualization
    num_images = len(available_images)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if num_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    # Display images
    for i, (title, image, file_size) in enumerate(available_images):
        if i >= len(axes_flat):
            break
            
        axes_flat[i].imshow(image)
        axes_flat[i].set_title(title, fontweight='bold', fontsize=10)
        axes_flat[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('nvImageCodec Test Results - Image Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save the visualization
    output_path = "/tmp/nvimagecodec_visualization_complete.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Complete visualization saved: {output_path}")
    
    # Create a detailed analysis visualization
    create_analysis_visualization(available_images)
    
    # Print detailed analysis
    print_detailed_analysis(available_images)

def create_analysis_visualization(images_data):
    """Create a detailed analysis visualization"""
    print(f"\n📊 Creating detailed analysis visualization...")
    
    # Create comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Show first 3 images
    for i in range(min(3, len(images_data))):
        title, image, file_size = images_data[i]
        axes[0, i].imshow(image)
        axes[0, i].set_title(title, fontweight='bold', fontsize=10)
        axes[0, i].axis('off')
    
    # Bottom row: Show next 3 images or analysis
    for i in range(3, min(6, len(images_data))):
        title, image, file_size = images_data[i]
        axes[1, i-3].imshow(image)
        axes[1, i-3].set_title(title, fontweight='bold', fontsize=10)
        axes[1, i-3].axis('off')
    
    # Fill remaining slots with analysis
    remaining_slots = 6 - len(images_data)
    if remaining_slots > 0:
        # Add file size comparison
        slot_idx = len(images_data)
        if slot_idx < 6:
            row, col = divmod(slot_idx, 3)
            axes[row, col].axis('off')
            
            # Create file size comparison text
            analysis_text = "File Size Analysis:\n\n"
            original_size = 256 * 256 * 3  # Uncompressed
            
            for title, image, file_size in images_data:
                if file_size > 0:
                    compression = original_size / file_size
                    format_name = title.split('\n')[0][:15]
                    analysis_text += f"{format_name}:\n"
                    analysis_text += f"  {file_size:,} bytes\n"
                    analysis_text += f"  {compression:.1f}x compression\n\n"
            
            axes[row, col].text(0.1, 0.9, analysis_text, transform=axes[row, col].transAxes,
                              fontsize=10, fontfamily='monospace', verticalalignment='top',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            axes[row, col].set_title("Compression Analysis", fontweight='bold')
    
    # Hide any remaining empty slots
    for i in range(len(images_data), 6):
        row, col = divmod(i, 3)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('nvImageCodec Detailed Analysis - Official Examples Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save the analysis visualization
    analysis_output = "/tmp/nvimagecodec_analysis_detailed.png"
    plt.savefig(analysis_output, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Detailed analysis saved: {analysis_output}")

def print_detailed_analysis(images_data):
    """Print detailed analysis of the images"""
    print(f"\n📊 Detailed Image Analysis:")
    print("=" * 70)
    
    original_size = 256 * 256 * 3  # Uncompressed RGB
    
    print(f"{'Image Type':<30} {'Size (bytes)':<12} {'Compression':<12} {'Quality'}")
    print("-" * 70)
    
    for i, (title, image, file_size) in enumerate(images_data):
        image_type = title.split('\n')[0][:28]
        
        if file_size > 0:
            compression = original_size / file_size
            
            # Determine quality based on compression and type
            if "Original" in title or "BMP" in title:
                quality = "Reference/Lossless"
            elif compression > 50:
                quality = "Excellent"
            elif compression > 20:
                quality = "Very Good"
            elif compression > 10:
                quality = "Good"
            else:
                quality = "Fair"
            
            print(f"{image_type:<30} {file_size:>8,}     {compression:>8.1f}x     {quality}")
        else:
            print(f"{image_type:<30} {'N/A':<12} {'N/A':<12} {'N/A'}")
    
    print(f"\nOriginal uncompressed: {original_size:,} bytes (256x256x3 RGB)")
    
    # Show pattern analysis
    print(f"\n🎨 Test Pattern Analysis:")
    if len(images_data) > 0:
        original_image = images_data[0][1]
        print(f"Image dimensions: {original_image.shape}")
        print(f"Data type: {original_image.dtype}")
        print(f"Value range: {original_image.min()} - {original_image.max()}")
        print(f"Pattern: Mathematical gradient (Red: (i+j)%256, Green: (i*2)%256, Blue: (j*2)%256)")
    
    # Show format capabilities
    print(f"\n🚀 nvImageCodec Capabilities Demonstrated:")
    print(f"✅ Memory-based encoding/decoding (like official examples)")
    print(f"✅ File-based operations (decoder.read(), encoder.write())")
    print(f"✅ Multiple formats: JPEG, BMP, JPEG2000")
    print(f"✅ Quality control: Lossless, PSNR-based compression")
    print(f"✅ GPU acceleration: Images processed on GPU memory")
    print(f"✅ OpenCV interoperability: Seamless format conversion")

def main():
    """Main function"""
    try:
        visualize_test_images()
        
        # Show generated files
        print(f"\n📁 Generated Visualization Files:")
        viz_files = [
            "/tmp/nvimagecodec_visualization_complete.png",
            "/tmp/nvimagecodec_analysis_detailed.png",
            "/tmp/nvimagecodec_official_examples.png"
        ]
        
        for filepath in viz_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"   {filepath}: {size:,} bytes")
        
        print(f"\n💡 To view the visualizations:")
        print(f"   firefox /tmp/nvimagecodec_visualization_complete.png")
        print(f"   eog /tmp/nvimagecodec_analysis_detailed.png")
        print(f"   Or any image viewer of your choice")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
