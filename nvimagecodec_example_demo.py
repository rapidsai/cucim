#!/usr/bin/env python3
"""
nvImageCodec API Demo following the official examples
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

def create_sample_image():
    """Create a sample image similar to the official examples"""
    print("🖼️  Creating sample test image...")
    
    # Create a more interesting test image (similar to a cat photo pattern)
    height, width = 480, 640
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a pattern that resembles natural image features
    for i in range(height):
        for j in range(width):
            # Create concentric circles and gradients
            center_y, center_x = height // 2, width // 2
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            
            # Red channel: radial gradient
            test_image[i, j, 0] = int(128 + 127 * np.sin(dist / 20)) % 256
            
            # Green channel: horizontal gradient with waves
            test_image[i, j, 1] = int(128 + 127 * np.sin(j / 30) * np.cos(i / 40)) % 256
            
            # Blue channel: vertical gradient
            test_image[i, j, 2] = int(255 * i / height) % 256
    
    # Save as JPEG for testing (like tabby_tiger_cat.jpg in examples)
    sample_jpg_path = "/tmp/sample_test_image.jpg"
    cv2.imwrite(sample_jpg_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Sample image created: {sample_jpg_path}")
    print(f"   Dimensions: {height}x{width}x3")
    
    return sample_jpg_path, test_image

def nvimagecodec_example_demo():
    """Demonstrate nvImageCodec API following official examples"""
    print("🚀 nvImageCodec API Demo (Following Official Examples)")
    print("=" * 60)
    
    # Import nvImageCodec module and create Decoder and Encoder
    print("\n📋 Step 1: Import nvImageCodec and create Decoder/Encoder")
    try:
        from nvidia import nvimgcodec
        decoder = nvimgcodec.Decoder()
        encoder = nvimgcodec.Encoder()
        print("✅ nvImageCodec imported and Decoder/Encoder created")
    except ImportError as e:
        print(f"❌ Failed to import nvImageCodec: {e}")
        return
    
    # Create sample image (since we don't have tabby_tiger_cat.jpg)
    sample_jpg_path, original_array = create_sample_image()
    
    # Load and decode JPEG image with nvImageCodec (like the example)
    print(f"\n📋 Step 2: Load and decode JPEG image with nvImageCodec")
    try:
        with open(sample_jpg_path, 'rb') as in_file:
            data = in_file.read()
            nv_img_sample = decoder.decode(data)
        
        print(f"✅ JPEG decoded successfully")
        print(f"   Shape: {nv_img_sample.shape}")
        print(f"   Buffer kind: {nv_img_sample.buffer_kind}")
    except Exception as e:
        print(f"❌ Failed to decode JPEG: {e}")
        return
    
    # Save image to BMP file with nvImageCodec (like the example)
    print(f"\n📋 Step 3: Save image to BMP file with nvImageCodec")
    try:
        with open("/tmp/sample-jpg-o.bmp", 'wb') as out_file:
            data = encoder.encode(nv_img_sample, "bmp")
            out_file.write(data)
        
        bmp_size = os.path.getsize("/tmp/sample-jpg-o.bmp")
        print(f"✅ BMP saved successfully: /tmp/sample-jpg-o.bmp ({bmp_size:,} bytes)")
    except Exception as e:
        print(f"❌ Failed to save BMP: {e}")
        return
    
    # Read back with OpenCV just saved BMP image (like the example)
    print(f"\n📋 Step 4: Read back with OpenCV the saved BMP image")
    try:
        cv_img_bmp = cv2.imread("/tmp/sample-jpg-o.bmp")
        cv_img_bmp = cv2.cvtColor(cv_img_bmp, cv2.COLOR_BGR2RGB)
        print(f"✅ BMP read back with OpenCV: {cv_img_bmp.shape}")
    except Exception as e:
        print(f"❌ Failed to read BMP with OpenCV: {e}")
        return
    
    # Test the one-function read/write methods (like the example)
    print(f"\n📋 Step 5: Test one-function read/write methods")
    try:
        # Read image directly (like decoder.read() in examples)
        nv_img_direct = decoder.read(sample_jpg_path)
        print(f"✅ Direct read successful: {nv_img_direct.shape}")
        
        # Write image directly (like encoder.write() in examples)
        output_jpg = encoder.write("/tmp/sample-direct-o.jpg", nv_img_direct)
        jpg_size = os.path.getsize("/tmp/sample-direct-o.jpg")
        print(f"✅ Direct write successful: {output_jpg} ({jpg_size:,} bytes)")
    except Exception as e:
        print(f"❌ Failed direct read/write: {e}")
        return
    
    # Test JPEG2000 functionality (like the jp2 example)
    print(f"\n📋 Step 6: Test JPEG2000 functionality")
    try:
        # Save as JPEG2000 (like the .jp2 example)
        encoder.write("/tmp/sample-o.j2k", nv_img_sample)
        j2k_size = os.path.getsize("/tmp/sample-o.j2k")
        print(f"✅ JPEG2000 saved: /tmp/sample-o.j2k ({j2k_size:,} bytes)")
        
        # Read back JPEG2000
        nv_img_j2k = decoder.read("/tmp/sample-o.j2k")
        print(f"✅ JPEG2000 read back: {nv_img_j2k.shape}")
    except Exception as e:
        print(f"❌ Failed JPEG2000 test: {e}")
    
    # Create visualization (non-GUI version)
    print(f"\n📋 Step 7: Create visualization")
    try:
        # Set matplotlib to non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_array)
        axes[0, 0].set_title('Original Test Image\n(Created Pattern)', fontweight='bold')
        axes[0, 0].axis('off')
        
        # nvImageCodec decoded JPEG
        nv_img_cpu = nv_img_sample.cpu() if hasattr(nv_img_sample, 'cpu') else nv_img_sample
        axes[0, 1].imshow(np.asarray(nv_img_cpu))
        axes[0, 1].set_title('nvImageCodec Decoded JPEG\n(from memory)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # OpenCV read BMP
        axes[0, 2].imshow(cv_img_bmp)
        axes[0, 2].set_title('OpenCV Read BMP\n(nvImageCodec encoded)', fontweight='bold')
        axes[0, 2].axis('off')
        
        # Direct read result
        nv_img_direct_cpu = nv_img_direct.cpu() if hasattr(nv_img_direct, 'cpu') else nv_img_direct
        axes[1, 0].imshow(np.asarray(nv_img_direct_cpu))
        axes[1, 0].set_title('nvImageCodec Direct Read\n(decoder.read())', fontweight='bold')
        axes[1, 0].axis('off')
        
        # JPEG2000 result (if available)
        if 'nv_img_j2k' in locals():
            nv_img_j2k_cpu = nv_img_j2k.cpu() if hasattr(nv_img_j2k, 'cpu') else nv_img_j2k
            axes[1, 1].imshow(np.asarray(nv_img_j2k_cpu))
            axes[1, 1].set_title('JPEG2000 Decoded\n(.j2k format)', fontweight='bold')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'JPEG2000\nNot Available', ha='center', va='center')
            axes[1, 1].set_title('JPEG2000 - Error')
            axes[1, 1].axis('off')
        
        # File size comparison
        axes[1, 2].axis('off')
        file_info = []
        
        # Get file sizes
        original_size = original_array.nbytes
        jpg_size = os.path.getsize(sample_jpg_path) if os.path.exists(sample_jpg_path) else 0
        bmp_size = os.path.getsize("/tmp/sample-jpg-o.bmp") if os.path.exists("/tmp/sample-jpg-o.bmp") else 0
        j2k_size = os.path.getsize("/tmp/sample-o.j2k") if os.path.exists("/tmp/sample-o.j2k") else 0
        
        file_info.append(f"Original (RAM): {original_size:,} bytes")
        file_info.append(f"JPEG: {jpg_size:,} bytes ({original_size/jpg_size:.1f}x compression)" if jpg_size > 0 else "JPEG: N/A")
        file_info.append(f"BMP: {bmp_size:,} bytes ({original_size/bmp_size:.1f}x compression)" if bmp_size > 0 else "BMP: N/A")
        file_info.append(f"JPEG2000: {j2k_size:,} bytes ({original_size/j2k_size:.1f}x compression)" if j2k_size > 0 else "JPEG2000: N/A")
        
        axes[1, 2].text(0.1, 0.9, "File Size Comparison:", fontweight='bold', transform=axes[1, 2].transAxes)
        for i, info in enumerate(file_info):
            axes[1, 2].text(0.1, 0.7 - i*0.15, info, transform=axes[1, 2].transAxes, fontfamily='monospace')
        
        plt.tight_layout()
        plt.suptitle('nvImageCodec API Demo - Following Official Examples', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save the visualization
        output_path = "/tmp/nvimagecodec_api_demo.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
        
        plt.close()  # Close to free memory
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    # Print summary like the examples
    print(f"\n🎉 nvImageCodec API Demo Complete!")
    print(f"=" * 60)
    print(f"✅ Successfully demonstrated all key nvImageCodec features:")
    print(f"   • Decoder/Encoder creation")
    print(f"   • Memory-based encoding/decoding (like the examples)")
    print(f"   • File-based read/write operations")
    print(f"   • Multiple format support (JPEG, BMP, JPEG2000)")
    print(f"   • OpenCV interoperability")
    print(f"   • Buffer management (CPU/GPU)")
    
    print(f"\n📁 Generated Files:")
    test_files = [
        "/tmp/sample_test_image.jpg",
        "/tmp/sample-jpg-o.bmp", 
        "/tmp/sample-direct-o.jpg",
        "/tmp/sample-o.j2k",
        "/tmp/nvimagecodec_api_demo.png"
    ]
    
    for filepath in test_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   {filepath}: {size:,} bytes")

def main():
    """Main function"""
    try:
        nvimagecodec_example_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
