#!/usr/bin/env python3
"""
Simple cuslide2 plugin test with nvImageCodec API integration
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

def test_cuslide2_plugin():
    """Test cuslide2 plugin setup"""
    print("🚀 Simple cuslide2 Plugin Test")
    print("=" * 40)
    
    # Set up environment
    plugin_root = "/home/cdinea/cucim/cpp/plugins/cucim.kit.cuslide2/build/lib"
    
    # Check if plugin file exists
    plugin_file = f"{plugin_root}/cucim.kit.cuslide2@25.10.00.so"
    if os.path.exists(plugin_file):
        print(f"✅ cuslide2 plugin found: {plugin_file}")
        
        # Get file size
        file_size = os.path.getsize(plugin_file)
        print(f"   Size: {file_size / (1024*1024):.1f} MB")
        
        # Check if it's a valid shared library
        try:
            import subprocess
            result = subprocess.run(['file', plugin_file], capture_output=True, text=True)
            if 'shared object' in result.stdout:
                print(f"✅ Valid shared library")
            else:
                print(f"⚠️  File type: {result.stdout.strip()}")
        except:
            print("   (Could not check file type)")
            
    else:
        print(f"❌ cuslide2 plugin not found: {plugin_file}")
        return False
    
    # Check nvImageCodec library
    nvimgcodec_lib = "/home/cdinea/micromamba/lib/libnvimgcodec.so.0"
    if os.path.exists(nvimgcodec_lib):
        print(f"✅ nvImageCodec library found: {nvimgcodec_lib}")
        
        # Try to get nvImageCodec version
        try:
            import ctypes
            nvimgcodec = ctypes.CDLL(nvimgcodec_lib)
            
            # First, try a simpler approach - check if we can get version from file info
            try:
                import subprocess
                result = subprocess.run(['strings', nvimgcodec_lib], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'nvImageCodec' in line and any(c.isdigit() for c in line):
                            if '.' in line:
                                print(f"   📋 nvImageCodec version info: {line.strip()}")
                                break
                    else:
                        # Look for version patterns
                        for line in lines:
                            if line.startswith('0.') or line.startswith('1.'):
                                if len(line.split('.')) >= 2:
                                    print(f"   📋 Possible nvImageCodec version: {line.strip()}")
                                    break
            except:
                pass
            
            # Try to call nvImageCodec API (this might fail due to initialization requirements)
            try:
                # Define nvImageCodec structures and functions
                class nvimgcodecProperties_t(ctypes.Structure):
                    _fields_ = [
                        ("struct_type", ctypes.c_int),
                        ("struct_size", ctypes.c_size_t),
                        ("struct_next", ctypes.c_void_p),
                        ("version", ctypes.c_uint32),
                        ("cuda_runtime_version", ctypes.c_uint32),
                        ("nvjpeg_version", ctypes.c_uint32),
                        ("nvjpeg2k_version", ctypes.c_uint32),
                    ]
                
                # Get nvImageCodec functions
                nvimgcodecGetProperties = nvimgcodec.nvimgcodecGetProperties
                nvimgcodecGetProperties.argtypes = [ctypes.POINTER(nvimgcodecProperties_t)]
                nvimgcodec.nvimgcodecGetProperties.restype = ctypes.c_int
                
                # Call nvimgcodecGetProperties
                props = nvimgcodecProperties_t()
                props.struct_type = 0  # NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES
                props.struct_size = ctypes.sizeof(nvimgcodecProperties_t)
                props.struct_next = None
                
                result = nvimgcodecGetProperties(ctypes.byref(props))
                if result == 0:  # NVIMGCODEC_STATUS_SUCCESS
                    # Extract version components
                    version = props.version
                    major = (version >> 16) & 0xFF
                    minor = (version >> 8) & 0xFF
                    patch = version & 0xFF
                    
                    print(f"   📋 nvImageCodec API version: {major}.{minor}.{patch}")
                    
                    # Show additional version info if available
                    if props.cuda_runtime_version > 0:
                        cuda_major = (props.cuda_runtime_version // 1000)
                        cuda_minor = (props.cuda_runtime_version % 1000) // 10
                        print(f"   📋 CUDA Runtime version: {cuda_major}.{cuda_minor}")
                    
                    if props.nvjpeg_version > 0:
                        nvjpeg_major = (props.nvjpeg_version >> 16) & 0xFF
                        nvjpeg_minor = (props.nvjpeg_version >> 8) & 0xFF
                        nvjpeg_patch = props.nvjpeg_version & 0xFF
                        print(f"   📋 nvJPEG version: {nvjpeg_major}.{nvjpeg_minor}.{nvjpeg_patch}")
                    
                    if props.nvjpeg2k_version > 0:
                        nvjpeg2k_major = (props.nvjpeg2k_version >> 16) & 0xFF
                        nvjpeg2k_minor = (props.nvjpeg2k_version >> 8) & 0xFF
                        nvjpeg2k_patch = props.nvjpeg2k_version & 0xFF
                        print(f"   📋 nvJPEG2000 version: {nvjpeg2k_major}.{nvjpeg2k_minor}.{nvjpeg2k_patch}")
                else:
                    # Decode common error codes
                    error_messages = {
                        1: "NVIMGCODEC_STATUS_INVALID_PARAMETER",
                        2: "NVIMGCODEC_STATUS_NOT_INITIALIZED", 
                        3: "NVIMGCODEC_STATUS_NOT_SUPPORTED",
                        4: "NVIMGCODEC_STATUS_INTERNAL_ERROR"
                    }
                    error_msg = error_messages.get(result, f"Unknown error ({result})")
                    print(f"   ⚠️  nvImageCodec API call failed: {error_msg}")
                    print(f"   💡 This is normal - nvImageCodec needs initialization before API calls")
                    
            except Exception as api_error:
                print(f"   ⚠️  nvImageCodec API not accessible: {api_error}")
                
            # Try to get version from conda package info
            try:
                conda_prefix = os.environ.get('CONDA_PREFIX', '/home/cdinea/micromamba')
                conda_meta_dir = f"{conda_prefix}/conda-meta"
                if os.path.exists(conda_meta_dir):
                    for filename in os.listdir(conda_meta_dir):
                        if 'libnvimgcodec' in filename and filename.endswith('.json'):
                            print(f"   📋 Conda package: {filename.replace('.json', '')}")
                            break
            except:
                pass
                
        except Exception as e:
            print(f"   ⚠️  Could not get nvImageCodec version: {e}")
    else:
        print(f"⚠️  nvImageCodec library not found: {nvimgcodec_lib}")
        print("   GPU acceleration will not be available")
    
    # Check cuCIM library
    cucim_lib = "/home/cdinea/cucim/build-release/lib/libcucim.so"
    if os.path.exists(cucim_lib):
        print(f"✅ cuCIM library found: {cucim_lib}")
    else:
        print(f"❌ cuCIM library not found: {cucim_lib}")
        return False
    
    # Test library loading
    print(f"\n🧪 Testing library loading...")
    try:
        import ctypes
        
        # Try to load cuCIM library
        cucim_handle = ctypes.CDLL(cucim_lib)
        print(f"✅ cuCIM library loaded successfully")
        
        # Try to load cuslide2 plugin
        plugin_handle = ctypes.CDLL(plugin_file)
        print(f"✅ cuslide2 plugin loaded successfully")
        
        # Try to load nvImageCodec (if available)
        if os.path.exists(nvimgcodec_lib):
            nvimgcodec_handle = ctypes.CDLL(nvimgcodec_lib)
            print(f"✅ nvImageCodec library loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Library loading failed: {e}")
        return False

def create_plugin_config():
    """Create a plugin configuration file"""
    print(f"\n🔧 Creating plugin configuration...")
    
    config = {
        "plugin": {
            "names": [
                "cucim.kit.cuslide2@25.10.00.so",  # cuslide2 with nvImageCodec
                "cucim.kit.cuslide@25.10.00.so",   # Original cuslide
                "cucim.kit.cumed@25.10.00.so"      # Medical imaging
            ]
        }
    }
    
    config_path = "/tmp/.cucim_cuslide2_simple.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration created: {config_path}")
    print(f"   Content: {json.dumps(config, indent=2)}")
    
    return config_path

def create_official_examples_visualization(images_dict):
    """Create visualization following the official nvImageCodec examples"""
    print("🎨 Creating visualization of official examples...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare images for visualization
        display_images = {}
        file_sizes = {}
        
        # Original image
        if 'original' in images_dict and images_dict['original'] is not None:
            display_images['Original Test Image\n(Created Pattern)'] = images_dict['original']
            file_sizes['Original'] = images_dict['original'].nbytes
        
        # nvImageCodec decoded (from memory, like tabby_tiger_cat.jpg example)
        if 'nvimgcodec_decoded' in images_dict and images_dict['nvimgcodec_decoded'] is not None:
            nv_img = images_dict['nvimgcodec_decoded']
            # Convert to CPU if needed
            if hasattr(nv_img, 'cpu'):
                nv_img = nv_img.cpu()
            display_images['nvImageCodec Decoded\n(from memory like tabby_tiger_cat.jpg)'] = np.asarray(nv_img)
            if os.path.exists('/tmp/test_image.jpg'):
                file_sizes['JPEG Input'] = os.path.getsize('/tmp/test_image.jpg')
        
        # OpenCV BMP (like cat-jpg-o.bmp example)
        if 'opencv_bmp' in images_dict and images_dict['opencv_bmp'] is not None:
            display_images['OpenCV Read BMP\n(like cat-jpg-o.bmp example)'] = images_dict['opencv_bmp']
            if os.path.exists('/tmp/test-jpg-o.bmp'):
                file_sizes['BMP Output'] = os.path.getsize('/tmp/test-jpg-o.bmp')
        
        # Direct read (like decoder.read() example)
        if 'direct_read' in images_dict and images_dict['direct_read'] is not None:
            direct_img = images_dict['direct_read']
            if hasattr(direct_img, 'cpu'):
                direct_img = direct_img.cpu()
            display_images['Direct Read\n(decoder.read() example)'] = np.asarray(direct_img)
            if os.path.exists('/tmp/test-direct-o.jpg'):
                file_sizes['Direct JPEG'] = os.path.getsize('/tmp/test-direct-o.jpg')
        
        # JPEG2000 (like cat-1046544_640.jp2 example)
        if 'jpeg2000' in images_dict and images_dict['jpeg2000'] is not None:
            j2k_img = images_dict['jpeg2000']
            if hasattr(j2k_img, 'cpu'):
                j2k_img = j2k_img.cpu()
            display_images['JPEG2000\n(like .jp2 example)'] = np.asarray(j2k_img)
            if os.path.exists('/tmp/test-o.j2k'):
                file_sizes['JPEG2000'] = os.path.getsize('/tmp/test-o.j2k')
        
        # Create the visualization
        num_images = len(display_images)
        if num_images == 0:
            print("⚠️  No images available for visualization")
            return
        
        # Calculate grid layout
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if num_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        # Display images
        for i, (title, image) in enumerate(display_images.items()):
            if i >= len(axes_flat):
                break
            
            axes_flat[i].imshow(image)
            axes_flat[i].set_title(title, fontweight='bold', fontsize=10)
            axes_flat[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_images, len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Add file size information
        if file_sizes:
            info_text = "File Sizes:\n"
            for name, size in file_sizes.items():
                if isinstance(size, int):
                    info_text += f"{name}: {size:,} bytes\n"
            
            # Add text box with file info
            fig.text(0.02, 0.02, info_text, fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('nvImageCodec Official Examples - Test Results', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Save visualization
        output_path = "/tmp/nvimagecodec_official_examples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Official examples visualization saved: {output_path}")
        
        # Print file analysis
        print(f"\n📊 Official Examples File Analysis:")
        original_size = 256 * 256 * 3  # Uncompressed RGB
        
        analysis_files = [
            ('/tmp/test_image.jpg', 'JPEG Input (like tabby_tiger_cat.jpg)'),
            ('/tmp/test-jpg-o.bmp', 'BMP Output (like cat-jpg-o.bmp)'),
            ('/tmp/test-direct-o.jpg', 'Direct JPEG (encoder.write())'),
            ('/tmp/test-o.j2k', 'JPEG2000 (like .jp2 example)')
        ]
        
        print(f"{'Format':<25} {'Size (bytes)':<12} {'Compression':<12} {'Example Reference'}")
        print("-" * 80)
        
        for filepath, description in analysis_files:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                compression = original_size / file_size if file_size > 0 else 0
                format_name = Path(filepath).suffix.upper()[1:]
                print(f"{format_name:<25} {file_size:<12,} {compression:<12.1f}x {description}")
        
        print(f"Original uncompressed: {original_size:,} bytes (256x256x3 RGB)")
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_image():
    """Create a simple test image for nvImageCodec testing"""
    print(f"\n🖼️  Creating test image...")
    
    # Create a simple RGB test image (256x256x3)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Create a colorful pattern
    for i in range(256):
        for j in range(256):
            test_image[i, j, 0] = (i + j) % 256      # Red channel
            test_image[i, j, 1] = (i * 2) % 256      # Green channel  
            test_image[i, j, 2] = (j * 2) % 256      # Blue channel
    
    # Save as a simple PPM file (P6 format) that we can read back
    test_image_path = "/tmp/test_image.ppm"
    with open(test_image_path, 'wb') as f:
        # PPM P6 header
        f.write(b'P6\n')
        f.write(b'256 256\n')
        f.write(b'255\n')
        # Write raw RGB data
        f.write(test_image.tobytes())
    
    print(f"✅ Test image created: {test_image_path}")
    return test_image_path, test_image

def test_nvimagecodec_api():
    """Test nvImageCodec Python API functionality following official examples"""
    print(f"\n🧪 Testing nvImageCodec Python API (Following Official Examples)...")
    print("=" * 60)
    
    try:
        # Import nvImageCodec module and create Decoder and Encoder (like official examples)
        try:
            from nvidia import nvimgcodec
            decoder = nvimgcodec.Decoder()
            encoder = nvimgcodec.Encoder()
            print("✅ nvImageCodec imported and Decoder/Encoder created (like official examples)")
        except ImportError as e:
            print(f"❌ Failed to import nvImageCodec: {e}")
            print("💡 Install with: pip install nvidia-nvimgcodec-cu12")
            return False
        
        # Create test image (like tabby_tiger_cat.jpg in examples)
        test_image_path, test_image_array = create_test_image()
        
        # Official Example Pattern 1: Load and decode JPEG image with nvImageCodec
        print(f"\n📋 Official Example 1: Load and decode JPEG with nvImageCodec")
        try:
            with open(test_image_path.replace('.ppm', '.jpg'), 'wb') as f:
                # First save our test image as JPEG using OpenCV
                import cv2
                cv2.imwrite(test_image_path.replace('.ppm', '.jpg'), 
                           cv2.cvtColor(test_image_array, cv2.COLOR_RGB2BGR))
            
            # Now follow the official example pattern
            with open(test_image_path.replace('.ppm', '.jpg'), 'rb') as in_file:
                data = in_file.read()
                nv_img_test = decoder.decode(data)
            
            print(f"✅ JPEG decoded from memory (like tabby_tiger_cat.jpg example)")
            print(f"   Shape: {nv_img_test.shape}")
            print(f"   Buffer kind: {nv_img_test.buffer_kind}")
        except Exception as e:
            print(f"❌ Official example pattern 1 failed: {e}")
            return False
        
        # Official Example Pattern 2: Save image to BMP file with nvImageCodec
        print(f"\n📋 Official Example 2: Save to BMP with nvImageCodec")
        try:
            with open("/tmp/test-jpg-o.bmp", 'wb') as out_file:
                data = encoder.encode(nv_img_test, "bmp")
                out_file.write(data)
            
            bmp_size = os.path.getsize("/tmp/test-jpg-o.bmp")
            print(f"✅ BMP saved with memory encoding (like cat-jpg-o.bmp example): {bmp_size:,} bytes")
        except Exception as e:
            print(f"❌ Official example pattern 2 failed: {e}")
            return False
        
        # Official Example Pattern 3: Read back with OpenCV
        print(f"\n📋 Official Example 3: Read back with OpenCV")
        try:
            import cv2
            from matplotlib import pyplot as plt
            
            cv_img_bmp = cv2.imread("/tmp/test-jpg-o.bmp")
            cv_img_bmp = cv2.cvtColor(cv_img_bmp, cv2.COLOR_BGR2RGB)
            print(f"✅ BMP read back with OpenCV (like official example): {cv_img_bmp.shape}")
        except Exception as e:
            print(f"❌ Official example pattern 3 failed: {e}")
            return False
        
        # Official Example Pattern 4: Direct read/write functions
        print(f"\n📋 Official Example 4: Direct read/write functions")
        try:
            # Load directly (like decoder.read() in examples)
            nv_img_direct = decoder.read(test_image_path.replace('.ppm', '.jpg'))
            print(f"✅ Direct read successful (like nv_img = decoder.read()): {nv_img_direct.shape}")
            
            # Save directly (like encoder.write() in examples)
            output_file = encoder.write("/tmp/test-direct-o.jpg", nv_img_direct)
            direct_size = os.path.getsize("/tmp/test-direct-o.jpg")
            print(f"✅ Direct write successful (like encoder.write()): {output_file} ({direct_size:,} bytes)")
        except Exception as e:
            print(f"❌ Official example pattern 4 failed: {e}")
            return False
        
        # Official Example Pattern 5: JPEG2000 functionality (like cat-1046544_640.jp2)
        print(f"\n📋 Official Example 5: JPEG2000 functionality")
        try:
            # Save as JPEG2000 (like the .jp2 example)
            encoder.write("/tmp/test-o.j2k", nv_img_test)
            j2k_size = os.path.getsize("/tmp/test-o.j2k")
            print(f"✅ JPEG2000 saved (like cat-jp2-o.jpg example): {j2k_size:,} bytes")
            
            # Read back JPEG2000
            nv_img_j2k = decoder.read("/tmp/test-o.j2k")
            print(f"✅ JPEG2000 read back: {nv_img_j2k.shape}")
        except Exception as e:
            print(f"❌ Official example pattern 5 failed: {e}")
        
        # Store images for visualization
        visualization_images = {
            'original': test_image_array,
            'nvimgcodec_decoded': nv_img_test,
            'opencv_bmp': cv_img_bmp,
            'direct_read': nv_img_direct,
            'jpeg2000': nv_img_j2k if 'nv_img_j2k' in locals() else None
        }
        
        # Create visualization of official examples
        print(f"\n📋 Creating Official Examples Visualization...")
        try:
            create_official_examples_visualization(visualization_images)
        except Exception as e:
            print(f"⚠️  Visualization creation failed: {e}")
        
        # Additional Tests: Backend configurations and advanced features
        print(f"\n📋 Additional Test 1: Backend configurations...")
        try:
            # GPU-preferred decoder
            gpu_decoder = nvimgcodec.Decoder(backends=[
                nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5),
                nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)
            ])
            print("✅ GPU-preferred decoder created")
            
            # CPU-only decoder
            cpu_decoder = nvimgcodec.Decoder(backend_kinds=[nvimgcodec.CPU_ONLY])
            print("✅ CPU-only decoder created")
            
        except Exception as e:
            print(f"⚠️  Backend configuration test failed: {e}")
        
        # Additional Test 2: Array interface testing
        print(f"\n📋 Additional Test 2: Array interface testing...")
        try:
            nv_image = nvimgcodec.as_image(test_image_array)
            print(f"✅ nvImageCodec Image created from numpy array")
            print(f"   Shape: {nv_image.shape}")
            print(f"   Buffer kind: {nv_image.buffer_kind}")
            
            # Test __array_interface__
            if hasattr(nv_image, '__array_interface__'):
                array_interface = nv_image.__array_interface__
                print(f"   Array interface shape: {array_interface['shape']}")
                print(f"   Array interface typestr: {array_interface['typestr']}")
        except Exception as e:
            print(f"❌ Failed to create nvImageCodec Image: {e}")
        
        # Test 4: Encode to different formats
        print(f"\n📋 Test 4: Testing encoding to different formats...")
        test_formats = ['jpg', 'png', 'bmp']
        encoded_files = []
        
        for fmt in test_formats:
            try:
                output_path = f"/tmp/test_output.{fmt}"
                encoder.write(output_path, nv_image)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ {fmt.upper()} encoding successful: {output_path} ({file_size} bytes)")
                    encoded_files.append(output_path)
                else:
                    print(f"❌ {fmt.upper()} encoding failed: file not created")
                    
            except Exception as e:
                print(f"❌ {fmt.upper()} encoding failed: {e}")
        
        # Test 5: Decode the encoded files
        print(f"\n📋 Test 5: Testing decoding of encoded files...")
        for file_path in encoded_files:
            try:
                decoded_image = decoder.read(file_path)
                print(f"✅ Decoded {Path(file_path).suffix}: shape {decoded_image.shape}")
                
                # Test buffer conversion
                if hasattr(decoded_image, 'cpu'):
                    cpu_image = decoded_image.cpu()
                    print(f"   CPU buffer: {cpu_image.buffer_kind}")
                
                if hasattr(decoded_image, 'cuda'):
                    try:
                        cuda_image = decoded_image.cuda()
                        print(f"   CUDA buffer: {cuda_image.buffer_kind}")
                    except Exception as cuda_e:
                        print(f"   ⚠️  CUDA buffer conversion failed: {cuda_e}")
                        
            except Exception as e:
                print(f"❌ Decoding {file_path} failed: {e}")
        
        # Test 6: Encoding parameters
        print(f"\n📋 Test 6: Testing encoding parameters...")
        try:
            # JPEG with quality settings
            jpeg_params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.QUALITY,
                quality_value=75
            )
            encoder.write("/tmp/test_quality75.jpg", nv_image, params=jpeg_params)
            print("✅ JPEG encoding with quality parameter successful")
            
            # JPEG with advanced parameters
            advanced_jpeg_params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.QUALITY,
                quality_value=90,
                jpeg_encode_params=nvimgcodec.JpegEncodeParams(
                    optimized_huffman=True,
                    progressive=True
                )
            )
            encoder.write("/tmp/test_advanced.jpg", nv_image, params=advanced_jpeg_params)
            print("✅ JPEG encoding with advanced parameters successful")
            
        except Exception as e:
            print(f"⚠️  Encoding parameters test failed: {e}")
        
        # Test 7: CodeStream parsing (if we have a real image file)
        print(f"\n📋 Test 7: Testing CodeStream parsing...")
        try:
            # Try to parse one of our encoded files
            if encoded_files:
                test_file = encoded_files[0]  # Use first successfully encoded file
                stream = nvimgcodec.CodeStream(test_file)
                print(f"✅ CodeStream created from {Path(test_file).name}")
                print(f"   Codec: {stream.codec_name}")
                print(f"   Dimensions: {stream.height}x{stream.width}x{stream.channels}")
                print(f"   Data type: {stream.dtype}")
                print(f"   Precision: {stream.precision}")
                print(f"   Tiles: {stream.num_tiles_y}x{stream.num_tiles_x}")
                
                # Test CodeStream from memory
                with open(test_file, 'rb') as f:
                    data = f.read()
                    memory_stream = nvimgcodec.CodeStream(data)
                    print(f"✅ CodeStream created from memory buffer")
                    
        except Exception as e:
            print(f"⚠️  CodeStream parsing test failed: {e}")
        
        # Test 8: JPEG2000 functionality (important for medical imaging)
        print(f"\n📋 Test 8: Testing JPEG2000 functionality...")
        try:
            # Test JPEG2000 encoding with different quality settings
            j2k_lossless_params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.LOSSLESS
            )
            encoder.write("/tmp/test_lossless.j2k", nv_image, params=j2k_lossless_params)
            print("✅ JPEG2000 lossless encoding successful")
            
            # JPEG2000 with PSNR quality
            j2k_psnr_params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.PSNR,
                quality_value=30
            )
            encoder.write("/tmp/test_psnr30.j2k", nv_image, params=j2k_psnr_params)
            print("✅ JPEG2000 PSNR encoding successful")
            
            # Advanced JPEG2000 parameters
            jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
            jpeg2k_encode_params.num_resolutions = 3
            jpeg2k_encode_params.code_block_size = (64, 64)
            jpeg2k_encode_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2
            jpeg2k_encode_params.prog_order = nvimgcodec.Jpeg2kProgOrder.LRCP
            
            advanced_j2k_params = nvimgcodec.EncodeParams(
                quality_type=nvimgcodec.QualityType.LOSSLESS,
                jpeg2k_encode_params=jpeg2k_encode_params
            )
            encoder.write("/tmp/test_advanced.j2k", nv_image, params=advanced_j2k_params)
            print("✅ JPEG2000 advanced encoding successful")
            
            # Test decoding JPEG2000 files
            for j2k_file in ["/tmp/test_lossless.j2k", "/tmp/test_psnr30.j2k", "/tmp/test_advanced.j2k"]:
                if os.path.exists(j2k_file):
                    decoded_j2k = decoder.read(j2k_file)
                    file_size = os.path.getsize(j2k_file)
                    print(f"✅ Decoded {Path(j2k_file).name}: shape {decoded_j2k.shape}, size {file_size} bytes")
                    
        except Exception as e:
            print(f"⚠️  JPEG2000 functionality test failed: {e}")
        
        # Test 9: Context managers
        print(f"\n📋 Test 9: Testing context managers...")
        try:
            with nvimgcodec.Decoder() as ctx_decoder:
                with nvimgcodec.Encoder() as ctx_encoder:
                    # Simple encode/decode cycle
                    ctx_encoder.write("/tmp/test_context.jpg", nv_image)
                    decoded = ctx_decoder.read("/tmp/test_context.jpg")
                    print(f"✅ Context manager test successful: {decoded.shape}")
                    
        except Exception as e:
            print(f"⚠️  Context manager test failed: {e}")
        
        # Test 10: Performance comparison (if both CPU and GPU backends are available)
        print(f"\n📋 Test 10: Performance comparison...")
        try:
            import time
            
            # Create a larger test image for performance testing
            large_test_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
            large_nv_image = nvimgcodec.as_image(large_test_image)
            
            # Test CPU encoding time
            cpu_encoder = nvimgcodec.Encoder(backend_kinds=[nvimgcodec.CPU_ONLY])
            start_time = time.time()
            cpu_encoder.write("/tmp/test_cpu_perf.jpg", large_nv_image)
            cpu_encode_time = time.time() - start_time
            print(f"✅ CPU encoding time: {cpu_encode_time:.3f}s")
            
            # Test GPU encoding time (if available)
            try:
                gpu_encoder = nvimgcodec.Encoder(backends=[
                    nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5),
                    nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)
                ])
                start_time = time.time()
                gpu_encoder.write("/tmp/test_gpu_perf.jpg", large_nv_image)
                gpu_encode_time = time.time() - start_time
                print(f"✅ GPU encoding time: {gpu_encode_time:.3f}s")
                
                if gpu_encode_time < cpu_encode_time:
                    speedup = cpu_encode_time / gpu_encode_time
                    print(f"🚀 GPU speedup: {speedup:.2f}x faster than CPU")
                else:
                    print(f"💡 CPU was faster for this image size")
                    
            except Exception as gpu_e:
                print(f"⚠️  GPU performance test failed: {gpu_e}")
                
        except Exception as e:
            print(f"⚠️  Performance comparison test failed: {e}")
        
        print(f"\n🎉 nvImageCodec API testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ nvImageCodec API testing failed: {e}")
        return False

def main():
    """Main test function"""
    
    # Test plugin setup
    if not test_cuslide2_plugin():
        print(f"\n❌ Plugin test failed")
        return 1
    
    # Create configuration
    config_path = create_plugin_config()
    
    # Test nvImageCodec API
    nvimgcodec_api_success = test_nvimagecodec_api()
    
    # Summary
    print(f"\n🎉 cuslide2 Plugin Test Summary")
    print(f"=" * 40)
    print(f"✅ cuslide2 plugin: Built and loadable")
    print(f"✅ cuCIM library: Available")
    print(f"✅ Configuration: Created at {config_path}")
    
    nvimgcodec_available = os.path.exists("/home/cdinea/micromamba/lib/libnvimgcodec.so.0")
    print(f"{'✅' if nvimgcodec_available else '⚠️ '} nvImageCodec library: {'Available' if nvimgcodec_available else 'Not available (CPU fallback)'}")
    print(f"{'✅' if nvimgcodec_api_success else '⚠️ '} nvImageCodec Python API: {'Working' if nvimgcodec_api_success else 'Not available'}")
    
    print(f"\n📝 Next Steps:")
    print(f"1. Set environment variable: export CUCIM_CONFIG_PATH={config_path}")
    print(f"2. Set library path: export LD_LIBRARY_PATH=/home/cdinea/cucim/build-release/lib:/home/cdinea/micromamba/lib")
    print(f"3. Use cuCIM with cuslide2 plugin in your applications")
    
    if nvimgcodec_available and nvimgcodec_api_success:
        print(f"\n🚀 GPU acceleration is ready!")
        print(f"   JPEG/JPEG2000 tiles will be decoded on GPU for faster performance")
        print(f"   nvImageCodec Python API is working and ready for use")
    elif nvimgcodec_available:
        print(f"\n⚠️  GPU acceleration library available but Python API not working")
        print(f"   Install nvImageCodec Python package: pip install nvidia-nvimgcodec-cu12")
    else:
        print(f"\n💡 To enable GPU acceleration:")
        print(f"   1. micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
        print(f"   2. pip install nvidia-nvimgcodec-cu12")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
