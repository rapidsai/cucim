#!/usr/bin/env python3
"""
Simple cuslide2 plugin test
"""

import os
import sys
import json

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

def main():
    """Main test function"""
    
    # Test plugin setup
    if not test_cuslide2_plugin():
        print(f"\n❌ Plugin test failed")
        return 1
    
    # Create configuration
    config_path = create_plugin_config()
    
    # Summary
    print(f"\n🎉 cuslide2 Plugin Test Summary")
    print(f"=" * 40)
    print(f"✅ cuslide2 plugin: Built and loadable")
    print(f"✅ cuCIM library: Available")
    print(f"✅ Configuration: Created at {config_path}")
    
    nvimgcodec_available = os.path.exists("/home/cdinea/micromamba/lib/libnvimgcodec.so.0")
    print(f"{'✅' if nvimgcodec_available else '⚠️ '} nvImageCodec: {'Available' if nvimgcodec_available else 'Not available (CPU fallback)'}")
    
    print(f"\n📝 Next Steps:")
    print(f"1. Set environment variable: export CUCIM_CONFIG_PATH={config_path}")
    print(f"2. Set library path: export LD_LIBRARY_PATH=/home/cdinea/cucim/build-release/lib:/home/cdinea/micromamba/lib")
    print(f"3. Use cuCIM with cuslide2 plugin in your applications")
    
    if nvimgcodec_available:
        print(f"\n🚀 GPU acceleration is ready!")
        print(f"   JPEG/JPEG2000 tiles will be decoded on GPU for faster performance")
    else:
        print(f"\n💡 To enable GPU acceleration:")
        print(f"   micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
