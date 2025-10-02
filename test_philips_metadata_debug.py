#!/usr/bin/env python3
"""Debug Philips TIFF metadata extraction"""

import sys
import cucim
from cucim.clara import _set_plugin_root
from pathlib import Path

def debug_philips_metadata(file_path):
    """Debug what metadata is actually available"""
    
    print("=" * 60)
    print("🔍 Philips TIFF Metadata Debug")
    print("=" * 60)
    print(f"📁 File: {file_path}")
    print()
    
    # Set plugin root
    plugin_lib = Path(__file__).parent / "cpp/plugins/cucim.kit.cuslide2/build-release/lib"
    _set_plugin_root(str(plugin_lib))
    
    # Load image
    img = cucim.CuImage(file_path)
    print(f"✅ Image loaded: {img.shape}")
    print()
    
    # Get ALL metadata
    print("📋 ALL Available Metadata:")
    print("-" * 60)
    metadata = img.metadata
    
    if metadata:
        print(f"Metadata type: {type(metadata)}")
        print(f"Metadata keys count: {len(metadata)}")
        print()
        
        # Print all keys
        for key in sorted(metadata.keys()):
            value = metadata[key]
            # Truncate long values
            if isinstance(value, str) and len(value) > 100:
                value_str = value[:100] + "..."
            else:
                value_str = str(value)
            print(f"  {key}: {value_str}")
    else:
        print("  ⚠️  No metadata found!")
    
    print()
    print("-" * 60)
    
    # Check for specific Philips keys
    print()
    print("🔍 Looking for Philips-specific metadata:")
    philips_keys = [k for k in metadata.keys() if 'philips' in k.lower()]
    if philips_keys:
        print(f"  ✅ Found {len(philips_keys)} Philips keys:")
        for key in philips_keys:
            print(f"    - {key}")
    else:
        print("  ❌ No 'philips' keys found")
    
    # Check for XML or ImageDescription
    print()
    print("🔍 Looking for ImageDescription or XML:")
    xml_keys = [k for k in metadata.keys() if 'description' in k.lower() or 'xml' in k.lower()]
    if xml_keys:
        print(f"  ✅ Found {len(xml_keys)} description/XML keys:")
        for key in xml_keys:
            value = metadata[key]
            if isinstance(value, str):
                print(f"    - {key}: {value[:200]}...")
            else:
                print(f"    - {key}: {value}")
    else:
        print("  ❌ No description/XML keys found")
    
    # Check openslide properties
    print()
    print("🔍 Looking for openslide properties:")
    openslide_keys = [k for k in metadata.keys() if 'openslide' in k.lower()]
    if openslide_keys:
        print(f"  ✅ Found {len(openslide_keys)} openslide keys:")
        for key in openslide_keys:
            print(f"    - {key}: {metadata[key]}")
    else:
        print("  ❌ No openslide keys found")
    
    # Check raw metadata
    print()
    print("🔍 Checking raw_metadata attribute:")
    if hasattr(img, 'raw_metadata'):
        print(f"  ✅ raw_metadata exists: {type(img.raw_metadata)}")
        if img.raw_metadata:
            if isinstance(img.raw_metadata, dict):
                print(f"     Keys: {list(img.raw_metadata.keys())[:10]}")
            elif isinstance(img.raw_metadata, str):
                print(f"     Length: {len(img.raw_metadata)} characters")
                print(f"     Preview: {img.raw_metadata[:200]}...")
    else:
        print("  ❌ No raw_metadata attribute")
    
    print()
    print("✅ Debug complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_philips_metadata_debug.py <philips_tiff_file>")
        sys.exit(1)
    
    debug_philips_metadata(sys.argv[1])

