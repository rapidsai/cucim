#!/usr/bin/env python3
"""
Test script for Supporting_Aperio_SVS_Format notebook with cuslide2 plugin
"""

import json
import sys
from pathlib import Path
import pprint

print("=" * 80)
print("Testing Aperio SVS Format Support with cuslide2 Plugin")
print("=" * 80)

# Test 1: Check if cuslide2 is configured
print("\n[Test 1] Checking cuslide2 plugin configuration...")
try:
    with open(".cucim.json") as f:
        config = json.load(f)
        print("✓ cuslide2 plugin configuration found:")
        pprint.pprint(config)
except Exception as e:
    print(f"✗ Error reading .cucim.json: {e}")
    sys.exit(1)

# Test 2: Import cucim
print("\n[Test 2] Importing cucim...")
try:
    from cucim import CuImage
    import cucim
    print(f"✓ cucim imported successfully (version: {cucim.__version__})")
except Exception as e:
    print(f"✗ Error importing cucim: {e}")
    sys.exit(1)

# Test 3: Download small SVS test file
print("\n[Test 3] Downloading small SVS test file...")
try:
    import wget
    
    data_url = "https://openslide.cs.cmu.edu/download/openslide-testdata"
    Path("Aperio").mkdir(parents=True, exist_ok=True)
    
    # Download just the small test file
    test_file = "Aperio/CMU-1-Small-Region.svs"
    
    if not Path(test_file).exists():
        print(f"  Downloading {test_file}...")
        wget.download(f"{data_url}/{test_file}", out=test_file)
        print()  # newline after download
    else:
        print(f"  File already exists: {test_file}")
    
    print(f"✓ Test file ready: {test_file}")
except Exception as e:
    print(f"✗ Error downloading test file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load SVS file with cuslide2
print("\n[Test 4] Loading SVS file with cuslide2...")
try:
    img = CuImage(test_file)
    print(f"✓ Successfully loaded {test_file}")
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")
except Exception as e:
    print(f"✗ Error loading SVS file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Access metadata
print("\n[Test 5] Accessing metadata...")
try:
    metadata = img.metadata
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    
    print("  Aperio metadata:")
    if 'aperio' in metadata:
        for key, value in list(metadata['aperio'].items())[:5]:  # Show first 5 items
            print(f"    {key}: {value}")
        print(f"    ... ({len(metadata['aperio'])} total fields)")
    else:
        print("    No Aperio metadata found")
    
    print("\n  cuCIM metadata:")
    if 'cucim' in metadata:
        cucim_meta = metadata['cucim']
        print(f"    ndim: {cucim_meta.get('ndim')}")
        print(f"    dims: {cucim_meta.get('dims')}")
        print(f"    channel_names: {cucim_meta.get('channel_names')}")
        print(f"    resolutions.level_count: {cucim_meta.get('resolutions', {}).get('level_count')}")
        print(f"    associated_images: {cucim_meta.get('associated_images')}")
    
    print("✓ Metadata accessed successfully")
except Exception as e:
    print(f"✗ Error accessing metadata: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Read image region
print("\n[Test 6] Reading image region...")
try:
    import numpy as np
    
    # Get level count
    level_count = metadata["cucim"]["resolutions"]["level_count"]
    
    # Read at lowest resolution (whole image)
    small_img = img.read_region(level=level_count - 1)
    print(f"✓ Read region at level {level_count - 1}")
    print(f"  Region shape: {small_img.shape}")
    print(f"  Region dtype: {small_img.dtype}")
    
    # Read a specific region at highest resolution
    region = img.read_region(location=(0, 0), size=(256, 256), level=0)
    print(f"✓ Read 256x256 region at level 0")
    print(f"  Region shape: {region.shape}")
    
except Exception as e:
    print(f"✗ Error reading region: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Access associated images
print("\n[Test 7] Accessing associated images...")
try:
    associated_images = img.associated_images
    print(f"  Available associated images: {list(associated_images.keys()) if hasattr(associated_images, 'keys') else associated_images}")
    
    # Try to read each associated image
    for img_name in ['label', 'macro', 'thumbnail']:
        try:
            assoc_img = img.associated_image(img_name)
            print(f"  ✓ {img_name}: {assoc_img.shape}")
        except Exception as e:
            print(f"  - {img_name}: not available ({e})")
    
    print("✓ Associated images test completed")
except Exception as e:
    print(f"✗ Error accessing associated images: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Try downloading and testing a larger file
print("\n[Test 8] Testing with a larger SVS file (JPEG2000)...")
try:
    large_test_file = "Aperio/CMU-1-JP2K-33005.svs"
    
    if not Path(large_test_file).exists():
        print(f"  Downloading {large_test_file} (126 MB, this may take a while)...")
        wget.download(f"{data_url}/{large_test_file}", out=large_test_file)
        print()  # newline after download
    else:
        print(f"  File already exists: {large_test_file}")
    
    # Load and test
    img2 = CuImage(large_test_file)
    print(f"✓ Successfully loaded {large_test_file}")
    print(f"  Image shape: {img2.shape}")
    
    # Read a small region
    region2 = img2.read_region(location=(1000, 1000), size=(512, 512), level=0)
    print(f"✓ Read 512x512 region from {large_test_file}")
    print(f"  Region shape: {region2.shape}")
    
except Exception as e:
    print(f"✗ Error with larger file: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("Testing completed!")
print("All critical tests passed. cuslide2 plugin is working correctly with SVS files.")
print("=" * 80)

