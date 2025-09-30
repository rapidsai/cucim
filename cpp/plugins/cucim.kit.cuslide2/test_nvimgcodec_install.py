#!/usr/bin/env python3
"""
Test script to validate nvImageCodec installation and detection logic
This simulates what the CMakeLists.txt does for nvImageCodec detection
"""

import os
import subprocess
import sys
from pathlib import Path

def find_conda_executable():
    """Find conda/micromamba executable"""
    search_paths = [
        Path(__file__).parent / "../../../bin/micromamba",
        Path(__file__).parent / "../../bin/micromamba", 
        Path(__file__).parent / "bin/micromamba",
        Path.home() / "micromamba/bin/micromamba",
        Path.home() / ".local/bin/micromamba",
        "/usr/local/bin/micromamba",
        "/opt/conda/bin/micromamba",
        "/opt/miniconda/bin/micromamba",
        Path.home() / "miniconda3/bin/conda",
        Path.home() / "anaconda3/bin/conda",
        "/opt/conda/bin/conda",
        "/opt/miniconda/bin/conda",
        "/usr/local/bin/conda"
    ]
    
    for path in search_paths:
        if Path(path).exists() and Path(path).is_file():
            return str(path)
    
    # Try system PATH
    for cmd in ["micromamba", "conda", "mamba"]:
        try:
            result = subprocess.run(["which", cmd], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
    
    return None

def check_nvimgcodec_installation(conda_cmd):
    """Check if nvImageCodec is installed"""
    try:
        result = subprocess.run(
            [conda_cmd, "list", "libnvimgcodec-dev"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print(f"✓ nvImageCodec found:")
            print(f"  Output: {result.stdout.strip()}")
            
            # Parse version
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'libnvimgcodec-dev' in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[1]
                        print(f"  Version: {version}")
                        return version
            return "unknown"
        else:
            print("✗ nvImageCodec not found")
            return None
            
    except Exception as e:
        print(f"✗ Error checking nvImageCodec: {e}")
        return None

def install_nvimgcodec(conda_cmd, version="0.6.0"):
    """Install nvImageCodec via conda"""
    print(f"Installing nvImageCodec {version}...")
    
    try:
        cmd = [
            conda_cmd, "install", 
            f"libnvimgcodec-dev={version}",
            f"libnvimgcodec0={version}",
            "-c", "conda-forge", "-y"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Installation successful")
            return True
        else:
            print(f"✗ Installation failed: {result.stderr}")
            
            # Try fallback without version constraint
            print("Trying fallback installation without version constraint...")
            fallback_cmd = [
                conda_cmd, "install", 
                "libnvimgcodec-dev", "libnvimgcodec0",
                "-c", "conda-forge", "-y"
            ]
            
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
            if fallback_result.returncode == 0:
                print("✓ Fallback installation successful")
                return True
            else:
                print(f"✗ Fallback installation also failed: {fallback_result.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("✗ Installation timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Installation error: {e}")
        return False

def detect_nvimgcodec_paths():
    """Detect nvImageCodec installation paths"""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    
    search_locations = []
    
    # Add conda environment paths
    if conda_prefix:
        search_locations.extend([
            Path(conda_prefix),  # Native conda package
            Path(conda_prefix) / "lib/python3.13/site-packages/nvidia/nvimgcodec",
            Path(conda_prefix) / "lib/python3.12/site-packages/nvidia/nvimgcodec", 
            Path(conda_prefix) / "lib/python3.11/site-packages/nvidia/nvimgcodec",
            Path(conda_prefix) / "lib/python3.10/site-packages/nvidia/nvimgcodec",
            Path(conda_prefix) / "lib/python3.9/site-packages/nvidia/nvimgcodec",
        ])
    
    # Add Python site-packages
    try:
        import site
        for site_pkg in site.getsitepackages():
            search_locations.append(Path(site_pkg) / "nvidia/nvimgcodec")
    except:
        pass
    
    print("Searching for nvImageCodec in:")
    for location in search_locations:
        print(f"  - {location}")
        
        header_path = location / "include/nvimgcodec.h"
        if header_path.exists():
            print(f"    ✓ Found headers: {header_path}")
            
            # Look for library
            lib_paths = [
                location / "lib/libnvimgcodec.so.0",
                location / "lib/libnvimgcodec.so", 
                location / "libnvimgcodec.so.0",
                location / "libnvimgcodec.so"
            ]
            
            for lib_path in lib_paths:
                if lib_path.exists():
                    print(f"    ✓ Found library: {lib_path}")
                    return str(location), str(lib_path)
            
            print(f"    ✗ Library not found in {location}")
        else:
            print(f"    ✗ Headers not found")
    
    return None, None

def main():
    print("=== nvImageCodec Installation Test ===")
    
    # Find conda executable
    conda_cmd = find_conda_executable()
    if not conda_cmd:
        print("✗ No conda/micromamba found")
        print("Please install conda, mamba, or micromamba")
        return 1
    
    print(f"✓ Found conda tool: {conda_cmd}")
    
    # Check current installation
    current_version = check_nvimgcodec_installation(conda_cmd)
    
    # Install if needed
    target_version = "0.6.0"
    if not current_version:
        print(f"\nInstalling nvImageCodec {target_version}...")
        if not install_nvimgcodec(conda_cmd, target_version):
            print("Installation failed")
            return 1
    elif current_version < target_version:
        print(f"\nUpgrading nvImageCodec from {current_version} to {target_version}...")
        if not install_nvimgcodec(conda_cmd, target_version):
            print("Upgrade failed")
            return 1
    else:
        print(f"✓ nvImageCodec {current_version} is already installed (>= {target_version})")
    
    # Detect installation paths
    print(f"\n=== Path Detection ===")
    nvimgcodec_root, nvimgcodec_lib = detect_nvimgcodec_paths()
    
    if nvimgcodec_root and nvimgcodec_lib:
        print(f"\n✓ nvImageCodec ready for CMake:")
        print(f"  NVIMGCODEC_ROOT: {nvimgcodec_root}")
        print(f"  NVIMGCODEC_LIBRARY: {nvimgcodec_lib}")
        print(f"\nCMake configuration:")
        print(f"  target_include_directories(target PRIVATE \"{nvimgcodec_root}/include\")")
        print(f"  target_link_libraries(target PRIVATE \"{nvimgcodec_lib}\")")
        print(f"  target_compile_definitions(target PRIVATE CUCIM_HAS_NVIMGCODEC)")
        return 0
    else:
        print("\n✗ nvImageCodec installation incomplete")
        return 1

if __name__ == "__main__":
    sys.exit(main())
