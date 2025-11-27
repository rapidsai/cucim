#!/usr/bin/env python3
"""
nvImageCodec Installation Verification Script

This script comprehensively tests nvImageCodec installation and functionality
to ensure the cuslide2 plugin will work correctly.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import platform

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def check_command_exists(command):
    """Check if a command exists in PATH"""
    try:
        subprocess.run([command, '--version'], 
                      capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        try:
            subprocess.run([command, '--help'], 
                          capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

def run_command(command, description=""):
    """Run a command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, 
                              text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_conda_environment():
    """Check conda environment and nvImageCodec packages"""
    print_section("Conda Environment Check")
    
    # Check if we're in a conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"✓ Conda environment: {conda_prefix}")
    else:
        print("⚠ Not in a conda environment")
        return False
    
    # Find conda executable
    conda_executables = ['micromamba', 'mamba', 'conda']
    conda_cmd = None
    
    for cmd in conda_executables:
        if check_command_exists(cmd):
            conda_cmd = cmd
            print(f"✓ Found conda manager: {cmd}")
            break
    
    if not conda_cmd:
        print("✗ No conda manager found (micromamba, mamba, conda)")
        return False
    
    # Check for nvImageCodec packages
    success, output, error = run_command(f"{conda_cmd} list libnvimgcodec")
    if success and "libnvimgcodec" in output:
        print("✓ nvImageCodec packages found:")
        for line in output.split('\n'):
            if 'libnvimgcodec' in line:
                print(f"  {line}")
        return True
    else:
        print("✗ nvImageCodec packages not found in conda environment")
        print(f"Error: {error}")
        return False

def check_python_packages():
    """Check Python nvImageCodec packages"""
    print_section("Python Package Check")
    
    # Check for nvidia-nvimgcodec packages
    success, output, error = run_command("pip list | grep nvidia-nvimgcodec")
    if success and output:
        print("✓ Python nvImageCodec packages found:")
        for line in output.split('\n'):
            if line.strip():
                print(f"  {line}")
        return True
    else:
        print("⚠ No Python nvImageCodec packages found")
        return False

def check_library_files():
    """Check for nvImageCodec library files"""
    print_section("Library File Check")
    
    search_paths = []
    
    # Add conda environment paths
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        search_paths.extend([
            f"{conda_prefix}/lib",
            f"{conda_prefix}/include"
        ])
        
        # Add Python site-packages paths
        for py_ver in ["3.13", "3.12", "3.11", "3.10", "3.9"]:
            search_paths.append(f"{conda_prefix}/lib/python{py_ver}/site-packages/nvidia/nvimgcodec")
    
    # Add system paths
    search_paths.extend([
        "/usr/local/lib",
        "/usr/lib",
        "/opt/conda/lib",
        "/usr/local/include",
        "/usr/include"
    ])
    
    # Look for header files
    header_found = False
    for path in search_paths:
        header_path = Path(path) / "nvimgcodec.h"
        if header_path.exists():
            print(f"✓ Header found: {header_path}")
            header_found = True
            break
        
        # Also check include subdirectory
        include_path = Path(path) / "include" / "nvimgcodec.h"
        if include_path.exists():
            print(f"✓ Header found: {include_path}")
            header_found = True
            break
    
    if not header_found:
        print("✗ nvimgcodec.h header file not found")
    
    # Look for library files
    library_found = False
    library_names = ["libnvimgcodec.so.0", "libnvimgcodec.so", "libnvimgcodec.dylib", "nvimgcodec.dll"]
    
    for path in search_paths:
        for lib_name in library_names:
            lib_path = Path(path) / lib_name
            if lib_path.exists():
                print(f"✓ Library found: {lib_path}")
                library_found = True
                break
            
            # Also check lib subdirectory
            lib_subpath = Path(path) / "lib" / lib_name
            if lib_subpath.exists():
                print(f"✓ Library found: {lib_subpath}")
                library_found = True
                break
        
        if library_found:
            break
    
    if not library_found:
        print("✗ nvImageCodec library file not found")
    
    return header_found and library_found

def test_c_compilation():
    """Test C compilation with nvImageCodec"""
    print_section("C Compilation Test")
    
    # Create a simple test program
    test_code = '''
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

// Try to include nvImageCodec header
#ifdef HAVE_NVIMGCODEC
#include <nvimgcodec.h>
#endif

int main() {
#ifdef HAVE_NVIMGCODEC
    printf("nvImageCodec header included successfully\\n");
    
    // Try to get version (if available)
    nvimgcodecProperties_t props;
    nvimgcodecStatus_t status = nvimgcodecGetProperties(&props);
    if (status == NVIMGCODEC_STATUS_SUCCESS) {
        printf("nvImageCodec version: %d.%d.%d\\n", 
               props.version.major, props.version.minor, props.version.patch);
    } else {
        printf("Could not get nvImageCodec version (status: %d)\\n", status);
    }
#else
    printf("nvImageCodec header not available\\n");
#endif
    return 0;
}

#ifdef __cplusplus
}
#endif
'''
    
    # Write test file
    test_file = Path("test_nvimgcodec.c")
    try:
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Find nvImageCodec paths
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        include_paths = []
        lib_paths = []
        
        if conda_prefix:
            include_paths.extend([
                f"-I{conda_prefix}/include",
                f"-I{conda_prefix}/lib/python3.12/site-packages/nvidia/nvimgcodec/include",
                f"-I{conda_prefix}/lib/python3.11/site-packages/nvidia/nvimgcodec/include",
                f"-I{conda_prefix}/lib/python3.10/site-packages/nvidia/nvimgcodec/include"
            ])
            lib_paths.extend([
                f"-L{conda_prefix}/lib",
                f"-L{conda_prefix}/lib/python3.12/site-packages/nvidia/nvimgcodec/lib",
                f"-L{conda_prefix}/lib/python3.11/site-packages/nvidia/nvimgcodec/lib",
                f"-L{conda_prefix}/lib/python3.10/site-packages/nvidia/nvimgcodec/lib"
            ])
        
        # Try compilation with nvImageCodec
        compile_cmd = f"gcc {' '.join(include_paths)} -DHAVE_NVIMGCODEC test_nvimgcodec.c {' '.join(lib_paths)} -lnvimgcodec -o test_nvimgcodec 2>&1"
        success, output, error = run_command(compile_cmd)
        
        if success:
            print("✓ C compilation with nvImageCodec successful")
            
            # Try to run the test
            success, output, error = run_command("./test_nvimgcodec")
            if success:
                print("✓ Test program execution successful:")
                print(f"  {output}")
            else:
                print("⚠ Test program compiled but failed to run:")
                print(f"  {error}")
        else:
            print("⚠ C compilation with nvImageCodec failed, trying without:")
            print(f"  {output}")
            
            # Try compilation without nvImageCodec
            compile_cmd = "gcc test_nvimgcodec.c -o test_nvimgcodec_simple 2>&1"
            success, output, error = run_command(compile_cmd)
            if success:
                print("✓ Basic C compilation successful (without nvImageCodec)")
                success, output, error = run_command("./test_nvimgcodec_simple")
                if success:
                    print(f"  Output: {output}")
            else:
                print("✗ Even basic C compilation failed")
    
    finally:
        # Cleanup
        for f in ["test_nvimgcodec.c", "test_nvimgcodec", "test_nvimgcodec_simple"]:
            try:
                Path(f).unlink(missing_ok=True)
            except:
                pass

def test_python_import():
    """Test Python import of nvImageCodec (if available)"""
    print_section("Python Import Test")
    
    # Try to import nvImageCodec Python bindings (if they exist)
    try:
        import nvidia.nvimgcodec
        print("✓ nvidia.nvimgcodec module imported successfully")
        
        # Try to get version
        if hasattr(nvidia.nvimgcodec, '__version__'):
            print(f"  Version: {nvidia.nvimgcodec.__version__}")
        
        return True
    except ImportError as e:
        print("⚠ nvidia.nvimgcodec Python module not available")
        print(f"  Error: {e}")
        return False

def check_cuda_availability():
    """Check CUDA availability"""
    print_section("CUDA Environment Check")
    
    # Check CUDA runtime
    success, output, error = run_command("nvidia-smi")
    if success:
        print("✓ NVIDIA GPU detected:")
        # Extract GPU info from nvidia-smi
        lines = output.split('\n')
        for line in lines:
            if 'NVIDIA' in line and ('GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line):
                print(f"  {line.strip()}")
                break
    else:
        print("⚠ nvidia-smi not available or no NVIDIA GPU detected")
    
    # Check CUDA version
    success, output, error = run_command("nvcc --version")
    if success:
        for line in output.split('\n'):
            if 'release' in line.lower():
                print(f"✓ CUDA compiler: {line.strip()}")
                break
    else:
        print("⚠ CUDA compiler (nvcc) not available")

def main():
    """Main verification function"""
    print_header("nvImageCodec Installation Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    results = {
        'conda_env': check_conda_environment(),
        'python_packages': check_python_packages(),
        'library_files': check_library_files(),
        'cuda': True  # We'll update this
    }
    
    check_cuda_availability()
    test_c_compilation()
    test_python_import()
    
    # Summary
    print_header("Installation Summary")
    
    if results['conda_env'] and results['library_files']:
        print("🎉 nvImageCodec appears to be properly installed!")
        print("✓ Conda packages found")
        print("✓ Library files accessible")
        print("✓ Ready for cuslide2 plugin usage")
        
        print("\n📋 Next Steps:")
        print("1. Build cuslide2 plugin: cd cpp/plugins/cucim.kit.cuslide2 && mkdir build && cd build")
        print("2. Configure with CMake: cmake -DAUTO_INSTALL_NVIMGCODEC=ON ..")
        print("3. Build: make -j$(nproc)")
        print("4. Test: python ../../../test_cuslide2_plugin.py")
        
        return True
    else:
        print("⚠ nvImageCodec installation incomplete or not found")
        
        print("\n🔧 Installation Options:")
        print("Option 1 (Conda): micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
        print("Option 2 (Pip): pip install nvidia-nvimgcodec-cu12[all]  # For CUDA 12.x")
        print("Option 3 (Auto): cmake -DAUTO_INSTALL_NVIMGCODEC=ON .. # During build")
        
        if not results['conda_env']:
            print("\n⚠ Consider using a conda environment for better dependency management")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
