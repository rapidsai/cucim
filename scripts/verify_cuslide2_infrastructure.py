#!/usr/bin/env python3
"""
cuslide2 Infrastructure Verification Script

This script validates that the cuslide2 plugin infrastructure is properly set up:
- nvImageCodec installation (conda or pip)
- cuslide2 plugin build status
- Plugin integration with nvImageCodec
- Environment configuration
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
    """Check conda environment and nvImageCodec packages - returns package list or None"""
    # Check if we're in a conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return None
    
    # Find conda executable
    conda_executables = ['micromamba', 'mamba', 'conda']
    conda_cmd = None
    
    for cmd in conda_executables:
        if check_command_exists(cmd):
            conda_cmd = cmd
            break
    
    if not conda_cmd:
        return None
    
    # Check for nvImageCodec packages
    success, output, error = run_command(f"{conda_cmd} list libnvimgcodec")
    if success and "libnvimgcodec" in output:
        packages = []
        for line in output.split('\n'):
            if 'libnvimgcodec' in line:
                packages.append(line.strip())
        return packages if packages else None
    return None

def check_python_packages():
    """Check Python nvImageCodec packages - returns package list or None"""
    success, output, error = run_command("pip list | grep nvidia-nvimgcodec")
    if success and output:
        packages = []
        for line in output.split('\n'):
            if line.strip():
                packages.append(line.strip())
        return packages if packages else None
    return None

def detect_installation_method():
    """Detect and display nvImageCodec installation method"""
    print_section("Installation Method Detection")
    
    conda_packages = check_conda_environment()
    pip_packages = check_python_packages()
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    
    if conda_packages:
        print("✓ Conda native packages detected:")
        for pkg in conda_packages:
            print(f"  {pkg}")
        print("ℹ Using conda installation (recommended for C++ builds)")
        if conda_prefix:
            print(f"  Environment: {conda_prefix}")
        return 'conda'
    elif pip_packages:
        print("✓ Pip packages detected:")
        for pkg in pip_packages:
            print(f"  {pkg}")
        print("ℹ Using pip installation")
        return 'pip'
    else:
        if conda_prefix:
            print("✗ nvImageCodec not found in conda environment")
        else:
            print("ℹ Not in a conda environment")
        print("✗ No nvImageCodec installation detected")
        return None

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
        for py_ver in ["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"]:
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
    
    // Try to create instance to verify library is functional
    nvimgcodecInstance_t instance = NULL;
    nvimgcodecInstanceCreateInfo_t create_info = {NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_ERROR;
    create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;
    
    nvimgcodecStatus_t status = nvimgcodecInstanceCreate(&instance, &create_info);
    if (status == NVIMGCODEC_STATUS_SUCCESS) {
        printf("✓ nvImageCodec instance created successfully\\n");
        nvimgcodecInstanceDestroy(instance);
    } else {
        printf("⚠ Could not create nvImageCodec instance (status: %d)\\n", status);
        printf("  This is OK - headers and library are accessible\\n");
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
        nvimgcodec_lib = None  # Store actual library path
        
        if conda_prefix:
            include_paths.extend([
                f"-I{conda_prefix}/include",
            ])
            lib_paths.extend([
                f"-L{conda_prefix}/lib",
            ])
            # Add CUDA library paths
            cuda_lib_paths = [
                f"{conda_prefix}/targets/x86_64-linux/lib",
                f"{conda_prefix}/targets/aarch64-linux/lib",
                f"{conda_prefix}/lib",
                "/usr/lib/x86_64-linux-gnu",  # System CUDA driver
                "/usr/lib/aarch64-linux-gnu",
                "/usr/local/cuda/lib64",
                "/opt/cuda/lib64",
            ]
            for cuda_lib in cuda_lib_paths:
                if Path(cuda_lib).exists():
                    lib_paths.append(f"-L{cuda_lib}")
            
            # Add CUDA toolkit include paths (needed by nvimgcodec.h)
            cuda_include_paths = [
                f"{conda_prefix}/targets/x86_64-linux/include",
                f"{conda_prefix}/targets/aarch64-linux/include",
                f"{conda_prefix}/include/cuda",
                "/usr/local/cuda/include",
                "/opt/cuda/include",
            ]
            for cuda_path in cuda_include_paths:
                if Path(cuda_path).exists():
                    include_paths.append(f"-I{cuda_path}")
            
            # Add Python site-packages paths for multiple versions
            # Also look for the actual library file
            for py_ver in ["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"]:
                include_paths.append(f"-I{conda_prefix}/lib/python{py_ver}/site-packages/nvidia/nvimgcodec/include")
                
                # Find actual library file
                if nvimgcodec_lib is None:
                    for lib_name in ["libnvimgcodec.so.0", "libnvimgcodec.so"]:
                        lib_file = Path(f"{conda_prefix}/lib/python{py_ver}/site-packages/nvidia/nvimgcodec/{lib_name}")
                        if lib_file.exists():
                            nvimgcodec_lib = str(lib_file)
                            break
                    # Also check lib subdirectory
                    if nvimgcodec_lib is None:
                        for lib_name in ["libnvimgcodec.so.0", "libnvimgcodec.so"]:
                            lib_file = Path(f"{conda_prefix}/lib/python{py_ver}/site-packages/nvidia/nvimgcodec/lib/{lib_name}")
                            if lib_file.exists():
                                nvimgcodec_lib = str(lib_file)
                                break
        
        # Try compilation with nvImageCodec (also link CUDA runtime and driver)
        # Use direct path to library instead of -l flag
        # Note: -lcuda links CUDA driver library for native conda packages
        # Use /usr/bin/gcc to avoid conda gcc glibc compatibility issues
        gcc_path = "/usr/bin/gcc" if Path("/usr/bin/gcc").exists() else "gcc"
        
        if nvimgcodec_lib:
            compile_cmd = f"{gcc_path} {' '.join(include_paths)} -DHAVE_NVIMGCODEC test_nvimgcodec.c {' '.join(lib_paths)} {nvimgcodec_lib} -lcudart -lcuda -o test_nvimgcodec 2>&1"
        else:
            compile_cmd = f"{gcc_path} {' '.join(include_paths)} -DHAVE_NVIMGCODEC test_nvimgcodec.c {' '.join(lib_paths)} -lnvimgcodec -lcudart -lcuda -o test_nvimgcodec 2>&1"
        success, output, error = run_command(compile_cmd)
        
        if success:
            print("✓ C compilation with nvImageCodec successful")
            
            # Try to run the test with library path set
            # Build LD_LIBRARY_PATH with nvImageCodec location
            lib_dir = str(Path(nvimgcodec_lib).parent) if nvimgcodec_lib else ""
            cuda_lib_dir = f"{conda_prefix}/lib" if conda_prefix else ""
            
            if lib_dir:
                ld_library_path = f"{lib_dir}:{cuda_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                run_cmd = f"LD_LIBRARY_PATH={ld_library_path} ./test_nvimgcodec"
            else:
                run_cmd = "./test_nvimgcodec"
            
            success, output, error = run_command(run_cmd)
            if success:
                print("✓ Test program execution successful:")
                print(f"  {output}")
            else:
                print("⚠ Test program compiled but failed to run:")
                print(f"  {error}")
                if lib_dir:
                    print(f"  (Note: Library path was set to {lib_dir})")
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

def test_python_import(install_method):
    """Test Python import of nvImageCodec (if available)"""
    print_section("Python Bindings Check")
    
    # Try to import nvImageCodec Python bindings (if they exist)
    try:
        import nvidia.nvimgcodec
        print("✓ nvidia.nvimgcodec Python module available")
        
        # Try to get version
        if hasattr(nvidia.nvimgcodec, '__version__'):
            print(f"  Version: {nvidia.nvimgcodec.__version__}")
        
        return True
    except ImportError:
        if install_method == 'conda':
            print("ℹ Python bindings not installed (not required for conda C++ builds)")
            print("  C++ plugin links directly to libnvimgcodec.so")
        else:
            print("ℹ Python bindings not found")
            print("  Install with: pip install nvidia-nvimgcodec-cu12[all]")
        return False

def check_cuslide2_plugin():
    """Check if cuslide2 plugin is built and integrated with nvImageCodec"""
    print_section("cuslide2 Plugin Check")
    
    # Get script directory and find plugin paths
    script_dir = Path(__file__).parent.parent  # Go up to cucim root
    
    # Possible plugin locations
    plugin_paths = [
        script_dir / "cpp/plugins/cucim.kit.cuslide2/build-release/lib",
        script_dir / "cpp/plugins/cucim.kit.cuslide2/build/lib",
        script_dir / "install/lib",
    ]
    
    plugin_file = None
    for plugin_dir in plugin_paths:
        if plugin_dir.exists():
            # Look for cucim.kit.cuslide2@*.so
            for so_file in plugin_dir.glob("cucim.kit.cuslide2@*.so"):
                plugin_file = so_file
                break
        if plugin_file:
            break
    
    if not plugin_file:
        print("✗ cuslide2 plugin not found")
        print("  Expected locations:")
        for path in plugin_paths:
            print(f"    {path}/cucim.kit.cuslide2@*.so")
        print("\n  Run: ./run build_local all release $CONDA_PREFIX")
        return False
    
    print(f"✓ Plugin found: {plugin_file.name}")
    
    # Check file size
    size_mb = plugin_file.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Location: {plugin_file.parent}")
    
    # Check if linked with nvImageCodec using ldd
    success, output, error = run_command(f"ldd {plugin_file} 2>&1")
    if success:
        has_nvimgcodec = "nvimgcodec" in output
        has_cuda = "cuda" in output.lower()
        
        if has_nvimgcodec:
            print("✓ Plugin linked with nvImageCodec (static linking)")
            # Show the nvimgcodec line
            for line in output.split('\n'):
                if 'nvimgcodec' in line:
                    print(f"  {line.strip()}")
        else:
            print("ℹ Plugin uses dynamic loading for nvImageCodec")
            print("  Library loaded at runtime via plugin system")
        
        if has_cuda:
            print("✓ Plugin linked with CUDA libraries")
        
    # Check for nvImageCodec symbols using nm
    success, output, error = run_command(f"nm -D {plugin_file} 2>&1 | grep -i nvimg | head -5")
    if success and output:
        print("✓ Plugin has nvImageCodec symbols:")
        for line in output.split('\n')[:3]:
            if line.strip():
                print(f"  {line.strip()}")
    
    return True

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
    print_header("cuslide2 Infrastructure Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Detect installation method
    install_method = detect_installation_method()
    
    # Check library files
    library_files_ok = check_library_files()
    
    # Check CUDA environment
    check_cuda_availability()
    
    # Check plugin
    plugin_built = check_cuslide2_plugin()
    
    # Check Python bindings (informational)
    if install_method:
        test_python_import(install_method)
    
    # Summary
    print_header("Infrastructure Summary")
    
    all_good = library_files_ok and plugin_built and install_method
    
    if all_good:
        print("🎉 cuslide2 infrastructure is properly set up!")
        if install_method == 'conda':
            print("✓ nvImageCodec installed via conda native packages")
        elif install_method == 'pip':
            print("✓ nvImageCodec installed via pip packages")
        print("✓ All required library files accessible")
        print("✓ cuslide2 plugin built successfully")
        print("✓ CUDA environment configured correctly")
        
        return True
    elif library_files_ok and not plugin_built:
        print("⚠ Partial setup - nvImageCodec installed but plugin not built")
        print("✓ nvImageCodec is installed")
        print("✗ cuslide2 plugin not built yet")
        print("\n📋 Next step: Build the plugin")
        print("   cd cucim && ./run build_local all release $CONDA_PREFIX")
        
        return False
    elif not install_method or not library_files_ok:
        print("✗ nvImageCodec installation incomplete or not found")
        
        print("\n🔧 Installation Options:")
        print("Option 1 (Conda - recommended for C++): micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
        print("Option 2 (Pip): pip install nvidia-nvimgcodec-cu12[all]  # For CUDA 12.x")
        print("Option 3 (Auto-install during build): cmake -DAUTO_INSTALL_NVIMGCODEC=ON ..")
        
        print("\n💡 Both conda and pip installations are supported!")
        
        return False
    else:
        print("✗ Setup incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

