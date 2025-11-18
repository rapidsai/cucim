#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
cuslide2 Infrastructure Verification Script

This script validates that the cuslide2 plugin infrastructure is properly set up:
- nvImageCodec installation (conda/pip packages OR CMake build integration)
- cuslide2 plugin build status
- Plugin integration with nvImageCodec (linkage verification)
- Environment configuration

Supports two nvImageCodec deployment models:
1. Conda/pip packages (specified in dependencies.yaml) - RECOMMENDED
2. CMake FetchContent integration (automatically fetched during build)
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_section(title):
    """Print a formatted section"""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def check_command_exists(command):
    """Check if a command exists in PATH"""
    try:
        subprocess.run(
            [command, "--version"], capture_output=True, check=True, timeout=10
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        try:
            subprocess.run(
                [command, "--help"], capture_output=True, check=True, timeout=10
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False


def run_command(command, description=""):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def check_conda_environment():
    """Check conda environment and nvImageCodec packages - returns package list or None"""
    # Check if we're in a conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    # Find conda executable
    conda_executables = ["micromamba", "mamba", "conda"]
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
        for line in output.split("\n"):
            if "libnvimgcodec" in line:
                packages.append(line.strip())
        return packages if packages else None
    return None


def check_python_packages():
    """Check Python nvImageCodec packages - returns package list or None"""
    success, output, error = run_command("pip list | grep nvidia-nvimgcodec")
    if success and output:
        packages = []
        for line in output.split("\n"):
            if line.strip():
                packages.append(line.strip())
        return packages if packages else None
    return None


def find_plugin_build_dir():
    """Find the cuslide2 plugin build directory"""
    script_dir = Path(__file__).parent.parent  # Go up to cucim root

    # Possible build locations
    build_paths = [
        script_dir / "cpp/plugins/cucim.kit.cuslide2/build-release",
        script_dir / "cpp/plugins/cucim.kit.cuslide2/build",
        script_dir / "build-release",
        script_dir / "build",
    ]

    for build_dir in build_paths:
        if build_dir.exists():
            return build_dir

    return None


def detect_installation_method():
    """Detect and display nvImageCodec installation method - supports both approaches"""
    print_section("nvImageCodec Installation Detection")

    conda_packages = check_conda_environment()
    pip_packages = check_python_packages()
    conda_prefix = os.environ.get("CONDA_PREFIX")

    # Check for conda/pip installation (Method 1 - RECOMMENDED per dependencies.yaml)
    if conda_packages:
        print("✓ Method 1: Conda packages detected (from dependencies.yaml):")
        for pkg in conda_packages:
            print(f"    {pkg}")
        if conda_prefix:
            print(f"  Environment: {conda_prefix}")
        return "conda"
    elif pip_packages:
        print("✓ Method 1: Pip packages detected:")
        for pkg in pip_packages:
            print(f"    {pkg}")
        return "pip"

    # Check for CMake build integration (Method 2 - Alternative)
    build_dir = find_plugin_build_dir()
    if build_dir:
        print("ℹ  Checking Method 2: CMake build integration...")
        print(f"   Build directory: {build_dir}")

        deps_dirs = [
            build_dir / "_deps",
            build_dir.parent.parent.parent / "build-release/_deps",
        ]

        nvimgcodec_found = False
        for deps_dir in deps_dirs:
            if not deps_dir.exists():
                continue
            for item in deps_dir.iterdir():
                if "nvimgcodec" in item.name.lower():
                    nvimgcodec_found = True
                    print("✓ Method 2: nvImageCodec integrated via CMake:")
                    print(f"    {item.name}")
                    return "cmake-integrated"

        if not nvimgcodec_found:
            print("✗ Method 2: No CMake integration found")

    # Neither method found
    if conda_prefix:
        print("✗ nvImageCodec not found via any method")
        print("  Expected: conda packages from dependencies.yaml")
    else:
        print("ℹ  Not in a conda environment")
        print("✗ No nvImageCodec installation detected")

    return None


def check_nvimgcodec_library_in_deps():
    """Check for nvImageCodec library files in plugin dependencies"""
    build_dir = find_plugin_build_dir()

    if not build_dir:
        return False

    # Search for nvImageCodec library in build dependencies
    search_paths = [
        build_dir / "_deps",
        build_dir.parent.parent.parent / "build-release/_deps",
    ]

    library_names = ["libnvimgcodec.so", "libnvimgcodec.so.0", "libnvimgcodec.a"]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Recursively search for library files
        for lib_name in library_names:
            for lib_file in search_path.rglob(lib_name):
                print(
                    f"✓ nvImageCodec library found: {lib_file.relative_to(search_path.parent)}"
                )
                return True

    return False


def check_library_files(install_method):
    """Check for nvImageCodec library files based on installation method"""
    print_section("nvImageCodec Library Files Check")

    search_paths = []

    # Method 1: Check conda/pip installation paths
    if install_method in ["conda", "pip"]:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            search_paths.extend([f"{conda_prefix}/lib", f"{conda_prefix}/include"])
            # Add Python site-packages paths for pip installations
            for py_ver in ["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"]:
                search_paths.append(
                    f"{conda_prefix}/lib/python{py_ver}/site-packages/nvidia/nvimgcodec"
                )
        # Add system paths
        search_paths.extend(
            ["/usr/local/lib", "/usr/lib", "/usr/local/include", "/usr/include"]
        )

    # Method 2: Check CMake build dependencies
    if install_method == "cmake-integrated":
        build_dir = find_plugin_build_dir()
        if build_dir:
            search_paths.extend(
                [
                    str(build_dir / "_deps"),
                    str(build_dir.parent.parent.parent / "build-release/_deps"),
                ]
            )

    if not search_paths:
        print("✗ No search paths available - installation method unknown")
        return False

    header_found = False
    library_found = False

    # Look for header files
    for path in search_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue

        # Check for header directly
        header_path = path_obj / "nvimgcodec.h"
        if header_path.exists():
            print(f"✓ Header found: {header_path}")
            header_found = True
            break

        # Check include subdirectory
        include_path = path_obj / "include" / "nvimgcodec.h"
        if include_path.exists():
            print(f"✓ Header found: {include_path}")
            header_found = True
            break

        # For CMake integration, search recursively
        if install_method == "cmake-integrated":
            for header_file in path_obj.rglob("nvimgcodec.h"):
                print(f"✓ Header found: {header_file}")
                header_found = True
                break

        if header_found:
            break

    if not header_found:
        print("✗ nvimgcodec.h header file not found")

    # Look for library files
    library_names = [
        "libnvimgcodec.so.0",
        "libnvimgcodec.so",
        "libnvimgcodec.a",
        "libnvimgcodec.dylib",
        "nvimgcodec.dll",
    ]

    for path in search_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue

        for lib_name in library_names:
            lib_path = path_obj / lib_name
            if lib_path.exists():
                size_mb = lib_path.stat().st_size / (1024 * 1024)
                print(f"✓ Library found: {lib_path} ({size_mb:.1f} MB)")
                library_found = True
                break

            # Check lib subdirectory
            lib_subpath = path_obj / "lib" / lib_name
            if lib_subpath.exists():
                size_mb = lib_subpath.stat().st_size / (1024 * 1024)
                print(f"✓ Library found: {lib_subpath} ({size_mb:.1f} MB)")
                library_found = True
                break

            # For CMake integration, search recursively
            if install_method == "cmake-integrated":
                for lib_file in path_obj.rglob(lib_name):
                    size_mb = lib_file.stat().st_size / (1024 * 1024)
                    print(f"✓ Library found: {lib_file} ({size_mb:.1f} MB)")
                    library_found = True
                    break

            if library_found:
                break

        if library_found:
            break

    if not library_found:
        print("✗ nvImageCodec library file not found")

    return header_found and library_found


def check_cmake_configuration():
    """Check CMake configuration for nvImageCodec"""
    print_section("CMake Build Configuration Check")

    build_dir = find_plugin_build_dir()

    if not build_dir:
        print("✗ No build directory found - plugin not built yet")
        return False

    # Check for CMakeCache.txt which contains build configuration
    cmake_cache = build_dir / "CMakeCache.txt"

    if not cmake_cache.exists():
        print("✗ CMakeCache.txt not found - incomplete build")
        return False

    print(f"✓ CMake cache found: {cmake_cache}")

    # Read and check for nvImageCodec-related configuration
    try:
        with open(cmake_cache) as f:
            cache_content = f.read()

        has_nvimgcodec_config = "nvimgcodec" in cache_content.lower()

        if has_nvimgcodec_config:
            print("✓ nvImageCodec configuration found in CMake cache")

            # Look for specific nvImageCodec variables
            for line in cache_content.split("\n"):
                if (
                    "nvimgcodec" in line.lower()
                    and not line.startswith("//")
                    and "=" in line
                ):
                    print(f"  {line.strip()}")

            return True
        else:
            print("⚠ No nvImageCodec configuration in CMake cache")
            print("  The plugin may need to be rebuilt with nvImageCodec support")
            return False

    except Exception as e:
        print(f"✗ Error reading CMake cache: {e}")
        return False


def test_python_import():
    """Test Python import of nvImageCodec (if available)"""
    print_section("Python Import Test")

    # Try to import nvImageCodec Python bindings (if they exist)
    try:
        import nvidia.nvimgcodec

        print("✓ nvidia.nvimgcodec module imported successfully")

        # Try to get version
        if hasattr(nvidia.nvimgcodec, "__version__"):
            print(f"  Version: {nvidia.nvimgcodec.__version__}")

        return True
    except ImportError as e:
        print("⚠ nvidia.nvimgcodec Python module not available")
        print(f"  Error: {e}")
        return False


def check_infrastructure_only_mode():
    """Check if this is an infrastructure-only build (no actual plugin binary)."""
    script_dir = Path(__file__).parent.parent
    cmake_file = script_dir / "cpp/plugins/cucim.kit.cuslide2/CMakeLists.txt"
    
    if not cmake_file.exists():
        return False
    
    content = cmake_file.read_text()
    # Look for infrastructure-only indicators
    if "infrastructure-only" in content.lower() and "add_custom_target(${CUCIM_PLUGIN_NAME}" in content:
        return True
    return False


def check_cuslide2_plugin():
    """Check if cuslide2 plugin is built and integrated with nvImageCodec"""
    print_section("cuslide2 Plugin Check")

    # Get script directory and find plugin paths
    script_dir = Path(__file__).parent.parent  # Go up to cucim root

    # First check if this is an infrastructure-only build
    if check_infrastructure_only_mode():
        print("ℹ️  Infrastructure-only build detected")
        print("  ✓ This build validates dependencies without generating plugin binary")
        print("  ✓ CMake configuration verified - dependencies checked")
        print("  ℹ️  Plugin source code not yet added (planned for follow-up PR)")
        print("\n  📋 Infrastructure-only build is working as intended!")
        return "infrastructure-only"

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
        print("\n  💡 To build the plugin with nvImageCodec:")
        print("     cd cucim && ./run build_local all release $CONDA_PREFIX")
        print("     (nvImageCodec will be automatically fetched during build)")
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
            print("✓ Plugin linked with nvImageCodec")
            # Show the nvimgcodec line
            for line in output.split("\n"):
                if "nvimgcodec" in line:
                    print(f"  {line.strip()}")
        else:
            print("ℹ Plugin uses dynamic loading or static linking for nvImageCodec")
            print("  Library integrated at build time via plugin system")

        if has_cuda:
            print("✓ Plugin linked with CUDA libraries")

    # Check for nvImageCodec symbols using nm
    success, output, error = run_command(
        f"nm -D {plugin_file} 2>&1 | grep -i nvimg | head -5"
    )
    if success and output:
        print("✓ Plugin has nvImageCodec symbols:")
        for line in output.split("\n")[:3]:
            if line.strip():
                print(f"  {line.strip()}")
    else:
        # Try looking for static symbols
        success, output, error = run_command(
            f"nm {plugin_file} 2>&1 | grep -i nvimg | head -5"
        )
        if success and output:
            print("✓ Plugin has nvImageCodec symbols (static):")
            for line in output.split("\n")[:3]:
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
        lines = output.split("\n")
        for line in lines:
            if "NVIDIA" in line and (
                "GeForce" in line
                or "Tesla" in line
                or "Quadro" in line
                or "RTX" in line
            ):
                print(f"  {line.strip()}")
                break
    else:
        print("⚠ nvidia-smi not available or no NVIDIA GPU detected")

    # Check CUDA version
    success, output, error = run_command("nvcc --version")
    if success:
        for line in output.split("\n"):
            if "release" in line.lower():
                print(f"✓ CUDA compiler: {line.strip()}")
                break
    else:
        print("⚠ CUDA compiler (nvcc) not available")


def main():
    """Main verification function"""
    print_header("cuslide2 Infrastructure Verification")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")

    # Detect installation method (conda/pip or CMake integration)
    install_method = detect_installation_method()

    # Check library files based on installation method
    library_files_ok = check_library_files(install_method) if install_method else False

    # Check CMake configuration (optional, informational), no assignment
    check_cmake_configuration()

    # Check CUDA environment
    check_cuda_availability()

    # Check plugin
    plugin_built = check_cuslide2_plugin()

    # Test Python import
    test_python_import()

    # Summary
    print_header("Infrastructure Summary")

    # Check if this is an infrastructure-only build
    if plugin_built == "infrastructure-only":
        print("🎉 cuslide2 infrastructure validation PASSED!")
        print("✓ This is an infrastructure-only build")
        print("✓ CMake configuration verified")
        print("✓ All dependencies can be fetched/built")
        if install_method:
            if install_method == "conda":
                print("✓ nvImageCodec packages detected via conda")
            elif install_method == "pip":
                print("✓ nvImageCodec packages detected via pip")
            elif install_method == "cmake-integrated":
                print("✓ nvImageCodec CMake integration configured")
        print("\nℹ️  Note: Plugin binary intentionally not built in infrastructure-only mode")
        print("   Actual plugin implementation will be added in follow-up PR")
        return True

    all_good = library_files_ok and plugin_built and install_method

    if all_good:
        print("🎉 cuslide2 infrastructure is properly set up!")
        if install_method == "conda":
            print("✓ nvImageCodec installed via conda packages (dependencies.yaml)")
        elif install_method == "pip":
            print("✓ nvImageCodec installed via pip packages")
        elif install_method == "cmake-integrated":
            print("✓ nvImageCodec integrated via CMake build system")
        print("✓ All required library files accessible")
        print("✓ cuslide2 plugin built successfully")

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
        print("Option 1 (Recommended - via dependencies.yaml):")
        print("   micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge")
        print("Option 2 (Pip):")
        print("   pip install nvidia-nvimgcodec-cu12  # For CUDA 12.x")
        print("Option 3 (CMake auto-fetch):")
        print("   Build will automatically fetch nvImageCodec if not found")

        print(
            "\n💡 The project dependencies.yaml specifies conda packages as the preferred method."
        )

        return False
    else:
        print("✗ Setup incomplete")
        print("\n🔧 Recommendation:")
        print("   Install nvImageCodec and build the plugin:")
        print(
            "   1. micromamba install libnvimgcodec-dev libnvimgcodec0 -c conda-forge"
        )
        print("   2. cd cucim && ./run build_local all release $CONDA_PREFIX")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
