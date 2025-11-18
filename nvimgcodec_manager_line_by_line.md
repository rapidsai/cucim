# Line-by-Line Description: nvimgcodec_manager.h

## File Overview
This header file defines a singleton manager class for nvImageCodec resources, providing thread-safe centralized access to the nvImageCodec instance and decoder.

---

## Detailed Line-by-Line Breakdown

### Lines 1-15: Copyright and License Header
```cpp
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 * ...
 */
```
**Description:** Standard NVIDIA copyright notice with Apache 2.0 license information. Establishes legal ownership and usage terms for the code.

---

### Line 17: Include Guard
```cpp
#pragma once
```
**Description:** Modern C++ include guard that prevents multiple inclusion of this header file during compilation. More concise than traditional `#ifndef`/`#define`/`#endif` guards.

---

### Lines 19-21: Conditional nvImageCodec Include
```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
#include <nvimgcodec.h>
#endif
```
**Description:** 
- **Line 19:** Preprocessor conditional checking if nvImageCodec support is enabled
- **Line 20:** Includes the main nvImageCodec library header if support is enabled
- **Line 21:** Closes the conditional block

This allows the code to compile even when nvImageCodec is not available.

---

### Lines 23-25: Standard Library Headers
```cpp
#include <string>
#include <mutex>
#include <fmt/format.h>
```
**Description:**
- **Line 23:** Includes `<string>` for `std::string` usage
- **Line 24:** Includes `<mutex>` for `std::mutex` used in thread synchronization
- **Line 25:** Includes fmt library for modern C++ string formatting (alternative to printf/iostreams)

---

### Lines 27-28: Namespace Declaration
```cpp
namespace cuslide2::nvimgcodec
{
```
**Description:** Opens a nested namespace `cuslide2::nvimgcodec`. This organizes code to avoid naming conflicts and indicates this code belongs to the cuslide2 project's nvImageCodec integration layer.

---

### Line 30: Conditional Compilation Guard
```cpp
#ifdef CUCIM_HAS_NVIMGCODEC
```
**Description:** Begins a large conditional block. All the following code up to line 173 will only be compiled if `CUCIM_HAS_NVIMGCODEC` is defined, allowing the codebase to work with or without nvImageCodec support.

---

### Lines 32-36: Class Documentation
```cpp
/**
 * @brief Singleton manager for nvImageCodec instance and decoder
 * 
 * Provides centralized access to nvImageCodec resources with thread-safe initialization.
 */
```
**Description:** Doxygen-style documentation comment explaining the class purpose. Clarifies that this is a singleton pattern implementation providing thread-safe access to nvImageCodec resources.

---

### Lines 37-38: Class Declaration
```cpp
class NvImageCodecManager
{
```
**Description:** Declares the `NvImageCodecManager` class. This will be the singleton that manages the lifecycle of nvImageCodec resources.

---

### Line 39: Public Interface Section
```cpp
public:
```
**Description:** Begins the public interface section containing methods accessible to external code.

---

### Lines 40-44: Singleton Instance Accessor
```cpp
static NvImageCodecManager& instance()
{
    static NvImageCodecManager instance;
    return instance;
}
```
**Description:**
- **Line 40:** Static method returning a reference to the singleton instance
- **Line 42:** Creates a static local variable - guaranteed to be initialized exactly once in a thread-safe manner (C++11 "magic statics")
- **Line 43:** Returns reference to the single instance
- This is the Meyer's Singleton pattern - lazy initialization with automatic lifetime management

---

### Line 46: Instance Getter
```cpp
nvimgcodecInstance_t get_instance() const { return instance_; }
```
**Description:** Const getter method that returns the raw nvImageCodec instance handle. Inline one-liner for performance.

---

### Line 47: Decoder Getter
```cpp
nvimgcodecDecoder_t get_decoder() const { return decoder_; }
```
**Description:** Const getter method that returns the raw nvImageCodec decoder handle. Also inline for performance.

---

### Line 48: Mutex Getter
```cpp
std::mutex& get_mutex() { return decoder_mutex_; }
```
**Description:** Returns a reference to the internal mutex. This allows external code to synchronize access to the decoder when performing operations. Non-const because locking modifies mutex state.

---

### Line 49: Initialization Status Check
```cpp
bool is_initialized() const { return initialized_; }
```
**Description:** Returns whether the manager successfully initialized nvImageCodec resources. Const method returning the boolean flag.

---

### Line 50: Status Message Getter
```cpp
const std::string& get_status() const { return status_message_; }
```
**Description:** Returns a const reference to the status message string, which contains success/failure information from initialization. Avoids copying the string.

---

### Lines 52-87: API Validation Test Method
```cpp
// Quick API validation test
bool test_nvimagecodec_api()
```
**Description:** Method header and comment for testing nvImageCodec API functionality.

---

### Lines 54-55: Early Return Check
```cpp
if (!initialized_) return false;
```
**Description:** Guard clause - if the manager didn't initialize successfully, return false immediately. No point testing an uninitialized API.

---

### Lines 57-63: Test Setup - Properties Structure
```cpp
try {
    // Test 1: Get nvImageCodec properties
    nvimgcodecProperties_t props{};
    props.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
    props.struct_size = sizeof(nvimgcodecProperties_t);
    props.struct_next = nullptr;
```
**Description:**
- **Line 57:** Begin try block for exception safety
- **Line 59:** Initialize a properties structure with zero-initialization
- **Lines 60-62:** Set up the structure following nvImageCodec's API pattern:
  - `struct_type`: Identifies the structure type (used for API versioning)
  - `struct_size`: Size validation for ABI compatibility
  - `struct_next`: Pointer for structure chaining (extensibility mechanism)

---

### Lines 64-72: Test 1 - Get Version Info
```cpp
if (nvimgcodecGetProperties(&props) == NVIMGCODEC_STATUS_SUCCESS)
{
    uint32_t version = props.version;
    uint32_t major = (version >> 16) & 0xFF;
    uint32_t minor = (version >> 8) & 0xFF;
    uint32_t patch = version & 0xFF;
    
    fmt::print("✅ nvImageCodec API Test: Version {}.{}.{}\n", major, minor, patch);
```
**Description:**
- **Line 64:** Calls nvImageCodec API to get properties, checks for success
- **Line 66:** Extracts version number from properties
- **Lines 67-69:** Decodes packed version number using bit shifting and masking:
  - Bits 16-23: Major version
  - Bits 8-15: Minor version
  - Bits 0-7: Patch version
- **Line 71:** Prints success message with version info using checkmark emoji

---

### Lines 73-78: Test 2 - Decoder Check
```cpp
// Test 2: Check decoder capabilities
if (decoder_)
{
    fmt::print("✅ nvImageCodec Decoder: Ready\n");
    return true;
}
```
**Description:**
- **Line 74:** Checks if decoder handle is valid (non-null)
- **Line 76:** Prints success message
- **Line 77:** Returns true indicating all tests passed

---

### Lines 81-84: Exception Handling
```cpp
catch (const std::exception& e)
{
    fmt::print("⚠️  nvImageCodec API Test failed: {}\n", e.what());
}
```
**Description:** Catches any standard exceptions during testing and prints a warning message with the error details. Uses warning emoji to indicate non-critical failure.

---

### Line 86: Default Return
```cpp
return false;
```
**Description:** Returns false if tests failed or threw an exception.

---

### Lines 89-93: Deleted Copy/Move Operations
```cpp
// Disable copy/move
NvImageCodecManager(const NvImageCodecManager&) = delete;
NvImageCodecManager& operator=(const NvImageCodecManager&) = delete;
NvImageCodecManager(NvImageCodecManager&&) = delete;
NvImageCodecManager& operator=(NvImageCodecManager&&) = delete;
```
**Description:** 
- **Line 90:** Deletes copy constructor
- **Line 91:** Deletes copy assignment operator
- **Line 92:** Deletes move constructor
- **Line 93:** Deletes move assignment operator

These deletions enforce singleton semantics - there can only be one instance, so copying or moving it would violate the pattern.

---

### Line 95: Private Section
```cpp
private:
```
**Description:** Begins private section - implementation details not accessible to external code.

---

### Lines 96-98: Private Constructor Declaration
```cpp
NvImageCodecManager() : initialized_(false)
{
    try {
```
**Description:**
- **Line 96:** Private constructor (part of singleton pattern - only `instance()` can create the object)
- Initializer list sets `initialized_` to false
- **Line 98:** Begin try block for exception-safe initialization

---

### Lines 99-111: Instance Creation Setup
```cpp
// Create nvImageCodec instance following official API pattern
nvimgcodecInstanceCreateInfo_t create_info{};
create_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
create_info.struct_size = sizeof(nvimgcodecInstanceCreateInfo_t);
create_info.struct_next = nullptr;
create_info.load_builtin_modules = 1;
create_info.load_extension_modules = 1;
create_info.extension_modules_path = nullptr;
create_info.create_debug_messenger = 1;
create_info.debug_messenger_desc = nullptr;
create_info.message_severity = 0;
create_info.message_category = 0;
```
**Description:** Sets up the configuration structure for creating an nvImageCodec instance:
- **Line 100:** Zero-initializes the structure
- **Lines 101-103:** Standard structure preamble (type, size, next pointer)
- **Line 104:** Enable built-in codec modules
- **Line 105:** Enable extension modules
- **Line 106:** Use default extension path (nullptr)
- **Line 107:** Enable debug messenger for diagnostics
- **Line 108:** Use default debug messenger configuration
- **Lines 109-110:** Set message filtering (0 = default/all messages)

---

### Lines 112-117: Instance Creation and Error Handling
```cpp
if (nvimgcodecInstanceCreate(&instance_, &create_info) != NVIMGCODEC_STATUS_SUCCESS)
{
    status_message_ = "Failed to create nvImageCodec instance";
    fmt::print("❌ {}\n", status_message_);
    return;
}
```
**Description:**
- **Line 112:** Attempts to create the nvImageCodec instance, checks if it failed
- **Line 114:** Sets error status message
- **Line 115:** Prints error with X emoji
- **Line 116:** Early return leaves object in uninitialized state

---

### Lines 119-133: Decoder Creation Setup
```cpp
// Create decoder with execution parameters following official API pattern
nvimgcodecExecutionParams_t exec_params{};
exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
exec_params.struct_size = sizeof(nvimgcodecExecutionParams_t);
exec_params.struct_next = nullptr;
exec_params.device_allocator = nullptr;
exec_params.pinned_allocator = nullptr;
exec_params.max_num_cpu_threads = 0; // Use default
exec_params.executor = nullptr;
exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
exec_params.pre_init = 0;
exec_params.skip_pre_sync = 0;
exec_params.num_backends = 0;
exec_params.backends = nullptr;
```
**Description:** Configures execution parameters for the decoder:
- **Line 120:** Zero-initializes execution parameters structure
- **Lines 121-123:** Standard structure preamble
- **Lines 124-125:** Use default memory allocators (nullptr = default)
- **Line 126:** Use default number of CPU threads (0 = auto-detect)
- **Line 127:** Use default executor (nullptr)
- **Line 128:** Use current CUDA device
- **Line 129:** Don't pre-initialize codecs
- **Line 130:** Don't skip pre-synchronization
- **Lines 131-132:** Use default backends (nullptr/0 = all available)

---

### Lines 134-141: Decoder Creation and Error Handling
```cpp
if (nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr) != NVIMGCODEC_STATUS_SUCCESS)
{
    nvimgcodecInstanceDestroy(instance_);
    instance_ = nullptr;
    status_message_ = "Failed to create nvImageCodec decoder";
    fmt::print("❌ {}\n", status_message_);
    return;
}
```
**Description:**
- **Line 134:** Attempts to create decoder, checks for failure (last nullptr is for decode parameters)
- **Line 136:** Cleans up the instance since decoder creation failed
- **Line 137:** Nullifies the instance pointer for safety
- **Line 138:** Sets error status message
- **Line 139:** Prints error message
- **Line 140:** Early return with uninitialized state

---

### Lines 143-145: Success Path
```cpp
initialized_ = true;
status_message_ = "nvImageCodec initialized successfully";
fmt::print("✅ {}\n", status_message_);
```
**Description:**
- **Line 143:** Sets initialization flag to true - all resources created successfully
- **Line 144:** Sets success status message
- **Line 145:** Prints success message with checkmark

---

### Lines 147-148: Initial API Test
```cpp
// Run quick API test
test_nvimagecodec_api();
```
**Description:** Immediately runs the API validation test to verify the resources are working correctly.

---

### Lines 150-155: Exception Handler
```cpp
catch (const std::exception& e)
{
    status_message_ = fmt::format("nvImageCodec initialization exception: {}", e.what());
    fmt::print("❌ {}\n", status_message_);
    initialized_ = false;
}
```
**Description:** Catches any exceptions during initialization:
- **Line 152:** Formats detailed error message with exception details
- **Line 153:** Prints error message
- **Line 154:** Ensures initialized flag is false

---

### Lines 158-164: Destructor
```cpp
~NvImageCodecManager()
{
    // Intentionally NOT destroying resources to avoid crashes during Python interpreter shutdown
    // The OS will reclaim these resources when the process exits.
    // This is a workaround for nvJPEG2000 cleanup issues during static destruction.
    // Resources are only held in a singleton that lives for the entire program lifetime anyway.
}
```
**Description:** Private destructor with intentionally empty implementation. The detailed comment explains this is a workaround:
- **Line 159:** Destructor declaration
- **Lines 160-163:** Comment explaining the rationale: nvJPEG2000 has issues with cleanup during Python interpreter shutdown and static destruction order. Since this is a singleton that lives for the program lifetime, letting the OS reclaim resources on process exit is safer than explicit cleanup.

---

### Lines 166-170: Member Variables
```cpp
nvimgcodecInstance_t instance_{nullptr};
nvimgcodecDecoder_t decoder_{nullptr};
bool initialized_{false};
std::string status_message_;
std::mutex decoder_mutex_;  // Protect decoder operations from concurrent access
```
**Description:** Private member variables storing the manager's state:
- **Line 166:** nvImageCodec instance handle, initialized to nullptr
- **Line 167:** Decoder handle, initialized to nullptr
- **Line 168:** Initialization success flag, initialized to false
- **Line 169:** Status message string for diagnostics (default-initialized empty)
- **Line 170:** Mutex for thread-safe decoder access (with explanatory comment)

---

### Line 171: End of Class
```cpp
};
```
**Description:** Closes the `NvImageCodecManager` class definition.

---

### Line 173: End of Conditional Block
```cpp
#endif // CUCIM_HAS_NVIMGCODEC
```
**Description:** Closes the `#ifdef CUCIM_HAS_NVIMGCODEC` block started on line 30. Comment indicates which conditional is being closed.

---

### Line 175: Namespace Closure
```cpp
} // namespace cuslide2::nvimgcodec
```
**Description:** Closes the `cuslide2::nvimgcodec` namespace. Comment documents which namespace is being closed for clarity.

---

### Line 177: End of File
```cpp

```
**Description:** Blank line at end of file (good practice for POSIX text files).

---

## Key Design Patterns and Concepts

### 1. **Singleton Pattern (Meyer's Singleton)**
- Single static instance created on first access
- Thread-safe initialization guaranteed by C++11
- Private constructor prevents external instantiation
- Deleted copy/move operations enforce uniqueness

### 2. **RAII (Resource Acquisition Is Initialization)**
- Resources acquired in constructor
- Resources intentionally NOT released in destructor (workaround for library issues)
- Exception-safe initialization with try-catch blocks

### 3. **Conditional Compilation**
- Entire implementation wrapped in `#ifdef CUCIM_HAS_NVIMGCODEC`
- Allows codebase to compile with or without nvImageCodec support
- Clean separation of optional dependencies

### 4. **Thread Safety**
- Meyer's singleton provides thread-safe initialization
- Mutex member allows external synchronization of decoder operations
- Const methods for getters (no state modification)

### 5. **Modern C++ Practices**
- `pragma once` instead of traditional include guards
- In-class member initializers (`instance_{nullptr}`)
- `= delete` for explicitly deleted functions
- Zero-initialization with `{}`
- Const correctness throughout

### 6. **Error Handling Strategy**
- Status flag (`initialized_`) tracks success/failure
- Status message provides diagnostic information
- Exceptions caught and converted to status messages
- Early returns prevent partial initialization

### 7. **API Pattern Compliance**
- Carefully follows nvImageCodec's structure-based API pattern
- Proper initialization of struct_type, struct_size, struct_next
- Uses official constants and enums
- Comprehensive parameter configuration

---

## Usage Example

```cpp
// Access the singleton
auto& manager = NvImageCodecManager::instance();

// Check if initialization succeeded
if (manager.is_initialized()) {
    // Get the decoder for use
    auto decoder = manager.get_decoder();
    
    // Lock for thread-safe operations
    std::lock_guard<std::mutex> lock(manager.get_mutex());
    
    // Use decoder...
}
else {
    // Handle initialization failure
    std::cerr << "Error: " << manager.get_status() << std::endl;
}
```

---

## Summary

This header defines a robust, thread-safe singleton manager for nvImageCodec resources. It provides centralized initialization, error handling, and access control for the nvImageCodec library within the cuslide2 codebase. The implementation carefully follows both modern C++ best practices and nvImageCodec's API requirements, while including workarounds for known issues with library cleanup during process shutdown.

