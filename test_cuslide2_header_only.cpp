#include "cuslide2_cpp_header_only.hpp"

int main() {
    std::cout << "🧪 cuslide2 Header-Only C++ Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // Run the demo
        cuslide2::demo_cuslide2_cpp();
        
        std::cout << "\n🎯 Advanced Usage Example:" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Create reader for a specific file
        cuslide2::CuSlide2Reader reader("example_slide.svs");
        
        // Read different region sizes
        std::vector<int> sizes = {1024, 2048, 4096};
        
        for (int size : sizes) {
            std::cout << "\n📏 Testing " << size << "x" << size << " regions:" << std::endl;
            
            // CPU region
            auto cpu_region = reader.read_region_cpu(0, 0, size, size);
            std::cout << "   CPU region: " << cpu_region->width << "x" << cpu_region->height 
                      << " on " << cpu_region->device << std::endl;
            
            // GPU region  
            auto gpu_region = reader.read_region_gpu(0, 0, size, size);
            std::cout << "   GPU region: " << gpu_region->width << "x" << gpu_region->height 
                      << " on " << gpu_region->device << std::endl;
        }
        
        std::cout << "\n✅ Header-only C++ test completed successfully!" << std::endl;
        std::cout << "\n📝 This demonstrates cuslide2 concepts without full plugin build" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
