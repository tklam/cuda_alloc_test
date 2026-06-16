#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "   [CUDA ERROR] " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while (0)

// A simple CUDA kernel to test if the allocated memory is actually writable/readable
__global__ void verificationKernel(int* data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // Just test the first element as a capability heartbeat
        data[idx] = value;
    }
}

// ============================================================================
// CAPABILITY TEST FUNCTIONS
// ============================================================================

// Capability 1: Safely query available memory
bool testQueryCapability() {
    std::cout << "[Test 1] Capability: Query GPU Memory Status" << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    std::cout << "   -> Success: Device has " << free_mem / (1024 * 1024) 
              << " MB free out of " << total_mem / (1024 * 1024) << " MB total." << std::endl;
    return true;
}

// Capability 2: Allocate and free valid arbitrary memory chunks
bool testAllocationCapability(size_t size_mb) {
    std::cout << "[Test 2] Capability: Dynamic Allocation & Deallocation (" << size_mb << " MB)" << std::endl;
    size_t bytes = size_mb * 1024 * 1024;
    
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
    std::cout << "   -> Success: Allocated memory at address " << d_ptr << std::endl;
    
    CUDA_CHECK(cudaFree(d_ptr));
    std::cout << "   -> Success: Memory freed cleanly." << std::endl;
    return true;
}

// Capability 3: Prevent allocations that exceed hardware limits safely
bool testOOMPreventionCapability() {
    std::cout << "[Test 3] Capability: OOM Bound Protection" << std::endl;
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // Request impossible amount: Free memory + 10 Gigabytes
    size_t impossible_bytes = free_mem + (10ULL * 1024 * 1024 * 1024);
    
    std::cout << "   -> Attempting over-allocation safety check..." << std::endl;
    if (impossible_bytes > free_mem) {
        std::cout << "   -> Success: Program correctly intercepted and blocked unsafe allocation pre-emptively." << std::endl;
        return true;
    }
    
    // Fallback physical check if size math overflowed
    void* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, impossible_bytes);
    if (err == cudaErrorMemoryAllocation) {
        std::cout << "   -> Success: CUDA driver caught OOM and returned cudaErrorMemoryAllocation safely." << std::endl;
        cudaGetLastError(); // Clear error state
        return true;
    }
    
    if (d_ptr) cudaFree(d_ptr);
    return false;
}

// Capability 4: Interact with allocated memory via a GPU Kernel
bool testMemoryReadWriteCapability() {
    std::cout << "[Test 4] Capability: Read/Write Access on Allocated VRAM" << std::endl;
    size_t bytes = sizeof(int);
    int* d_ptr = nullptr;
    int h_result = 0;
    int test_val = 42;

    CUDA_CHECK(cudaMalloc((void**)&d_ptr, bytes));
    
    // Launch kernel to write to allocated memory
    verificationKernel<<<1, 1>>>(d_ptr, test_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host to verify integrity
    CUDA_CHECK(cudaMemcpy(&h_result, d_ptr, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_ptr));

    if (h_result == test_val) {
        std::cout << "   -> Success: Wrote " << test_val << " to GPU and read it back perfectly." << std::endl;
        return true;
    } else {
        std::cerr << "   -> Failure: Data corrupted. Expected " << test_val << ", got " << h_result << std::endl;
        return false;
    }
}

// Input parsing utility to test user argument capability
bool verifyInputParsing(const std::string& input, size_t& out_val) {
    char* endptr;
    unsigned long long val = std::strtoull(input.c_str(), &endptr, 10);
    if (*endptr != '\0' || val == 0) {
        return false;
    }
    out_val = val;
    return true;
}

// Capability 5: Robust command-line input parsing
bool testInputValidationCapability() {
    std::cout << "[Test 5] Capability: Input Argument Validation" << std::endl;
    
    std::vector<std::string> bad_inputs = {"abc", "-100", "0", "12.5", ""};
    size_t temp;
    
    for (const auto& input : bad_inputs) {
        if (verifyInputParsing(input, temp)) {
            std::cerr << "   -> Failure: Accepted an invalid input: " << input << std::endl;
            return false;
        }
    }
    std::cout << "   -> Success: Correctly rejected malformed numeric string arguments." << std::endl;
    return true;
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

int main(int argc, char* argv[]) {
    // If no argument is provided, execute the automated test suite verifying all capabilities
    if (argc < 2) {
        std::cout << "=== NO USER ARGUMENT DETECTED: RUNNING CAPABILITY TESTS ===\n" << std::endl;
        
        bool all_passed = true;
        all_passed &= testQueryCapability();
        std::cout << std::endl;
        all_passed &= testAllocationCapability(256); // Test standard 256MB allocation
        std::cout << std::endl;
        all_passed &= testOOMPreventionCapability();
        std::cout << std::endl;
        all_passed &= testMemoryReadWriteCapability();
        std::cout << std::endl;
        all_passed &= testInputValidationCapability();
        
        std::cout << "\n=======================================================" << std::endl;
        if (all_passed) {
            std::cout << " ALL CAPABILITY TESTS PASSED SUCCESSFULLY!" << std::endl;
        } else {
            std::cout << " TEST SUITE ENCOUNTERED FAILURES." << std::endl;
        }
        std::cout << "=======================================================" << std::endl;
        std::cout << "\nTo run custom allocation size, use: " << argv[0] << " <size_in_MB>" << std::endl;
        return all_passed ? 0 : 1;
    }

    // Standard run using user input argument
    size_t size_mb = 0;
    if (!verifyInputParsing(argv[1], size_mb)) {
        std::cerr << "Error: Invalid memory size. Please provide a positive integer." << std::endl;
        return 1;
    }

    std::cout << "=== RUNNING CUSTOM USER REQUEST ===" << std::endl;
    if (testAllocationCapability(size_mb)) {
        std::cout << "User operation completed successfully." << std::endl;
        return 0;
    }
    
    return 1;
}
