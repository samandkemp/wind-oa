// Headless Vulkan device discovery. The first thing the engine must do, and
// the P0 gate: the toolchain builds and a compute-capable GPU is visible.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace windoa {

struct GpuInfo {
    std::string name;
    std::uint32_t api_version = 0; // VK_MAKE_API_VERSION packed
    bool discrete = false;
    std::uint32_t compute_queue_family = UINT32_MAX; // UINT32_MAX = none
    std::uint32_t async_compute_family = UINT32_MAX; // compute without graphics
    std::uint64_t device_local_bytes = 0;
    bool shader_float16 = false; // f16 arithmetic (P4 storage plan)
    bool storage_16bit = false;  // 16-bit storage buffers
    std::uint32_t subgroup_size = 0;
    bool r32f_linear_filter = false; // 3-D R32F textures sampled with hardware trilinear
};

// Enumerate every physical device. Throws std::runtime_error if Vulkan
// itself is unavailable (no loader / no ICD).
std::vector<GpuInfo> enumerate_gpus();

// The capability facts for one physical device.
GpuInfo describe_gpu(VkPhysicalDevice pd);

} // namespace windoa
