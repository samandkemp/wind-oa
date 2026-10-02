#include "windoa/device.hpp"

#include <stdexcept>

#include <vulkan/vulkan.h>

namespace windoa {

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(r)) + ")");
    }
}

} // namespace

GpuInfo describe_gpu(VkPhysicalDevice pd) {
    GpuInfo g;

    // Feature / property chains: 1.1 and 1.2 structs carry the f16 and
    // subgroup facts the P4 performance plan depends on.
    VkPhysicalDeviceSubgroupProperties sub{};
    sub.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &sub;
    vkGetPhysicalDeviceProperties2(pd, &props);

    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.pNext = &f11;
    VkPhysicalDeviceFeatures2 feats{};
    feats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feats.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(pd, &feats);

    g.name = props.properties.deviceName;
    g.api_version = props.properties.apiVersion;
    g.discrete = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    g.subgroup_size = sub.subgroupSize;
    g.shader_float16 = f12.shaderFloat16 == VK_TRUE;
    g.storage_16bit = f11.storageBuffer16BitAccess == VK_TRUE;
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(pd, VK_FORMAT_R32_SFLOAT, &fp);
    g.r32f_linear_filter =
        (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;

    std::uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    for (std::uint32_t i = 0; i < nq; ++i) {
        const bool compute = (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        const bool graphics = (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (compute && g.compute_queue_family == UINT32_MAX)
            g.compute_queue_family = i;
        if (compute && !graphics && g.async_compute_family == UINT32_MAX)
            g.async_compute_family = i;
    }

    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (std::uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
        if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            g.device_local_bytes += mem.memoryHeaps[h].size;
    }
    return g;
}

std::vector<GpuInfo> enumerate_gpus() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "wind-oa";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
#ifdef WINDOA_VALIDATION_LAYERS
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    ci.enabledLayerCount = 1;
    ci.ppEnabledLayerNames = layers;
#endif

    VkInstance inst = VK_NULL_HANDLE;
    check(vkCreateInstance(&ci, nullptr, &inst), "vkCreateInstance");

    std::vector<GpuInfo> out;
    std::uint32_t n = 0;
    VkResult r = vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    if (r == VK_SUCCESS && n > 0)
        r = vkEnumeratePhysicalDevices(inst, &n, pds.data());
    if (r == VK_SUCCESS) {
        for (VkPhysicalDevice pd : pds)
            out.push_back(describe_gpu(pd));
    }
    vkDestroyInstance(inst, nullptr);
    check(r, "vkEnumeratePhysicalDevices");
    return out;
}

} // namespace windoa
