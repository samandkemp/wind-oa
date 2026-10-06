// Owning wrappers for the Vulkan objects a VkDevice creates. Each frees its
// handle when destroyed, so a constructor that throws part-way leaks nothing
// it had already made: C++ destroys the members constructed so far. Members
// are destroyed in reverse order of declaration, so declare a resource after
// what it depends on (a buffer after its memory, a pipeline after its
// layout).
//
// The destroy function is a template argument wrapping the Vulkan call: the
// address of the loader's own (imported) function is not a constant
// expression, so it cannot be one directly.
#pragma once

#include <utility>

#include <vulkan/vulkan.h>

namespace windoa {

namespace vk_destroy {
inline void buffer(VkDevice d, VkBuffer h) {
    vkDestroyBuffer(d, h, nullptr);
}
inline void memory(VkDevice d, VkDeviceMemory h) {
    vkFreeMemory(d, h, nullptr); // also unmaps a mapped allocation
}
inline void image(VkDevice d, VkImage h) {
    vkDestroyImage(d, h, nullptr);
}
inline void image_view(VkDevice d, VkImageView h) {
    vkDestroyImageView(d, h, nullptr);
}
inline void sampler(VkDevice d, VkSampler h) {
    vkDestroySampler(d, h, nullptr);
}
inline void shader_module(VkDevice d, VkShaderModule h) {
    vkDestroyShaderModule(d, h, nullptr);
}
inline void set_layout(VkDevice d, VkDescriptorSetLayout h) {
    vkDestroyDescriptorSetLayout(d, h, nullptr);
}
inline void pipeline_layout(VkDevice d, VkPipelineLayout h) {
    vkDestroyPipelineLayout(d, h, nullptr);
}
inline void pipeline(VkDevice d, VkPipeline h) {
    vkDestroyPipeline(d, h, nullptr);
}
inline void descriptor_pool(VkDevice d, VkDescriptorPool h) {
    vkDestroyDescriptorPool(d, h, nullptr); // frees its sets too
}
inline void query_pool(VkDevice d, VkQueryPool h) {
    vkDestroyQueryPool(d, h, nullptr);
}
} // namespace vk_destroy

template <class T, void (*Destroy)(VkDevice, T)> class DeviceHandle {
  public:
    DeviceHandle() = default;
    explicit DeviceHandle(VkDevice device) : device_(device) {}
    ~DeviceHandle() { reset(); }
    DeviceHandle(const DeviceHandle&) = delete;
    DeviceHandle& operator=(const DeviceHandle&) = delete;
    DeviceHandle(DeviceHandle&& o) noexcept
        : device_(o.device_), handle_(std::exchange(o.handle_, T(VK_NULL_HANDLE))) {}
    DeviceHandle& operator=(DeviceHandle&& o) noexcept {
        if (this != &o) {
            reset();
            device_ = o.device_;
            handle_ = std::exchange(o.handle_, T(VK_NULL_HANDLE));
        }
        return *this;
    }

    T get() const { return handle_; }
    // The slot a vkCreate* call fills (any current handle is freed first).
    T* put() {
        reset();
        return &handle_;
    }
    void reset() {
        if (handle_ != VK_NULL_HANDLE)
            Destroy(device_, handle_);
        handle_ = VK_NULL_HANDLE;
    }

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    T handle_ = VK_NULL_HANDLE;
};

using BufferHandle = DeviceHandle<VkBuffer, vk_destroy::buffer>;
using MemoryHandle = DeviceHandle<VkDeviceMemory, vk_destroy::memory>;
using ImageHandle = DeviceHandle<VkImage, vk_destroy::image>;
using ImageViewHandle = DeviceHandle<VkImageView, vk_destroy::image_view>;
using SamplerHandle = DeviceHandle<VkSampler, vk_destroy::sampler>;
using ShaderModuleHandle = DeviceHandle<VkShaderModule, vk_destroy::shader_module>;
using SetLayoutHandle = DeviceHandle<VkDescriptorSetLayout, vk_destroy::set_layout>;
using PipelineLayoutHandle = DeviceHandle<VkPipelineLayout, vk_destroy::pipeline_layout>;
using PipelineHandle = DeviceHandle<VkPipeline, vk_destroy::pipeline>;
using DescriptorPoolHandle = DeviceHandle<VkDescriptorPool, vk_destroy::descriptor_pool>;
using QueryPoolHandle = DeviceHandle<VkQueryPool, vk_destroy::query_pool>;

} // namespace windoa
