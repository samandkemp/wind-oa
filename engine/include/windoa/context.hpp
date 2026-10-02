// The engine's Vulkan compute context and the two objects every solver is
// built from: a storage Buffer and a ComputeKernel. Headless -- no window,
// no swapchain of its own.
//
// A window app shares this Context (ContextOptions::graphics) and adds its
// own swapchain on top; see app/.
//
// Ownership: every object here holds raw Vulkan handles and frees them in
// its destructor (RAII), so copying is deleted -- two owners would free the
// same handle twice. Destroy kernels and buffers before their Context
// (declare the Context first; C++ destroys members / locals in reverse).
//
// Memory is allocated with plain vkAllocateMemory, one allocation per
// buffer. The engine holds a handful of large buffers, so a sub-allocator
// (VMA) would buy nothing here -- and it is one less dependency.
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.h>

#include "windoa/device.hpp"

namespace windoa {

#ifdef WINDOA_VALIDATION_LAYERS
inline constexpr bool kValidationDefault = true;
#else
inline constexpr bool kValidationDefault = false;
#endif

// Throws std::runtime_error naming `what` unless r == VK_SUCCESS.
void vk_check(VkResult r, const char* what);

class Buffer;

// How to build a Context. The defaults are the headless engine. A window
// app passes its surface extensions as plain names and asks for a graphics
// queue -- the engine never sees a window type, so it stays headless.
struct ContextOptions {
    bool validation_layers = kValidationDefault;
    std::vector<const char*> instance_extensions;
    std::vector<const char*> device_extensions;
    // The main queue must also do graphics (and the Vulkan 1.3 dynamic
    // rendering + synchronization2 features are enabled).
    bool graphics = false;
    // With graphics: put the engine's work (submit_and_wait) on the async
    // compute family, when the GPU has one, so a solver running on it never
    // queues behind the frames (graphics_queue()) and vice versa.
    bool async_compute = false;
};

class Context {
  public:
    explicit Context(bool validation_layers = kValidationDefault)
        : Context(ContextOptions{validation_layers}) {}
    explicit Context(const ContextOptions& opts);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance() const { return instance_; }
    VkDevice device() const { return device_; }
    VkPhysicalDevice physical() const { return physical_; }
    VkQueue queue() const { return queue_; } // the engine's queue (submit_and_wait)
    // The window's queue: queue() unless async_compute split them.
    VkQueue graphics_queue() const { return graphics_queue_; }
    std::uint32_t graphics_family() const { return graphics_family_; }
    // Families a buffer is shared between: empty (one family) or both.
    std::span<const std::uint32_t> sharing_families() const {
        return family_ == graphics_family_ ? std::span<const std::uint32_t>{}
                                           : std::span<const std::uint32_t>{sharing_, 2};
    }
    const GpuInfo& gpu() const { return gpu_; }
    std::uint32_t queue_family() const { return family_; }
    std::uint32_t max_group_count_x() const { return max_groups_x_; }

    // Record commands, submit them, block until the GPU is done. Every batch
    // is bracketed by full memory barriers, so work in one batch always sees
    // the writes of the previous one and the host sees everything after the
    // wait. The interactive loop will get its own non-blocking path.
    // Thread-safe (serialised): a solver thread and the UI thread may both
    // submit. The overload signals a timeline semaphore to `value` when the
    // batch completes -- the hand-off of its results to another queue.
    void submit_and_wait(const std::function<void(VkCommandBuffer)>& record);
    void submit_and_wait(const std::function<void(VkCommandBuffer)>& record, VkSemaphore signal,
                         std::uint64_t value);

    // Timeline semaphores (caller destroys with vkDestroySemaphore).
    VkSemaphore create_timeline(std::uint64_t initial = 0) const;
    void wait_timeline(VkSemaphore s, std::uint64_t value) const; // host wait

    // Host <-> device copies through a temporary staging buffer (blocking).
    void upload(Buffer& dst, const void* src, VkDeviceSize bytes, VkDeviceSize dst_offset = 0);
    void download(const Buffer& src, void* dst, VkDeviceSize bytes, VkDeviceSize src_offset = 0);
    void fill_zero(Buffer& dst) { fill_u32(dst, 0u); }
    void fill_u32(Buffer& dst, std::uint32_t value); // every 32-bit word = value

    // Memory type with all of `want`, among the types allowed by `bits`;
    // returns UINT32_MAX when there is none.
    std::uint32_t find_memory_type(std::uint32_t bits, VkMemoryPropertyFlags want) const;

    // Pipeline barriers, named for what they order.
    static void barrier_compute_to_compute(VkCommandBuffer cmd);
    static void barrier_full(VkCommandBuffer cmd);

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    std::uint32_t family_ = 0;
    std::uint32_t graphics_family_ = 0;
    std::uint32_t sharing_[2] = {};
    std::mutex submit_mutex_;
    std::uint32_t max_groups_x_ = 65535;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_{};
    GpuInfo gpu_;
};

enum class MemoryUse {
    Device,   // device-local; the solvers' fields (fastest for the GPU)
    Upload,   // host-visible, coherent; staging for host -> device
    Readback, // host-visible, coherent, cached if possible; device -> host
};

// A storage buffer. Upload / Readback buffers are persistently mapped.
class Buffer {
  public:
    Buffer(Context& ctx, VkDeviceSize bytes, MemoryUse use = MemoryUse::Device);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    bool mapped() const { return mapped_ != nullptr; }
    void* data() const { return mapped_; }

    // Typed view of the mapped memory (host-visible buffers only).
    template <class T> std::span<T> as() const {
        if (!mapped_)
            throw std::runtime_error("Buffer::as: buffer is not host-visible");
        return {static_cast<T*>(mapped_), static_cast<std::size_t>(size_ / sizeof(T))};
    }

  private:
    VkDevice device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
};

// Workgroup counts for n items at `local` threads per group, split over x
// and y when x alone would exceed the device limit. The shader rebuilds the
// linear group id as gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x.
struct Groups {
    std::uint32_t x = 1, y = 1, total = 1;
};
Groups groups_for(const Context& ctx, std::uint64_t n, std::uint32_t local);

// A 3-D R32F texture for hardware trilinear sampling in compute shaders
// (the renderer's ray march): the texture units fetch and weight the 8
// corners in one instruction, from 3-D-tiled memory. Filled by copying a
// storage buffer in the engine's cell order, [x][y][z] with z fastest --
// so the image extent is (nz, ny, nx) and a shader samples it at
// vec3(z, y, x) / vec3(nz, ny, nx) (normalised coordinates; cell centres
// at (i + 0.5) / n; clamp to edge). Lives in VK_IMAGE_LAYOUT_GENERAL, so a
// copy and the sampling need only memory barriers. Owns its linear
// sampler. Shared between the engine and graphics families like Buffer.
class Image3D {
  public:
    Image3D(Context& ctx, int nx, int ny, int nz);
    ~Image3D();
    Image3D(const Image3D&) = delete;
    Image3D& operator=(const Image3D&) = delete;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    // Record the copy of `src` (nx ny nz floats, cell order) into the image.
    void record_copy_from(VkCommandBuffer cmd, const Buffer& src) const;

  private:
    VkDevice device_;
    VkExtent3D extent_{};
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

// Descriptor kinds a ComputeKernel binding can hold.
enum class Slot : std::uint8_t {
    Buffer,  // storage buffer
    Texture, // combined image sampler (an Image3D)
};

// One bound resource: a Buffer or an Image3D (implicitly converted).
struct Resource {
    const Buffer* buffer = nullptr;
    const Image3D* image = nullptr;
    Resource(const Buffer* b) : buffer(b) {}
    Resource(const Image3D* i) : image(i) {}
};

// One compute pipeline: storage buffers at set 0, bindings 0..n-1, an
// optional push-constant block, and specialisation constants 0..k-1
// (compile-time feature switches: the driver folds them, so a disabled
// feature costs nothing). `n_sets` descriptor sets let one pipeline
// alternate between buffer bindings, e.g. the A -> B / B -> A ping-pong of
// the LBM.
class ComputeKernel {
  public:
    ComputeKernel(Context& ctx, std::span<const std::uint32_t> spirv, std::uint32_t n_buffers,
                  std::uint32_t push_bytes, std::uint32_t n_sets = 1,
                  std::span<const std::uint32_t> spec = {});
    // Mixed bindings: slot kinds in binding order.
    ComputeKernel(Context& ctx, std::span<const std::uint32_t> spirv, std::span<const Slot> slots,
                  std::uint32_t push_bytes, std::uint32_t n_sets = 1,
                  std::span<const std::uint32_t> spec = {});
    ~ComputeKernel();
    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    // Point set `set`'s bindings 0..n-1 at these buffers. Must not be called
    // while a command buffer using that set is still executing.
    void bind(std::uint32_t set, std::initializer_list<const Buffer*> buffers);
    void bind(std::initializer_list<const Buffer*> buffers) { bind(0, buffers); }
    // Mixed buffers and textures, in binding order.
    void bind_resources(std::uint32_t set, std::initializer_list<Resource> resources);

    // Record one dispatch.
    void record_set(VkCommandBuffer cmd, std::uint32_t set, const void* push, std::uint32_t gx,
                    std::uint32_t gy = 1, std::uint32_t gz = 1) const;
    void record(VkCommandBuffer cmd, const void* push, std::uint32_t gx, std::uint32_t gy = 1,
                std::uint32_t gz = 1) const {
        record_set(cmd, 0, push, gx, gy, gz);
    }

  private:
    void bind_list(std::uint32_t set, const std::vector<Resource>& res);

    VkDevice device_;
    std::vector<Slot> slots_;
    std::uint32_t push_bytes_;
    std::uint32_t n_sets_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet sets_[4] = {};
};

} // namespace windoa
