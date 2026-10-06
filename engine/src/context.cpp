#include "windoa/context.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace windoa {

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(r)) + ")");
    }
}

// -- Context -----------------------------------------------------------------

// First queue family with compute and graphics, or UINT32_MAX.
static std::uint32_t graphics_compute_family(VkPhysicalDevice pd) {
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, fams.data());
    const VkQueueFlags want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (std::uint32_t i = 0; i < n; ++i) {
        if ((fams[i].queueFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

Context::Context(const ContextOptions& opts) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "wind-oa";
    app.apiVersion = VK_API_VERSION_1_3;

    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (opts.validation_layers) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = layers;
    }
    ici.enabledExtensionCount = static_cast<std::uint32_t>(opts.instance_extensions.size());
    ici.ppEnabledExtensionNames = opts.instance_extensions.data();
    vk_check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");
    try {

        // Pick the GPU: a discrete one with a compute queue beats anything else.
        std::uint32_t n = 0;
        vk_check(vkEnumeratePhysicalDevices(instance_, &n, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> pds(n);
        vk_check(vkEnumeratePhysicalDevices(instance_, &n, pds.data()),
                 "vkEnumeratePhysicalDevices");
        int best_score = 0;
        for (VkPhysicalDevice pd : pds) {
            GpuInfo g = describe_gpu(pd);
            if (opts.graphics)
                g.compute_queue_family = graphics_compute_family(pd);
            if (g.compute_queue_family == UINT32_MAX)
                continue;
            const int score = g.discrete ? 2 : 1;
            if (score > best_score) {
                best_score = score;
                physical_ = pd;
                gpu_ = g;
            }
        }
        if (physical_ == VK_NULL_HANDLE)
            throw std::runtime_error(opts.graphics
                                         ? "no Vulkan device with a graphics+compute queue"
                                         : "no Vulkan device with a compute queue");
        family_ = gpu_.compute_queue_family;
        graphics_family_ = family_;
        if (opts.graphics && opts.async_compute && gpu_.async_compute_family != UINT32_MAX)
            family_ = gpu_.async_compute_family; // engine work on its own queue
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem_);
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_, &props);
        max_groups_x_ = props.limits.maxComputeWorkGroupCount[0];

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci[2]{};
        for (VkDeviceQueueCreateInfo& q : qci) {
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueCount = 1;
            q.pQueuePriorities = &priority;
        }
        qci[0].queueFamilyIndex = family_;
        qci[1].queueFamilyIndex = graphics_family_;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = family_ != graphics_family_ ? 2u : 1u;
        dci.pQueueCreateInfos = qci;
        dci.enabledExtensionCount = static_cast<std::uint32_t>(opts.device_extensions.size());
        dci.ppEnabledExtensionNames = opts.device_extensions.data();
        // Features are switched on by chaining feature structs through pNext.
        // Timeline semaphores (core 1.2): cross-queue hand-offs, always on.
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.timelineSemaphore = VK_TRUE;
        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        // 16-bit storage buffers (1.1): the LBM kernels carry an f16 view of the
        // distributions (lbm::Config::storage_f16), so the capability is in
        // their SPIR-V whichever storage is chosen.
        VkPhysicalDeviceVulkan11Features f11{};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.storageBuffer16BitAccess = gpu_.storage_16bit ? VK_TRUE : VK_FALSE;
        f11.pNext = &f12;
        dci.pNext = &f11;
        if (opts.graphics) {
            f13.dynamicRendering = VK_TRUE; // no VkRenderPass / VkFramebuffer objects
            f13.synchronization2 = VK_TRUE;
            f12.pNext = &f13;
        }
        vk_check(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, family_, 0, &queue_);
        vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
        sharing_[0] = family_;
        sharing_[1] = graphics_family_;

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family_;
        vk_check(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device_, &cai, &cmd_), "vkAllocateCommandBuffers");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vk_check(vkCreateFence(device_, &fci, nullptr, &fence_), "vkCreateFence");
        for (AsyncSlot& a : async_) {
            vk_check(vkAllocateCommandBuffers(device_, &cai, &a.cmd), "vkAllocateCommandBuffers");
            vk_check(vkCreateFence(device_, &fci, nullptr, &a.fence), "vkCreateFence");
        }
    } catch (...) { // a destructor does not run for a half-built object
        release();
        throw;
    }
}

Context::~Context() {
    release();
}

void Context::release() {
    upload_staging_.reset(); // Buffers go before the device
    download_staging_.reset();
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyFence(device_, fence_, nullptr); // a null handle is a no-op
        for (AsyncSlot& a : async_)
            vkDestroyFence(device_, a.fence, nullptr);
        vkDestroyCommandPool(device_, pool_, nullptr); // frees cmd_ too
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void Context::submit_and_wait(const std::function<void(VkCommandBuffer)>& record) {
    submit_and_wait(record, VK_NULL_HANDLE, 0);
}

void Context::submit_and_wait(const std::function<void(VkCommandBuffer)>& record,
                              VkSemaphore signal, std::uint64_t value) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    record_batch(cmd_, record);
    vk_check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    submit_batch(cmd_, fence_, signal, value);
    vk_check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

std::uint64_t Context::submit_async(const std::function<void(VkCommandBuffer)>& record,
                                    VkSemaphore signal, std::uint64_t value) {
    std::lock_guard<std::mutex> lock(submit_mutex_);
    AsyncSlot& a = async_[async_tickets_ % kAsyncSlots];
    if (a.ticket != 0) // its last batch must be done before the buffer is reused
        vk_check(vkWaitForFences(device_, 1, &a.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    record_batch(a.cmd, record);
    vk_check(vkResetFences(device_, 1, &a.fence), "vkResetFences");
    submit_batch(a.cmd, a.fence, signal, value);
    a.ticket = ++async_tickets_;
    return a.ticket;
}

void Context::wait(std::uint64_t ticket) {
    for (AsyncSlot& a : async_)
        if (a.ticket == ticket) {
            vk_check(vkWaitForFences(device_, 1, &a.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
            return;
        }
    // not found: its slot has been reused, so it completed before that
}

// A fence wait makes a batch's writes available to the host, not visible to
// the next batch's shaders -- hence the barriers at both ends.
void Context::record_batch(VkCommandBuffer cmd,
                           const std::function<void(VkCommandBuffer)>& record) {
    vk_check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    barrier_full(cmd);
    record(cmd);
    barrier_full(cmd);
    vk_check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
}

// Optional timeline signal: the hand-off to another queue (the signal makes
// the batch's writes available to whoever waits on it there).
void Context::submit_batch(VkCommandBuffer cmd, VkFence fence, VkSemaphore signal,
                           std::uint64_t value) {
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkTimelineSemaphoreSubmitInfo ti{};
    ti.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ti.signalSemaphoreValueCount = 1;
    ti.pSignalSemaphoreValues = &value;
    if (signal != VK_NULL_HANDLE) {
        si.pNext = &ti;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &signal;
    }
    vk_check(vkQueueSubmit(queue_, 1, &si, fence), "vkQueueSubmit");
}

void Context::upload(Buffer& dst, const void* src, VkDeviceSize bytes, VkDeviceSize dst_offset) {
    if (bytes == 0)
        return;
    if (dst.mapped()) {
        std::memcpy(static_cast<char*>(dst.data()) + dst_offset, src, bytes);
        return;
    }
    std::unique_lock<std::mutex> lock(staging_mutex_, std::defer_lock);
    std::unique_ptr<Buffer> temporary;
    Buffer* staging = nullptr;
    if (bytes <= kStagingKeep) {
        lock.lock();
        if (!upload_staging_)
            upload_staging_ = std::make_unique<Buffer>(*this, kStagingKeep, MemoryUse::Upload);
        staging = upload_staging_.get();
    } else {
        temporary = std::make_unique<Buffer>(*this, bytes, MemoryUse::Upload);
        staging = temporary.get();
    }
    std::memcpy(staging->data(), src, bytes);
    submit_and_wait([&](VkCommandBuffer cmd) {
        VkBufferCopy region{0, dst_offset, bytes};
        vkCmdCopyBuffer(cmd, staging->handle(), dst.handle(), 1, &region);
    });
}

void Context::download(const Buffer& src, void* dst, VkDeviceSize bytes, VkDeviceSize src_offset) {
    if (bytes == 0)
        return;
    if (src.mapped()) {
        submit_and_wait([](VkCommandBuffer) {}); // make device writes host-visible
        std::memcpy(dst, static_cast<const char*>(src.data()) + src_offset, bytes);
        return;
    }
    std::unique_lock<std::mutex> lock(staging_mutex_, std::defer_lock);
    std::unique_ptr<Buffer> temporary;
    Buffer* staging = nullptr;
    if (bytes <= kStagingKeep) {
        lock.lock();
        if (!download_staging_)
            download_staging_ = std::make_unique<Buffer>(*this, kStagingKeep, MemoryUse::Readback);
        staging = download_staging_.get();
    } else {
        temporary = std::make_unique<Buffer>(*this, bytes, MemoryUse::Readback);
        staging = temporary.get();
    }
    submit_and_wait([&](VkCommandBuffer cmd) {
        VkBufferCopy region{src_offset, 0, bytes};
        vkCmdCopyBuffer(cmd, src.handle(), staging->handle(), 1, &region);
    });
    std::memcpy(dst, staging->data(), bytes);
}

void Context::fill_u32(Buffer& dst, std::uint32_t value) {
    submit_and_wait(
        [&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, dst.handle(), 0, VK_WHOLE_SIZE, value); });
}

std::uint32_t Context::find_memory_type(std::uint32_t bits, VkMemoryPropertyFlags want) const {
    for (std::uint32_t i = 0; i < mem_.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mem_.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    return UINT32_MAX;
}

void Context::barrier_compute_to_compute(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Context::barrier_full(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb,
                         0, nullptr, 0, nullptr);
}

Groups groups_for(const Context& ctx, std::uint64_t n, std::uint32_t local) {
    const std::uint64_t total = std::max<std::uint64_t>(1, (n + local - 1) / local);
    const std::uint64_t max_x = std::min<std::uint64_t>(ctx.max_group_count_x(), 65535u);
    Groups g;
    g.x = static_cast<std::uint32_t>(std::min(total, max_x));
    g.y = static_cast<std::uint32_t>((total + g.x - 1) / g.x);
    g.total = g.x * g.y;
    return g;
}

VkSemaphore Context::create_timeline(std::uint64_t initial) const {
    VkSemaphoreTypeCreateInfo ti{};
    ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    ti.initialValue = initial;
    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    si.pNext = &ti;
    VkSemaphore s = VK_NULL_HANDLE;
    vk_check(vkCreateSemaphore(device_, &si, nullptr, &s), "vkCreateSemaphore (timeline)");
    return s;
}

void Context::wait_timeline(VkSemaphore s, std::uint64_t value) const {
    VkSemaphoreWaitInfo wi{};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores = &s;
    wi.pValues = &value;
    vk_check(vkWaitSemaphores(device_, &wi, UINT64_MAX), "vkWaitSemaphores");
}

// -- Buffer ------------------------------------------------------------------

Buffer::Buffer(Context& ctx, VkDeviceSize bytes, MemoryUse use)
    : device_(ctx.device()), memory_(ctx.device()), buffer_(ctx.device()),
      size_(std::max<VkDeviceSize>(bytes, 16)) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size_;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // Engine and graphics queues in different families: let both use every
    // buffer without ownership transfers (CONCURRENT costs ~nothing on AMD).
    const std::span<const std::uint32_t> fams = ctx.sharing_families();
    if (fams.size() == 2) {
        bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bci.queueFamilyIndexCount = 2;
        bci.pQueueFamilyIndices = fams.data();
    }
    vk_check(vkCreateBuffer(device_, &bci, nullptr, buffer_.put()), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buffer_.get(), &req);
    const VkMemoryPropertyFlags visible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t type = UINT32_MAX;
    switch (use) {
    case MemoryUse::Device:
        type = ctx.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        break;
    case MemoryUse::Upload:
        type = ctx.find_memory_type(req.memoryTypeBits, visible);
        break;
    case MemoryUse::Readback:
        type =
            ctx.find_memory_type(req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (type == UINT32_MAX)
            type = ctx.find_memory_type(req.memoryTypeBits, visible);
        break;
    }
    if (type == UINT32_MAX)
        throw std::runtime_error("no suitable Vulkan memory type for a buffer");

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    vk_check(vkAllocateMemory(device_, &mai, nullptr, memory_.put()), "vkAllocateMemory");
    vk_check(vkBindBufferMemory(device_, buffer_.get(), memory_.get(), 0), "vkBindBufferMemory");
    if (use != MemoryUse::Device) {
        vk_check(vkMapMemory(device_, memory_.get(), 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
    }
}

// The handles free themselves: the buffer, then its memory (which unmaps it).
Buffer::~Buffer() = default;

// -- Image3D -------------------------------------------------------------------

Image3D::Image3D(Context& ctx, int nx, int ny, int nz)
    : device_(ctx.device()), extent_{std::uint32_t(nz), std::uint32_t(ny), std::uint32_t(nx)},
      memory_(ctx.device()), image_(ctx.device()), view_(ctx.device()),
      sampler_(ctx.device()) { // extent: z fastest (see the header)
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = VK_FORMAT_R32_SFLOAT;
    ici.extent = extent_;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    const std::span<const std::uint32_t> fams = ctx.sharing_families();
    if (fams.size() == 2) {
        ici.sharingMode = VK_SHARING_MODE_CONCURRENT;
        ici.queueFamilyIndexCount = 2;
        ici.pQueueFamilyIndices = fams.data();
    }
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_check(vkCreateImage(device_, &ici, nullptr, image_.put()), "vkCreateImage (3-D)");
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, image_.get(), &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        ctx.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX)
        throw std::runtime_error("no device-local Vulkan memory type for a 3-D image");
    vk_check(vkAllocateMemory(device_, &mai, nullptr, memory_.put()), "vkAllocateMemory (3-D)");
    vk_check(vkBindImageMemory(device_, image_.get(), memory_.get(), 0), "vkBindImageMemory");

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image_.get();
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = VK_FORMAT_R32_SFLOAT;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk_check(vkCreateImageView(device_, &vi, nullptr, view_.put()), "vkCreateImageView (3-D)");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    vk_check(vkCreateSampler(device_, &si, nullptr, sampler_.put()), "vkCreateSampler");

    // UNDEFINED -> GENERAL once; copies and sampling then need only memory
    // barriers.
    ctx.submit_and_wait([&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image_.get();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, image_.get(), VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    });
}

Image3D::~Image3D() = default; // sampler, view, image, then the memory

void Image3D::record_copy_from(VkCommandBuffer cmd, const Buffer& src) const {
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = extent_;
    vkCmdCopyBufferToImage(cmd, src.handle(), image_.get(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

// -- ComputeKernel -----------------------------------------------------------

ComputeKernel::ComputeKernel(Context& ctx, std::span<const std::uint32_t> spirv,
                             std::uint32_t n_buffers, std::uint32_t push_bytes,
                             std::uint32_t n_sets, std::span<const std::uint32_t> spec)
    : ComputeKernel(ctx, spirv, std::vector<Slot>(n_buffers, Slot::Buffer), push_bytes, n_sets,
                    spec) {}

ComputeKernel::ComputeKernel(Context& ctx, std::span<const std::uint32_t> spirv,
                             std::span<const Slot> slots, std::uint32_t push_bytes,
                             std::uint32_t n_sets, std::span<const std::uint32_t> spec)
    : device_(ctx.device()), slots_(slots.begin(), slots.end()), push_bytes_(push_bytes),
      n_sets_(n_sets), module_(ctx.device()), set_layout_(ctx.device()), layout_(ctx.device()),
      pipeline_(ctx.device()), pool_(ctx.device()) {
    if (n_sets < 1 || n_sets > 4)
        throw std::runtime_error("ComputeKernel: 1..4 sets");
    const std::uint32_t n = static_cast<std::uint32_t>(slots_.size());

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = spirv.size_bytes();
    smi.pCode = spirv.data();
    vk_check(vkCreateShaderModule(device_, &smi, nullptr, module_.put()), "vkCreateShaderModule");

    std::uint32_t n_buf = 0, n_tex = 0;
    std::vector<VkDescriptorSetLayoutBinding> binds(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const bool tex = slots_[i] == Slot::Texture;
        (tex ? n_tex : n_buf) += 1;
        binds[i] = {};
        binds[i].binding = i;
        binds[i].descriptorType =
            tex ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = n;
    dli.pBindings = binds.data();
    vk_check(vkCreateDescriptorSetLayout(device_, &dli, nullptr, set_layout_.put()),
             "vkCreateDescriptorSetLayout");
    const VkDescriptorSetLayout set_layout = set_layout_.get();

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = push_bytes;
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout;
    pli.pushConstantRangeCount = push_bytes > 0 ? 1 : 0;
    pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, layout_.put()),
             "vkCreatePipelineLayout");

    // Specialisation constant k <- spec[k] (4 bytes each; a GLSL bool
    // constant takes a VkBool32, i.e. also 4 bytes).
    std::vector<VkSpecializationMapEntry> entries(spec.size());
    for (std::uint32_t k = 0; k < spec.size(); ++k) {
        entries[k] = {k, k * 4u, 4u};
    }
    VkSpecializationInfo si{};
    si.mapEntryCount = static_cast<std::uint32_t>(entries.size());
    si.pMapEntries = entries.data();
    si.dataSize = spec.size_bytes();
    si.pData = spec.data();

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module_.get();
    cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    cpi.layout = layout_.get();
    vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, pipeline_.put()),
             "vkCreateComputePipelines");

    VkDescriptorPoolSize ps[2]{};
    std::uint32_t nps = 0;
    ps[nps++] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max(1u, n_buf) * n_sets};
    if (n_tex > 0)
        ps[nps++] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n_tex * n_sets};
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets = n_sets;
    dpi.poolSizeCount = nps;
    dpi.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &dpi, nullptr, pool_.put()), "vkCreateDescriptorPool");

    std::vector<VkDescriptorSetLayout> layouts(n_sets, set_layout);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool_.get();
    dai.descriptorSetCount = n_sets;
    dai.pSetLayouts = layouts.data();
    vk_check(vkAllocateDescriptorSets(device_, &dai, sets_), "vkAllocateDescriptorSets");
}

ComputeKernel::~ComputeKernel() = default; // the handles, in reverse order of creation

void ComputeKernel::bind(std::uint32_t set, std::initializer_list<const Buffer*> buffers) {
    std::vector<Resource> r;
    for (const Buffer* b : buffers)
        r.emplace_back(b);
    bind_list(set, r);
}

void ComputeKernel::bind_resources(std::uint32_t set, std::initializer_list<Resource> resources) {
    bind_list(set, std::vector<Resource>(resources));
}

void ComputeKernel::bind_list(std::uint32_t set, const std::vector<Resource>& res) {
    if (set >= n_sets_)
        throw std::runtime_error("ComputeKernel::bind: set out of range");
    if (res.size() != slots_.size()) {
        throw std::runtime_error("ComputeKernel::bind: expected " + std::to_string(slots_.size()) +
                                 " resources, got " + std::to_string(res.size()));
    }
    // writes point into these: reserve so they never reallocate
    std::vector<VkDescriptorBufferInfo> buf_infos;
    std::vector<VkDescriptorImageInfo> img_infos;
    buf_infos.reserve(res.size());
    img_infos.reserve(res.size());
    std::vector<VkWriteDescriptorSet> writes;
    for (std::uint32_t i = 0; i < res.size(); ++i) {
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets_[set];
        w.dstBinding = i;
        w.descriptorCount = 1;
        if (slots_[i] == Slot::Texture) {
            if (!res[i].image)
                throw std::runtime_error("ComputeKernel::bind: binding " + std::to_string(i) +
                                         " wants an Image3D");
            img_infos.push_back(
                {res[i].image->sampler(), res[i].image->view(), VK_IMAGE_LAYOUT_GENERAL});
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &img_infos.back();
        } else {
            if (!res[i].buffer)
                throw std::runtime_error("ComputeKernel::bind: binding " + std::to_string(i) +
                                         " wants a Buffer");
            buf_infos.push_back({res[i].buffer->handle(), 0, VK_WHOLE_SIZE});
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &buf_infos.back();
        }
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                           nullptr);
}

void ComputeKernel::record_set(VkCommandBuffer cmd, std::uint32_t set, const void* push,
                               std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.get());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_.get(), 0, 1, &sets_[set],
                            0, nullptr);
    if (push_bytes_ > 0) {
        vkCmdPushConstants(cmd, layout_.get(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
    }
    vkCmdDispatch(cmd, gx, gy, gz);
}

} // namespace windoa
