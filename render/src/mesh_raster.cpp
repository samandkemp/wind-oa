#include "mesh_raster.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "mesh_fs_spv.hpp"
#include "mesh_vs_spv.hpp"

namespace windoa::render {

namespace {

constexpr VkFormat kColourFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // normal, distance
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
static_assert(sizeof(RasterCamera) == 64);

using V3 = std::array<float, 3>;

V3 corner(const geometry::Mesh& m, std::size_t t, int k) {
    const float* p = &m.xyz[9 * t + 3 * std::size_t(k)];
    return {p[0], p[1], p[2]};
}

// A welding key: the corner's coordinates to the bit (+0 for -0).
struct Key {
    std::uint32_t b[3];
    bool operator==(const Key& o) const {
        return b[0] == o.b[0] && b[1] == o.b[1] && b[2] == o.b[2];
    }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const {
        std::uint64_t h = k.b[0];
        h = h * 0x9E3779B97F4A7C15ull ^ k.b[1];
        h = h * 0x9E3779B97F4A7C15ull ^ k.b[2];
        return std::size_t(h ^ (h >> 29));
    }
};
Key key_of(const V3& p) {
    Key k{};
    for (int i = 0; i < 3; ++i) {
        const float v = p[i] + 0.0f;
        std::memcpy(&k.b[i], &v, sizeof(float));
    }
    return k;
}

void memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage, VkAccessFlags src,
                    VkPipelineStageFlags dst_stage, VkAccessFlags dst) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src;
    mb.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                   VkImageLayout from, VkImageLayout to, VkPipelineStageFlags src_stage,
                   VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
                   VkAccessFlags dst_access) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void create_module(VkDevice device, std::span<const std::uint32_t> spirv, ShaderModuleHandle& h) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size_bytes();
    ci.pCode = spirv.data();
    vk_check(vkCreateShaderModule(device, &ci, nullptr, h.put()), "vkCreateShaderModule (mesh)");
}

} // namespace

std::vector<float> mesh_corners(const geometry::Mesh& m) {
    const std::size_t nt = m.triangles();
    // face normals (unit) and weights (twice the area)
    std::vector<V3> fn(nt);
    std::vector<float> fw(nt);
    for (std::size_t t = 0; t < nt; ++t) {
        const V3 a = corner(m, t, 0), b = corner(m, t, 1), c = corner(m, t, 2);
        const V3 e1{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const V3 e2{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const V3 n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                   e1[0] * e2[1] - e1[1] * e2[0]};
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        fw[t] = len;
        fn[t] = len > 0.0f ? V3{n[0] / len, n[1] / len, n[2] / len} : V3{0.0f, 0.0f, 0.0f};
    }
    // weld the corners by position, then list the faces round each vertex
    // (compressed rows: start[v] .. start[v + 1] in faces)
    std::unordered_map<Key, std::uint32_t, KeyHash> ids;
    ids.reserve(nt * 3 / 2 + 16);
    std::vector<std::uint32_t> vid(3 * nt);
    for (std::size_t i = 0; i < 3 * nt; ++i) {
        const auto it =
            ids.try_emplace(key_of(corner(m, i / 3, int(i % 3))), std::uint32_t(ids.size()));
        vid[i] = it.first->second;
    }
    std::vector<std::uint32_t> start(ids.size() + 1, 0), faces(3 * nt);
    for (std::uint32_t v : vid)
        ++start[v + 1];
    for (std::size_t v = 0; v < ids.size(); ++v)
        start[v + 1] += start[v];
    std::vector<std::uint32_t> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < 3 * nt; ++i)
        faces[fill[vid[i]]++] = std::uint32_t(i / 3);

    const float crease = std::cos(kCreaseDeg * 3.14159265f / 180.0f);
    std::vector<float> out(3 * nt * 8);
    for (std::size_t i = 0; i < 3 * nt; ++i) {
        const std::size_t t = i / 3;
        const V3 p = corner(m, t, int(i % 3));
        V3 s{0.0f, 0.0f, 0.0f};
        for (std::uint32_t k = start[vid[i]]; k < start[vid[i] + 1]; ++k) {
            const std::uint32_t f = faces[k];
            if (fn[f][0] * fn[t][0] + fn[f][1] * fn[t][1] + fn[f][2] * fn[t][2] >= crease)
                for (int j = 0; j < 3; ++j)
                    s[j] += fw[f] * fn[f][j];
        }
        const float len = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        const V3 n = len > 0.0f ? V3{s[0] / len, s[1] / len, s[2] / len} : fn[t];
        float* o = &out[8 * i];
        o[0] = p[0];
        o[1] = p[1];
        o[2] = p[2];
        o[3] = 1.0f;
        o[4] = n[0];
        o[5] = n[1];
        o[6] = n[2];
        o[7] = 0.0f;
    }
    return out;
}

MeshRaster::MeshRaster(Context& ctx)
    : ctx_(ctx), device_(ctx.device()), vs_(device_), fs_(device_), set_layout_(device_),
      layout_(device_), pipeline_(device_), pool_(device_), colour_memory_(device_),
      colour_(device_), colour_view_(device_), depth_memory_(device_), depth_(device_),
      depth_view_(device_) {
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(ctx_.physical(), kDepthFormat, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        throw std::runtime_error("MeshRaster: no D32_SFLOAT depth attachment on this device");
    create_module(device_, spv::mesh_vs, vs_);
    create_module(device_, spv::mesh_fs, fs_);

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo sl{};
    sl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sl.bindingCount = 1;
    sl.pBindings = &b;
    vk_check(vkCreateDescriptorSetLayout(device_, &sl, nullptr, set_layout_.put()),
             "vkCreateDescriptorSetLayout (mesh)");
    const VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                 sizeof(RasterCamera)};
    const VkDescriptorSetLayout set_layout = set_layout_.get();
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &set_layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pr;
    vk_check(vkCreatePipelineLayout(device_, &pl, nullptr, layout_.put()),
             "vkCreatePipelineLayout (mesh)");

    // The graphics pipeline: no vertex input (the shader pulls its corners),
    // both faces drawn, nearest wins under reversed depth (GREATER).
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (int k = 0; k < 2; ++k) {
        stages[k].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[k].pName = "main";
    }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_.get();
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_.get();
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{};
    dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    const VkFormat colour_format = kColourFormat;
    VkPipelineRenderingCreateInfo ri{}; // dynamic rendering: no render pass object
    ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachmentFormats = &colour_format;
    ri.depthAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo gp{};
    gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext = &ri;
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dy;
    gp.layout = layout_.get();
    vk_check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, pipeline_.put()),
             "vkCreateGraphicsPipelines (mesh)");

    const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(device_, &dp, nullptr, pool_.put()),
             "vkCreateDescriptorPool (mesh)");
    VkDescriptorSetAllocateInfo da{};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = pool_.get();
    da.descriptorSetCount = 1;
    da.pSetLayouts = &set_layout;
    vk_check(vkAllocateDescriptorSets(device_, &da, &set_), "vkAllocateDescriptorSets (mesh)");
}

MeshRaster::~MeshRaster() = default;

void MeshRaster::set_mesh(const geometry::Mesh& m) {
    const std::vector<float> data = mesh_corners(m);
    vkQueueWaitIdle(ctx_.graphics_queue()); // frames in flight draw the old corners
    corners_ = 0;
    upload_pending_ = false;
    staging_.reset();
    corner_buf_.reset();
    if (data.empty())
        return;
    const VkDeviceSize bytes = data.size() * sizeof(float);
    corner_buf_ = std::make_unique<Buffer>(ctx_, bytes);
    staging_ = std::make_unique<Buffer>(ctx_, bytes, MemoryUse::Upload);
    std::memcpy(staging_->data(), data.data(), bytes);
    VkDescriptorBufferInfo bi{corner_buf_->handle(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set_;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    corners_ = std::uint32_t(data.size() / 8);
    upload_pending_ = true;
}

void MeshRaster::resize(std::uint32_t width, std::uint32_t height) {
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    if (width == width_ && height == height_)
        return;
    vkQueueWaitIdle(ctx_.graphics_queue());
    width_ = width;
    height_ = height;
    create_targets();
}

void MeshRaster::create_targets() {
    auto make = [&](VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                    MemoryHandle& memory, ImageHandle& image, ImageViewHandle& view) {
        view.reset();
        image.reset();
        memory.reset();
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {width_, height_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(device_, &ici, nullptr, image.put()), "vkCreateImage (mesh)");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, image.get(), &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex =
            ctx_.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
            throw std::runtime_error("no device-local Vulkan memory type for the mesh pass");
        vk_check(vkAllocateMemory(device_, &mai, nullptr, memory.put()), "vkAllocateMemory (mesh)");
        vk_check(vkBindImageMemory(device_, image.get(), memory.get(), 0),
                 "vkBindImageMemory (mesh)");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = image.get();
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        vk_check(vkCreateImageView(device_, &vci, nullptr, view.put()), "vkCreateImageView (mesh)");
    };
    make(kColourFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         VK_IMAGE_ASPECT_COLOR_BIT, colour_memory_, colour_, colour_view_);
    make(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
         depth_memory_, depth_, depth_view_);
}

void MeshRaster::record(VkCommandBuffer cmd, const RasterCamera& cam, const Buffer& dst) {
    if (width_ == 0 || dst.size() < VkDeviceSize{width_} * height_ * 16)
        throw std::runtime_error("MeshRaster::record: no target, or `dst` too small");
    if (upload_pending_) {
        const VkBufferCopy c{0, 0, staging_->size()};
        vkCmdCopyBuffer(cmd, staging_->handle(), corner_buf_->handle(), 1, &c);
        memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        upload_pending_ = false;
    }
    // Last frame's contents are not needed (cleared): from UNDEFINED.
    image_barrier(cmd, colour_.get(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    image_barrier(
        cmd, depth_.get(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ca.imageView = colour_view_.get();
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ca.clearValue.color = {{0.0f, 1.0f, 0.0f, 1e30f}}; // no triangle: distance 1e30
    VkRenderingAttachmentInfo da{};
    da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    da.imageView = depth_view_.get();
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    da.clearValue.depthStencil = {0.0f, 0}; // reversed depth: 0 = infinitely far
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, {width_, height_}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cmd, &ri);
    if (corners_ > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.get());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_.get(), 0, 1, &set_, 0,
                                nullptr);
        vkCmdPushConstants(cmd, layout_.get(),
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(RasterCamera), &cam);
        const VkViewport viewport{0.0f, 0.0f, float(width_), float(height_), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, {width_, height_}};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdDraw(cmd, corners_, 1, 0, 0);
    }
    vkCmdEndRendering(cmd);
    image_barrier(cmd, colour_.get(), VK_IMAGE_ASPECT_COLOR_BIT,
                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyImageToBuffer(cmd, colour_.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.handle(),
                           1, &region);
    memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

} // namespace windoa::render
