#include "windoa/volume.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

#include "vol_blur_spv.hpp"
#include "vol_march_spv.hpp"
#include "vol_occ_spv.hpp"
#include "vol_prepare_spv.hpp"
#include "vol_q_spv.hpp"
#include "vol_qstat_spv.hpp"
#include "vol_smooth_spv.hpp"
#include "vol_splat_spv.hpp"

namespace windoa::render {

namespace {

constexpr int kOccBlock = 4; // must match OCC_BLOCK in vol_common.glsl
constexpr std::uint32_t kLocal = 256;
constexpr int kMaxSplatSources = 4;

// vol_march.comp bindings: buffers, and the 3-D textures it samples
const Slot kMarchLayout[] = {Slot::Buffer,  Slot::Buffer,  Slot::Texture, Slot::Buffer,
                             Slot::Texture, Slot::Texture, Slot::Texture, Slot::Buffer,
                             Slot::Texture, Slot::Buffer,  Slot::Buffer,  Slot::Texture,
                             Slot::Texture};
const std::span<const Slot> kMarchSlots{kMarchLayout};

// Push-constant blocks; each must match its shader's `Params`.
struct Dims {
    std::int32_t d[4];
};
struct PrepareParams {
    std::int32_t dims[4]; // nx, ny, nz, mode
    float ref[4];         // u_ref, rho_ref, pscale, gamma
    float stats[4];       // 1 / weight, recirc output on, -, -
};
struct QStatParams {
    std::uint32_t n_partials;
    float sense;
};
struct MarchParams {
    std::int32_t dims[4];  // nx, ny, nz, n_steps
    std::int32_t img[4];   // width, height, feature bits, surface | paint << 4
    std::int32_t modes[4]; // cmode, slice_axis, pscale (float bits), -
    float eye[4];          // xyz, tan(fov / 2)
    float fwd[4];          // xyz, aspect
    float right[4];        // xyz, u_ref
    float up[4];           // xyz, rho_ref
    float haze[4];         // gain, floor, slice_pos, dye_gain
};
struct SplatParams {
    float eye[4];
    float fwd[4];
    float right[4];
    float up[4];
    std::int32_t img[4]; // width, height, count, mode
    float misc[4];       // radius, alpha, depth bias, -
};
static_assert(sizeof(PrepareParams) == 48);
static_assert(sizeof(MarchParams) == 128, "Vulkan guarantees only 128 push-constant bytes");
static_assert(sizeof(SplatParams) == 96);

// Feature bits (vol_march.comp)
constexpr std::int32_t F_BOX = 1, F_HAZE = 2, F_PAINT_CP = 4, F_Q = 8, F_DYE = 16,
                       F_DYE_BY_SPEED = 32, F_RECIRC = 64, F_SLICE_LIC = 128;

// Colour mode per field: 0 coolwarm, 1 vort, 2 grey, 3 Mach, 4 sequential.
int cmode_of(Field f) {
    switch (f) {
    case Field::Turbulence:
        return 4;
    case Field::VortX:
        return 1;
    case Field::Schlieren:
        return 2;
    case Field::Mach:
        return 3;
    default:
        return 0;
    }
}

std::size_t block_count(int nx, int ny, int nz) {
    auto nb = [](int n) { return std::size_t((n + kOccBlock - 1) / kOccBlock); };
    return nb(nx) * nb(ny) * nb(nz);
}

using Vec = std::array<float, 3>;
Vec sub(Vec a, Vec b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec cross(Vec a, Vec b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
float norm(Vec a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
Vec normalised(Vec a, Vec fallback) {
    const float n = norm(a);
    return n > 1e-12f ? Vec{a[0] / n, a[1] / n, a[2] / n} : fallback;
}

void memory_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src_stage, VkAccessFlags src,
                    VkPipelineStageFlags dst_stage, VkAccessFlags dst) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src;
    mb.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void image_layout(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                  VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                  VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

float default_floor(Field f) {
    switch (f) {
    case Field::Speed:
        return 0.06f;
    case Field::Pressure:
        return 0.12f;
    case Field::Vorticity:
        return 0.02f;
    case Field::VortX:
        return 0.10f;
    case Field::Mach:
        return 0.0f;
    case Field::Schlieren:
        return 0.15f;
    case Field::MeanSpeed:
        return 0.06f;
    case Field::Turbulence:
        return 0.10f; // 2 % intensity
    }
    return 0.06f;
}

VolumeRenderer::VolumeRenderer(Context& ctx, int nx, int ny, int nz)
    : ctx_(ctx), n_{nx, ny, nz}, cells_(std::size_t(nx) * ny * nz),
      cell_groups_(groups_for(ctx, cells_, kLocal)), field_(ctx, cells_ * sizeof(float)),
      occ_(ctx, block_count(nx, ny, nz) * sizeof(std::uint32_t)),
      solid_raw_(ctx, cells_ * sizeof(float)), solid_blur_(ctx, cells_ * sizeof(float)),
      blur_tmp_(ctx, cells_ * sizeof(float)), vsmooth_(ctx, cells_ * 16),
      smooth_tmp_(ctx, cells_ * 16), qfield_(ctx, cells_ * sizeof(float)),
      qpartials_(ctx, std::size_t{cell_groups_.total} * 8), qstat_(ctx, 16),
      zero_(ctx, cells_ * sizeof(float)), speed_(ctx, cells_ * sizeof(float)),
      recirc_(ctx, cells_ * sizeof(float)), t_field_(ctx, nx, ny, nz), t_raw_(ctx, nx, ny, nz),
      t_blur_(ctx, nx, ny, nz), t_q_(ctx, nx, ny, nz), t_dye_(ctx, nx, ny, nz),
      t_speed_(ctx, nx, ny, nz), t_recirc_(ctx, nx, ny, nz),
      prepare_(ctx, spv::vol_prepare, 8, sizeof(PrepareParams), kSlots),
      occ_kernel_(ctx, spv::vol_occ, 3, sizeof(Dims), kSlots),
      blur_(ctx, spv::vol_blur, 2, sizeof(Dims), 3),
      smooth_(ctx, spv::vol_smooth, 5, sizeof(Dims), kSlots),
      q_(ctx, spv::vol_q, 4, sizeof(Dims), kSlots),
      qstat_kernel_(ctx, spv::vol_qstat, 2, sizeof(QStatParams)),
      march_(ctx, spv::vol_march, kMarchSlots, sizeof(MarchParams), kSlots),
      splat_(ctx, spv::vol_splat, 4, sizeof(SplatParams), kMaxSplatSources) {
    ctx_.fill_zero(zero_);
    ctx_.fill_zero(qstat_);
    // blur x raw -> blur, y blur -> tmp, z tmp -> blur
    blur_.bind(0, {&solid_raw_, &solid_blur_});
    blur_.bind(1, {&solid_blur_, &blur_tmp_});
    blur_.bind(2, {&blur_tmp_, &solid_blur_});
    qstat_kernel_.bind({&qpartials_, &qstat_});
    resize(640, 360);
}

VolumeRenderer::~VolumeRenderer() {
    vkQueueWaitIdle(ctx_.graphics_queue());
    destroy_target();
}

void VolumeRenderer::set_sources(int slot, const Sources& s) {
    if (slot < 0 || slot >= kSlots || !s.flags || !s.macro)
        throw std::runtime_error("VolumeRenderer::set_sources: bad slot or missing buffers");
    sources_[slot] = s;
    const std::uint32_t k = std::uint32_t(slot);
    const Buffer* aux = s.aux ? s.aux : &zero_;
    // without statistics the macro buffer stands in (never read: weight 0)
    const Buffer* mean = s.mean ? s.mean : s.macro;
    const Buffer* m2 = s.m2 ? s.m2 : s.macro;
    prepare_.bind(k, {s.flags, s.macro, &field_, aux, &speed_, mean, m2, &recirc_});
    occ_kernel_.bind(k, {s.flags, &occ_, &solid_raw_});
    // separable smoothing: x pass -> vsmooth_ (as scratch), y -> smooth_tmp_,
    // z -> vsmooth_
    smooth_.bind(k, {s.flags, s.macro, &vsmooth_, &vsmooth_, &smooth_tmp_});
    q_.bind(k, {s.flags, &vsmooth_, &qfield_, &qpartials_});
    if (pixels_)
        bind_target_dependent();
    geometry_seen_ = UINT64_MAX; // new flags buffer: rebuild the solid fields
}

int VolumeRenderer::add_splat_source(const SplatSource& s) {
    if (int(splat_sources_.size()) >= kMaxSplatSources)
        throw std::runtime_error("VolumeRenderer: too many splat sources");
    splat_sources_.push_back(s);
    if (pixels_)
        bind_target_dependent();
    return int(splat_sources_.size()) - 1;
}

void VolumeRenderer::destroy_target() {
    if (image_ != VK_NULL_HANDLE)
        vkDestroyImage(ctx_.device(), image_, nullptr);
    if (image_memory_ != VK_NULL_HANDLE)
        vkFreeMemory(ctx_.device(), image_memory_, nullptr);
    image_ = VK_NULL_HANDLE;
    image_memory_ = VK_NULL_HANDLE;
    pixels_.reset();
    depth_.reset();
}

void VolumeRenderer::resize(std::uint32_t width, std::uint32_t height) {
    width = std::max(width, 1u);
    height = std::max(height, 1u);
    if (width == width_ && height == height_)
        return;
    // In-flight frames (graphics queue only) still use the old target. Not
    // vkDeviceWaitIdle: that would race a solver thread on the other queue.
    vkQueueWaitIdle(ctx_.graphics_queue());
    destroy_target();
    width_ = width;
    height_ = height;
    create_target();
}

void VolumeRenderer::create_target() {
    pixels_ = std::make_unique<Buffer>(ctx_, VkDeviceSize{width_} * height_ * 4);
    depth_ = std::make_unique<Buffer>(ctx_, VkDeviceSize{width_} * height_ * 4);

    // RGBA8 image: the packed pixels are copied in, then blitted (with
    // scaling and the RGBA -> BGRA swizzle) into the swapchain.
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {width_, height_, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_check(vkCreateImage(ctx_.device(), &ici, nullptr, &image_), "vkCreateImage");
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(ctx_.device(), image_, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex =
        ctx_.find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_check(vkAllocateMemory(ctx_.device(), &mai, nullptr, &image_memory_), "vkAllocateMemory");
    vk_check(vkBindImageMemory(ctx_.device(), image_, image_memory_, 0), "vkBindImageMemory");
    bind_target_dependent();
}

void VolumeRenderer::bind_target_dependent() {
    for (int k = 0; k < kSlots; ++k) {
        const Sources& s = sources_[k];
        if (!s.flags)
            continue;
        march_.bind_resources(std::uint32_t(k), {s.flags, s.macro, &t_field_, &occ_, &t_raw_,
                                                 &t_blur_, &t_q_, &qstat_, &t_dye_, pixels_.get(),
                                                 depth_.get(), &t_speed_, &t_recirc_});
    }
    for (std::size_t k = 0; k < splat_sources_.size(); ++k) {
        splat_.bind(std::uint32_t(k), {pixels_.get(), depth_.get(), splat_sources_[k].verts,
                                       splat_sources_[k].colours});
    }
}

std::vector<std::uint32_t> VolumeRenderer::read_pixels() {
    std::vector<std::uint32_t> px(std::size_t{width_} * height_);
    ctx_.download(*pixels_, px.data(), px.size() * sizeof(std::uint32_t));
    return px;
}

void VolumeRenderer::record(VkCommandBuffer cmd, const View& view, const Settings& s, int slot,
                            std::uint64_t geometry_version, std::span<const SplatDraw> splats) {
    if (slot < 0 || slot >= kSlots || !sources_[slot].flags)
        throw std::runtime_error("VolumeRenderer::record: slot has no sources");
    const std::uint32_t set = std::uint32_t(slot);
    const Groups& g = cell_groups_;
    const std::int32_t nx = n_[0], ny = n_[1], nz = n_[2];
    const Dims dims{{nx, ny, nz, 0}};

    // The previous frame (earlier on this queue) may still be reading the
    // pixel buffer / image: order this frame after all of it.
    Context::barrier_full(cmd);

    // Solid-derived fields
    bool geometry_copy = false;
    if (geometry_version != geometry_seen_) {
        vkCmdFillBuffer(cmd, occ_.handle(), 0, VK_WHOLE_SIZE, 0u);
        memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        occ_kernel_.record_set(cmd, set, &dims, g.x, g.y);
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            Context::barrier_compute_to_compute(cmd);
            const Dims b{{nx, ny, nz, axis}};
            blur_.record_set(cmd, std::uint32_t(axis), &b, g.x, g.y);
        }
        geometry_seen_ = geometry_version;
        q_valid_ = false;
        geometry_copy = true;
    }

    // Per-cell field
    const Field field = s.field;
    const bool dye_on = s.dye && sources_[slot].dye;
    const bool stats = s.stats_inv_weight > 0.0f && sources_[slot].mean && sources_[slot].m2;
    const bool recirc = s.recirculation && stats;
    const bool prepared = s.haze || s.slice_axis >= 0 || (dye_on && s.dye_by_speed) || recirc;
    if (prepared) {
        const PrepareParams pp{
            {nx, ny, nz, static_cast<std::int32_t>(field)},
            {s.u_ref, s.rho_ref, s.pscale, s.gamma},
            {stats ? s.stats_inv_weight : 0.0f, recirc ? 1.0f : 0.0f, 0.0f, 0.0f}};
        prepare_.record_set(cmd, set, &pp, g.x, g.y);
    }
    // Vortex cores: smooth -> Q (+ partials) -> threshold on the device
    // Q is refreshed every kQEvery frames (cores move little in a few 14 ms
    // frames; the pass costs ~3 ms on the fine grid), and at once when it
    // is switched on, the sensitivity changes, or the geometry changes.
    const bool q_due =
        s.vortex_cores && (!q_valid_ || ++q_age_ >= kQEvery || s.q_sense != q_sense_seen_);
    if (!s.vortex_cores)
        q_valid_ = false;
    if (q_due) {
        q_valid_ = true;
        q_age_ = 0;
        q_sense_seen_ = s.q_sense;
        for (std::int32_t pass = 0; pass < 3; ++pass) {
            if (pass > 0)
                Context::barrier_compute_to_compute(cmd);
            const Dims sd{{nx, ny, nz, pass}};
            smooth_.record_set(cmd, set, &sd, g.x, g.y);
        }
        Context::barrier_compute_to_compute(cmd);
        q_.record_set(cmd, set, &dims, g.x, g.y);
        Context::barrier_compute_to_compute(cmd);
        const QStatParams qp{g.total, s.q_sense};
        qstat_kernel_.record(cmd, &qp, 1);
    }
    Context::barrier_compute_to_compute(cmd);

    // What the march samples goes into its 3-D textures (hardware trilinear).
    if (geometry_copy || prepared || dye_on || q_due) {
        memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (geometry_copy) {
            t_raw_.record_copy_from(cmd, solid_raw_);
            t_blur_.record_copy_from(cmd, solid_blur_);
        }
        if (s.haze || s.slice_axis >= 0)
            t_field_.record_copy_from(cmd, field_);
        if (dye_on && s.dye_by_speed)
            t_speed_.record_copy_from(cmd, speed_);
        if (dye_on)
            t_dye_.record_copy_from(cmd, *sources_[slot].dye);
        if (q_due)
            t_q_.record_copy_from(cmd, qfield_);
        if (recirc)
            t_recirc_.record_copy_from(cmd, recirc_);
        memory_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    // Camera basis (lattice cells)
    const Vec fwd = normalised(sub(view.target, view.eye), {1.0f, 0.0f, 0.0f});
    Vec right = cross(fwd, view.up);
    if (norm(right) < 1e-9f)
        right = cross(fwd, {1.0f, 0.0f, 0.0f}); // looking straight up / down
    right = normalised(right, {0.0f, 0.0f, 1.0f});
    const Vec up = cross(right, fwd);
    const float tan_half = std::tan(view.fov_deg * 0.5f * 3.14159265f / 180.0f);
    const float aspect = float(width_) / float(std::max(height_, 1u));
    const float floor_v = s.haze_floor >= 0.0f ? s.haze_floor : default_floor(field);

    std::int32_t feat = 0;
    if (s.box)
        feat |= F_BOX;
    if (s.haze)
        feat |= F_HAZE;
    if (s.paint_surface)
        feat |= F_PAINT_CP;
    if (recirc)
        feat |= F_RECIRC;
    if (s.slice_lic && s.slice_axis >= 0)
        feat |= F_SLICE_LIC;
    if (s.vortex_cores)
        feat |= F_Q;
    if (s.dye && sources_[slot].dye)
        feat |= F_DYE;
    if (s.dye_by_speed)
        feat |= F_DYE_BY_SPEED;

    MarchParams mp{};
    mp.dims[0] = nx;
    mp.dims[1] = ny;
    mp.dims[2] = nz;
    mp.dims[3] = std::max(s.steps, 1);
    mp.img[0] = std::int32_t(width_);
    mp.img[1] = std::int32_t(height_);
    mp.img[2] = feat;
    mp.img[3] = static_cast<std::int32_t>(s.surface) | (static_cast<std::int32_t>(s.paint) << 4);
    mp.modes[0] = cmode_of(field);
    mp.modes[1] = s.slice_axis;
    mp.modes[2] = std::bit_cast<std::int32_t>(s.pscale);
    for (int k = 0; k < 3; ++k) {
        mp.eye[k] = view.eye[k];
        mp.fwd[k] = fwd[k];
        mp.right[k] = right[k];
        mp.up[k] = up[k];
    }
    mp.eye[3] = tan_half;
    mp.fwd[3] = aspect;
    mp.right[3] = s.u_ref;
    mp.up[3] = s.rho_ref;
    mp.haze[0] = s.haze_gain;
    mp.haze[1] = floor_v;
    mp.haze[2] = s.slice_pos;
    mp.haze[3] = s.dye_gain;
    march_.record_set(cmd, set, &mp, (width_ + 7) / 8, (height_ + 7) / 8);

    // Depth-tested splats over the marched image
    for (const SplatDraw& d : splats) {
        if (d.count == 0 || d.source < 0 || d.source >= int(splat_sources_.size()))
            continue;
        Context::barrier_compute_to_compute(cmd);
        SplatParams sp{};
        std::copy(std::begin(mp.eye), std::end(mp.eye), sp.eye);
        std::copy(std::begin(mp.fwd), std::end(mp.fwd), sp.fwd);
        std::copy(std::begin(mp.right), std::end(mp.right), sp.right);
        std::copy(std::begin(mp.up), std::end(mp.up), sp.up);
        sp.img[0] = std::int32_t(width_);
        sp.img[1] = std::int32_t(height_);
        sp.img[2] = std::int32_t(d.count);
        sp.img[3] = d.segments ? 1 : 0;
        sp.misc[0] = d.radius;
        sp.misc[1] = d.alpha;
        sp.misc[2] = d.depth_bias;
        splat_.record_set(cmd, std::uint32_t(d.source), &sp, (d.count + 63) / 64);
    }

    // pixels -> image, left in TRANSFER_SRC for the app's blit.
    memory_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    image_layout(cmd, image_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width_, height_, 1};
    vkCmdCopyBufferToImage(cmd, pixels_->handle(), image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);
    image_layout(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
}

} // namespace windoa::render
