#include "windoa/voxeliser.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "vox_dist_band_spv.hpp"
#include "vox_phi_spv.hpp"
#include "vox_shell_apply_spv.hpp"
#include "vox_shell_mark_spv.hpp"
#include "vox_votes_spv.hpp"
#include "vox_winding_spv.hpp"

namespace windoa {

namespace {

// Mirrors VoxParams in shaders/vox_common.glsl.
struct Params {
    std::int32_t dims[4]; // nx, ny, nz, n_tris
    std::int32_t misc[4]; // x: casting axis
    float fparam[4];      // x: band
};
static_assert(sizeof(Params) == 48, "Params must match vox_common.glsl");

constexpr std::uint32_t kShellEmpty = 0x7FFFFFFFu;
constexpr std::uint32_t kBindings = 6;

} // namespace

Voxeliser::Voxeliser(Context& ctx, int nx, int ny, int nz)
    : ctx_(ctx), nx_(nx), ny_(ny), nz_(nz), n_(static_cast<std::size_t>(nx) * ny * nz),
      tris_(std::make_unique<Buffer>(ctx, 3 * 16 * 1024)), vote_(ctx, n_ * 4), shell_(ctx, n_ * 4),
      flags_(ctx, n_ * 4), phi_(ctx, n_ * 4), counter_(ctx, 16) {
    const std::uint32_t pb = sizeof(Params);
    winding_ = std::make_unique<ComputeKernel>(ctx, spv::vox_winding, kBindings, pb);
    votes_ = std::make_unique<ComputeKernel>(ctx, spv::vox_votes, kBindings, pb);
    shell_mark_ = std::make_unique<ComputeKernel>(ctx, spv::vox_shell_mark, kBindings, pb);
    shell_apply_ = std::make_unique<ComputeKernel>(ctx, spv::vox_shell_apply, kBindings, pb);
    dist_band_ = std::make_unique<ComputeKernel>(ctx, spv::vox_dist_band, kBindings, pb);
    phi_kernel_ = std::make_unique<ComputeKernel>(ctx, spv::vox_phi, kBindings, pb);
    bind_all();
}

Voxeliser::~Voxeliser() = default;

void Voxeliser::bind_all() {
    for (ComputeKernel* k : {winding_.get(), votes_.get(), shell_mark_.get(), shell_apply_.get(),
                             dist_band_.get(), phi_kernel_.get()}) {
        k->bind({tris_.get(), &vote_, &shell_, &flags_, &phi_, &counter_});
    }
}

void Voxeliser::run(const ComputeKernel& k, const void* params, std::uint64_t items,
                    std::uint32_t local) {
    const Groups g = groups_for(ctx_, items, local);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) { k.record(cmd, params, g.x, g.y); });
}

void Voxeliser::upload_flags(const std::vector<std::uint8_t>& flags) {
    if (flags.size() != n_)
        throw std::runtime_error("Voxeliser: flags size mismatch");
    std::size_t lo = 0, hi = n_;
    if (device_flags_.size() == n_) {
        lo = std::size_t(std::mismatch(flags.begin(), flags.end(), device_flags_.begin()).first -
                         flags.begin());
        hi = lo;
        for (std::size_t c = n_; c > lo; --c)
            if (flags[c - 1] != device_flags_[c - 1]) {
                hi = c;
                break;
            }
    }
    if (hi > lo) {
        std::vector<std::uint32_t> words(flags.begin() + std::ptrdiff_t(lo),
                                         flags.begin() + std::ptrdiff_t(hi));
        ctx_.upload(flags_, words.data(), words.size() * 4, lo * 4);
    }
    device_flags_ = flags;
}

// The x-planes within `margin` cells of the last mesh, clamped to the grid.
std::pair<int, int> Voxeliser::slab(float margin) const {
    if (tri_xmax_ < tri_xmin_)
        return {0, 0};
    const int x0 = std::clamp(int(std::floor(tri_xmin_ - margin)), 0, nx_);
    const int x1 = std::clamp(int(std::ceil(tri_xmax_ + margin)) + 1, x0, nx_);
    return {x0, x1};
}

Voxeliser::Stats Voxeliser::voxelise(const geometry::Mesh& mesh, std::vector<std::uint8_t>& flags) {
    n_tris_ = mesh.triangles();
    if (n_tris_ > kMaxTriangles)
        throw std::runtime_error("Voxeliser: too many triangles (max 2^20)");

    // Triangles as vec4 (std430 vec3 arrays pad to 16 bytes anyway).
    const std::size_t bytes = std::max<std::size_t>(n_tris_, 1) * 3 * 16;
    if (tris_->size() < bytes) {
        tris_ = std::make_unique<Buffer>(ctx_, bytes);
        bind_all();
    }
    std::vector<float> packed(n_tris_ * 12, 0.0f);
    for (std::size_t t = 0; t < n_tris_; ++t)
        for (int v = 0; v < 3; ++v)
            for (int k = 0; k < 3; ++k)
                packed[t * 12 + v * 4 + k] = mesh.xyz[t * 9 + v * 3 + k];
    ctx_.upload(*tris_, packed.data(), packed.size() * sizeof(float));
    tri_xmin_ = 1e30f;
    tri_xmax_ = -1e30f;
    for (std::size_t i = 0; i < mesh.xyz.size(); i += 3) {
        tri_xmin_ = std::min(tri_xmin_, mesh.xyz[i]);
        tri_xmax_ = std::max(tri_xmax_, mesh.xyz[i]);
    }

    upload_flags(flags);

    Params p{};
    p.dims[0] = nx_;
    p.dims[1] = ny_;
    p.dims[2] = nz_;
    p.dims[3] = static_cast<std::int32_t>(n_tris_);
    const std::uint64_t columns[3] = {std::uint64_t(ny_) * nz_, std::uint64_t(nx_) * nz_,
                                      std::uint64_t(nx_) * ny_};
    // Every pass in one submission (the same kernels in the same order).
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, vote_.handle(), 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, shell_.handle(), 0, VK_WHOLE_SIZE, kShellEmpty);
        vkCmdFillBuffer(cmd, counter_.handle(), 0, VK_WHOLE_SIZE, 0u);
        Context::barrier_full(cmd);
        auto pass = [&](const ComputeKernel& k, std::uint64_t items, std::uint32_t local) {
            const Groups g = groups_for(ctx_, items, local);
            k.record(cmd, &p, g.x, g.y);
            Context::barrier_compute_to_compute(cmd);
        };
        for (int axis = 0; axis < 3; ++axis) {
            p.misc[0] = axis;
            pass(*winding_, columns[axis], 64);
        }
        pass(*votes_, n_, 256);
        pass(*shell_mark_, n_tris_, 64);
        pass(*shell_apply_, n_, 256);
    });

    // The passes change no cell more than two cells off the mesh: only that
    // slab comes back (the rest is what went up).
    const auto [x0, x1] = slab(4.0f);
    const std::size_t plane = std::size_t(ny_) * nz_, lo = std::size_t(x0) * plane,
                      len = std::size_t(x1 - x0) * plane;
    std::vector<std::uint32_t> words(len);
    if (len > 0)
        ctx_.download(flags_, words.data(), len * 4, lo * 4);
    std::uint32_t n_thin = 0;
    ctx_.download(counter_, &n_thin, 4);
    for (std::size_t i = 0; i < len; ++i)
        flags[lo + i] = static_cast<std::uint8_t>(words[i]);
    device_flags_ = flags;

    Stats s;
    s.n_tris = n_tris_;
    s.n_thin = n_thin;
    s.n_solid = std::size_t(std::count(flags.begin(), flags.end(), std::uint8_t(1)));
    return s;
}

std::vector<float> Voxeliser::signed_distance(const std::vector<std::uint8_t>& flags, float band) {
    upload_flags(flags);
    Params p{};
    p.dims[0] = nx_;
    p.dims[1] = ny_;
    p.dims[2] = nz_;
    p.dims[3] = static_cast<std::int32_t>(n_tris_);
    p.fparam[0] = band;
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, shell_.handle(), 0, VK_WHOLE_SIZE, kShellEmpty);
        Context::barrier_full(cmd);
        const Groups gt = groups_for(ctx_, n_tris_, 64), gc = groups_for(ctx_, n_, 256);
        dist_band_->record(cmd, &p, gt.x, gt.y);
        Context::barrier_compute_to_compute(cmd);
        phi_kernel_->record(cmd, &p, gc.x, gc.y);
    });
    // Beyond band + 1 of the mesh every cell holds vox_phi.comp's far value,
    // +-min(band + 1, 4) by its flag: filled here, the mesh's slab read back.
    std::vector<float> phi(n_);
    const float far = std::min(band + 1.0f, 4.0f);
    for (std::size_t c = 0; c < n_; ++c)
        phi[c] = flags[c] != 0 ? -far : far;
    const auto [x0, x1] = slab(band + 2.0f);
    const std::size_t plane = std::size_t(ny_) * nz_;
    if (x1 > x0)
        ctx_.download(phi_, phi.data() + std::size_t(x0) * plane, std::size_t(x1 - x0) * plane * 4,
                      std::size_t(x0) * plane * 4);
    return phi;
}

} // namespace windoa
