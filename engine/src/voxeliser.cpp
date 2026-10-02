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

constexpr std::size_t kMaxTris = std::size_t{1} << 20; // TRI_BITS in the shell key
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
    std::vector<std::uint32_t> words(flags.begin(), flags.end());
    ctx_.upload(flags_, words.data(), words.size() * 4);
}

Voxeliser::Stats Voxeliser::voxelise(const geometry::Mesh& mesh, std::vector<std::uint8_t>& flags) {
    n_tris_ = mesh.triangles();
    if (n_tris_ > kMaxTris)
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

    upload_flags(flags);
    ctx_.fill_zero(vote_);
    ctx_.fill_u32(shell_, kShellEmpty);
    ctx_.fill_zero(counter_);

    Params p{};
    p.dims[0] = nx_;
    p.dims[1] = ny_;
    p.dims[2] = nz_;
    p.dims[3] = static_cast<std::int32_t>(n_tris_);
    const std::uint64_t columns[3] = {std::uint64_t(ny_) * nz_, std::uint64_t(nx_) * nz_,
                                      std::uint64_t(nx_) * ny_};
    for (int axis = 0; axis < 3; ++axis) {
        p.misc[0] = axis;
        run(*winding_, &p, columns[axis], 64);
    }
    run(*votes_, &p, n_, 256);
    run(*shell_mark_, &p, n_tris_, 64);
    run(*shell_apply_, &p, n_, 256);

    std::vector<std::uint32_t> words(n_);
    ctx_.download(flags_, words.data(), n_ * 4);
    std::uint32_t n_thin = 0;
    ctx_.download(counter_, &n_thin, 4);

    Stats s;
    s.n_tris = n_tris_;
    s.n_thin = n_thin;
    for (std::size_t c = 0; c < n_; ++c) {
        flags[c] = static_cast<std::uint8_t>(words[c]);
        if (words[c] == 1u)
            ++s.n_solid;
    }
    return s;
}

std::vector<float> Voxeliser::signed_distance(const std::vector<std::uint8_t>& flags, float band) {
    upload_flags(flags);
    ctx_.fill_u32(shell_, kShellEmpty);
    Params p{};
    p.dims[0] = nx_;
    p.dims[1] = ny_;
    p.dims[2] = nz_;
    p.dims[3] = static_cast<std::int32_t>(n_tris_);
    p.fparam[0] = band;
    run(*dist_band_, &p, n_tris_, 64);
    run(*phi_kernel_, &p, n_, 256);
    std::vector<float> phi(n_);
    ctx_.download(phi_, phi.data(), n_ * 4);
    return phi;
}

} // namespace windoa
