#include "windoa/tracers.hpp"

#include <algorithm>

#include "smoke_spv.hpp"
#include "streamlines_spv.hpp"

namespace windoa::render {

namespace {

struct SmokeParams {
    std::int32_t dims[4]; // nx, ny, nz, count
    float rake[4];        // x, cy, cz, -
    float half_[4];       // half_y, half_z, u_ref, max_age
    std::int32_t misc[4]; // n_steps, n_side, seed, mode
};
struct LineParams {
    std::int32_t dims[4]; // nx, ny, nz, n_seeds
    float misc[4];        // seed_x, u_ref, step, -
    std::int32_t n[4];    // steps per line
};

constexpr float kMarkerColour[4] = {0.25f, 0.85f, 1.0f, 1.0f};

} // namespace

Tracers::Tracers(Context& ctx, int nx, int ny, int nz)
    : ctx_(ctx), n_{nx, ny, nz}, rake_cy_(ny * 0.5f), rake_cz_(nz * 0.5f), half_y_(ny * 0.10f),
      half_z_(ny * 0.10f), smoke_pos_(ctx, kSmokeParticles * 16),
      smoke_col_(ctx, kSmokeParticles * 16),
      line_verts_(ctx, std::size_t{line_segments()} * 2 * 16),
      line_col_(ctx, std::size_t{line_segments()} * 16), marker_verts_(ctx, 8 * 16),
      marker_col_(ctx, 4 * 16), smoke_(ctx, spv::smoke, 4, sizeof(SmokeParams), kSlots),
      lines_(ctx, spv::streamlines, 3, sizeof(LineParams), kSlots) {
    float cols[16];
    for (int e = 0; e < 4; ++e)
        std::copy(std::begin(kMarkerColour), std::end(kMarkerColour), cols + 4 * e);
    ctx_.upload(marker_col_, cols, sizeof(cols));
    ctx_.fill_zero(line_col_);
}

void Tracers::set_sources(int slot, const Buffer& flags, const Buffer& macro) {
    smoke_.bind(std::uint32_t(slot), {&flags, &macro, &smoke_pos_, &smoke_col_});
    lines_.bind(std::uint32_t(slot), {&macro, &line_verts_, &line_col_});
}

void Tracers::set_rake(float x, float cy, float cz, float half_y, float half_z) {
    half_y = std::max(half_y, 1.0f);
    half_z = std::max(half_z, 1.0f);
    if (x == rake_x_ && cy == rake_cy_ && cz == rake_cz_ && half_y == half_y_ && half_z == half_z_)
        return;
    rake_x_ = x;
    rake_cy_ = cy;
    rake_cz_ = cz;
    half_y_ = half_y;
    half_z_ = half_z;
    marker_dirty_ = true;
}

void Tracers::record(VkCommandBuffer cmd, int slot, int steps, float u_ref, bool smoke,
                     bool streamlines) {
    const std::uint32_t set = std::uint32_t(slot);
    ++frame_;
    Context::barrier_full(cmd); // the previous frame's splats read these buffers
    if (marker_dirty_) {
        // Wand outline: 4 edges of the emission rectangle (vkCmdUpdateBuffer
        // keeps the host write ordered with the frames in flight).
        const float c[4][2] = {{rake_cy_ - half_y_, rake_cz_ - half_z_},
                               {rake_cy_ + half_y_, rake_cz_ - half_z_},
                               {rake_cy_ + half_y_, rake_cz_ + half_z_},
                               {rake_cy_ - half_y_, rake_cz_ + half_z_}};
        float v[32];
        for (int e = 0; e < 4; ++e) {
            const int f = (e + 1) % 4;
            const float a[4] = {rake_x_, c[e][0], c[e][1], 1.0f};
            const float b[4] = {rake_x_, c[f][0], c[f][1], 1.0f};
            std::copy(a, a + 4, v + 8 * e);
            std::copy(b, b + 4, v + 8 * e + 4);
        }
        vkCmdUpdateBuffer(cmd, marker_verts_.handle(), 0, sizeof(v), v);
        marker_dirty_ = false;
        reset_pending_ = true; // the change shows at once, not as particles cycle out
    }
    if (smoke && (steps > 0 || reset_pending_)) {
        SmokeParams sp{};
        sp.dims[0] = n_[0];
        sp.dims[1] = n_[1];
        sp.dims[2] = n_[2];
        sp.dims[3] = std::int32_t(kSmokeParticles);
        sp.rake[0] = rake_x_;
        sp.rake[1] = rake_cy_;
        sp.rake[2] = rake_cz_;
        sp.half_[0] = half_y_;
        sp.half_[1] = half_z_;
        sp.half_[2] = u_ref;
        sp.half_[3] = 2.5f * n_[0] / std::max(u_ref, 1e-6f); // max age: 2.5 flow-throughs
        sp.misc[0] = steps;
        sp.misc[1] = kRakeSide;
        sp.misc[2] = std::int32_t(frame_);
        sp.misc[3] = reset_pending_ ? 1 : 0;
        smoke_.record_set(cmd, set, &sp, (kSmokeParticles + 63) / 64);
        reset_pending_ = false;
    }
    if (streamlines) {
        LineParams lp{};
        lp.dims[0] = n_[0];
        lp.dims[1] = n_[1];
        lp.dims[2] = n_[2];
        lp.dims[3] = std::int32_t(kSeeds);
        lp.misc[0] = 6.0f; // seed plane x
        lp.misc[1] = u_ref;
        lp.misc[2] = kLineStep;
        lp.n[0] = std::int32_t(kLineSteps);
        lines_.record_set(cmd, set, &lp, (kSeeds + 63) / 64);
    }
    Context::barrier_compute_to_compute(cmd);
}

} // namespace windoa::render
