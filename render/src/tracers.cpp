#include "windoa/tracers.hpp"

#include <algorithm>

#include "arrows_spv.hpp"
#include "smoke_spv.hpp"
#include "streamlines_spv.hpp"

namespace windoa::render {

namespace {

struct SmokeParams {
    std::int32_t dims[4];  // nx, ny, nz, count
    float rake[4];         // x, cy, cz, -
    float half_[4];        // half_y, half_z, u_ref, max_age
    std::int32_t misc[4];  // n_steps, n_side, seed, mode
    std::int32_t pulse[4]; // period (0 = streaklines), per pulse, clock, -
};
struct LineParams {
    std::int32_t dims[4]; // nx, ny, nz, n_seeds
    float misc[4];        // seed_x, u_ref, step, -
    std::int32_t n[4];    // steps per line
};
struct ArrowParams {
    std::int32_t dims[4]; // nx, ny, nz, axis
    std::int32_t grid[4]; // arrows along the two in-plane axes
    float misc[4];        // plane position, spacing, u_ref, scale
};
static_assert(sizeof(SmokeParams) == 80);
static_assert(sizeof(ArrowParams) == 48);

constexpr float kMarkerColour[4] = {0.25f, 0.85f, 1.0f, 1.0f};
constexpr float kSurveyColour[4] = {1.0f, 0.55f, 0.20f, 0.9f};
constexpr float kUpstreamColour[4] = {0.55f, 0.60f, 0.75f, 0.7f};
// segments: wand 4 + planes 8 + probes 12 + rotor blades (4 per blade)
constexpr std::uint32_t kMaxMarkers = 160;
constexpr float kBladeColour[4] = {0.95f, 0.95f, 1.0f, 1.0f};
constexpr std::int32_t kPerPulse = 200; // timeline particles per line
constexpr float kArrowScale = 0.8f;     // freestream arrow / grid spacing

} // namespace

Tracers::Tracers(Context& ctx, int nx, int ny, int nz)
    : ctx_(ctx), n_{nx, ny, nz}, rake_cy_(ny * 0.5f), rake_cz_(nz * 0.5f), half_y_(ny * 0.10f),
      half_z_(ny * 0.10f), max_arrows_(std::size_t(std::max({nx * ny, nx * nz, ny * nz})) / 4 + 1),
      smoke_pos_(ctx, kSmokeParticles * 16), smoke_col_(ctx, kSmokeParticles * 16),
      line_verts_(ctx, std::size_t{line_segments()} * 2 * 16),
      line_col_(ctx, std::size_t{line_segments()} * 16), marker_verts_(ctx, kMaxMarkers * 2 * 16),
      marker_col_(ctx, kMaxMarkers * 16), arrow_verts_(ctx, max_arrows_ * 6 * 16),
      arrow_col_(ctx, max_arrows_ * 3 * 16),
      smoke_(ctx, spv::smoke, 4, sizeof(SmokeParams), kSlots),
      lines_(ctx, spv::streamlines, 3, sizeof(LineParams), kSlots),
      arrows_(ctx, spv::arrows, 4, sizeof(ArrowParams), kSlots) {
    ctx_.fill_zero(line_col_);
    ctx_.fill_zero(arrow_col_);
    ctx_.fill_zero(marker_col_);
}

void Tracers::set_sources(int slot, const Buffer& flags, const Buffer& macro) {
    smoke_.bind(std::uint32_t(slot), {&flags, &macro, &smoke_pos_, &smoke_col_});
    lines_.bind(std::uint32_t(slot), {&macro, &line_verts_, &line_col_});
    arrows_.bind(std::uint32_t(slot), {&flags, &macro, &arrow_verts_, &arrow_col_});
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

void Tracers::set_timelines(int pulse_steps) {
    pulse_steps = std::max(pulse_steps, 0);
    if (pulse_steps == pulse_steps_)
        return;
    pulse_steps_ = pulse_steps;
    reset_pending_ = true;
}

void Tracers::set_overlay(int x_upstream, int x_survey,
                          const std::vector<std::array<float, 3>>& probes) {
    if (x_upstream == x_upstream_ && x_survey == x_survey_ && probes == probes_)
        return;
    x_upstream_ = x_upstream;
    x_survey_ = x_survey;
    probes_ = probes;
    marker_dirty_ = true;
}

void Tracers::set_blades(const std::vector<std::array<std::array<float, 3>, 2>>& segments) {
    if (segments == blades_)
        return;
    blades_ = segments;
    redraw_ = true;
}

void Tracers::show_wand(bool on) {
    if (on == show_wand_)
        return;
    show_wand_ = on;
    redraw_ = true;
}

void Tracers::record(VkCommandBuffer cmd, int slot, int steps, float u_ref, bool smoke,
                     bool streamlines) {
    const std::uint32_t set = std::uint32_t(slot);
    ++frame_;
    Context::barrier_full(cmd); // the previous frame's splats read these buffers
    if (marker_dirty_ || redraw_) {
        // The wand outline, the survey planes and the probe crosses as
        // segments (vkCmdUpdateBuffer keeps the host write ordered with the
        // frames in flight).
        std::vector<float> v, c;
        auto seg = [&](std::array<float, 3> a, std::array<float, 3> b, const float* col) {
            v.insert(v.end(), {a[0], a[1], a[2], 1.0f, b[0], b[1], b[2], 1.0f});
            c.insert(c.end(), col, col + 4);
        };
        auto rect_x = [&](float x, float y0, float z0, float y1, float z1, const float* col) {
            seg({x, y0, z0}, {x, y1, z0}, col);
            seg({x, y1, z0}, {x, y1, z1}, col);
            seg({x, y1, z1}, {x, y0, z1}, col);
            seg({x, y0, z1}, {x, y0, z0}, col);
        };
        if (show_wand_)
            rect_x(rake_x_, rake_cy_ - half_y_, rake_cz_ - half_z_, rake_cy_ + half_y_,
                   rake_cz_ + half_z_, kMarkerColour);
        const float ny = float(n_[1]), nz = float(n_[2]);
        if (x_upstream_ >= 0)
            rect_x(float(x_upstream_) + 0.5f, 0.5f, 0.5f, ny - 0.5f, nz - 0.5f, kUpstreamColour);
        if (x_survey_ >= 0)
            rect_x(float(x_survey_) + 0.5f, 0.5f, 0.5f, ny - 0.5f, nz - 0.5f, kSurveyColour);
        for (std::size_t p = 0; p < probes_.size() && p < kProbeColours.size(); ++p) {
            const auto& q = probes_[p];
            const float* col = kProbeColours[p].data();
            const float r = 2.5f;
            seg({q[0] - r, q[1], q[2]}, {q[0] + r, q[1], q[2]}, col);
            seg({q[0], q[1] - r, q[2]}, {q[0], q[1] + r, q[2]}, col);
            seg({q[0], q[1], q[2] - r}, {q[0], q[1], q[2] + r}, col);
        }
        for (const auto& b : blades_)
            seg(b[0], b[1], kBladeColour);
        marker_count_ = std::uint32_t(std::min<std::size_t>(c.size() / 4, kMaxMarkers));
        if (marker_count_ > 0) { // a zero-byte update is invalid
            vkCmdUpdateBuffer(cmd, marker_verts_.handle(), 0, marker_count_ * 32, v.data());
            vkCmdUpdateBuffer(cmd, marker_col_.handle(), 0, marker_count_ * 16, c.data());
        }
        if (marker_dirty_)
            reset_pending_ = true; // the change shows at once, not as particles cycle out
        marker_dirty_ = redraw_ = false;
    }
    if (smoke && (steps > 0 || reset_pending_)) {
        if (reset_pending_)
            clock_ = 0;
        else
            clock_ += steps;
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
        sp.pulse[0] = pulse_steps_;
        sp.pulse[1] = kPerPulse;
        sp.pulse[2] = std::int32_t(clock_);
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

std::uint32_t Tracers::record_arrows(VkCommandBuffer cmd, int slot, int axis, float pos,
                                     float spacing, float u_ref) {
    if (axis < 0 || axis > 2)
        return 0;
    spacing = std::max(spacing, 2.0f);
    const int a = axis == 0 ? 1 : 0, b = axis == 2 ? 1 : 2;
    const int ga = int(float(n_[a]) / spacing), gb = int(float(n_[b]) / spacing);
    const std::size_t count = std::min(std::size_t(ga) * std::size_t(gb), max_arrows_);
    if (count == 0)
        return 0;
    ArrowParams ap{};
    ap.dims[0] = n_[0];
    ap.dims[1] = n_[1];
    ap.dims[2] = n_[2];
    ap.dims[3] = axis;
    ap.grid[0] = ga;
    ap.grid[1] = std::int32_t(count / std::size_t(ga));
    ap.misc[0] = pos;
    ap.misc[1] = spacing;
    ap.misc[2] = u_ref;
    ap.misc[3] = kArrowScale;
    Context::barrier_full(cmd); // the previous frame's splats read these buffers
    arrows_.record_set(cmd, std::uint32_t(slot), &ap, std::uint32_t((count + 63) / 64));
    Context::barrier_compute_to_compute(cmd);
    return std::uint32_t(count * 3);
}

} // namespace windoa::render
