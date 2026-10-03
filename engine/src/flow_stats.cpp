#include "windoa/flow_stats.hpp"

#include <stdexcept>

#include "stats_accumulate_spv.hpp"

namespace windoa {

namespace {

constexpr std::uint32_t kLocal = 256;

struct StatsParams {
    std::uint32_t n[4]; // cells
    float w[4];         // w / W', w
};
static_assert(sizeof(StatsParams) == 32);

} // namespace

FlowStats::FlowStats(Context& ctx, std::size_t cells)
    : ctx_(ctx), n_(cells), groups_(groups_for(ctx, cells, kLocal)), mean_(ctx, cells * 16),
      m2_(ctx, cells * 16), kernel_(ctx, spv::stats_accumulate, 3, sizeof(StatsParams), 2) {
    reset();
}

void FlowStats::reset() {
    ctx_.fill_zero(mean_);
    ctx_.fill_zero(m2_);
    samples_ = 0;
    weight_ = 0.0;
}

void FlowStats::add(const Buffer& macro, double weight) {
    if (weight <= 0.0)
        return;
    if (macro.size() < n_ * 16)
        throw std::runtime_error("FlowStats::add: macro buffer too small");
    // Two descriptor sets, one per buffer the caller alternates between (the
    // solver's double-buffered macro); a third buffer replaces the older one.
    int set = macro.handle() == (bound_[0] ? bound_[0]->handle() : VK_NULL_HANDLE)   ? 0
              : macro.handle() == (bound_[1] ? bound_[1]->handle() : VK_NULL_HANDLE) ? 1
                                                                                     : -1;
    if (set < 0) {
        set = bound_[0] == nullptr ? 0 : 1;
        if (bound_[0] && bound_[1]) {
            bound_[0] = bound_[1];
            kernel_.bind(0, {bound_[0], &mean_, &m2_});
            set = 1;
        }
        bound_[set] = &macro;
        kernel_.bind(std::uint32_t(set), {&macro, &mean_, &m2_});
    }
    weight_ += weight;
    ++samples_;
    StatsParams p{};
    p.n[0] = std::uint32_t(n_);
    p.w[0] = float(weight / weight_);
    p.w[1] = float(weight);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        kernel_.record_set(cmd, std::uint32_t(set), &p, groups_.x, groups_.y);
    });
}

void FlowStats::read(std::size_t first, std::size_t count, std::vector<float>& mean,
                     std::vector<float>& variance) {
    if (first + count > n_)
        throw std::runtime_error("FlowStats::read: range outside the grid");
    mean.resize(count * 4);
    variance.resize(count * 4);
    ctx_.download(mean_, mean.data(), count * 16, first * 16);
    ctx_.download(m2_, variance.data(), count * 16, first * 16);
    const float inv_w = weight_ > 0.0 ? float(1.0 / weight_) : 0.0f;
    for (std::size_t i = 0; i < count * 4; ++i)
        variance[i] *= inv_w;
}

PlaneFlux plane_momentum_flux(FlowStats& stats, std::span<const std::uint8_t> flags, int nx, int ny,
                              int nz, int x, double nu) {
    if (x < 1 || x > nx - 2)
        throw std::runtime_error("plane_momentum_flux: plane outside 1 .. nx - 2");
    const std::size_t plane = std::size_t(ny) * nz;
    std::vector<float> mean, var;
    stats.read(std::size_t(x - 1) * plane, 3 * plane, mean, var); // planes x - 1, x, x + 1
    PlaneFlux out;
    for (std::size_t c = 0; c < plane; ++c) {
        const std::size_t cell = std::size_t(x) * plane + c;
        if (flags[cell] != 0) // FLUID only
            continue;
        const float* m = &mean[(plane + c) * 4];
        const double rho = 1.0 + double(m[3]), ux = m[0];
        const double ux_dn = mean[(2 * plane + c) * 4], ux_up = mean[c * 4];
        const double dudx = 0.5 * (ux_dn - ux_up);
        out.momentum += rho * (ux * ux + var[(plane + c) * 4]) + rho / 3.0 - 2.0 * nu * rho * dudx;
        out.mass += rho * ux;
        ++out.cells;
    }
    return out;
}

} // namespace windoa
