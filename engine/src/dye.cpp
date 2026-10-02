#include "windoa/dye.hpp"

#include <stdexcept>

#include "dye_step_spv.hpp"

namespace windoa {

namespace {

constexpr std::uint32_t kLocal = 256;
constexpr float kW7[7] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f};

struct DyeParams {
    std::int32_t dims[4]; // nx, ny, nz, periodic bits
    float tau[4];         // omega_minus, omega_plus
};

} // namespace

Dye::Dye(Context& ctx, const lbm::Solver& solver, float tau, float tau_plus)
    : ctx_(ctx), solver_(solver), tau_(tau), tau_plus_(tau_plus > 0.0f ? tau_plus : tau),
      n_(solver.cells()), groups_(groups_for(ctx, n_, kLocal)),
      g_{Buffer(ctx, 7 * n_ * sizeof(float)), Buffer(ctx, 7 * n_ * sizeof(float))},
      conc_(ctx, n_ * sizeof(float)), src_(ctx, n_ * sizeof(float)),
      step_(ctx, spv::dye_step, 6, sizeof(DyeParams), 4) {
    for (int par = 0; par < 2; ++par)
        for (int m = 0; m < 2; ++m)
            step_.bind(std::uint32_t(par * 2 + m), {&g_[par], &g_[1 - par], &solver.flag_buffer(),
                                                    &solver.macro_buffer(m), &conc_, &src_});
    clear();
    ctx_.fill_zero(src_);
}

void Dye::clear() {
    ctx_.fill_zero(g_[0]);
    ctx_.fill_zero(g_[1]);
    ctx_.fill_zero(conc_);
    parity_ = 0;
}

void Dye::set_concentration(std::span<const float> c) {
    if (c.size() != n_)
        throw std::runtime_error("Dye::set_concentration: wrong size");
    std::vector<float> g(7 * n_);
    for (int i = 0; i < 7; ++i)
        for (std::size_t k = 0; k < n_; ++k)
            g[i * n_ + k] = kW7[i] * c[k];
    ctx_.upload(g_[0], g.data(), g.size() * sizeof(float));
    ctx_.upload(g_[1], g.data(), g.size() * sizeof(float));
    ctx_.upload(conc_, c.data(), c.size_bytes());
    parity_ = 0;
}

void Dye::set_sources(std::span<const float> rates) {
    if (rates.size() != n_)
        throw std::runtime_error("Dye::set_sources: wrong size");
    ctx_.upload(src_, rates.data(), rates.size_bytes());
}

void Dye::record_step(VkCommandBuffer cmd, int macro_index) {
    const lbm::Config& c = solver_.config();
    DyeParams p{};
    p.dims[0] = c.nx;
    p.dims[1] = c.ny;
    p.dims[2] = c.nz;
    p.dims[3] = (c.mode_x == lbm::AxisX::Periodic ? 1 : 0) |
                (c.mode_y == lbm::AxisYZ::Periodic ? 2 : 0) |
                (c.mode_z == lbm::AxisYZ::Periodic ? 4 : 0);
    p.tau[0] = 1.0f / tau_;
    p.tau[1] = 1.0f / tau_plus_;
    step_.record_set(cmd, std::uint32_t(parity_ * 2 + macro_index), &p, groups_.x, groups_.y);
    Context::barrier_compute_to_compute(cmd);
    parity_ ^= 1;
}

void Dye::step_with(lbm::Solver& solver, int n) {
    solver.step(n, [&](VkCommandBuffer cmd, int macro_index) { record_step(cmd, macro_index); });
}

std::vector<float> Dye::concentration() {
    std::vector<float> c(n_);
    ctx_.download(conc_, c.data(), c.size() * sizeof(float));
    return c;
}

} // namespace windoa
