#include "windoa/lbm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "lbm_init_spv.hpp"
#include "lbm_reduce_spv.hpp"
#include "lbm_refresh_spv.hpp"
#include "lbm_scale_spv.hpp"
#include "lbm_stats_spv.hpp"
#include "lbm_step_spv.hpp"
#include "lbm_turb_spv.hpp"
#include "turb_blur_spv.hpp"
#include "windoa/half.hpp"

#include <random>

namespace windoa::lbm {

namespace {

// Mirrors `Params` in shaders/lbm_params.glsl (all 16-byte vectors, so the
// std430 push-constant layout is exactly this struct).
struct Params {
    std::int32_t dims[4];
    float flow[4];
    float g[4];
    float u_lid[4];
    float torque_ref[4];
    float aux[4];
};
static_assert(sizeof(Params) == 96, "Params must match lbm_params.glsl");

constexpr std::uint32_t kLocal = 256;     // workgroup size of the LBM kernels...
constexpr std::uint32_t kStepLocal = 256; // ...and of lbm_step.comp (must match its local_size_x)
constexpr int kStepsPerSubmit = 128;      // keeps each GPU batch well under the OS watchdog
constexpr std::size_t kAccumVec4 = 5;     // see lbm_reduce.comp

// Specialisation constants of lbm_step.comp.
std::array<std::uint32_t, 13> step_spec(const Config& c, bool force_field = false) {
    return {static_cast<std::uint32_t>(c.mode_x),
            static_cast<std::uint32_t>(c.mode_y),
            static_cast<std::uint32_t>(c.mode_z),
            (c.regularised || c.recursive) ? 1u : 0u,
            c.recursive ? 1u : 0u,
            c.use_ibb ? 1u : 0u,
            c.moving_boundaries ? 1u : 0u,
            static_cast<std::uint32_t>(std::max(0, c.outlet_sponge)),
            static_cast<std::uint32_t>(c.sponge_target),
            c.opt_lazy_macro ? 1u : 0u,
            c.opt_sparse_forces ? 1u : 0u,
            c.storage_f16 ? 1u : 0u,
            force_field ? 1u : 0u};
}

// Bytes per stored distribution value, and the kernels' F16 constant.
std::size_t f_bytes(const Config& c) {
    return c.storage_f16 ? 2 : 4;
}
std::array<std::uint32_t, 1> f16_spec(const Config& c) {
    return {c.storage_f16 ? 1u : 0u};
}

// D3Q19 rest weights (the f16 storage holds f - w_i)
constexpr float kW[19] = {1.0f / 3,  1.0f / 18, 1.0f / 18, 1.0f / 18, 1.0f / 18,
                          1.0f / 18, 1.0f / 18, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                          1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                          1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36};

std::size_t cell_count(const Config& c) {
    if (c.nx < 3 || c.ny < 1 || c.nz < 1) {
        throw std::runtime_error("lbm::Solver: grid must be at least 3 x 1 x 1");
    }
    return static_cast<std::size_t>(c.nx) * c.ny * c.nz;
}

} // namespace

Solver::Solver(Context& ctx, const Config& cfg)
    : ctx_(ctx), cfg_(cfg), n_(cell_count(cfg)), groups_(groups_for(ctx, n_, kLocal)),
      flags_(n_, FLUID), f_{Buffer(ctx, Q * n_ * f_bytes(cfg)), Buffer(ctx, Q * n_ * f_bytes(cfg))},
      macro_{Buffer(ctx, n_ * 16), Buffer(ctx, n_ * 16)},
      flag_buf_(ctx, n_ * sizeof(std::uint32_t)),
      link_q_(ctx, cfg.use_ibb ? (Q * n_ + 3) / 4 * 4 : 16),
      u_wall_(ctx, cfg.moving_boundaries ? n_ * 16 : 16),
      force_field_(std::make_unique<Buffer>(ctx, 16)),
      partials_(ctx, std::size_t{2} * groups_for(ctx, n_, kStepLocal).total * 16),
      accum_(ctx, kAccumVec4 * 16), stats_partials_(ctx, std::size_t{groups_.total} * 16),
      step_(ctx, spv::lbm_step, 9, sizeof(Params), 2, step_spec(cfg)),
      step_forced_(ctx, spv::lbm_step, 9, sizeof(Params), 2, step_spec(cfg, true)),
      reduce_(ctx, spv::lbm_reduce, 2, sizeof(Params)),
      init_(ctx, spv::lbm_init, 4, sizeof(Params), 1, f16_spec(cfg)),
      stats_(ctx, spv::lbm_stats, 3, sizeof(Params), 2),
      scale_(ctx, spv::lbm_scale, 3, sizeof(Params), 2, f16_spec(cfg)),
      refresh_(ctx, spv::lbm_refresh, 3, sizeof(Params), 2, f16_spec(cfg)) {
    if (cfg.storage_f16 && !ctx.gpu().storage_16bit)
        throw std::runtime_error("lbm::Solver: f16 storage needs 16-bit storage buffers");
    // set k reads f_[k] / macro_[k] and writes the other pair.
    for (int k = 0; k < 2; ++k) {
        const int o = 1 - k;
        for (ComputeKernel* sk : {&step_, &step_forced_})
            sk->bind(k, {&f_[k], &f_[o], &flag_buf_, &macro_[k], &macro_[o], &link_q_, &u_wall_,
                         &partials_, force_field_.get()});
        stats_.bind(k, {&flag_buf_, &macro_[k], &stats_partials_});
        scale_.bind(k, {&f_[k], &flag_buf_, &macro_[k]});
        refresh_.bind(k, {&f_[k], &flag_buf_, &macro_[k]});
    }
    reduce_.bind({&partials_, &accum_});
    init_.bind({&f_[0], &f_[1], &macro_[0], &macro_[1]});

    torque_ref_ = {cfg.nx / 2.0f, cfg.ny / 2.0f, cfg.nz / 2.0f};
    if (cfg.moving_boundaries)
        u_wall_host_.assign(n_ * 4, 0.0f);

    set_flags(flags_);
    if (cfg.use_ibb) {
        std::vector<std::uint8_t> half(Q * n_, 128); // 128 = exactly half-way: a no-op
        set_link_q(half);
    }
    if (cfg.moving_boundaries)
        ctx_.fill_zero(u_wall_);
    init_equilibrium(1.0f, {cfg.u_inlet, 0.0f, 0.0f});
}

Solver::~Solver() = default;

void Solver::push_params(void* out, const float aux[4]) const {
    Params p{};
    p.dims[0] = cfg_.nx;
    p.dims[1] = cfg_.ny;
    p.dims[2] = cfg_.nz;
    p.flow[0] = cfg_.u_inlet;
    p.flow[1] = cfg_.tau;
    // Smagorinsky prefactor, Hou et al. form: 18 sqrt(2) Cs^2
    p.flow[2] = 18.0f * 1.41421356f * cfg_.smagorinsky_cs * cfg_.smagorinsky_cs;
    for (int k = 0; k < 3; ++k) {
        p.g[k] = body_force_[k];
        p.u_lid[k] = lid_velocity_[k];
        p.torque_ref[k] = torque_ref_[k];
    }
    if (aux)
        std::memcpy(p.aux, aux, sizeof(p.aux));
    std::memcpy(out, &p, sizeof(p));
}

// -- Setup ---------------------------------------------------------------------

void Solver::set_flags(std::span<const std::uint8_t> flags) {
    if (flags.size() != n_)
        throw std::runtime_error("set_flags: wrong size");
    if (flags.data() != flags_.data())
        flags_.assign(flags.begin(), flags.end());
    std::vector<std::uint32_t> words(flags_.begin(), flags_.end());
    ctx_.upload(flag_buf_, words.data(), words.size() * sizeof(std::uint32_t));
    ++geometry_version_;
}

void Solver::set_link_q(std::span<const std::uint8_t> q) {
    if (!cfg_.use_ibb)
        throw std::runtime_error("set_link_q: solver built without use_ibb");
    if (q.size() != Q * n_)
        throw std::runtime_error("set_link_q: wrong size");
    // Byte k lands in word k / 4 at bits 8 (k % 4) on a little-endian host --
    // exactly what the shader's link_fraction() unpacks.
    ctx_.upload(link_q_, q.data(), q.size());
}

void Solver::init_equilibrium(float rho0, Vec3 u) {
    Params p;
    const float aux[4] = {u[0], u[1], u[2], rho0};
    push_params(&p, aux);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) { init_.record(cmd, &p, groups_.x, groups_.y); });
    ctx_.fill_zero(accum_);
    parity_ = 0;
    steps_ = 0;
}

// -- Stepping --------------------------------------------------------------------

void Solver::step(int n_steps) {
    step(n_steps, StepHook{});
}

void Solver::step(int n_steps, const StepHook& after_each) {
    Params p;
    const Groups sg = groups_for(ctx_, n_, kStepLocal);
    const float aux[4] = {static_cast<float>(sg.total), 0.0f, 0.0f, 0.0f};
    push_params(&p, aux);
    const bool turb = turb_on_ && cfg_.mode_x == AxisX::InletOutlet && turb_;
    const std::uint32_t plane = static_cast<std::uint32_t>(cfg_.ny * cfg_.nz);
    while (n_steps > 0) {
        const int batch = std::min(n_steps, kStepsPerSubmit);
        ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
            int par = parity_;
            for (int s = 0; s < batch; ++s) {
                // rho / u are written on the steps someone reads them: the
                // last of the submit, or every step for a per-step hook
                p.aux[1] = (after_each || s == batch - 1) ? 1.0f : 0.0f;
                (force_field_on_ ? step_forced_ : step_)
                    .record_set(cmd, static_cast<std::uint32_t>(par), &p, sg.x, sg.y);
                Context::barrier_compute_to_compute(cmd);
                reduce_.record(cmd, &p, 1);
                if (turb) { // overwrite the inlet plane the step has written
                    Params tp;
                    const float ta[4] = {static_cast<float>(turb_phase_), turb_intensity_,
                                         static_cast<float>(turb_span_), 0.0f};
                    push_params(&tp, ta);
                    turb_->record_set(cmd, static_cast<std::uint32_t>(par), &tp,
                                      (plane + kLocal - 1) / kLocal);
                    turb_phase_ = std::fmod(turb_phase_ + turb_u_, double(turb_span_));
                }
                Context::barrier_compute_to_compute(cmd);
                par ^= 1;
                if (after_each)
                    after_each(cmd, par); // macro_[par] is this step's output
            }
        });
        if (batch % 2)
            parity_ ^= 1;
        steps_ += batch;
        n_steps -= batch;
    }
}

// -- Inlet turbulence --------------------------------------------------------------

void Solver::set_inlet_turbulence(float intensity, float length, int span, float u_conv,
                                  std::uint32_t seed) {
    turb_intensity_ = intensity;
    turb_u_ = u_conv;
    if (intensity <= 0.0f) {
        turb_on_ = false;
        return;
    }
    const std::array<float, 3> params{length, float(span), float(seed)};
    if (params != turb_params_)
        build_turbulence_patch(length, span, seed);
    turb_on_ = true;
}

// Three Gaussian-filtered white-noise potentials A (the Fourier filter
// exp(-2 (pi L)^2 k^2) is a periodic Gaussian of sigma L cells), then
// u' = curl A by periodic central differences (s plays x), mean removed,
// scaled to unit rms per component. V21 checks its statistics.
void Solver::build_turbulence_patch(float length, int span, std::uint32_t seed) {
    const int ny = cfg_.ny, nz = cfg_.nz;
    const std::size_t cells = std::size_t(span) * ny * nz;
    std::vector<float> a(cells * 4, 0.0f);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (std::size_t c = 0; c < cells; ++c)
        for (int k = 0; k < 3; ++k)
            a[4 * c + k] = normal(rng);

    // Separable periodic Gaussian on the GPU (3 axes, ping-pong).
    const int radius = std::max(1, int(std::ceil(4.0f * length)));
    std::vector<float> w(2 * radius + 1);
    double wsum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        w[k + radius] = float(std::exp(-0.5 * double(k) * k / (double(length) * length)));
        wsum += w[k + radius];
    }
    for (float& v : w)
        v = float(v / wsum);
    Buffer b0(ctx_, cells * 16), b1(ctx_, cells * 16), wbuf(ctx_, w.size() * sizeof(float));
    ctx_.upload(b0, a.data(), a.size() * sizeof(float));
    ctx_.upload(wbuf, w.data(), w.size() * sizeof(float));
    struct BlurParams {
        std::int32_t dims[4];
        std::int32_t r[4];
    };
    ComputeKernel blur(ctx_, spv::turb_blur, 3, sizeof(BlurParams), 2);
    blur.bind(0, {&b0, &b1, &wbuf});
    blur.bind(1, {&b1, &b0, &wbuf});
    const Groups g = groups_for(ctx_, cells, kLocal);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        for (int ax = 0; ax < 3; ++ax) {
            const BlurParams bp{{span, ny, nz, ax}, {radius, 0, 0, 0}};
            blur.record_set(cmd, std::uint32_t(ax % 2), &bp, g.x, g.y);
            Context::barrier_compute_to_compute(cmd);
        }
    });
    ctx_.download(b1, a.data(), a.size() * sizeof(float)); // 3 passes: 0->1->0->1

    // u' = curl A (periodic central differences), mean-free, unit rms.
    auto at = [&](int s, int y, int z, int k) {
        s = (s + span) % span;
        y = (y + ny) % ny;
        z = (z + nz) % nz;
        return a[4 * ((std::size_t(s) * ny + y) * nz + z) + k];
    };
    std::vector<float> up(cells * 4, 0.0f);
    double mean[3] = {0, 0, 0};
    for (int s = 0; s < span; ++s)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                auto d = [&](int k, int ax) { // d A_k / d axis
                    const int o[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                    return 0.5f * (at(s + o[ax][0], y + o[ax][1], z + o[ax][2], k) -
                                   at(s - o[ax][0], y - o[ax][1], z - o[ax][2], k));
                };
                const std::size_t c = (std::size_t(s) * ny + y) * nz + z;
                const float u = d(2, 1) - d(1, 2);
                const float v = d(0, 2) - d(2, 0);
                const float ww = d(1, 0) - d(0, 1);
                up[4 * c] = u;
                up[4 * c + 1] = v;
                up[4 * c + 2] = ww;
                mean[0] += u;
                mean[1] += v;
                mean[2] += ww;
            }
    double ss = 0.0;
    for (int k = 0; k < 3; ++k)
        mean[k] /= double(cells);
    for (std::size_t c = 0; c < cells; ++c)
        for (int k = 0; k < 3; ++k) {
            up[4 * c + k] = float(up[4 * c + k] - mean[k]);
            ss += double(up[4 * c + k]) * up[4 * c + k];
        }
    const double rms = std::sqrt(ss / (3.0 * double(cells))); // per-component rms
    for (std::size_t c = 0; c < cells; ++c)
        for (int k = 0; k < 3; ++k)
            up[4 * c + k] = float(up[4 * c + k] / std::max(rms, 1e-12));

    turb_patch_ = std::make_unique<Buffer>(ctx_, cells * 16);
    ctx_.upload(*turb_patch_, up.data(), up.size() * sizeof(float));
    if (!turb_)
        turb_ =
            std::make_unique<ComputeKernel>(ctx_, spv::lbm_turb, 4u, std::uint32_t(sizeof(Params)),
                                            2u, std::span<const std::uint32_t>(f16_spec(cfg_)));
    for (int k = 0; k < 2; ++k)
        turb_->bind(std::uint32_t(k), {&f_[1 - k], &flag_buf_, &macro_[1 - k], turb_patch_.get()});
    turb_span_ = span;
    turb_phase_ = 0.0;
    turb_params_ = {length, float(span), float(seed)};
}

void Solver::set_turbulence_patch(std::span<const float> patch) {
    const std::size_t cells = std::size_t(turb_span_) * cfg_.ny * cfg_.nz;
    if (!turb_patch_ || patch.size() != cells * 3)
        throw std::runtime_error("set_turbulence_patch: needs set_inlet_turbulence first and "
                                 "span x ny x nz x 3 values");
    std::vector<float> v(cells * 4, 0.0f);
    for (std::size_t c = 0; c < cells; ++c)
        for (int k = 0; k < 3; ++k)
            v[4 * c + k] = patch[3 * c + k];
    ctx_.upload(*turb_patch_, v.data(), v.size() * sizeof(float));
}

std::vector<float> Solver::turbulence_patch() {
    if (!turb_patch_)
        return {};
    const std::size_t cells = std::size_t(turb_span_) * cfg_.ny * cfg_.nz;
    std::vector<float> v(cells * 4), out(cells * 3);
    ctx_.download(*turb_patch_, v.data(), v.size() * sizeof(float));
    for (std::size_t c = 0; c < cells; ++c)
        for (int k = 0; k < 3; ++k)
            out[3 * c + k] = v[4 * c + k];
    return out;
}

// -- Forces ------------------------------------------------------------------------

Vec3 Solver::obstacle_force() {
    float a[4];
    ctx_.download(accum_, a, sizeof(a));
    return {a[0], a[1], a[2]};
}

MeanForces Solver::read_mean_forces() {
    float a[kAccumVec4 * 4];
    ctx_.download(accum_, a, sizeof(a));
    const float zeros[12] = {};
    ctx_.upload(accum_, zeros, sizeof(zeros), 2 * 16); // restart the window
    MeanForces m;
    m.steps = static_cast<int>(a[16]);
    if (m.steps > 0) {
        for (int k = 0; k < 3; ++k) {
            m.force[k] = a[8 + k] / m.steps;
            m.torque[k] = a[12 + k] / m.steps;
        }
    }
    return m;
}

// -- Statistics ----------------------------------------------------------------------

Solver::StatsSum Solver::stats(int plane) {
    Params p;
    const float aux[4] = {static_cast<float>(plane), 0.0f, 0.0f, 0.0f};
    push_params(&p, aux);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        stats_.record_set(cmd, static_cast<std::uint32_t>(live()), &p, groups_.x, groups_.y);
    });
    std::vector<float> v(std::size_t{groups_.total} * 4);
    ctx_.download(stats_partials_, v.data(), v.size() * sizeof(float));
    StatsSum s;
    for (std::size_t g = 0; g < groups_.total; ++g) {
        s.max_speed = std::max(s.max_speed, static_cast<double>(v[4 * g]));
        s.bad += v[4 * g + 1];
        s.excess += v[4 * g + 2];
        s.count += v[4 * g + 3];
    }
    return s;
}

double Solver::mean_fluid_density() {
    const StatsSum s = stats(-1);
    return s.count > 0 ? 1.0 + s.excess / s.count : 0.0;
}

double Solver::plane_mean_density(int x) {
    const StatsSum s = stats(std::clamp(x, 0, cfg_.nx - 1));
    return 1.0 + (s.count > 0 ? s.excess / s.count : 0.0);
}

double Solver::enforce_mass(double target_rho) {
    const double mean = mean_fluid_density();
    if (mean > 1e-6) {
        Params p;
        const float aux[4] = {static_cast<float>(target_rho / mean), 0.0f, 0.0f, 0.0f};
        push_params(&p, aux);
        ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
            scale_.record_set(cmd, static_cast<std::uint32_t>(live()), &p, groups_.x, groups_.y);
        });
    }
    return mean;
}

Health Solver::health() {
    const StatsSum s = stats(-1);
    return {static_cast<float>(s.max_speed), static_cast<int>(s.bad)};
}

// -- State ---------------------------------------------------------------------------

std::vector<float> Solver::get_state() {
    std::vector<float> f(Q * n_);
    if (cfg_.storage_f16) { // binary16 of f - w_i
        std::vector<std::uint16_t> h(Q * n_);
        ctx_.download(f_[live()], h.data(), h.size() * sizeof(std::uint16_t));
        for (int i = 0; i < Q; ++i)
            for (std::size_t c = 0; c < n_; ++c)
                f[i * n_ + c] = from_half(h[i * n_ + c]) + kW[i];
        return f;
    }
    ctx_.download(f_[live()], f.data(), f.size() * sizeof(float));
    return f;
}

void Solver::set_state(std::span<const float> f) {
    if (f.size() != Q * n_)
        throw std::runtime_error("set_state: wrong size");
    if (cfg_.storage_f16) {
        std::vector<std::uint16_t> h(Q * n_);
        for (int i = 0; i < Q; ++i)
            for (std::size_t c = 0; c < n_; ++c)
                h[i * n_ + c] = to_half(f[i * n_ + c] - kW[i]);
        ctx_.upload(f_[0], h.data(), h.size() * sizeof(std::uint16_t));
        ctx_.upload(f_[1], h.data(), h.size() * sizeof(std::uint16_t));
    } else {
        ctx_.upload(f_[0], f.data(), f.size_bytes());
        ctx_.upload(f_[1], f.data(), f.size_bytes());
    }
    parity_ = 0;
    Params p;
    push_params(&p, nullptr);
    ctx_.submit_and_wait(
        [&](VkCommandBuffer cmd) { refresh_.record_set(cmd, 0, &p, groups_.x, groups_.y); });
}

std::vector<float> Solver::velocity() {
    std::vector<float> m(n_ * 4);
    ctx_.download(macro_[live()], m.data(), m.size() * sizeof(float));
    std::vector<float> u(n_ * 3);
    for (std::size_t c = 0; c < n_; ++c) {
        u[3 * c] = m[4 * c];
        u[3 * c + 1] = m[4 * c + 1];
        u[3 * c + 2] = m[4 * c + 2];
    }
    return u;
}

std::vector<float> Solver::density() {
    std::vector<float> m(n_ * 4);
    ctx_.download(macro_[live()], m.data(), m.size() * sizeof(float));
    std::vector<float> rho(n_);
    for (std::size_t c = 0; c < n_; ++c)
        rho[c] = m[4 * c + 3];
    return rho;
}

std::vector<std::array<float, 4>> Solver::macro_at(std::span<const std::size_t> cells) {
    std::vector<std::array<float, 4>> out(cells.size());
    if (cells.empty())
        return out;
    const VkDeviceSize bytes = cells.size() * 16;
    if (!probe_buf_ || probe_buf_->size() < bytes)
        probe_buf_ = std::make_unique<Buffer>(ctx_, bytes, MemoryUse::Readback);
    std::vector<VkBufferCopy> regions(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        if (cells[i] >= n_)
            throw std::runtime_error("Solver::macro_at: cell outside the grid");
        regions[i] = {cells[i] * 16, i * 16, 16};
    }
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        vkCmdCopyBuffer(cmd, macro_[live()].handle(), probe_buf_->handle(),
                        std::uint32_t(regions.size()), regions.data());
    });
    std::memcpy(out.data(), probe_buf_->data(), bytes);
    return out;
}

// -- Moving boundaries ------------------------------------------------------------------

void Solver::enable_force_field(bool on) {
    if (on == force_field_on_)
        return;
    force_field_on_ = on;
    if (on && force_field_->size() < n_ * 16) {
        force_field_ = std::make_unique<Buffer>(ctx_, n_ * 16);
        for (int k = 0; k < 2; ++k) {
            const int o = 1 - k;
            for (ComputeKernel* sk : {&step_, &step_forced_})
                sk->bind(k, {&f_[k], &f_[o], &flag_buf_, &macro_[k], &macro_[o], &link_q_, &u_wall_,
                             &partials_, force_field_.get()});
        }
    }
    if (on)
        ctx_.fill_zero(*force_field_);
}

void Solver::clear_wall_velocity() {
    if (!cfg_.moving_boundaries)
        throw std::runtime_error("solver built without moving_boundaries");
    std::fill(u_wall_host_.begin(), u_wall_host_.end(), 0.0f);
    ctx_.fill_zero(u_wall_);
}

int Solver::set_rotation(Vec3 axis_point, Vec3 omega, std::optional<float> radius,
                         std::optional<float> half_len) {
    if (!cfg_.moving_boundaries)
        throw std::runtime_error("solver built without moving_boundaries");
    const double on = std::sqrt(double(omega[0]) * omega[0] + double(omega[1]) * omega[1] +
                                double(omega[2]) * omega[2]);
    const double ax[3] = {omega[0] / std::max(on, 1e-12), omega[1] / std::max(on, 1e-12),
                          omega[2] / std::max(on, 1e-12)};
    const double hl = half_len.value_or(radius.value_or(0.0f));
    int count = 0;
    for (int x = 0; x < cfg_.nx; ++x) {
        for (int y = 0; y < cfg_.ny; ++y) {
            for (int z = 0; z < cfg_.nz; ++z) {
                const std::size_t c = (static_cast<std::size_t>(x) * cfg_.ny + y) * cfg_.nz + z;
                if (flags_[c] != OBSTACLE)
                    continue;
                const double r[3] = {x + 0.5 - axis_point[0], y + 0.5 - axis_point[1],
                                     z + 0.5 - axis_point[2]};
                if (radius) {
                    const double along = r[0] * ax[0] + r[1] * ax[1] + r[2] * ax[2];
                    const double px = r[0] - along * ax[0], py = r[1] - along * ax[1],
                                 pz = r[2] - along * ax[2];
                    const double perp = std::sqrt(px * px + py * py + pz * pz);
                    if (perp > *radius || std::abs(along) > hl)
                        continue;
                }
                // u_wall = omega x r
                u_wall_host_[4 * c] = static_cast<float>(omega[1] * r[2] - omega[2] * r[1]);
                u_wall_host_[4 * c + 1] = static_cast<float>(omega[2] * r[0] - omega[0] * r[2]);
                u_wall_host_[4 * c + 2] = static_cast<float>(omega[0] * r[1] - omega[1] * r[0]);
                ++count;
            }
        }
    }
    if (count > 0) {
        ctx_.upload(u_wall_, u_wall_host_.data(), u_wall_host_.size() * sizeof(float));
    }
    return count;
}

int Solver::set_wall_velocity(Vec3 point, Vec3 axis, float radius, float half_len, Vec3 u) {
    if (!cfg_.moving_boundaries)
        throw std::runtime_error("solver built without moving_boundaries");
    const double an = std::sqrt(double(axis[0]) * axis[0] + double(axis[1]) * axis[1] +
                                double(axis[2]) * axis[2]);
    const double ax[3] = {axis[0] / std::max(an, 1e-12), axis[1] / std::max(an, 1e-12),
                          axis[2] / std::max(an, 1e-12)};
    int count = 0;
    const int r_cells = int(std::ceil(radius + half_len)) + 1;
    const int lo[3] = {std::max(0, int(point[0]) - r_cells), std::max(0, int(point[1]) - r_cells),
                       std::max(0, int(point[2]) - r_cells)};
    const int hi[3] = {std::min(cfg_.nx - 1, int(point[0]) + r_cells),
                       std::min(cfg_.ny - 1, int(point[1]) + r_cells),
                       std::min(cfg_.nz - 1, int(point[2]) + r_cells)};
    for (int x = lo[0]; x <= hi[0]; ++x)
        for (int y = lo[1]; y <= hi[1]; ++y)
            for (int z = lo[2]; z <= hi[2]; ++z) {
                const std::size_t c = (static_cast<std::size_t>(x) * cfg_.ny + y) * cfg_.nz + z;
                if (flags_[c] != OBSTACLE)
                    continue;
                const double r[3] = {x + 0.5 - point[0], y + 0.5 - point[1], z + 0.5 - point[2]};
                const double along = r[0] * ax[0] + r[1] * ax[1] + r[2] * ax[2];
                const double px = r[0] - along * ax[0], py = r[1] - along * ax[1],
                             pz = r[2] - along * ax[2];
                if (std::sqrt(px * px + py * py + pz * pz) > radius || std::abs(along) > half_len)
                    continue;
                for (int k = 0; k < 3; ++k)
                    u_wall_host_[4 * c + std::size_t(k)] = u[std::size_t(k)];
                ++count;
            }
    if (count > 0)
        ctx_.upload(u_wall_, u_wall_host_.data(), u_wall_host_.size() * sizeof(float));
    return count;
}

} // namespace windoa::lbm
