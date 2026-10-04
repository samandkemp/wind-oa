#include "windoa/euler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "euler_diag_spv.hpp"
#include "euler_dt_spv.hpp"
#include "euler_force_spv.hpp"
#include "euler_ghost_spv.hpp"
#include "euler_stage_spv.hpp"
#include "euler_update_spv.hpp"

namespace windoa::euler {

namespace {
constexpr std::uint32_t kLocal = 256;
constexpr int kStepsPerSubmit = 32; // ~12 dispatches each: well under the OS watchdog
constexpr float kPhiFar = 4.0f;
constexpr std::uint8_t kFluid = 0;
} // namespace

// Mirrors `Params` in shaders/euler_common.glsl.
struct Solver::Params {
    std::int32_t dims[4];
    std::int32_t bc[4];
    std::int32_t opt[4];
    float flow[4];
    float aux[4];
    float aux2[4];
};

Solver::Solver(Context& ctx, const Config& cfg)
    : ctx_(ctx), cfg_(cfg), n_(std::size_t(cfg.nx) * cfg.ny * cfg.nz),
      groups_(groups_for(ctx, n_, kLocal)), host_flags_(n_, kFluid), u_(ctx, n_ * 20),
      u1_(ctx, n_ * 20), flags_(ctx, n_ * 4), phi_(ctx, n_ * 4), gs_(ctx, n_ * 20),
      gn_(ctx, n_ * 16), gpw_(ctx, n_ * 4), dt_(ctx, 16),
      partials_(ctx, std::size_t{groups_.total} * 32), macro_(ctx, n_ * 16), rho_(ctx, n_ * 4),
      ports_(ctx, kMaxPorts * 5 * sizeof(float)),
      ghost_(ctx, spv::euler_ghost, 13, sizeof(Params), 2),
      update_(ctx, spv::euler_update, 13, sizeof(Params), 2),
      stage_(ctx, spv::euler_stage, 13, sizeof(Params)),
      dt_kernel_(ctx, spv::euler_dt, 13, sizeof(Params)),
      force_(ctx, spv::euler_force, 13, sizeof(Params)),
      diag_(ctx, spv::euler_diag, 13, sizeof(Params)) {
    static_assert(sizeof(Params) == 96, "Params must match euler_common.glsl");
    if (cfg.nx < 1 || cfg.ny < 1 || cfg.nz < 1)
        throw std::runtime_error("euler::Solver: empty grid");
    // Set 0: the state U is the input (stencil) and U0; U1 the output --
    // stage 1, and every other kernel. Set 1: U1 is the input, U both U0
    // and the output -- stage 2 (see euler_update.comp).
    for (ComputeKernel* k : {&ghost_, &update_, &stage_, &dt_kernel_, &force_, &diag_})
        k->bind(0, {&u_, &u_, &u1_, &flags_, &phi_, &gs_, &gn_, &gpw_, &dt_, &partials_, &macro_,
                    &rho_, &ports_});
    for (ComputeKernel* k : {&ghost_, &update_})
        k->bind(1, {&u1_, &u_, &u_, &flags_, &phi_, &gs_, &gn_, &gpw_, &dt_, &partials_, &macro_,
                    &rho_, &ports_});
    ctx_.fill_zero(flags_);
    ctx_.fill_zero(gn_);
    ctx_.fill_zero(dt_);
    ctx_.fill_zero(u1_);
    ctx_.fill_zero(ports_);
}

void Solver::set_ports(std::span<const std::array<float, 5>> states) {
    if (states.size() > kMaxPorts)
        throw std::runtime_error("euler::Solver::set_ports: too many ports");
    std::vector<float> w(kMaxPorts * 5, 0.0f);
    for (std::size_t k = 0; k < states.size(); ++k)
        for (std::size_t i = 0; i < 5; ++i)
            w[5 * k + i] = states[k][i];
    ctx_.upload(ports_, w.data(), w.size() * sizeof(float));
}

Solver::Params Solver::params(int axis, int first, int mode, float aux0, float aux1) const {
    Params p{};
    p.dims[0] = cfg_.nx;
    p.dims[1] = cfg_.ny;
    p.dims[2] = cfg_.nz;
    p.dims[3] = axis;
    p.bc[0] = static_cast<int>(cfg_.x_bc);
    p.bc[1] = static_cast<int>(cfg_.side_bc);
    p.bc[2] = cfg_.z_bc < 0 ? static_cast<int>(cfg_.side_bc) : cfg_.z_bc;
    p.bc[3] = static_cast<int>(cfg_.wall);
    p.opt[0] = static_cast<int>(cfg_.limiter);
    p.opt[1] = first;
    p.opt[2] = mode;
    p.flow[0] = cfg_.mach;
    p.flow[1] = flow_angle_;
    p.flow[2] = cfg_.cfl;
    p.flow[3] = dt_cap_;
    p.aux[0] = aux0;
    p.aux[1] = aux1;
    return p;
}

void Solver::set_flags(std::span<const std::uint8_t> flags) {
    if (flags.size() != n_)
        throw std::runtime_error("euler::Solver::set_flags: wrong size");
    host_flags_.assign(flags.begin(), flags.end());
    std::vector<std::uint32_t> w(host_flags_.begin(), host_flags_.end());
    ctx_.upload(flags_, w.data(), w.size() * 4);
    phi_exact_ = phi_ready_ = false;
    ++geometry_version_;
}

void Solver::set_distance(std::span<const float> phi) {
    if (phi.size() != n_)
        throw std::runtime_error("euler::Solver::set_distance: wrong size");
    std::vector<float> c(phi.begin(), phi.end());
    for (float& v : c)
        v = std::clamp(v, -kPhiFar, kPhiFar);
    ctx_.upload(phi_, c.data(), c.size() * 4);
    phi_exact_ = phi_ready_ = true;
}

// No exact distance: phi ~ (1/2 - s) / |grad s| near the surface from a
// [1,2,1]^2-per-axis smoothed indicator s (its 1/2 level set is the half-
// way wall); +-PHI_FAR elsewhere by the flag. Fine for thick bodies, too
// coarse for 1-cell features (which then fall back to the mirror). Once per
// geometry, on the host.
void Solver::phi_from_flags() {
    const int nx = cfg_.nx, ny = cfg_.ny, nz = cfg_.nz;
    auto id = [&](int x, int y, int z) { return (std::size_t(x) * ny + y) * nz + z; };
    std::vector<float> a(n_), b(n_);
    for (std::size_t c = 0; c < n_; ++c)
        a[c] = host_flags_[c] != kFluid ? 1.0f : 0.0f;
    auto blur = [&](const std::vector<float>& src, std::vector<float>& dst, int ax) {
        const int n[3] = {nx, ny, nz};
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y)
                for (int z = 0; z < nz; ++z) {
                    int p[3] = {x, y, z}, lo[3] = {x, y, z}, hi[3] = {x, y, z};
                    lo[ax] = std::max(p[ax] - 1, 0);
                    hi[ax] = std::min(p[ax] + 1, n[ax] - 1);
                    dst[id(x, y, z)] =
                        0.25f * (src[id(lo[0], lo[1], lo[2])] + 2.0f * src[id(x, y, z)] +
                                 src[id(hi[0], hi[1], hi[2])]);
                }
    };
    for (int it = 0; it < 2; ++it) {
        blur(a, b, 0);
        blur(b, a, 1);
        blur(a, b, 2);
        a.swap(b);
    }
    std::vector<float> phi(n_);
    const int n[3] = {nx, ny, nz};
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                float g2 = 0.0f;
                const int p[3] = {x, y, z};
                for (int ax = 0; ax < 3; ++ax) {
                    int lo[3] = {x, y, z}, hi[3] = {x, y, z};
                    lo[ax] = std::max(p[ax] - 1, 0);
                    hi[ax] = std::min(p[ax] + 1, n[ax] - 1);
                    const float g = (a[id(hi[0], hi[1], hi[2])] - a[id(lo[0], lo[1], lo[2])]) /
                                    std::max(float(hi[ax] - lo[ax]), 1.0f);
                    g2 += g * g;
                }
                const float g = std::sqrt(g2);
                float v = host_flags_[id(x, y, z)] != kFluid ? -kPhiFar : kPhiFar;
                if (g > 0.05f)
                    v = std::clamp((0.5f - a[id(x, y, z)]) / g, -kPhiFar, kPhiFar);
                phi[id(x, y, z)] = v;
            }
    ctx_.upload(phi_, phi.data(), phi.size() * 4);
    phi_ready_ = true;
}

void Solver::ensure_geometry() {
    if (!phi_ready_)
        phi_from_flags();
}

void Solver::set_flow_angle(float degrees) {
    flow_angle_ = degrees * 3.14159265f / 180.0f;
}

void Solver::init_freestream() {
    ensure_geometry();
    const Params p = params(0, 0, 0);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, dt_.handle(), 0, VK_WHOLE_SIZE, 0u); // time 0
        stage_.record(cmd, &p, groups_.x, groups_.y);
    });
    refresh();
}

void Solver::set_primitive(std::span<const float> w) {
    if (w.size() != n_ * 5)
        throw std::runtime_error("euler::Solver::set_primitive: wrong size");
    ensure_geometry();
    std::vector<float> u(n_ * 5);
    for (std::size_t c = 0; c < n_; ++c) {
        const float rho = w[5 * c], vx = w[5 * c + 1], vy = w[5 * c + 2], vz = w[5 * c + 3],
                    p = w[5 * c + 4];
        u[5 * c] = rho;
        u[5 * c + 1] = rho * vx;
        u[5 * c + 2] = rho * vy;
        u[5 * c + 3] = rho * vz;
        u[5 * c + 4] = p / (kGamma - 1.0f) + 0.5f * rho * (vx * vx + vy * vy + vz * vz);
    }
    ctx_.upload(u_, u.data(), u.size() * 4);
    ctx_.fill_zero(dt_);
    refresh();
}

void Solver::record_step(VkCommandBuffer cmd, const Marker* mark) {
    const Groups& g = groups_;
    auto done = [&](int category) {
        Context::barrier_compute_to_compute(cmd);
        if (mark)
            (*mark)(cmd, category);
    };
    const Params dt0 = params(0, 0, 0, 0.0f), dt1 = params(0, 0, 0, 1.0f, float(g.total));
    dt_kernel_.record(cmd, &dt0, g.x, g.y);
    Context::barrier_compute_to_compute(cmd);
    dt_kernel_.record(cmd, &dt1, 1);
    done(0);
    // Two SSP-RK2 stages, each: ghost states from the stage input, then the
    // fused flux + update (euler_update.comp). Set 0 reads U, writes U1;
    // set 1 reads U1, writes U.
    for (int stage = 1; stage <= 2; ++stage) {
        const std::uint32_t set = std::uint32_t(stage - 1);
        const Params gp = params();
        ghost_.record_set(cmd, set, &gp, g.x, g.y);
        done(1);
        const Params up = params(0, 0, stage);
        update_.record_set(cmd, set, &up, g.x, g.y);
        done(2);
    }
}

// GPU time per kernel category over `steps` steps (timestamps after each
// category's dispatch + barrier; a development measurement).
std::array<double, 4> Solver::profile(int steps) {
    ensure_geometry();
    const int per_step = 1 + 2 * (1 + 1);
    const std::uint32_t nq = std::uint32_t(steps * per_step + 1);
    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = nq;
    VkQueryPool pool = VK_NULL_HANDLE;
    vk_check(vkCreateQueryPool(ctx_.device(), &qi, nullptr, &pool), "vkCreateQueryPool");
    std::vector<int> cats;
    std::uint32_t q = 0;
    const Marker mark = [&](VkCommandBuffer cmd, int category) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, q++);
        cats.push_back(category);
    };
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        vkCmdResetQueryPool(cmd, pool, 0, nq);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, q++);
        for (int s = 0; s < steps; ++s)
            record_step(cmd, &mark);
    });
    std::vector<std::uint64_t> ts(q);
    vkGetQueryPoolResults(ctx_.device(), pool, 0, q, ts.size() * 8, ts.data(), 8,
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    vkDestroyQueryPool(ctx_.device(), pool, nullptr);
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physical(), &props);
    std::array<double, 4> ms{};
    for (std::size_t k = 1; k < ts.size(); ++k)
        ms[cats[k - 1]] += double(ts[k] - ts[k - 1]) * props.limits.timestampPeriod * 1e-6;
    steps_ += steps;
    return ms;
}

void Solver::step(int n) {
    ensure_geometry();
    while (n > 0) {
        const int batch = std::min(n, kStepsPerSubmit);
        ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
            for (int s = 0; s < batch; ++s)
                record_step(cmd);
        });
        steps_ += batch;
        n -= batch;
    }
}

double Solver::time() {
    float d[4];
    ctx_.download(dt_, d, sizeof(d));
    return d[1];
}

std::array<double, 3> Solver::body_force() {
    return body_loads({0.0f, 0.0f, 0.0f}).force;
}

Solver::Loads Solver::body_loads(const std::array<float, 3>& ref) {
    ensure_geometry();
    const Params p = params();
    Params pf = p;
    for (int a = 0; a < 3; ++a)
        pf.aux2[a] = ref[std::size_t(a)];
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        ghost_.record(cmd, &p, groups_.x, groups_.y);
        Context::barrier_compute_to_compute(cmd);
        force_.record(cmd, &pf, groups_.x, groups_.y);
    });
    std::vector<float> v(std::size_t{groups_.total} * 8);
    ctx_.download(partials_, v.data(), v.size() * 4);
    Loads l;
    for (std::size_t k = 0; k < groups_.total; ++k)
        for (std::size_t a = 0; a < 3; ++a) {
            l.force[a] += v[8 * k + a];
            l.moment[a] += v[8 * k + 4 + a];
        }
    return l;
}

void Solver::refresh() {
    const Params p = params();
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) { diag_.record(cmd, &p, groups_.x, groups_.y); });
}

Solver::Health Solver::health() {
    refresh();
    std::vector<float> v(std::size_t{groups_.total} * 4);
    ctx_.download(partials_, v.data(), v.size() * 4);
    Health h;
    double bad = 0.0;
    for (std::size_t k = 0; k < groups_.total; ++k) {
        h.max_mach = std::max(h.max_mach, v[4 * k]);
        bad += v[4 * k + 1];
    }
    h.bad_cells = int(bad);
    return h;
}

std::vector<float> Solver::primitive() {
    std::vector<float> u(n_ * 5), w(n_ * 5);
    ctx_.download(u_, u.data(), u.size() * 4);
    for (std::size_t c = 0; c < n_; ++c) {
        const float rho = std::max(u[5 * c], 1e-8f);
        const float vx = u[5 * c + 1] / rho, vy = u[5 * c + 2] / rho, vz = u[5 * c + 3] / rho;
        const float p =
            (kGamma - 1.0f) * (u[5 * c + 4] - 0.5f * rho * (vx * vx + vy * vy + vz * vz));
        w[5 * c] = rho;
        w[5 * c + 1] = vx;
        w[5 * c + 2] = vy;
        w[5 * c + 3] = vz;
        w[5 * c + 4] = std::max(p, 1e-8f);
    }
    return w;
}

} // namespace windoa::euler
