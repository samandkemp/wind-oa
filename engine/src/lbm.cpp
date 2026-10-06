#include "windoa/lbm.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "lbm_init_spv.hpp"
#include "lbm_linkq_spv.hpp"
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
constexpr std::uint32_t kStepBindings = 13; // lbm_step.comp: 9 + the fused dye's 4

std::array<std::uint32_t, 14> step_spec(const Config& c, bool force_field = false,
                                        bool dye = false) {
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
            force_field ? 1u : 0u,
            dye ? 1u : 0u};
}

// Bytes per stored distribution value, and the kernels' F16 constant.
std::size_t f_bytes(const Config& c) {
    return c.storage_f16 ? 2 : 4;
}
std::array<std::uint32_t, 1> f16_spec(const Config& c) {
    return {c.storage_f16 ? 1u : 0u};
}

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
      ring_slices_(ring_slices(groups_for(ctx, n_, kStepLocal).total)),
      partials_(ctx, std::size_t{2} * ring_slices_ * groups_for(ctx, n_, kStepLocal).total * 16),
      totals_(ctx, std::size_t{2} * ring_slices_ * 16), accum_(ctx, kAccumVec4 * 16),
      accum_mirror_{Buffer(ctx, kAccumVec4 * 16, MemoryUse::Readback),
                    Buffer(ctx, kAccumVec4 * 16, MemoryUse::Readback)},
      stats_partials_(ctx, std::size_t{groups_.total} * 16),
      step_(ctx, spv::lbm_step, kStepBindings, sizeof(Params), 2, step_spec(cfg)),
      step_forced_(ctx, spv::lbm_step, kStepBindings, sizeof(Params), 2, step_spec(cfg, true)),
      reduce_(ctx, spv::lbm_reduce, 3, sizeof(Params)),
      init_(ctx, spv::lbm_init, 4, sizeof(Params), 1, f16_spec(cfg)),
      stats_(ctx, spv::lbm_stats, 3, sizeof(Params), 2),
      scale_(ctx, spv::lbm_scale, 3, sizeof(Params), 2, f16_spec(cfg)),
      refresh_(ctx, spv::lbm_refresh, 3, sizeof(Params), 2, f16_spec(cfg)),
      linkq_(ctx, spv::lbm_linkq, 2, sizeof(Params)), dye_dummy_(ctx, 16) {
    if (cfg.storage_f16 && !ctx.gpu().storage_16bit)
        throw std::runtime_error("lbm::Solver: f16 storage needs 16-bit storage buffers");
    bind_step_sets();
    // set k reads f_[k] / macro_[k] and writes the other pair.
    for (int k = 0; k < 2; ++k) {
        stats_.bind(k, {&flag_buf_, &macro_[k], &stats_partials_});
        scale_.bind(k, {&f_[k], &flag_buf_, &macro_[k]});
        refresh_.bind(k, {&f_[k], &flag_buf_, &macro_[k]});
    }
    reduce_.bind({&partials_, &totals_, &accum_});
    for (Buffer& m : accum_mirror_)
        std::memset(m.data(), 0, kAccumVec4 * 16);
    if (cfg.use_ibb) // every link half-way until set (set_link_q_slab writes a slab)
        ctx_.fill_u32(link_q_, 0x80808080u);
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
    if (fused_on_) { // the fused dye's TRT rates (lbm_params.glsl)
        p.u_lid[3] = fused_.omega_minus;
        p.torque_ref[3] = fused_.omega_plus;
    }
    if (aux)
        std::memcpy(p.aux, aux, sizeof(p.aux));
    std::memcpy(out, &p, sizeof(p));
}

// -- Setup ---------------------------------------------------------------------

void Solver::set_flags(std::span<const std::uint8_t> flags) {
    if (flags.size() != n_)
        throw std::runtime_error("set_flags: wrong size");
    // Only the range that changed goes up (a new model rewrites its slab of
    // the grid, not the grid): all of it the first time, or when the caller
    // passes the solver's own copy.
    std::size_t lo = 0, hi = n_;
    if (flags_uploaded_ && flags.data() != flags_.data()) {
        lo = std::size_t(std::mismatch(flags.begin(), flags.end(), flags_.begin()).first -
                         flags.begin());
        hi = lo;
        for (std::size_t c = n_; c > lo; --c)
            if (flags[c - 1] != flags_[c - 1]) {
                hi = c;
                break;
            }
    }
    if (flags.data() != flags_.data())
        std::copy(flags.begin() + std::ptrdiff_t(lo), flags.begin() + std::ptrdiff_t(hi),
                  flags_.begin() + std::ptrdiff_t(lo));
    if (hi > lo) {
        std::vector<std::uint32_t> words(flags_.begin() + std::ptrdiff_t(lo),
                                         flags_.begin() + std::ptrdiff_t(hi));
        ctx_.upload(flag_buf_, words.data(), words.size() * sizeof(std::uint32_t),
                    lo * sizeof(std::uint32_t));
    }
    flags_uploaded_ = true;
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
    link_x0_ = 0; // all of it may differ from half-way now
    link_x1_ = cfg_.nx;
}

void Solver::set_link_q_sparse(int x0, int x1, std::span<const std::uint32_t> index,
                               std::span<const std::uint8_t> q) {
    if (!cfg_.use_ibb)
        throw std::runtime_error("set_link_q_sparse: solver built without use_ibb");
    if (index.size() != q.size())
        throw std::runtime_error("set_link_q_sparse: index and value counts differ");
    const std::size_t plane = std::size_t(cfg_.ny) * cfg_.nz;
    if (plane % 4 != 0) { // the fill below needs whole words a plane: the dense path
        std::vector<std::uint8_t> dense(Q * n_, 128);
        for (std::size_t k = 0; k < index.size(); ++k)
            dense.at(index[k]) = q[k];
        set_link_q(dense);
        return;
    }
    x0 = std::clamp(x0, 0, cfg_.nx);
    x1 = std::clamp(x1, x0, cfg_.nx);
    // The union with the slab set before: half-way where the new one ends.
    const int lo = link_x1_ > link_x0_ ? std::min(x0, link_x0_) : x0;
    const int hi = link_x1_ > link_x0_ ? std::max(x1, link_x1_) : x1;
    link_x0_ = x0;
    link_x1_ = x1;
    const std::size_t n = index.size();
    if (n > 0) {
        if (!link_entries_ || link_entries_->size() < n * 8)
            link_entries_ = std::make_unique<Buffer>(ctx_, std::max<std::size_t>(n * 8, 4096),
                                                     MemoryUse::Upload);
        auto* e = static_cast<std::uint32_t*>(link_entries_->data());
        for (std::size_t k = 0; k < n; ++k) {
            if (index[k] >= Q * n_)
                throw std::runtime_error("set_link_q_sparse: link outside the grid");
            e[2 * k] = index[k];
            e[2 * k + 1] = q[k];
        }
        linkq_.bind({&link_q_, link_entries_.get()});
    }
    Params p;
    const float aux[4] = {static_cast<float>(n), 0.0f, 0.0f, 0.0f};
    push_params(&p, aux);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) {
        if (hi > lo) {
            for (int d = 0; d < Q; ++d) {
                const VkDeviceSize off = VkDeviceSize(d) * n_ + VkDeviceSize(lo) * plane;
                const VkDeviceSize len = VkDeviceSize(hi - lo) * plane;
                vkCmdFillBuffer(cmd, link_q_.handle(), off, len, 0x80808080u);
            }
            Context::barrier_full(cmd);
        }
        if (n > 0)
            linkq_.record(cmd, &p, std::uint32_t((n + 255) / 256));
    });
}

void Solver::init_equilibrium(float rho0, Vec3 u) {
    Params p;
    const float aux[4] = {u[0], u[1], u[2], rho0};
    push_params(&p, aux);
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) { init_.record(cmd, &p, groups_.x, groups_.y); });
    ctx_.fill_zero(accum_);
    for (Buffer& m : accum_mirror_)
        std::memset(m.data(), 0, kAccumVec4 * 16);
    window_reset_pending_ = false;
    parity_ = 0;
    steps_ = 0;
}

// -- Stepping --------------------------------------------------------------------

// Set k reads f_[k] / macro_[k] and writes the other pair; with a fused
// scalar it reads the scalar's live populations in the same parity (offset
// by fused_bound_, which follows the scalar's own parity), else placeholders.
void Solver::bind_step_sets() {
    const int d = fused_on_ ? ((*fused_.parity ^ parity_) & 1) : 0;
    for (int k = 0; k < 2; ++k) {
        const int o = 1 - k;
        const Buffer* g_src = fused_on_ ? fused_.g[k ^ d] : &dye_dummy_;
        const Buffer* g_dst = fused_on_ ? fused_.g[1 - (k ^ d)] : &dye_dummy_;
        const Buffer* conc = fused_on_ ? fused_.conc : &dye_dummy_;
        const Buffer* src = fused_on_ ? fused_.src : &dye_dummy_;
        for (ComputeKernel* sk : {&step_, &step_forced_, step_dye_.get(), step_forced_dye_.get()})
            if (sk)
                sk->bind(k, {&f_[k], &f_[o], &flag_buf_, &macro_[k], &macro_[o], &link_q_, &u_wall_,
                             &partials_, force_field_.get(), g_src, g_dst, conc, src});
    }
    fused_bound_ = d;
}

void Solver::fuse_scalar(const FusedScalar* s) {
    fused_on_ = s != nullptr;
    if (s) {
        if (!s->g[0] || !s->g[1] || !s->conc || !s->src || !s->parity)
            throw std::runtime_error("Solver::fuse_scalar: missing buffers");
        fused_ = *s;
        if (!step_dye_) { // built on first use: most runs never fuse a scalar
            constexpr auto push = static_cast<std::uint32_t>(sizeof(Params));
            step_dye_ = std::make_unique<ComputeKernel>(ctx_, spv::lbm_step, kStepBindings, push, 2,
                                                        step_spec(cfg_, false, true));
            step_forced_dye_ = std::make_unique<ComputeKernel>(
                ctx_, spv::lbm_step, kStepBindings, push, 2u, step_spec(cfg_, true, true));
        }
    } else {
        fused_ = {};
    }
    bind_step_sets();
}

// The steps' force partials go into consecutive slices of a ring, folded by
// one reduce pass for all of them (lbm_reduce.comp): a one-workgroup reduce
// after every step left the GPU idle for 5.8 % of the time on the fast grid.
// The ring holds up to 32 MB (the fast grid's whole submit; the ultra
// grid's in steps of 14).
int Solver::ring_slices(std::uint32_t step_groups) {
    const std::size_t slice_bytes = std::size_t{2} * step_groups * 16;
    return int(std::clamp<std::size_t>((32u << 20) / slice_bytes, 1, kStepsPerSubmit));
}

void Solver::record_reduce(VkCommandBuffer cmd, std::uint32_t step_groups, int steps) {
    Params p;
    float aux[4] = {static_cast<float>(step_groups), 0.0f, static_cast<float>(steps), 0.0f};
    push_params(&p, aux);
    reduce_.record(cmd, &p, std::uint32_t(steps)); // fold: a workgroup per step
    Context::barrier_compute_to_compute(cmd);
    aux[3] = 1.0f;
    push_params(&p, aux);
    reduce_.record(cmd, &p, 1); // accumulate, in step order
    Context::barrier_compute_to_compute(cmd);
}

void Solver::step(int n_steps) {
    step(n_steps, StepHook{});
}

void Solver::step(int n_steps, const StepHook& after_each) {
    run_steps(n_steps, after_each, false, 0);
}

Solver::Ticket Solver::step_async(int n_steps, const StepHook& after_each, const Tail& tail) {
    const int slot = async_slot_;
    async_slot_ ^= 1;
    return {run_steps(n_steps, after_each, true, slot, &tail), slot};
}

void Solver::finish(const Ticket& t) {
    ctx_.wait(t.submit);
    finish_readings(t.slot);
}

std::uint64_t Solver::run_steps(int n_steps, const StepHook& after_each, bool async, int slot,
                                const Tail* tail) {
    std::uint64_t ticket = 0;
    bool first = true;
    Params p;
    const Groups sg = groups_for(ctx_, n_, kStepLocal);
    const float aux[4] = {static_cast<float>(sg.total), 0.0f, 0.0f, 0.0f};
    push_params(&p, aux);
    const bool turb = turb_on_ && cfg_.mode_x == AxisX::InletOutlet && turb_;
    if (fused_on_ && ((*fused_.parity ^ parity_) & 1) != fused_bound_)
        bind_step_sets(); // the scalar's parity moved (a clear): realign its buffers
    ComputeKernel& kernel = fused_on_ ? (force_field_on_ ? *step_forced_dye_ : *step_dye_)
                                      : (force_field_on_ ? step_forced_ : step_);
    const std::uint32_t plane = static_cast<std::uint32_t>(cfg_.ny * cfg_.nz);
    while (n_steps > 0) {
        const int batch = std::min(n_steps, kStepsPerSubmit);
        const bool last = batch == n_steps;
        auto record = [&](VkCommandBuffer cmd) {
            // The window restarts where read_mean_forces() asked, and at the
            // start of every pipelined batch (each keeps its own window).
            if (window_reset_pending_ || (async && first)) {
                vkCmdFillBuffer(cmd, accum_.handle(), 2 * 16, 3 * 16, 0u);
                Context::barrier_full(cmd);
                window_reset_pending_ = false;
            }
            int par = parity_;
            int slice = 0; // the partials ring slice this step writes
            for (int s = 0; s < batch; ++s) {
                // rho / u are written on the steps someone reads them: the
                // last of the submit, or every step for a per-step hook
                p.aux[1] = (after_each || s == batch - 1) ? 1.0f : 0.0f;
                p.dims[3] = slice * static_cast<std::int32_t>(sg.total);
                kernel.record_set(cmd, static_cast<std::uint32_t>(par), &p, sg.x, sg.y);
                Context::barrier_compute_to_compute(cmd);
                if (++slice == ring_slices_ || s == batch - 1) { // fold the ring's steps
                    record_reduce(cmd, sg.total, slice);
                    slice = 0;
                }
                if (turb) { // overwrite the inlet plane the step has written
                    Params tp;
                    const float ta[4] = {static_cast<float>(turb_phase_), turb_intensity_,
                                         static_cast<float>(turb_span_), 0.0f};
                    push_params(&tp, ta);
                    turb_->record_set(cmd, static_cast<std::uint32_t>(par), &tp,
                                      (plane + kLocal - 1) / kLocal);
                    turb_phase_ = std::fmod(turb_phase_ + turb_u_, double(turb_span_));
                }
                if (turb) // the next step or the hook reads the plane it wrote
                    Context::barrier_compute_to_compute(cmd);
                par ^= 1;
                if (after_each)
                    after_each(cmd, par); // macro_[par] is this step's output
            }
            if (last)
                record_readings(cmd, par, slot);
            if (last && tail && tail->record) {
                Context::barrier_full(cmd);
                tail->record(cmd, par);
            }
        };
        const bool signal = last && tail && tail->signal != VK_NULL_HANDLE;
        if (async)
            ticket = ctx_.submit_async(record, signal ? tail->signal : VK_NULL_HANDLE,
                                       signal ? tail->value : 0);
        else
            ctx_.submit_and_wait(record);
        first = false;
        if (batch % 2) {
            parity_ ^= 1;
            if (fused_on_)
                *fused_.parity ^= 1; // its live buffer moves with the flow's
        }
        steps_ += batch;
        n_steps -= batch;
        if (last && !async)
            finish_readings(slot);
    }
    return ticket;
}

void Solver::request_plane_density(int x) {
    want_plane_ = std::clamp(x, 0, cfg_.nx - 1);
}

void Solver::request_probes(std::span<const std::size_t> cells) {
    for (std::size_t c : cells)
        if (c >= n_)
            throw std::runtime_error("Solver::request_probes: cell outside the grid");
    want_probes_.assign(cells.begin(), cells.end());
}

// The end of a step() submission: the force accumulator into its mirror
// (after the last reduce), then any requested statistics and probes on the
// state the steps leave (`live`).
void Solver::record_readings(VkCommandBuffer cmd, int live, int slot) {
    const VkBufferCopy acc{0, 0, kAccumVec4 * 16};
    vkCmdCopyBuffer(cmd, accum_.handle(), accum_mirror_[slot].handle(), 1, &acc);
    const int planes[2] = {want_health_ ? -1 : INT_MIN, want_plane_};
    for (int k = 0; k < 2; ++k) {
        if (planes[k] == INT_MIN || (k == 1 && planes[k] < 0))
            continue;
        std::unique_ptr<Buffer>& mirror = stats_mirror_[slot][k];
        if (!mirror)
            mirror = std::make_unique<Buffer>(ctx_, stats_partials_.size(), MemoryUse::Readback);
        Params p;
        const float aux[4] = {static_cast<float>(planes[k]), 0.0f, 0.0f, 0.0f};
        push_params(&p, aux);
        stats_.record_set(cmd, static_cast<std::uint32_t>(live), &p, groups_.x, groups_.y);
        Context::barrier_full(cmd); // compute writes -> copy; and the copy before a reuse
        const VkBufferCopy all{0, 0, stats_partials_.size()};
        vkCmdCopyBuffer(cmd, stats_partials_.handle(), mirror->handle(), 1, &all);
        Context::barrier_full(cmd);
    }
    if (!want_probes_.empty()) {
        const VkDeviceSize bytes = want_probes_.size() * 16;
        std::unique_ptr<Buffer>& mirror = probe_mirror_[slot];
        if (!mirror || mirror->size() < bytes)
            mirror = std::make_unique<Buffer>(ctx_, bytes, MemoryUse::Readback);
        std::vector<VkBufferCopy> regions(want_probes_.size());
        for (std::size_t i = 0; i < want_probes_.size(); ++i)
            regions[i] = {want_probes_[i] * 16, i * 16, 16};
        vkCmdCopyBuffer(cmd, macro_[live].handle(), mirror->handle(), std::uint32_t(regions.size()),
                        regions.data());
    }
    slot_read_[slot] = {true, want_health_, want_plane_, want_probes_.size()};
    want_health_ = false;
    want_plane_ = -1;
    want_probes_.clear();
}

void Solver::finish_readings(int slot) {
    SlotReadings& r = slot_read_[slot];
    read_slot_ = slot;
    if (!r.recorded)
        return;
    r.recorded = false;
    if (r.health) {
        const StatsSum s = sum_stats(static_cast<const float*>(stats_mirror_[slot][0]->data()));
        last_health_ = {static_cast<float>(s.max_speed), static_cast<int>(s.bad)};
    }
    if (r.plane >= 0) {
        const StatsSum s = sum_stats(static_cast<const float*>(stats_mirror_[slot][1]->data()));
        last_plane_density_ = 1.0 + (s.count > 0 ? s.excess / s.count : 0.0);
    }
    last_probes_.resize(r.probes);
    if (r.probes > 0)
        std::memcpy(last_probes_.data(), probe_mirror_[slot]->data(), r.probes * 16);
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
    const Grid patch{span, ny, nz};
    auto at = [&](int s, int y, int z, int k) {
        s = (s + span) % span;
        y = (y + ny) % ny;
        z = (z + nz) % nz;
        return a[4 * patch.index(s, y, z) + k];
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
                const std::size_t c = patch.index(s, y, z);
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
    const float* a = static_cast<const float*>(accum_mirror_[read_slot_].data());
    return {a[0], a[1], a[2]};
}

// The window as the last step() submission left it; its restart is the next
// submission's first command, and the mirror's copy is cleared so a second
// read before any step finds an empty window.
MeanForces Solver::read_mean_forces() {
    float a[kAccumVec4 * 4];
    Buffer& mirror = accum_mirror_[read_slot_];
    std::memcpy(a, mirror.data(), sizeof(a));
    std::memset(static_cast<char*>(mirror.data()) + 2 * 16, 0, 3 * 16);
    window_reset_pending_ = true;
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
    return sum_stats(v.data());
}

Solver::StatsSum Solver::sum_stats(const float* v) const {
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
                f[i * n_ + c] = from_half(h[i * n_ + c]) + lattice::W[std::size_t(i)];
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
                h[i * n_ + c] = to_half(f[i * n_ + c] - lattice::W[std::size_t(i)]);
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
        bind_step_sets();
    }
    if (on)
        ctx_.fill_zero(*force_field_);
}

void Solver::clear_wall_velocity() {
    if (!cfg_.moving_boundaries)
        throw std::runtime_error("solver built without moving_boundaries");
    std::fill(u_wall_host_.begin(), u_wall_host_.end(), 0.0f);
    ctx_.fill_zero(u_wall_);
    wall_set_lo_ = wall_set_hi_ = 0;
}

void Solver::begin_wall_update() {
    if (!cfg_.moving_boundaries)
        throw std::runtime_error("solver built without moving_boundaries");
    if (wall_set_hi_ > wall_set_lo_)
        std::fill(u_wall_host_.begin() + std::ptrdiff_t(4 * wall_set_lo_),
                  u_wall_host_.begin() + std::ptrdiff_t(4 * wall_set_hi_), 0.0f);
    wall_dirty_lo_ = wall_set_lo_;
    wall_dirty_hi_ = wall_set_hi_;
    wall_set_lo_ = wall_set_hi_ = 0;
    wall_deferred_ = true;
}

void Solver::end_wall_update() {
    wall_deferred_ = false;
    if (wall_dirty_hi_ > wall_dirty_lo_)
        upload_wall_range(wall_dirty_lo_, wall_dirty_hi_);
    wall_dirty_lo_ = wall_dirty_hi_ = 0;
}

void Solver::mark_wall_cells(std::size_t lo, std::size_t hi) {
    auto extend = [&](std::size_t& a, std::size_t& b) {
        if (b <= a) {
            a = lo;
            b = hi;
        } else {
            a = std::min(a, lo);
            b = std::max(b, hi);
        }
    };
    extend(wall_set_lo_, wall_set_hi_);
    if (wall_deferred_)
        extend(wall_dirty_lo_, wall_dirty_hi_);
}

// The host copy and the device buffer agree outside [lo, hi), so uploading
// only that range leaves the device holding exactly the host copy.
void Solver::upload_wall_range(std::size_t lo, std::size_t hi) {
    ctx_.upload(u_wall_, u_wall_host_.data() + 4 * lo, (hi - lo) * 16, lo * 16);
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
    // Only the capture cylinder's bounding box can hold its cells (the whole
    // grid when there is no cylinder).
    int lo[3] = {0, 0, 0}, hi[3] = {cfg_.nx - 1, cfg_.ny - 1, cfg_.nz - 1};
    if (radius) {
        const int dims[3] = {cfg_.nx, cfg_.ny, cfg_.nz};
        for (int k = 0; k < 3; ++k) {
            const double ext = hl * std::abs(ax[k]) +
                               double(*radius) * std::sqrt(std::max(0.0, 1.0 - ax[k] * ax[k])) +
                               1.0;
            lo[k] = std::max(0, int(std::floor(axis_point[std::size_t(k)] - ext)));
            hi[k] = std::min(dims[k] - 1, int(std::ceil(axis_point[std::size_t(k)] + ext)));
        }
    }
    int count = 0;
    std::size_t first = n_, last = 0;
    for (int x = lo[0]; x <= hi[0]; ++x) {
        for (int y = lo[1]; y <= hi[1]; ++y) {
            for (int z = lo[2]; z <= hi[2]; ++z) {
                const std::size_t c = grid().index(x, y, z);
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
                first = std::min(first, c);
                last = std::max(last, c);
            }
        }
    }
    if (count > 0) {
        mark_wall_cells(first, last + 1);
        if (!wall_deferred_)
            upload_wall_range(first, last + 1);
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
    std::size_t first = n_, last = 0;
    const int r_cells = int(std::ceil(radius + half_len)) + 1;
    const int lo[3] = {std::max(0, int(point[0]) - r_cells), std::max(0, int(point[1]) - r_cells),
                       std::max(0, int(point[2]) - r_cells)};
    const int hi[3] = {std::min(cfg_.nx - 1, int(point[0]) + r_cells),
                       std::min(cfg_.ny - 1, int(point[1]) + r_cells),
                       std::min(cfg_.nz - 1, int(point[2]) + r_cells)};
    for (int x = lo[0]; x <= hi[0]; ++x)
        for (int y = lo[1]; y <= hi[1]; ++y)
            for (int z = lo[2]; z <= hi[2]; ++z) {
                const std::size_t c = grid().index(x, y, z);
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
                first = std::min(first, c);
                last = std::max(last, c);
            }
    if (count > 0) {
        mark_wall_cells(first, last + 1);
        if (!wall_deferred_)
            upload_wall_range(first, last + 1);
    }
    return count;
}

} // namespace windoa::lbm
