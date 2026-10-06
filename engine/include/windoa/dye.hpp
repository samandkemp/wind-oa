// Dye -- a passive scalar (smoke / dye concentration) carried by the flow on
// its own D3Q7 lattice, TRT collision. It uses the same pull streaming and
// the same bounce-back on the same flags as the flow solver, so dye never
// leaks into a solid by construction; at tau 0.53 the diffusivity is small
// and physical (a Schmidt number), not a scheme's numerical smear.
// Specification: THEORY 6.
//
// Defaults by measurement: plain BGK at tau 0.515 flips population signs
// and the positivity clip then manufactures 5 % dye; TRT with the usual
// large tau_plus is unstable at the tunnel's cell Peclet ~80; tau 0.53 /
// tau_plus 1.0 keeps a blob's mass to 0.04 % and C bounded (<= 1.07)
// round the Ahmed body (V20).
//
// Stepping: once per flow step. attach() fuses it into the flow's own step
// kernel (lbm::Solver::fuse_scalar: no separate pass, and the flow's rho / u
// stay lazy), the default; record_step() is the reference path, a pass after
// each flow step through the solver's per-step hook. The two agree bit for
// bit (tools/lbm_equiv).
#pragma once

#include <span>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/lbm.hpp"

namespace windoa {

class Dye {
  public:
    Dye(Context& ctx, const lbm::Solver& solver, float tau = 0.53f, float tau_plus = 1.0f);
    ~Dye();
    Dye(const Dye&) = delete;
    Dye& operator=(const Dye&) = delete;

    double diffusivity() const { return (tau_ - 0.5) / 4.0; }

    void clear();
    // Initialise C with rest-equilibrium populations (validation use).
    void set_concentration(std::span<const float> c);
    // Per-cell relaxation rates towards C = 1, 0..1 (0 = no source).
    void set_sources(std::span<const float> rates);

    // Fused into `solver`'s steps until detach() (or destruction).
    void attach(lbm::Solver& solver);
    void detach();
    bool attached() const { return attached_ != nullptr; }

    // Reference path: one dye step after a flow step whose output is
    // macro_buffer(k).
    void record_step(VkCommandBuffer cmd, int macro_index);
    // Headless: n coupled steps, fused (attached for the call when not
    // already); step_with_reference() through the per-step hook instead.
    void step_with(lbm::Solver& solver, int n);
    void step_with_reference(lbm::Solver& solver, int n);

    const Buffer& concentration_buffer() const { return conc_; }
    std::vector<float> concentration();

  private:
    Context& ctx_;
    const lbm::Solver& solver_;
    float tau_, tau_plus_;
    std::size_t n_;
    Groups groups_;
    int parity_ = 0;
    lbm::Solver* attached_ = nullptr;
    Buffer g_[2];
    Buffer conc_;
    Buffer src_;
    ComputeKernel step_; // set = parity * 2 + macro index
};

} // namespace windoa
