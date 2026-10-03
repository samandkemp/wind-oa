// Time-averaged flow: the running, step-weighted mean and variance of the
// velocity and density in every cell, accumulated on the GPU from a solver's
// macroscopic buffer, and the control-volume momentum balance built on them
// (the wake survey). Specification: THEORY 12.1, 12.2.
//
// Welford's update (weighted, as West 1979) keeps the f32 mean relative to
// the sample rather than to a growing total, so a long window loses no
// precision; the density is stored as rho - 1 for the same reason.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "windoa/context.hpp"

namespace windoa {

class FlowStats {
  public:
    FlowStats(Context& ctx, std::size_t cells);
    FlowStats(const FlowStats&) = delete;
    FlowStats& operator=(const FlowStats&) = delete;

    // Forget every sample.
    void reset();
    // Add one sample: `macro` (vec4 per cell: u.xyz, rho) standing for
    // `weight` solver steps. Submits and waits.
    void add(const Buffer& macro, double weight);

    int samples() const { return samples_; }
    double weight() const { return weight_; } // total solver steps represented
    // vec4 per cell: <u>, <rho> - 1
    const Buffer& mean() const { return mean_; }
    // vec4 per cell: sum of w (x - <x>)^2 per component; variance = m2 / weight()
    const Buffer& m2() const { return m2_; }

    // Host copies of the cells [first, first + count), 4 floats per cell:
    // the mean as stored (u.xyz, rho - 1: add the 1 in double) and the
    // variance.
    void read(std::size_t first, std::size_t count, std::vector<float>& mean,
              std::vector<float>& variance);

  private:
    Context& ctx_;
    std::size_t n_;
    Groups groups_;
    Buffer mean_, m2_;
    ComputeKernel kernel_;
    const Buffer* bound_[2] = {nullptr, nullptr}; // macro buffers bound to sets 0, 1
    int samples_ = 0;
    double weight_ = 0.0;
};

// The x-momentum flux of the mean flow through the plane x = const, over its
// FLUID cells (THEORY 12.2): the momentum flux <rho>(<u_x>^2 + var u_x),
// the pressure <rho> / 3 and the viscous normal stress -2 nu <rho> d<u_x>/dx
// (central difference; nu the molecular viscosity). `flags` [x][y][z];
// 1 <= x <= nx - 2. Reads the planes x - 1 .. x + 1.
struct PlaneFlux {
    double momentum = 0.0; // the x-momentum flux, lattice units per step
    double mass = 0.0;     // sum of <rho> <u_x>
    std::size_t cells = 0; // FLUID cells in the plane
};
PlaneFlux plane_momentum_flux(FlowStats& stats, std::span<const std::uint8_t> flags, int nx, int ny,
                              int nz, int x, double nu);

} // namespace windoa
