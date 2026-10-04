// Rotors as actuator lines (THEORY 3.12): each blade a line of elements that
// sample the local flow, take lift and drag from a section polar (blade-
// element theory) and give the reaction to the air, smeared by a Gaussian of
// width eps, as a per-cell body force (lbm::Solver::enable_force_field). The
// blades are forces, not geometry, so the tip speed is not a wall speed:
// tip-speed ratios of a real rotor (a turbine's 7) stay within the lattice's
// Mach limit, which a moving wall's 0.08 would not.
//
// Stepping: once per flow step, after it -- lbm::Solver::step's per-step
// hook calls record_step() with the step's fresh macro buffer; the forces
// act on the next step.
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/lbm.hpp"

namespace windoa {

// One rotor, in cells. axis is the through-flow direction (unit): a turbine
// slows the air along it, a propeller drives it along it.
struct RotorSpec {
    std::array<float, 3> hub{}, axis{1.0f, 0.0f, 0.0f};
    float r_tip = 10.0f, r_hub = 1.0f;
    int blades = 3;
    int elements = 0;                    // per blade (0: about one per cell)
    std::vector<float> chord, twist_deg; // root to tip, evenly spaced (cells, degrees)
    float omega = 0.0f;                  // rad / step about the axis
    float cl_alpha = 6.2832f, alpha_stall_deg = 12.0f, cd0 = 0.012f;
    // Validation: a uniformly loaded actuator disc, disc_force along the axis
    // (on the disc; the air gets the reaction), rings x spokes elements.
    bool disc = false;
    float disc_force = 0.0f;
    int disc_rings = 12, disc_spokes = 48;
};

class ActuatorLines {
  public:
    ActuatorLines(Context& ctx, lbm::Solver& solver, std::vector<RotorSpec> rotors,
                  float eps = 2.0f);
    ~ActuatorLines(); // turns the solver's force field off
    ActuatorLines(const ActuatorLines&) = delete;
    ActuatorLines& operator=(const ActuatorLines&) = delete;

    void set_omega(std::size_t rotor, float omega);
    // After a flow step whose output is macro_buffer(macro_index): the
    // element forces, then the body force for the next step.
    void record_step(VkCommandBuffer cmd, int macro_index);
    // Headless: n coupled steps.
    void step_with(lbm::Solver& solver, int n);

    // On each rotor's blades at the last step: the force, the thrust (along
    // the axis) and the torque about it; power = torque x omega (> 0: the
    // flow drives the rotor, a turbine).
    struct Loads {
        std::array<double, 3> force{};
        double thrust = 0.0, torque = 0.0, power = 0.0;
    };
    std::vector<Loads> loads();
    // Hub -> tip of every blade at the current angle (drawing).
    std::vector<std::array<std::array<float, 3>, 2>> blade_lines() const;
    // Each blade's planform at the current angle as four segments (root and tip
    // chords, leading and trailing edges; the leading edge c/4 ahead of the
    // line in the direction of rotation).
    std::vector<std::array<std::array<float, 3>, 2>> blade_outlines() const;
    const std::vector<RotorSpec>& rotors() const { return rotors_; }

  private:
    struct Box {
        std::array<int, 3> lo, size;
    };
    Context& ctx_;
    lbm::Solver& solver_;
    std::vector<RotorSpec> rotors_;
    std::vector<int> n_el_, offset_, sec_offset_;
    std::vector<double> angle_;
    std::vector<std::array<float, 3>> e1_;
    Box box_{}; // every rotor's reach
    float eps_;
    int total_ = 0;
    std::unique_ptr<Buffer> elements_, sections_;
    ComputeKernel elem_, spread_;
};

} // namespace windoa
