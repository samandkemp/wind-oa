// D3Q19 lattice Boltzmann solver with Smagorinsky LES on Vulkan compute.
// Memory layout: f direction-major [Q][x][y][z], C order.
// Specification: docs/THEORY.md, THEORY 1 - 4 (performance switches: 11).
//
// Not implemented: a per-cell surface force field for painting Cp (the
// volume renderer samples the pressure instead).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "windoa/context.hpp"

namespace windoa::lbm {

inline constexpr int Q = 19;

// Cell flags (host side u8; stored one u32 per cell on the GPU).
enum Flag : std::uint8_t {
    FLUID = 0,    // computed every step
    OBSTACLE = 1, // bounce-back, contributes to the model force / torque
    WALL = 2,     // bounce-back, no force bookkeeping
    LID = 3,      // bounce-back with the prescribed lid velocity
};

enum class AxisX : int { InletOutlet = 0, Periodic = 1 };
enum class AxisYZ : int { FreeSlip = 0, Periodic = 1 };

using Vec3 = std::array<float, 3>;

struct Config {
    int nx = 0, ny = 0, nz = 0;
    float tau = 0.504f;          // the app's default (TunnelSettings)
    float u_inlet = 0.0f;        // lattice units, +x
    float smagorinsky_cs = 0.1f; // 0 disables the LES model
    AxisX mode_x = AxisX::InletOutlet;
    AxisYZ mode_y = AxisYZ::FreeSlip;
    AxisYZ mode_z = AxisYZ::FreeSlip;
    bool regularised = false;       // Latt & Chopard 2006
    bool recursive = false;         // recursive-regularised (implies regularised)
    bool use_ibb = false;           // Bouzidi interpolated bounce-back (needs link_q)
    bool moving_boundaries = false; // Ladd moving walls on OBSTACLE cells
    int outlet_sponge = 0;          // absorbing sponge width in cells (0 = off)
    int sponge_target = 0;          // 0 = rho only at local u, 1 = full freestream
    // P4 performance switches -- exact identities (tools/lbm_equiv checks
    // every bit against them off): write rho / u only when read; skip the
    // force tree in workgroups with no model link.
    bool opt_lazy_macro = true;
    bool opt_sparse_forces = true;
    // f16 storage of the distributions (as f - w_i): halves the dominant
    // memory traffic. Not an identity -- an approximation, judged by the
    // physics gates (tools/lbm_equiv --f16 reports its deviations).
    bool storage_f16 = false;
};

struct MeanForces {
    Vec3 force{};  // mean per step over the window, lattice units
    Vec3 torque{}; // about torque_ref
    int steps = 0;
};

struct Health {
    float max_speed = 0.0f; // max |u| over FLUID cells (sound speed 0.577)
    int bad_cells = 0;      // non-finite FLUID cells
};

class Solver {
  public:
    Solver(Context& ctx, const Config& cfg);
    ~Solver();
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    const Config& config() const { return cfg_; }
    std::size_t cells() const { return n_; }
    std::int64_t steps_taken() const { return steps_; }

    // -- Setup ---------------------------------------------------------------
    // Flags in C order [x][y][z]. Bumps geometry_version().
    void set_flags(std::span<const std::uint8_t> flags);
    const std::vector<std::uint8_t>& flags() const { return flags_; }
    std::uint64_t geometry_version() const { return geometry_version_; }
    // Link fractions [Q][x][y][z] (0..255, 128 ~ half-way). Needs use_ibb.
    void set_link_q(std::span<const std::uint8_t> q);
    // Fill both buffers with the equilibrium at (rho0, u); resets the step
    // count and the force window. Call after the flags are set.
    void init_equilibrium(float rho0, Vec3 u);

    // -- Runtime parameters (no recompilation; pushed per dispatch) -----------
    void set_inlet_velocity(float u) { cfg_.u_inlet = u; }
    float inlet_velocity() const { return cfg_.u_inlet; }
    void set_body_force(Vec3 g) { body_force_ = g; }
    void set_lid_velocity(Vec3 u) { lid_velocity_ = u; }
    void set_torque_ref(Vec3 p) { torque_ref_ = p; }

    // -- Stepping --------------------------------------------------------------
    void step(int n_steps = 1);
    // With a hook recorded after every step, in the same submission, given
    // the index of the macro buffer that step has written. A passive scalar
    // advected with the flow (Dye, for example) steps in lock-step this way.
    using StepHook = std::function<void(VkCommandBuffer cmd, int macro_index)>;
    void step(int n_steps, const StepHook& after_each);

    // -- Synthetic inlet turbulence -------------------------------------------
    // A periodic, divergence-free patch u'(s, y, z) (the curl of a Gaussian-
    // filtered random vector potential, unit rms per component) convected
    // past the inlet at u_conv cells / step (Taylor): the inlet becomes
    // (1, u_in (1 + intensity u')). intensity = rms fraction of U (0 = off:
    // the step is then bit-identical to a solver that never had it).
    // Changing only the intensity reuses the patch.
    void set_inlet_turbulence(float intensity, float length = 8.0f, int span = 512,
                              float u_conv = 0.05f, std::uint32_t seed = 11);
    void set_turbulence_convection(float u_conv) { turb_u_ = u_conv; }
    float inlet_turbulence() const { return turb_on_ ? turb_intensity_ : 0.0f; }
    std::vector<float> turbulence_patch(); // [s][y][z][3], for validation
    // Replace the patch (same shape; validation only: V21 injects a non-
    // solenoidal field to show the curl is what keeps the inlet quiet).
    void set_turbulence_patch(std::span<const float> patch);

    // -- Forces ----------------------------------------------------------------
    Vec3 obstacle_force();         // the most recent step
    MeanForces read_mean_forces(); // mean since the last call; restarts the window

    // -- Mass (closed domains) -------------------------------------------------
    double mean_fluid_density();
    double plane_mean_density(int x);             // upstream reference static pressure
    double enforce_mass(double target_rho = 1.0); // returns the mean found before

    // -- Health ----------------------------------------------------------------
    Health health();

    // -- State (the complete simulation state; rho, u are its moments) --------
    std::vector<float> get_state();           // [Q][x][y][z]
    void set_state(std::span<const float> f); // refreshes rho / u too
    std::vector<float> velocity();            // [x][y][z][3]
    std::vector<float> density();             // [x][y][z]
    // The live (u.xyz, rho) of a few cells, e.g. probes: one small copy per
    // cell, not a full-grid download.
    std::vector<std::array<float, 4>> macro_at(std::span<const std::size_t> cells);

    // -- GPU views (read-only; for the renderer) ---------------------------------
    // macro_buffer(k): (u.xyz, rho) per cell as vec4, for the pair k = 0, 1;
    // live_index() names the one holding the current state (it alternates
    // every step). Stable objects, so a consumer binds each once and picks by
    // live_index() -- no descriptor rewrites while frames are in flight.
    const Buffer& macro_buffer(int k) const { return macro_[k]; }
    int live_index() const { return parity_; }
    const Buffer& flag_buffer() const { return flag_buf_; } // one uint per cell

    // -- Per-cell body force -----------------------------------------------------
    // On: a vec4 per cell (xyz = force per unit volume, added to the uniform
    // body force) that the next steps read; whoever owns it writes it (the
    // actuator lines, THEORY 3.12). Off: the plain step, bit for bit. Not
    // while a submission is in flight.
    void enable_force_field(bool on);
    bool force_field_on() const { return force_field_on_; }
    Buffer& force_field_buffer() { return *force_field_; }

    // -- Moving boundaries -------------------------------------------------------
    void clear_wall_velocity();
    // u_wall = omega x (cell_centre - axis_point) on OBSTACLE cells, limited
    // to a cylinder (radius about the omega axis, half_len along it) when
    // radius is given. Overwrites those cells (call clear_wall_velocity()
    // first, then once per spinning part); returns the cells set.
    int set_rotation(Vec3 axis_point, Vec3 omega, std::optional<float> radius = {},
                     std::optional<float> half_len = {});
    // u_wall = u on OBSTACLE cells within a cylinder (radius about `axis`
    // through `point`, half_len along it): an engine port's face, a velocity
    // boundary (blowing along its normal, suction against it; THEORY 3.11).
    // Overwrites those cells; returns the cells set.
    int set_wall_velocity(Vec3 point, Vec3 axis, float radius, float half_len, Vec3 u);

  private:
    struct StatsSum {
        double max_speed = 0, bad = 0, excess = 0, count = 0;
    };
    StatsSum stats(int plane);
    void build_turbulence_patch(float length, int span, std::uint32_t seed);
    void push_params(void* out, const float aux[4]) const;
    int live() const { return parity_; } // index of the buffer holding the state

    Context& ctx_;
    Config cfg_;
    std::size_t n_;
    Groups groups_;

    Vec3 body_force_{};
    Vec3 lid_velocity_{};
    Vec3 torque_ref_{};

    std::vector<std::uint8_t> flags_;
    std::vector<float> u_wall_host_; // [cell][4], only with moving_boundaries
    std::uint64_t geometry_version_ = 0;
    int parity_ = 0;
    std::int64_t steps_ = 0;

    // GPU buffers
    Buffer f_[2];
    Buffer macro_[2];
    Buffer flag_buf_;
    Buffer link_q_;
    Buffer u_wall_;
    std::unique_ptr<Buffer> force_field_; // 16 bytes until enabled
    bool force_field_on_ = false;
    Buffer partials_;
    Buffer accum_;
    Buffer stats_partials_;

    // Kernels
    ComputeKernel step_, step_forced_; // without / with the per-cell force
    ComputeKernel reduce_;
    ComputeKernel init_;
    ComputeKernel stats_;
    ComputeKernel scale_;
    ComputeKernel refresh_;

    // Inlet turbulence (allocated on first use)
    bool turb_on_ = false;
    float turb_intensity_ = 0.0f, turb_u_ = 0.05f;
    double turb_phase_ = 0.0;
    int turb_span_ = 0;
    std::array<float, 3> turb_params_{-1.0f, 0.0f, 0.0f}; // length, span, seed built
    std::unique_ptr<Buffer> turb_patch_;
    std::unique_ptr<Buffer> probe_buf_; // macro_at readback
    std::unique_ptr<ComputeKernel> turb_;
};

} // namespace windoa::lbm
