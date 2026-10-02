// Compressible Euler solver (finite volume) -- the transonic path.
//
// The LBM is weakly compressible (error ~Ma^2, trusted to Ma ~0.3, no energy
// equation, no shocks). Transonic flow is shocks and supersonic pockets, so:
//   - conserved U = (rho, rho u, rho v, rho w, E), ideal gas gamma 1.4, on
//     the same grid and voxel flags as the LBM;
//   - MUSCL reconstruction of (rho, u, v, w, p) with a slope limiter, HLLC
//     at every face (keeps contacts / shear layers sharp);
//   - SSP-RK2 in time, the CFL dt computed on the device each step;
//   - image-point ghost-cell walls from an exact signed distance
//     (Voxeliser::signed_distance), curvature-corrected (Dadone & Grossman),
//     with an impermeable grid-axis mirror for 1-cell sheets (see the
//     shaders for the measured failures each rule fixes).
// Units: freestream rho = 1, sound speed 1 (p = 1/gamma), cell 1: the
// freestream speed is the Mach number; time is in cell sound-crossings.
// Inviscid: shocks, expansions, wave drag; no boundary layers.
// Specification: docs/THEORY.md, THEORY 8.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "windoa/context.hpp"

namespace windoa::euler {

inline constexpr float kGamma = 1.4f;

enum class XBC : int { Tunnel = 0, Transmissive = 1 }; // inflow / outflow, or zero-gradient
enum class SideBC : int { Farfield = 0, Slip = 1 };    // open air, or an inviscid wall
enum class Wall : int { Mirror = 0, Image = 1 };
enum class Limiter : int { Minmod = 0, VanLeer = 1 };

struct Config {
    int nx = 0, ny = 0, nz = 0;
    float mach = 0.8f;
    float cfl = 0.4f;
    Limiter limiter = Limiter::VanLeer;
    SideBC side_bc = SideBC::Farfield; // y faces
    int z_bc = -1;                     // z faces (-1: as side_bc). A 2-D
                                       // section needs Slip (farfield z
                                       // made it a 2-cell wing: Cl ~0)
    XBC x_bc = XBC::Tunnel;
    Wall wall = Wall::Image;
};

class Solver {
  public:
    Solver(Context& ctx, const Config& cfg);
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    const Config& config() const { return cfg_; }
    std::size_t cells() const { return n_; }

    // Flags [x][y][z] (lbm::Flag values). Clears an exact distance: set it
    // again, or the indicator-based estimate is used.
    void set_flags(std::span<const std::uint8_t> flags);
    // Exact signed distance to the surface (cells, negative in the solid).
    void set_distance(std::span<const float> phi);

    void init_freestream();                       // uniform freestream (impulsive start)
    void set_primitive(std::span<const float> w); // [cell][5] (rho, u, v, w, p)
    void set_mach(float m) { cfg_.mach = m; }
    float mach() const { return cfg_.mach; }
    void set_flow_angle(float degrees); // freestream tilt in x-y (+ = up)
    void set_dt_cap(float cap) { dt_cap_ = cap; }

    void step(int n = 1);
    // GPU milliseconds per kernel category over `steps` steps: dt, ghost,
    // update (fused fluxes + RK stage), - (a development measurement;
    // advances the flow).
    std::array<double, 4> profile(int steps);
    std::int64_t steps_taken() const { return steps_; }
    double time(); // simulated time (reads back)

    std::array<double, 3> body_force(); // pressure force on OBSTACLE cells
    struct Health {
        float max_mach = 0.0f;
        int bad_cells = 0;
    };
    Health health();                // also refreshes the render fields
    std::vector<float> primitive(); // [cell][5]

    // Render fields (refreshed by health() / refresh()): vec4 (u, p), rho.
    void refresh();
    const Buffer& macro_buffer() const { return macro_; }
    const Buffer& rho_buffer() const { return rho_; }
    const Buffer& flag_buffer() const { return flags_; }
    std::uint64_t geometry_version() const { return geometry_version_; }

  private:
    struct Params;
    Params params(int axis = 0, int first = 0, int mode = 0, float aux0 = 0.0f,
                  float aux1 = 0.0f) const;
    void ensure_geometry();
    void phi_from_flags();
    using Marker = std::function<void(VkCommandBuffer, int category)>;
    void record_step(VkCommandBuffer cmd, const Marker* mark = nullptr);

    Context& ctx_;
    Config cfg_;
    std::size_t n_;
    Groups groups_;
    float flow_angle_ = 0.0f, dt_cap_ = 1e9f;
    std::int64_t steps_ = 0;
    std::uint64_t geometry_version_ = 0;
    bool phi_exact_ = false, phi_ready_ = false;
    std::vector<std::uint8_t> host_flags_;

    Buffer u_, u1_, flags_, phi_, gs_, gn_, gpw_, dt_, partials_, macro_, rho_;
    ComputeKernel ghost_, update_, stage_, dt_kernel_, force_, diag_;
};

} // namespace windoa::euler
