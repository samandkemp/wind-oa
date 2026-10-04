// Case setups shared by several gates, so that each configuration is
// written once.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "gate.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/lbm.hpp"
#include "windoa/mesh.hpp"
#include "windoa/shapes.hpp"
#include "windoa/tunnel.hpp"
#include "windoa/voxeliser.hpp"

namespace windoa::cases {

// -- Body-force Poiseuille channel (V1, V4 A, V5 A, V9) -------------------------
namespace poiseuille {
inline constexpr int NX = 4, NY = 34, NZ = 4;
inline constexpr float TAU = 0.8f; // nu = 0.1
inline constexpr float G_X = 1.0e-5f;
inline constexpr int STEPS = 40'000;

struct Result {
    std::vector<double> u;     // u_x(y) at (NX/2, :, NZ/2), all NY cells
    std::vector<double> exact; // the parabola at the fluid cells y = 1..NY-2
    double l2 = 0.0;           // relative L2 over the fluid cells
};

// The steady solution between half-way bounce-back walls (wall planes half a
// cell inside the WALL / fluid interface, H = NY - 2):
// u(y') = g / (2 nu) y' (H - y'), y' the distance from the wall plane.
inline Result run(Context& ctx, bool regularised, bool recursive, float g_x = G_X) {
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = TAU;
    c.smagorinsky_cs = 0.0f; // laminar
    c.mode_x = lbm::AxisX::Periodic;
    c.mode_y = lbm::AxisYZ::FreeSlip; // irrelevant: y edges are WALL cells
    c.mode_z = lbm::AxisYZ::Periodic;
    c.regularised = regularised;
    c.recursive = recursive;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
    for (int x = 0; x < NX; ++x)
        for (int z = 0; z < NZ; ++z) {
            flags[(std::size_t(x) * NY + 0) * NZ + z] = lbm::WALL;
            flags[(std::size_t(x) * NY + NY - 1) * NZ + z] = lbm::WALL;
        }
    s.set_flags(flags);
    s.set_body_force({g_x, 0.0f, 0.0f});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    s.step(STEPS);

    const auto vel = s.velocity();
    Result r;
    for (int y = 0; y < NY; ++y)
        r.u.push_back(vel[((std::size_t(NX / 2) * NY + y) * NZ + NZ / 2) * 3 + 0]);
    const double nu = (TAU - 0.5) / 3.0, H = NY - 2;
    std::vector<double> sim;
    for (int y = 1; y < NY - 1; ++y) {
        const double yw = y - 0.5;
        r.exact.push_back(g_x / (2.0 * nu) * yw * (H - yw));
        sim.push_back(r.u[std::size_t(y)]);
    }
    r.l2 = gate::rel_l2(sim, r.exact);
    return r;
}
} // namespace poiseuille

// -- Lid-driven cavity, Re 1000 (V3, V4, V5 B) ----------------------------------
namespace cavity {
inline constexpr int N = 128; // fluid cells per side
inline constexpr int NX = N + 2, NY = N + 2, NZ = 4;
inline constexpr float U_LID = 0.1f;

inline float tau_for(double re) {
    return float(3.0 * U_LID * N / re + 0.5);
}

// Ghia, Ghia & Shin (1982), Re 1000: u_x / U along the vertical centreline.
inline const std::vector<double> GHIA_Y = {1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516,
                                           0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719,
                                           0.1016, 0.0703, 0.0625, 0.0547, 0.0000};
inline const std::vector<double> GHIA_U = {1.0000,  0.6593,  0.5749,  0.5112,  0.4660,  0.3330,
                                           0.1872,  0.0570,  -0.0608, -0.1065, -0.2781, -0.3829,
                                           -0.2973, -0.2222, -0.2020, -0.1811, 0.0000};

struct Setup {
    float tau = tau_for(1000.0);
    float cs = 0.0f;
    bool regularised = false, recursive = false;
    int steps = 200'000;
    int anchor_every = 500; // closed-domain mass anchor; 0 = none
};

struct Result {
    std::vector<double> u;       // u_x / U_LID along x = NX/2, z = NZ/2, all NY
    std::vector<double> at_ghia; // interpolated to GHIA_Y (lid 1, floor 0)
    double rms = 0.0;
    double mean_rho = 0.0;
    bool finite = true;
    float max_speed = 0.0f;
};

inline std::vector<std::uint8_t> flags() {
    std::vector<std::uint8_t> f(std::size_t(NX) * NY * NZ, lbm::FLUID);
    auto at = [&](int x, int y, int z) -> std::uint8_t& {
        return f[(std::size_t(x) * NY + y) * NZ + z];
    };
    for (int y = 0; y < NY; ++y)
        for (int z = 0; z < NZ; ++z)
            at(0, y, z) = at(NX - 1, y, z) = lbm::WALL;
    for (int x = 0; x < NX; ++x)
        for (int z = 0; z < NZ; ++z) {
            at(x, 0, z) = lbm::WALL;
            at(x, NY - 1, z) = lbm::LID; // set last: the top corners are LID
        }
    return f;
}

inline lbm::Config config(const Setup& su) {
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = su.tau;
    c.smagorinsky_cs = su.cs;
    c.mode_x = lbm::AxisX::Periodic; // x edges are WALL cells anyway
    c.mode_y = lbm::AxisYZ::FreeSlip;
    c.mode_z = lbm::AxisYZ::Periodic;
    c.regularised = su.regularised;
    c.recursive = su.recursive;
    return c;
}

// Fluid cells y = 1..N; wall plane y = 0.5, lid plane y = N + 0.5.
inline void score(Result& r) {
    std::vector<double> yh, uf;
    for (int y = 1; y < NY - 1; ++y) {
        yh.push_back((y - 0.5) / N);
        uf.push_back(r.u[std::size_t(y)]);
    }
    r.at_ghia.clear();
    double ss = 0.0;
    for (std::size_t k = 0; k < GHIA_Y.size(); ++k) {
        double v = gate::interp(GHIA_Y[k], yh, uf);
        if (k == 0)
            v = 1.0; // the lid point itself
        if (k + 1 == GHIA_Y.size())
            v = 0.0; // the floor point
        r.at_ghia.push_back(v);
        ss += (v - GHIA_U[k]) * (v - GHIA_U[k]);
    }
    r.rms = std::sqrt(ss / double(GHIA_Y.size()));
}

inline Result run(Context& ctx, const Setup& su) {
    lbm::Solver s(ctx, config(su));
    s.set_flags(flags());
    s.set_lid_velocity({U_LID, 0.0f, 0.0f});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    const int chunk = su.anchor_every > 0 ? su.anchor_every : 1000;
    Result r;
    for (int done = 0; done < su.steps;) {
        const int n = std::min(chunk, su.steps - done);
        s.step(n);
        done += n;
        if (su.anchor_every > 0)
            s.enforce_mass(1.0);
    }
    const auto vel = s.velocity();
    for (int y = 0; y < NY; ++y)
        r.u.push_back(vel[((std::size_t(NX / 2) * NY + y) * NZ + NZ / 2) * 3 + 0] / U_LID);
    for (const double v : r.u)
        r.finite = r.finite && std::isfinite(v);
    r.mean_rho = s.mean_fluid_density();
    r.max_speed = s.health().max_speed;
    score(r);
    return r;
}
} // namespace cavity

// -- Ahmed body 25 deg in free flow (V14, V15) ----------------------------------
namespace ahmed {
inline constexpr int NX = 256, NY = 128, NZ = 128;
inline constexpr float U_IN = 0.05f;
inline constexpr float LENGTH = 100.0f; // body length, cells
inline constexpr float CLEAR = 5.0f;    // clearance above the y = 0 free-slip plane

struct Result {
    double cd = 0.0, cl = 0.0, a_ref = 0.0;
    std::size_t n_solid = 0;
    double min_ux_wake = 0.0; // min u_x behind the base, mid-span plane
    bool finite = true;
};

inline Result run(Context& ctx, float tau, bool regularised) {
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = tau;
    c.u_inlet = 0.0f;
    c.smagorinsky_cs = 0.1f;
    c.regularised = regularised;
    lbm::Solver s(ctx, c);
    const double height = double(LENGTH) * 288.0 / 1044.0;
    // the centre in double, rounded once to float (see shapes.hpp on f32
    // placement)
    const geometry::Vec3 centre{float(NX * 0.30), float(CLEAR + height / 2.0), float(NZ / 2.0)};
    const auto mesh = geometry::fit_to_box(catalogue::find("ahmed_25deg")->build(), centre, LENGTH);
    Voxeliser vox(ctx, NX, NY, NZ);
    std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
    Result r;
    r.n_solid = vox.voxelise(mesh, flags).n_solid;
    s.set_flags(flags);
    for (int y = 0; y < NY; ++y) // frontal area: OBSTACLE anywhere along x
        for (int z = 0; z < NZ; ++z)
            for (int x = 0; x < NX; ++x)
                if (flags[(std::size_t(x) * NY + y) * NZ + z] == lbm::OBSTACLE) {
                    r.a_ref += 1.0;
                    break;
                }
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    constexpr int RAMP = 2000, N_STEPS = 40'000, N_AVG = 10'000;
    for (int k = 0; k < RAMP; ++k) {
        s.set_inlet_velocity(U_IN * float(k + 1) / RAMP);
        s.step(1);
    }
    s.step(N_STEPS - RAMP - N_AVG);
    s.read_mean_forces();
    s.step(N_AVG);
    const auto f = s.read_mean_forces();
    const double q = 0.5 * U_IN * U_IN * r.a_ref;
    r.cd = f.force[0] / q;
    r.cl = f.force[1] / q;
    r.finite = std::isfinite(r.cd) && std::isfinite(r.cl);
    // the near wake: from the base to 0.3 L behind it, body height, mid-span
    const auto vel = s.velocity();
    const int x_base = int(NX * 0.30f + LENGTH / 2.0f) + 1;
    r.min_ux_wake = 1e30;
    for (int x = x_base; x < x_base + int(0.3f * LENGTH); ++x)
        for (int y = int(CLEAR); y < int(CLEAR + float(height)); ++y)
            r.min_ux_wake =
                std::min(r.min_ux_wake, double(vel[((std::size_t(x) * NY + y) * NZ + NZ / 2) * 3]));
    return r;
}
} // namespace ahmed

// -- The app's scenario: the Ahmed body, 64 cells, on a preset grid ---------------
// (V20 C, V21 D, V22, V23)
namespace app {

struct Placed {
    std::vector<std::uint8_t> flags;  // OBSTACLE where the body is
    std::vector<std::uint8_t> link_q; // the app's sub-cell walls (empty when off)
    double area = 0.0;                // frontal: OBSTACLE anywhere along x
};

// The body into a solver built from solver_config(t), walls as the app sets them.
inline void place(lbm::Solver& s, const Placed& p) {
    s.set_flags(p.flags);
    if (!p.link_q.empty())
        s.set_link_q(p.link_q);
}

// fit_to_box(ahmed, (0.35 nx, 0.5 ny, 0.5 nz), 64), pitched by -aoa about
// that centre.
inline Placed ahmed(Voxeliser& vox, const TunnelSettings& t, float aoa_deg = 0.0f) {
    const geometry::Vec3 centre{float(0.35 * t.nx), float(0.5 * t.ny), float(0.5 * t.nz)};
    auto mesh = geometry::fit_to_box(catalogue::find("ahmed_25deg")->build(), centre, 64.0f);
    if (aoa_deg != 0.0f)
        mesh = geometry::transform(mesh, 0.0f, -aoa_deg, 0.0f, centre);
    Placed p;
    p.flags.assign(std::size_t(t.nx) * t.ny * t.nz, lbm::FLUID);
    vox.voxelise(mesh, p.flags);
    if (t.sub_cell_walls)
        p.link_q =
            shapes::link_fractions(p.flags, vox.signed_distance(p.flags), t.nx, t.ny, t.nz, &mesh);
    for (int y = 0; y < t.ny; ++y)
        for (int z = 0; z < t.nz; ++z)
            for (int x = 0; x < t.nx; ++x)
                if (p.flags[(std::size_t(x) * t.ny + y) * t.nz + z] == lbm::OBSTACLE) {
                    p.area += 1.0;
                    break;
                }
    return p;
}

} // namespace app

} // namespace windoa::cases
