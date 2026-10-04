// V9 -- interpolated bounce-back, Bouzidi et al. 2001 linear.
// Specified in THEORY 3.6, 3.10 (docs/THEORY.md).
//
// A. Closed form (the certification): channel walls at a known sub-voxel
//    link fraction q; the Poiseuille profile must match the analytic
//    solution for the shifted wall, H = (n_fluid_rows - 1) + 2q, u_max =
//    g H^2 / (8 nu), within 1.5 % for q 0.25, 0.5, 0.75.
// B. Informational: sphere Cd at low blockage (0.8 %), half-way vs IBB,
//    Re 100, D 16 (ref 1.09). The staircase's half-way wall costs about 4
//    points of the error here; the interpolated wall removes them (THEORY
//    3.9; V28 B measures the same on a voxelised mesh).
#include "cases.hpp"

using namespace windoa;

namespace {

// e_y of each D3Q19 direction (engine/shaders/lattice.glsl order).
constexpr int kEy[lbm::Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};

double certify(Context& ctx, double q) {
    constexpr int NX = 4, NY = 34, NZ = 4;
    constexpr float TAU = 0.8f, GX = 1.0e-5f;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = TAU;
    c.smagorinsky_cs = 0.0f;
    c.mode_x = lbm::AxisX::Periodic;
    c.mode_z = lbm::AxisYZ::Periodic;
    c.use_ibb = true;
    lbm::Solver s(ctx, c);
    const std::size_t n = std::size_t(NX) * NY * NZ;
    std::vector<std::uint8_t> flags(n, lbm::FLUID);
    for (int x = 0; x < NX; ++x)
        for (int z = 0; z < NZ; ++z) {
            flags[(std::size_t(x) * NY + 0) * NZ + z] = lbm::OBSTACLE; // IBB acts on OBSTACLE
            flags[(std::size_t(x) * NY + NY - 1) * NZ + z] = lbm::OBSTACLE;
        }
    s.set_flags(flags);
    std::vector<std::uint8_t> qa(lbm::Q * n, 128);
    const auto qv = std::uint8_t(std::lround(q * 255.0));
    for (int d = 1; d < lbm::Q; ++d)
        for (int x = 0; x < NX; ++x)
            for (int z = 0; z < NZ; ++z) {
                if (kEy[d] == -1) // links into the bottom wall
                    qa[d * n + (std::size_t(x) * NY + 1) * NZ + z] = qv;
                if (kEy[d] == +1) // links into the top wall
                    qa[d * n + (std::size_t(x) * NY + NY - 2) * NZ + z] = qv;
            }
    s.set_link_q(qa);
    s.set_body_force({GX, 0.0f, 0.0f});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    s.step(40'000);
    const auto vel = s.velocity();
    double umax = -1e30;
    for (std::size_t i = 0; i < n; ++i)
        umax = std::max(umax, double(vel[i * 3]));
    return umax;
}

double sphere_cd(Context& ctx, bool ibb) {
    constexpr int NX = 200, NY = 160, NZ = 160;
    constexpr float D = 16.0f, U = 0.05f, RE = 100.0f;
    constexpr int RAMP = 1500, TOTAL = 26'000, AVG = 5'000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = shapes::tau_for_reynolds(U, D, RE);
    c.u_inlet = 0.0f;
    c.smagorinsky_cs = 0.0f;
    c.use_ibb = ibb;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
    const double cx = NX * 0.3, cy = NY / 2.0, cz = NZ / 2.0; // double (shapes.hpp)
    shapes::add_sphere(flags, NX, NY, NZ, cx, cy, cz, D / 2.0);
    s.set_flags(flags);
    // add_sphere tests the cell index, the fractions the cell centre (shapes.hpp):
    // in centre coordinates the flagged sphere sits half a cell further along
    // each axis, and the fractions must describe that sphere.
    if (ibb)
        s.set_link_q(shapes::sphere_link_fractions(flags, NX, NY, NZ, cx + 0.5, cy + 0.5, cz + 0.5,
                                                   D / 2.0));
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < RAMP; ++k) {
        s.set_inlet_velocity(U * float(k + 1) / RAMP);
        s.step(1);
    }
    s.step(TOTAL - RAMP - AVG);
    s.read_mean_forces();
    s.step(AVG);
    const auto f = s.read_mean_forces();
    return f.force[0] / (0.5 * U * U * gate::kPi * (D / 2.0) * (D / 2.0));
}

} // namespace

int main() {
    return gate::run("V9_ibb", [](gate::Gate& g) {
        Context ctx;
        g.section("A: certification (shifted-wall Poiseuille)");
        const double nu = (0.8 - 0.5) / 3.0;
        for (const double q : {0.25, 0.5, 0.75}) {
            const double umax = certify(ctx, q);
            const double H = (34 - 3) + 2.0 * q;
            const double exact = 1.0e-5 * H * H / (8.0 * nu);
            const double err = std::abs(umax - exact) / exact;
            g.check(err < 0.015,
                    "q %.2f: H_eff %5.2f, u_max %.6f vs exact %.6f, %.2f %% (< 1.5 %%)", q, H, umax,
                    exact, err * 100.0);
        }

        g.section("B: sphere Cd, half-way vs IBB (0.8 % blockage; informational)");
        for (const bool ibb : {false, true}) {
            const double cd = sphere_cd(ctx, ibb);
            g.note("%-8s: Cd %.3f (ref 1.09, err %.1f %%)", ibb ? "IBB" : "half-way", cd,
                   std::abs(cd - 1.09) / 1.09 * 100.0);
        }
    });
}
