// V6 -- sphere drag at Re 100 and 300.
// Specified in THEORY 4.6 (docs/THEORY.md).
//
// Published reference: the Schiller-Naumann correlation
// Cd = 24/Re (1 + 0.15 Re^0.687) (Re < 800, +-5 %): 1.09 at Re 100, 0.685
// at Re 300. A voxel sphere (index-convention shapes::add_sphere) in the
// tunnel: velocity inlet ramped from rest over 2,000 steps,
// pressure outlet, free-slip sides; BGK (Re 100 laminar, Re 300 with
// cs 0.1). Pass: within 15 % (voxel stair-step at D 16-24 plus ~3 %
// blockage justify the loose band; the trend matters most).
#include "cases.hpp"

using namespace windoa;

namespace {

struct Case {
    double re, d;
    int nx, ny, nz;
    float cs;
    int steps, n_avg;
    double cd_ref, tol;
};

const Case kCases[] = {
    {100.0, 16.0, 160, 80, 80, 0.0f, 40'000, 5'000, 1.09, 0.15},
    {300.0, 24.0, 192, 96, 96, 0.1f, 40'000, 10'000, 0.685, 0.15},
};

struct Result {
    double cd = 0.0, u_free = 0.0;
    float tau = 0.0f;
    int solid = 0;
};

Result run(Context& ctx, const Case& k) {
    constexpr float U_IN = 0.05f;
    constexpr int RAMP = 2000;
    lbm::Config c;
    c.nx = k.nx;
    c.ny = k.ny;
    c.nz = k.nz;
    c.tau = shapes::tau_for_reynolds(U_IN, float(k.d), float(k.re));
    c.u_inlet = U_IN;
    c.smagorinsky_cs = k.cs;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(std::size_t(k.nx) * k.ny * k.nz, lbm::FLUID);
    Result r;
    r.tau = c.tau;
    r.solid =
        shapes::add_sphere(flags, k.nx, k.ny, k.nz, k.nx * 0.3, k.ny / 2.0, k.nz / 2.0, k.d / 2.0);
    s.set_flags(flags);
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f}); // still air
    for (int i = 0; i < RAMP; ++i) {
        s.set_inlet_velocity(U_IN * float(i + 1) / RAMP);
        s.step(1);
    }
    s.step(k.steps - RAMP - k.n_avg);
    s.read_mean_forces(); // open the averaging window
    s.step(k.n_avg);
    const auto f = s.read_mean_forces();
    r.cd = f.force[0] / (0.5 * U_IN * U_IN * gate::kPi * (k.d / 2.0) * (k.d / 2.0));
    const auto vel = s.velocity();
    const int xf = int(k.nx * 0.1);
    double sum = 0.0;
    for (int y = 0; y < k.ny; ++y)
        for (int z = 0; z < k.nz; ++z)
            sum += vel[((std::size_t(xf) * k.ny + y) * k.nz + z) * 3];
    r.u_free = sum / (double(k.ny) * k.nz);
    return r;
}

} // namespace

int main() {
    return gate::run("V6_sphere", [](gate::Gate& g) {
        Context ctx;
        for (const auto& k : kCases) {
            g.section(k.re < 200 ? "Re 100" : "Re 300");
            const auto r = run(ctx, k);
            g.note("D %.0f cells, tau %.4f, %d solid cells, %d steps; freestream u %.4f (0.05)",
                   k.d, r.tau, r.solid, k.steps, r.u_free);
            const double err = std::abs(r.cd - k.cd_ref) / k.cd_ref;
            g.check(err < k.tol, "Schiller-Naumann: Cd %.3f vs %.3f, %.1f %% (< %.0f %%)", r.cd,
                    k.cd_ref, err * 100.0, k.tol * 100.0);
        }
    });
}
