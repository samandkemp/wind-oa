// V30 -- an engine port in the compressible regime: the under-expanded jet.
// Specified in THEORY 8.12 (docs/THEORY.md).
//
// Published reference (Ashkenas & Sherman 1966; Crist, Sherman & Glass
// 1966): a sonic jet issuing at stagnation pressure p0 into still air at p_a
// expands, then recompresses through a normal shock, the Mach disc, at
//   x_M / D = 0.67 sqrt(p0 / p_a)
// from the exit, D the exit diameter, for p0 / p_a from about 15 upwards.
//
// Case: a nozzle body (a solid cylinder, radius 14 cells) whose +x face holds
// a round port (D 20 cells) emitting the sonic exit state of a cold jet
// (stagnation temperature that of the air: T_e = T0 / 1.2, p_e = p0 / 1.893)
// into still air, 256 x 128 x 128. The Mach disc is the steepest fall of the
// local Mach number along the jet's axis. Pass: within 10 % of the
// correlation at p0 / p_a = 20 and 40 (the sqrt scaling too), and the jet's
// peak axial Mach number past 3 (a real expansion, not a smeared one).
#include "euler_cases.hpp"

using namespace windoa;
using namespace windoa::euler_cases;

namespace {

constexpr int NX = 256, NY = 128, NZ = 128;
constexpr double D = 20.0, X_EXIT = 40.0;

struct Jet {
    double x_disc, peak_mach;
};

Jet run(gate::Gate& g, Context& ctx, double npr) {
    euler::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.mach = 0.0f;
    c.side_bc = euler::SideBC::Farfield;
    euler::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(s.cells(), 0); // fluid; 1 = the nozzle body
    const double cy = NY / 2.0, cz = NZ / 2.0;
    for (int x = 8; x < int(X_EXIT); ++x)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const double r2 = (y + 0.5 - cy) * (y + 0.5 - cy) + (z + 0.5 - cz) * (z + 0.5 - cz);
                if (r2 > 14.0 * 14.0)
                    continue;
                const bool port = x >= int(X_EXIT) - 2 && r2 <= (D / 2) * (D / 2);
                flags[Grid{NX, NY, NZ}.index(x, y, z)] = port ? euler::kPortFlag : std::uint8_t(1);
            }
    s.set_flags(flags);
    // sonic, cold: T_e = T0 / 1.2 = T_a / 1.2, p_e = p0 / 1.893; Euler units
    // rho_a = 1, c_a = 1, p_a = 1 / gamma
    const double p_ratio = npr / std::pow(1.2, G / (G - 1.0)), t_ratio = 1.0 / 1.2;
    const std::array<float, 5> exit = {float(p_ratio / t_ratio), float(std::sqrt(t_ratio)), 0.0f,
                                       0.0f, float(p_ratio / G)};
    s.set_ports(std::span<const std::array<float, 5>>(&exit, 1));
    s.init_freestream();
    Jet j{};
    for (int k = 0; k < 4; ++k) {
        s.step(1000);
        const auto w = s.primitive();
        // the axis: the four cells round it, averaged
        std::vector<double> mach(NX, 0.0);
        for (int x = 0; x < NX; ++x)
            for (int dy = -1; dy <= 0; ++dy)
                for (int dz = -1; dz <= 0; ++dz) {
                    const std::size_t i = (std::size_t(x) * NY + std::size_t(NY / 2 + dy)) * NZ +
                                          std::size_t(NZ / 2 + dz);
                    const double rho = w[5 * i], u = w[5 * i + 1], v = w[5 * i + 2],
                                 ww = w[5 * i + 3], p = w[5 * i + 4];
                    mach[std::size_t(x)] +=
                        0.25 * std::sqrt(u * u + v * v + ww * ww) / std::sqrt(G * p / rho);
                }
        double steepest = 0.0, peak = 0.0;
        int at = 0;
        for (int x = int(X_EXIT) + 2; x < NX - 20; ++x) {
            peak = std::max(peak, mach[std::size_t(x)]);
            const double fall = mach[std::size_t(x) - 1] - mach[std::size_t(x) + 1];
            if (fall > steepest) {
                steepest = fall;
                at = x;
            }
        }
        j = {at + 0.5 - X_EXIT, peak};
        g.note("  p0/p_a %.0f, %dk steps: Mach disc %.1f cells past the exit (%.2f D), peak axial "
               "Mach %.2f",
               npr, k + 1, j.x_disc, j.x_disc / D, j.peak_mach);
    }
    return j;
}

} // namespace

int main() {
    return gate::run("V30_underexpanded_jet", [](gate::Gate& g) {
        Context ctx;
        for (const double npr : {20.0, 40.0}) {
            const Jet j = run(g, ctx, npr);
            const double ref = 0.67 * std::sqrt(npr);
            const double err = std::abs(j.x_disc / D - ref) / ref;
            g.check(err < 0.10 && j.peak_mach > 3.0,
                    "p0/p_a %.0f: Mach disc at %.2f D vs Ashkenas-Sherman %.2f D (%.1f %%, < 10 "
                    "%%); peak axial Mach %.2f (> 3)",
                    npr, j.x_disc / D, ref, 100.0 * err, j.peak_mach);
        }
    });
}
