// V20 -- the D3Q7 dye field (engine Dye).
// Specified in THEORY 6 (docs/THEORY.md).
//
//   A. Closed form: a Gaussian blob in uniform periodic flow moves at U,
//      spreads with per-axis variance sigma0^2 + 2 D t (D = (tau - 1/2) / 4)
//      and conserves its mass. Two diffusivities (tau 0.53, the app's, and
//      0.6), so the D formula itself is tested. Pass: centre within 0.5
//      cell, variances within 3 %, mass within 0.5 %.
//   B. Invariant: no leak through walls -- a WALL slab across a periodic
//      box, a smooth dye band on one side, flow along it: the far side stays
//      exactly zero over 3,000 steps and the total within 0.1 %.
//   C. Invariant: bounded and positive with sources in the app's own tunnel
//      at the top speed round the Ahmed body: 0 <= C <= 1.1 (a few %
//      overshoot at steep fronts is expected of this non-monotone scheme;
//      what must never happen is growth without bound). The step's
//      C <= 1.25 limiter (for spinning walls, THEORY 6.3) cannot mask a
//      failure here: the band is tighter.
#include "cases.hpp"
#include "windoa/dye.hpp"

using namespace windoa;

namespace {

struct Blob {
    double xc, var_x, var_y, mass_err, d;
};

Blob blob(Context& ctx, float tau_d) {
    constexpr int NX = 160, NY = 56, NZ = 56; // wide enough that the tails do not wrap
    constexpr double U = 0.05, X0 = 40.0, S0 = 4.0;
    constexpr int T = 1000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.8f;
    c.smagorinsky_cs = 0.0f;
    c.mode_x = lbm::AxisX::Periodic;
    c.mode_y = lbm::AxisYZ::Periodic;
    c.mode_z = lbm::AxisYZ::Periodic;
    lbm::Solver s(ctx, c);
    s.set_flags(std::vector<std::uint8_t>(s.cells(), lbm::FLUID));
    s.init_equilibrium(1.0f, {float(U), 0.0f, 0.0f});
    Dye d(ctx, s, tau_d);
    std::vector<float> c0(s.cells());
    double m0 = 0.0;
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const double r2 = (x + 0.5 - X0) * (x + 0.5 - X0) +
                                  (y + 0.5 - NY / 2.0) * (y + 0.5 - NY / 2.0) +
                                  (z + 0.5 - NZ / 2.0) * (z + 0.5 - NZ / 2.0);
                const float v = float(std::exp(-r2 / (2 * S0 * S0)));
                c0[(std::size_t(x) * NY + y) * NZ + z] = v;
                m0 += v;
            }
    d.set_concentration(c0);
    d.step_with(s, T);
    const auto cc = d.concentration();
    std::vector<double> px(NX, 0.0), py(NY, 0.0);
    double m = 0.0;
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const double v = cc[(std::size_t(x) * NY + y) * NZ + z];
                px[x] += v;
                py[y] += v;
                m += v;
            }
    auto moments = [](const std::vector<double>& p, double& mean, double& var) {
        double s0 = 0, s1 = 0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            s0 += p[i];
            s1 += p[i] * (i + 0.5);
        }
        mean = s1 / s0;
        double s2 = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            s2 += p[i] * (i + 0.5 - mean) * (i + 0.5 - mean);
        var = s2 / s0;
    };
    Blob b;
    double yc;
    moments(px, b.xc, b.var_x);
    moments(py, yc, b.var_y);
    b.mass_err = std::abs(m - m0) / m0;
    b.d = d.diffusivity();
    return b;
}

} // namespace

int main() {
    return gate::run("V20_dye", [](gate::Gate& g) {
        Context ctx;
        g.section("A: Gaussian blob in uniform flow vs analytic");
        for (const float tau_d : {0.53f, 0.6f}) {
            const auto b = blob(ctx, tau_d);
            const double var_exp = 4.0 * 4.0 + 2.0 * b.d * 1000, x_exp = 40.0 + 0.05 * 1000;
            const double e_vx = std::abs(b.var_x - var_exp) / var_exp,
                         e_vy = std::abs(b.var_y - var_exp) / var_exp;
            g.check(
                std::abs(b.xc - x_exp) < 0.5 && e_vx < 0.03 && e_vy < 0.03 && b.mass_err < 0.005,
                "tau %.2f (D %.5f): centre %.2f vs %.2f; var x %.2f / y %.2f vs %.2f (%.1f %%); "
                "mass %.3f %%",
                tau_d, b.d, b.xc, x_exp, b.var_x, b.var_y, var_exp, std::max(e_vx, e_vy) * 100.0,
                b.mass_err * 100.0);
        }

        g.section("B: wall slab, 3,000 steps");
        {
            constexpr int NX = 96, NY = 48, NZ = 16;
            lbm::Config c;
            c.nx = NX;
            c.ny = NY;
            c.nz = NZ;
            c.tau = 0.6f;
            c.smagorinsky_cs = 0.0f;
            c.mode_x = lbm::AxisX::Periodic;
            c.mode_y = lbm::AxisYZ::Periodic;
            c.mode_z = lbm::AxisYZ::Periodic;
            lbm::Solver s(ctx, c);
            std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
            auto at = [&](int x, int y, int z) { return (std::size_t(x) * NY + y) * NZ + z; };
            for (int x = 0; x < NX; ++x)
                for (int z = 0; z < NZ; ++z) {
                    flags[at(x, NY / 2 - 1, z)] = flags[at(x, NY / 2, z)] =
                        lbm::WALL;                  // two-cell slab
                    flags[at(x, 0, z)] = lbm::WALL; // and the periodic seam
                }
            s.set_flags(flags);
            s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
            Dye d(ctx, s);
            // a smooth band in the lower half (a sharp step gains ~0.6 %: the
            // positivity clip at a discontinuity)
            std::vector<float> c0(s.cells(), 0.0f);
            double m0 = 0.0;
            for (int x = 0; x < NX; ++x)
                for (int y = 0; y < NY / 2; ++y)
                    for (int z = 0; z < NZ; ++z) {
                        if (flags[at(x, y, z)] != lbm::FLUID)
                            continue;
                        const float v =
                            float(std::exp(-((y - NY / 4.0) / 4.0) * ((y - NY / 4.0) / 4.0)));
                        c0[at(x, y, z)] = v;
                        m0 += v;
                    }
            d.set_concentration(c0);
            d.step_with(s, 3000);
            const auto cc = d.concentration();
            double upper = 0.0, total = 0.0;
            for (int x = 0; x < NX; ++x)
                for (int y = 0; y < NY; ++y)
                    for (int z = 0; z < NZ; ++z) {
                        total += cc[at(x, y, z)];
                        if (y >= NY / 2 + 1)
                            upper += cc[at(x, y, z)];
                    }
            const double e_m = std::abs(total - m0) / m0;
            g.check(upper == 0.0 && e_m < 0.001,
                    "dye on the far side %.3e (must be 0); total change %.4f %%", upper,
                    e_m * 100.0);
        }

        g.section("C: nozzle sources round the Ahmed body, top speed, app solver");
        {
            const TunnelSettings t = tunnel_preset("fast");
            Voxeliser vox(ctx, t.nx, t.ny, t.nz);
            const auto placed = cases::app::ahmed(vox, t);
            lbm::Solver s(ctx, solver_config(t));
            cases::app::place(s, placed);
            s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
            Dye d(ctx, s);
            std::vector<float> src(s.cells(), 0.0f);
            const double C0 = 0.35 * t.nx, C1 = 0.5 * t.ny, C2 = 0.5 * t.nz;
            const int xr = int(C0 - 0.18 * t.nx);
            for (int a = 0; a < 7; ++a)
                for (int b = 0; b < 7; ++b) {
                    const int j = int(C1 - 12 + 4.0 * a),
                              k = int(C2 - 12 + 4.0 * b); // offsets -12, -8, ..., 12
                    for (int x = xr - 1; x < xr + 1; ++x)
                        for (int y = j - 1; y < j + 1; ++y)
                            for (int z = k - 1; z < k + 1; ++z)
                                src[(std::size_t(x) * t.ny + y) * t.nz + z] = 0.5f;
                }
            d.set_sources(src);
            double cmin = 0.0, cmax = 0.0;
            bool finite = true;
            std::vector<float> cc;
            for (int k = 0; k < 40; ++k) {
                s.set_inlet_velocity(t.u_max *
                                     std::min(1.0f, float(k + 1) * 150.0f / t.ramp_steps));
                d.step_with(s, 150);
                cc = d.concentration();
                for (const float v : cc) {
                    finite = finite && std::isfinite(v);
                    cmin = std::min(cmin, double(v));
                    cmax = std::max(cmax, double(v));
                }
            }
            std::size_t fluid = 0, reached = 0;
            for (std::size_t i = 0; i < cc.size(); ++i)
                if (placed.flags[i] == lbm::FLUID) {
                    ++fluid;
                    reached += cc[i] > 0.01f;
                }
            g.check(finite && cmin >= 0.0 && cmax <= 1.1,
                    "U %.2f, 6,000 steps: C in [%.2e, %.4f] (0 .. 1.1); dye reaches %.1f %% of the "
                    "fluid",
                    t.u_max, cmin, cmax, 100.0 * double(reached) / double(fluid));
        }
    });
}
