// V21 -- synthetic inlet turbulence (lbm::Solver::set_inlet_turbulence),
// on the app's solver and fast grid.
// Specified in THEORY 7 (docs/THEORY.md).
//
//   A. Closed form: the patch is divergence-free (the curl of a vector
//      potential, central differences) and unit rms, components balanced.
//   B. Invariant: in an empty tunnel the intensity measured 8 cells past
//      the inlet matches the request (2 %) within 25 %; it then decays.
//   C. Invariant (the reason for the curl): solenoidal fluctuations make
//      much less sound (density rms, excess over a clean inlet) than the
//      same intensity of non-solenoidal noise: < 1/3 of it. The noise is
//      the patch scrambled (component 0 <- 1, shifted in y, fixed signs
//      flipped).
//   D. Sanity bound: Ahmed Cd with 1 % turbulence within 10 % of clean.
#include <memory>

#include "cases.hpp"

using namespace windoa;

namespace {

struct Measured {
    double tu8 = 0, tu60 = 0, rho_rms = 0;
};

// develop, then record the velocity at two planes and the density rms of
// the interior every 10 steps
Measured measure(lbm::Solver& s, const TunnelSettings& t, int n_dev = 3000, int n_rec = 3000) {
    s.step(n_dev);
    const std::size_t plane = std::size_t(t.ny) * t.nz;
    std::vector<double> sum8(plane * 3, 0.0), sq8(plane * 3, 0.0), sum60(plane * 3, 0.0),
        sq60(plane * 3, 0.0);
    double rho_acc = 0.0;
    const int n = n_rec / 10;
    for (int k = 0; k < n; ++k) {
        s.step(10);
        const auto v = s.velocity();
        for (std::size_t i = 0; i < plane * 3; ++i) {
            const double a = v[8 * plane * 3 + i], b = v[60 * plane * 3 + i];
            sum8[i] += a;
            sq8[i] += a * a;
            sum60[i] += b;
            sq60[i] += b * b;
        }
        const auto r = s.density();
        double m = 0, m2 = 0;
        const std::size_t lo = 20 * plane, hi = std::size_t(t.nx - 40) * plane;
        for (std::size_t i = lo; i < hi; ++i) {
            m += r[i];
            m2 += double(r[i]) * r[i];
        }
        const double cnt = double(hi - lo);
        rho_acc += std::sqrt(std::max(0.0, m2 / cnt - (m / cnt) * (m / cnt)));
    }
    auto tu = [&](const std::vector<double>& s1, const std::vector<double>& s2) {
        double var = 0.0; // time variance per (cell, component), averaged
        for (std::size_t i = 0; i < s1.size(); ++i)
            var += s2[i] / n - (s1[i] / n) * (s1[i] / n);
        return std::sqrt(std::max(0.0, var / double(s1.size()))) / t.u_inlet;
    };
    return {tu(sum8, sq8), tu(sum60, sq60), rho_acc / n};
}

std::unique_ptr<lbm::Solver> empty_tunnel(Context& ctx, const TunnelSettings& t) {
    auto c = solver_config(t);
    c.u_inlet = t.u_inlet;
    auto s = std::make_unique<lbm::Solver>(ctx, c);
    s->set_flags(std::vector<std::uint8_t>(s->cells(), lbm::FLUID));
    s->init_equilibrium(1.0f, {t.u_inlet, 0.0f, 0.0f});
    return s;
}

// periodic central difference of component k along axis ax of a [s][y][z][3] field
double dcomp(const std::vector<float>& f, int S, int NY, int NZ, int s, int y, int z, int k,
             int ax) {
    auto at = [&](int a, int b, int c) {
        a = (a + S) % S;
        b = (b + NY) % NY;
        c = (c + NZ) % NZ;
        return double(f[((std::size_t(a) * NY + b) * NZ + c) * 3 + k]);
    };
    const int o[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    return 0.5 * (at(s + o[ax][0], y + o[ax][1], z + o[ax][2]) -
                  at(s - o[ax][0], y - o[ax][1], z - o[ax][2]));
}

double rms_div(const std::vector<float>& f, int S, int NY, int NZ) {
    double ss = 0.0;
    for (int s = 0; s < S; ++s)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const double d = dcomp(f, S, NY, NZ, s, y, z, 0, 0) +
                                 dcomp(f, S, NY, NZ, s, y, z, 1, 1) +
                                 dcomp(f, S, NY, NZ, s, y, z, 2, 2);
                ss += d * d;
            }
    return std::sqrt(ss / (double(S) * NY * NZ));
}

double ahmed_cd(Context& ctx, const TunnelSettings& t, const cases::app::Placed& p, float tu) {
    lbm::Solver s(ctx, solver_config(t));
    cases::app::place(s, p);
    if (tu > 0)
        s.set_inlet_turbulence(tu, 8.0f);
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < 50; ++k) {
        s.set_inlet_velocity(t.u_inlet * std::min(1.0f, float(k + 1) * 400.0f / t.ramp_steps));
        s.step(400);
    }
    s.read_mean_forces();
    s.step(10000);
    return s.read_mean_forces().force[0] / (0.5 * t.u_inlet * t.u_inlet * p.area);
}

} // namespace

int main() {
    return gate::run("V21_turbulence", [](gate::Gate& g) {
        Context ctx;
        const TunnelSettings t = tunnel_preset("fast");

        g.section("A: the patch");
        {
            auto s = empty_tunnel(ctx, t);
            s->set_inlet_turbulence(0.02f, 8.0f, 256);
            const auto up = s->turbulence_patch();
            const int S = 256;
            const double div = rms_div(up, S, t.ny, t.nz);
            double gss = 0.0, csq[3] = {0, 0, 0};
            const std::size_t cells = std::size_t(S) * t.ny * t.nz;
            for (int a = 0; a < S; ++a)
                for (int y = 0; y < t.ny; ++y)
                    for (int z = 0; z < t.nz; ++z) {
                        const double d = dcomp(up, S, t.ny, t.nz, a, y, z, 0, 0);
                        gss += d * d;
                    }
            for (std::size_t c = 0; c < cells; ++c)
                for (int k = 0; k < 3; ++k)
                    csq[k] += double(up[c * 3 + k]) * up[c * 3 + k];
            const double grad = std::sqrt(gss / double(cells));
            double comp[3], all = 0;
            for (int k = 0; k < 3; ++k) {
                comp[k] = std::sqrt(csq[k] / double(cells));
                all += csq[k];
            }
            all = std::sqrt(all / (3.0 * double(cells)));
            const double cmax = std::max({comp[0], comp[1], comp[2]}),
                         cmin = std::min({comp[0], comp[1], comp[2]});
            g.check(div < 1e-4 * grad && std::abs(all - 1.0) < 1e-3 && cmax / cmin < 1.25,
                    "rms div %.2e vs rms du/dx %.2e; rms %.4f; components %.3f %.3f %.3f", div,
                    grad, all, comp[0], comp[1], comp[2]);
        }

        g.section("B + C: intensity and quietness in an empty tunnel");
        constexpr float TU = 0.02f;
        auto sol = empty_tunnel(ctx, t);
        sol->set_inlet_turbulence(TU, 8.0f);
        const auto ms = measure(*sol, t);
        g.check(std::abs(ms.tu8 - TU) / TU < 0.25,
                "requested Tu %.1f %%: measured %.2f %% at x = 8, %.2f %% at x = 60 (decaying)",
                TU * 100.0, ms.tu8 * 100.0, ms.tu60 * 100.0);
        auto non = empty_tunnel(ctx, t);
        non->set_inlet_turbulence(TU, 8.0f);
        {
            const auto raw = non->turbulence_patch();
            const int S = int(raw.size() / (std::size_t(t.ny) * t.nz * 3));
            std::vector<float> noise(raw.size());
            const float sign[3] = {-1.0f, 1.0f, -1.0f};
            for (int a = 0; a < S; ++a)
                for (int y = 0; y < t.ny; ++y)
                    for (int z = 0; z < t.nz; ++z) {
                        const std::size_t dst = ((std::size_t(a) * t.ny + y) * t.nz + z) * 3;
                        const std::size_t src =
                            ((std::size_t(a) * t.ny + (y - 5 + t.ny) % t.ny) * t.nz + z) * 3;
                        for (int k = 0; k < 3; ++k)
                            noise[dst + k] =
                                raw[src + k] * sign[k]; // rolled 5 cells in y, signs flipped
                        noise[dst] = raw[dst + 1];      // scramble: no longer solenoidal
                    }
            double ss = 0.0;
            for (const float v : noise)
                ss += double(v) * v;
            const double r = std::sqrt(ss / double(noise.size()));
            for (float& v : noise)
                v = float(v / r);
            g.note("non-solenoidal noise: rms div %.1e", rms_div(noise, S, t.ny, t.nz));
            non->set_turbulence_patch(noise);
        }
        const auto mn = measure(*non, t);
        auto clean = empty_tunnel(ctx, t);
        const auto mc = measure(*clean, t);
        g.check(ms.rho_rms - mc.rho_rms < 0.33 * (mn.rho_rms - mc.rho_rms),
                "density rms (sound): clean %.2e | solenoidal %.2e | non-solenoidal %.2e (excess < "
                "1/3)",
                mc.rho_rms, ms.rho_rms, mn.rho_rms);

        g.section("D: Ahmed Cd, 1 % turbulence vs clean");
        Voxeliser vox(ctx, t.nx, t.ny, t.nz);
        const auto placed = cases::app::ahmed(vox, t);
        const double cd0 = ahmed_cd(ctx, t, placed, 0.0f), cd1 = ahmed_cd(ctx, t, placed, 0.01f);
        g.check(std::abs(cd1 - cd0) / cd0 < 0.10,
                "clean %.4f, Tu 1 %% %.4f (%+.2f %%; within 10 %%)", cd0, cd1,
                (cd1 - cd0) / cd0 * 100.0);
    });
}
