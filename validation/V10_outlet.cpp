// V10 -- the absorbing outlet sponge.
// Specified in THEORY 3.3, 3.10 (docs/THEORY.md).
//
// The pressure outlet pins rho = 1 on the last plane: acoustically a
// pressure-release end, which reflects sound (R ~0.68 without the sponge,
// part A). The app runs a 24-cell band relaxing to the full freestream
// (TunnelSettings outlet_sponge / sponge_target).
//   A. Invariant (reflection): a planar Gaussian pressure pulse in an empty
//      400-cell tunnel, recorded at a probe plane between the pulse and the
//      outlet; incident and outlet-reflected passes separated in time by the
//      sound speed. Pass: R < 0.05 with the app's sponge (bare outlet shown).
//   B. Invariant (no change to the answer): the Ahmed body on the fast
//      preset with the app's own solver (tunnel.hpp solver_config), mean Cd
//      over 40 x 250 steps with and without the sponge. Pass: within 1 %,
//      and the sponged force signal no noisier (std <= 1.1x).
#include <array>

#include "cases.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/mesh.hpp"
#include "windoa/tunnel.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

namespace {

// e_x and weights per D3Q19 direction (engine/shaders/lattice.glsl order).
constexpr int kEx[lbm::Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
constexpr double kWq[lbm::Q] = {1.0 / 3,  1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
                                1.0 / 18, 1.0 / 18, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                                1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                                1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};

double reflection(Context& ctx, int sponge, int target) {
    constexpr int NX = 400, NY = 16, NZ = 16;
    constexpr double U = 0.05, X0 = 250, SIG = 6.0, AMP = 1e-3;
    constexpr int XP = 200;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.52f;
    c.u_inlet = float(U);
    c.smagorinsky_cs = 0.0f;
    c.regularised = true;
    c.outlet_sponge = sponge;
    c.sponge_target = target;
    lbm::Solver s(ctx, c);
    s.set_flags(std::vector<std::uint8_t>(std::size_t(NX) * NY * NZ, lbm::FLUID));
    s.init_equilibrium(1.0f, {float(U), 0.0f, 0.0f});
    const std::size_t n = std::size_t(NX) * NY * NZ;
    std::vector<float> f0(lbm::Q * n);
    for (int i = 0; i < lbm::Q; ++i) {
        const double eu = kEx[i] * U;
        for (int x = 0; x < NX; ++x) {
            const double rho = 1.0 + AMP * std::exp(-((x - X0) / SIG) * ((x - X0) / SIG));
            const auto v = float(kWq[i] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * U * U));
            for (std::size_t yz = 0; yz < std::size_t(NY) * NZ; ++yz)
                f0[i * n + std::size_t(x) * NY * NZ + yz] = v;
        }
    }
    s.set_state(f0);
    // left-going half passes the probe at ~t 95 (c_s - U); the outlet echo
    // returns at ~t 620; the inlet's own echo only at ~t 790
    double inc = 0.0, ref = 0.0;
    for (int t = 0; t < 900; ++t) {
        s.step(1);
        if (t % 4 == 0) {
            const double d = std::abs(s.plane_mean_density(XP) - 1.0);
            if (t > 60 && t < 140)
                inc = std::max(inc, d);
            if (t > 540 && t < 720)
                ref = std::max(ref, d);
        }
    }
    return ref / inc;
}

struct CdStats {
    double mean = 0.0, std = 0.0;
};

CdStats ahmed(Context& ctx, const TunnelSettings& ts, int sponge) {
    TunnelSettings t = ts;
    t.outlet_sponge = sponge;
    lbm::Solver s(ctx, solver_config(t));
    const auto* e = catalogue::find("ahmed_25deg");
    const auto mesh =
        geometry::fit_to_box(e->build(), {0.35f * t.nx, 0.5f * t.ny, 0.5f * t.nz}, 64.0f);
    Voxeliser vox(ctx, t.nx, t.ny, t.nz);
    std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
    vox.voxelise(mesh, flags);
    s.set_flags(flags);
    int area = 0; // frontal: OBSTACLE anywhere along x
    for (int y = 0; y < t.ny; ++y)
        for (int z = 0; z < t.nz; ++z)
            for (int x = 0; x < t.nx; ++x)
                if (flags[(std::size_t(x) * t.ny + y) * t.nz + z] == lbm::OBSTACLE) {
                    ++area;
                    break;
                }
    const double q = 0.5 * t.u_inlet * t.u_inlet * area;
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < 50; ++k) {
        s.set_inlet_velocity(t.u_inlet * std::min(1.0f, float(k + 1) * 400.0f / t.ramp_steps));
        s.step(400);
    }
    s.read_mean_forces();
    std::vector<double> cds;
    for (int k = 0; k < 40; ++k) {
        s.step(250);
        cds.push_back(s.read_mean_forces().force[0] / q);
    }
    CdStats r;
    for (const double v : cds)
        r.mean += v;
    r.mean /= double(cds.size());
    for (const double v : cds)
        r.std += (v - r.mean) * (v - r.mean);
    r.std = std::sqrt(r.std / double(cds.size())); // population standard deviation
    return r;
}

} // namespace

int main() {
    return gate::run("V10_outlet", [](gate::Gate& g) {
        Context ctx;
        const TunnelSettings ts = tunnel_preset("fast");

        g.section("A: acoustic reflection at the outlet");
        const double r_bare = reflection(ctx, 0, 0);
        const double r_app = reflection(ctx, ts.outlet_sponge, ts.sponge_target);
        g.note("bare pressure outlet: R = %.3f", r_bare);
        g.check(r_app < 0.05, "app sponge (%d cells, target %d): R = %.3f (< 0.05)",
                ts.outlet_sponge, ts.sponge_target, r_app);

        g.section("B: Ahmed body forces, fast preset, app solver");
        const auto c0 = ahmed(ctx, ts, 0);
        const auto c1 = ahmed(ctx, ts, ts.outlet_sponge);
        g.note("bare outlet: Cd %.4f (250-step window std %.4f)", c0.mean, c0.std);
        g.note("app sponge : Cd %.4f (250-step window std %.4f)", c1.mean, c1.std);
        const double dcd = std::abs(c1.mean - c0.mean) / c0.mean;
        g.check(dcd < 0.01 && c1.std <= c0.std * 1.1,
                "sponge changes Cd by %.2f %% (< 1 %%), std ratio %.2f (<= 1.1)", dcd * 100.0,
                c1.std / c0.std);
    });
}
