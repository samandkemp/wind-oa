// V25 -- the wake survey: drag from a control-volume momentum balance of the
// mean flow (engine plane_momentum_flux, Tunnel analysis).
// Specified in THEORY 12.2, 12.6 (docs/THEORY.md).
//
// Conservation-law reference: between an upstream plane and a survey plane
// behind the body, with free-slip side walls carrying no x-force, the drag
// equals the drop in the x-momentum flux (momentum flux + pressure - viscous
// normal stress) of the mean flow. Two independent measurements of one
// force: this survey, and the momentum-exchange force balance averaged over
// the same window.
//   A. A steady wake (the sphere of V6 at Re 100, laminar): survey drag at
//      two planes behind the body against the force balance, within 1 %
//      (measured 0.34 % and 0.56 %); the mass flux through the planes equal
//      to 2e-4.
//   B. What the app runs (the Ahmed body on the fast preset, LES, sponge,
//      an unsteady wake) through Tunnel's averaging and analysis: the
//      survey's Cd beside the force balance's mean Cd. The Reynolds stress
//      enters through the variance; the eddy viscosity's normal stress is
//      left out (THEORY 12.5), so the band is wider: 2 % (measured 0.52 %).
//
// The survey found a defect when it was first run: the free-slip side faces
// took a reflected population from one row inside the domain, creating and
// destroying mass along the walls, and the survey read 3 - 5 % low. With the
// faces mirroring about the face (THEORY 3.4; V2 now checks it) it closes.
#include <filesystem>

#include "cases.hpp"
#include "windoa/flow_stats.hpp"

using namespace windoa;

namespace {

void part_a(Context& ctx, gate::Gate& g) {
    g.section("A: steady sphere, Re 100 -- survey vs force balance");
    constexpr int NX = 160, NY = 80, NZ = 80;
    constexpr float D = 16.0f, U = 0.05f, RE = 100.0f;
    constexpr int RAMP = 2000, STEPS = 40'000, AVG = 5'000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = shapes::tau_for_reynolds(U, D, RE);
    c.u_inlet = U;
    c.smagorinsky_cs = 0.0f;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
    shapes::add_sphere(flags, NX, NY, NZ, NX * 0.3, NY / 2.0, NZ / 2.0, D / 2.0);
    s.set_flags(flags);
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < RAMP; ++k) {
        s.set_inlet_velocity(U * float(k + 1) / RAMP);
        s.step(1);
    }
    s.step(STEPS - RAMP - AVG);
    FlowStats st(ctx, s.cells());
    s.read_mean_forces(); // open the force window with the statistics
    for (int k = 0; k < AVG / 100; ++k) {
        s.step(100);
        st.add(s.macro_buffer(s.live_index()), 100.0);
    }
    const double fx = s.read_mean_forces().force[0];
    const double q = 0.5 * U * U * gate::kPi * (D / 2.0) * (D / 2.0);
    const double nu = (c.tau - 0.5) / 3.0;
    const int x1 = NX / 10;
    const PlaneFlux f1 = plane_momentum_flux(st, flags, NX, NY, NZ, x1, nu);
    g.note("force balance: Cd %.4f; upstream plane x = %d", fx / q, x1);
    for (const int x2 : {int(NX * 0.3 + 3 * D), NX - 30}) {
        const PlaneFlux f2 = plane_momentum_flux(st, flags, NX, NY, NZ, x2, nu);
        const double drag = f1.momentum - f2.momentum;
        const double err = std::abs(drag - fx) / fx;
        const double dm = std::abs(f2.mass - f1.mass) / f1.mass;
        g.check(err < 0.01 && dm < 2e-4,
                "survey plane x = %3d: Cd %.4f, %.2f %% from the balance (< 1 %%); mass flux "
                "%.1e (< 2e-4)",
                x2, drag / q, err * 100.0, dm);
    }
}

void part_b(Context& ctx, gate::Gate& g) {
    g.section("B: the app's tunnel (Ahmed, fast preset) -- Tunnel averaging and analysis");
    const auto dir = std::filesystem::temp_directory_path() / "windoa_v25_cache";
    std::filesystem::remove_all(dir);
    {
        const TunnelSettings ts = tunnel_preset("fast");
        Tunnel t(ctx, ts, dir);
        const auto* e = catalogue::find("ahmed_25deg");
        const Model m = model_from_catalogue(*e);
        t.set_model(m, default_placement(*e, ts, m.mesh));
        t.set_averaging(true);
        int batches = 0;
        while (batches < 2000) {
            t.advance(100);
            ++batches;
            const TunnelStatus st = t.status();
            if (st.averaging_active && st.avg_flow_throughs >= 3.0)
                break;
        }
        t.refresh_analysis();
        const TunnelAnalysis& a = t.analysis();
        const TunnelStatus st = t.status();
        g.note("%lld steps; averaged %.2f flow-throughs (%d samples); planes x = %d and %d",
               static_cast<long long>(st.steps), a.avg_flow_throughs, a.avg_samples, a.x_upstream,
               a.x_survey);
        const double err = std::abs(a.cd_wake - a.cd_balance) / a.cd_balance;
        g.check(a.wake_valid && err < 0.02 && std::abs(a.mass_imbalance) < 2e-4,
                "survey Cd %.4f vs force-balance mean %.4f: %.2f %% (< 2 %%); mass flux %.1e "
                "(< 2e-4)",
                a.cd_wake, a.cd_balance, err * 100.0, a.mass_imbalance);
        double umin = 1e30;
        for (const float v : a.profile_uy)
            umin = std::min(umin, double(v));
        g.note("mean wake profile at the survey plane: min <u_x>/U %.3f over %zu cells", umin,
               a.profile_uy.size());
    }
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    return gate::run("V25_wake_survey", [](gate::Gate& g) {
        Context ctx;
        part_a(ctx, g);
        part_b(ctx, g);
    });
}
