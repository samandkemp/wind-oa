// V29 -- engine ports (jets and intakes) in the lattice Boltzmann regime.
// Specified in THEORY 3.11 (docs/THEORY.md).
//
//   A. Conservation, steady and laminar (tau 0.8): a block (24 cells
//      square, 24 long) in a stream U 0.05 with a round port (radius 6) on its
//      rear face. As an exhaust blowing at 2 U: (a) the mass the face
//      injects -- the mass flux behind the block less the flux ahead -- equals
//      rho U_j A, A the face's cells (the moving-wall term passes exactly
//      rho u_w per face cell through a half-way wall), within 1 %; (b) the
//      force on the block, the jet's reaction included, equals the momentum
//      balance between the planes (V25's survey: rho u^2 + rho cs^2 -
//      2 rho nu du/dx) within 2 %. (c) The same balance with the port as an
//      intake drawing at U, and (d) with the port off, as the control.
//   B. The app's tunnel: a block with an exhaust (Tunnel, power on, fast
//      grid, sub-cell walls, LES): the time-averaged wake survey equals the
//      force balance within 2 %, thrust included, and the mass flux grows
//      across the model by the port's injection within 5 %.
#include <filesystem>

#include "cases.hpp"
#include "windoa/flow_stats.hpp"

using namespace windoa;

namespace {

constexpr int NX = 192, NY = 80, NZ = 80;
constexpr float U = 0.05f;

struct Result {
    double force, survey, injected, expected;
};

// u_port: the face velocity along +x (exhaust > 0, intake < 0, off 0).
Result block(Context& ctx, float u_port) {
    constexpr int RAMP = 1000, STEPS = 24'000, AVG = 4'000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.8f;
    c.u_inlet = U;
    c.smagorinsky_cs = 0.0f;
    c.moving_boundaries = true;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
    shapes::add_box(flags, NX, NY, NZ, 56, 79, 28, 51, 28, 51); // rear face at x = 80
    s.set_flags(flags);
    int face = 0; // the port's face cells: x = 79, within radius 6 of the axis
    for (int y = 0; y < NY; ++y)
        for (int z = 0; z < NZ; ++z) {
            const double dy = y + 0.5 - 40.0, dz = z + 0.5 - 40.0;
            face += dy * dy + dz * dz <= 36.0 ? 1 : 0;
        }
    if (u_port != 0.0f)
        s.set_wall_velocity({79.5f, 40.0f, 40.0f}, {1.0f, 0.0f, 0.0f}, 6.0f, 1.5f,
                            {u_port, 0.0f, 0.0f});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < RAMP; ++k) {
        s.set_inlet_velocity(U * float(k + 1) / RAMP);
        s.step(1);
    }
    s.step(STEPS - RAMP - AVG);
    FlowStats st(ctx, s.cells());
    s.read_mean_forces();
    for (int k = 0; k < AVG / 100; ++k) {
        s.step(100);
        st.add(s.macro_buffer(s.live_index()), 100.0);
    }
    const double fx = s.read_mean_forces().force[0];
    const double nu = (c.tau - 0.5) / 3.0;
    const PlaneFlux up = plane_momentum_flux(st, flags, NX, NY, NZ, 24, nu);
    const PlaneFlux dn = plane_momentum_flux(st, flags, NX, NY, NZ, 130, nu);
    return {fx, up.momentum - dn.momentum, dn.mass - up.mass, double(u_port) * face};
}

void part_a(Context& ctx, gate::Gate& g) {
    g.section("A: a block with a rear port, laminar -- mass and momentum");
    const struct {
        const char* name;
        float u;
    } cases[] = {{"exhaust at 2 U", 2.0f * U}, {"intake at U", -U}, {"port off", 0.0f}};
    for (const auto& k : cases) {
        const Result r = block(ctx, k.u);
        const double e_mom = std::abs(r.survey - r.force) / std::abs(r.force);
        if (k.u != 0.0f) {
            const double e_mass = std::abs(r.injected - r.expected) / std::abs(r.expected);
            g.check(e_mass < 0.01 && e_mom < 0.02,
                    "%-14s: injected mass %+.4f vs rho u_w A %+.4f (%.2f %%, < 1 %%); force %+.5f "
                    "vs survey %+.5f (%.2f %%, < 2 %%)",
                    k.name, r.injected, r.expected, 100.0 * e_mass, r.force, r.survey,
                    100.0 * e_mom);
        } else {
            g.check(e_mom < 0.02, "%-14s: force %+.5f vs survey %+.5f (%.2f %%, < 2 %%)", k.name,
                    r.force, r.survey, 100.0 * e_mom);
        }
    }
}

void part_b(Context& ctx, gate::Gate& g) {
    g.section("B: the app's tunnel -- a block with an exhaust, averaged");
    const auto dir = std::filesystem::temp_directory_path() / "windoa_v29_cache";
    std::filesystem::remove_all(dir);
    {
        const TunnelSettings ts = tunnel_preset("fast");
        Tunnel t(ctx, ts, dir);
        Model m;
        m.id = m.label = "v29_block";
        m.mesh = catalogue::make_box(0.0, 0.0, 0.0, 2.0, 1.0, 1.0);
        m.hash = "v29_block";
        catalogue::Port port;
        port.c = {1.0, 0.0, 0.0};
        port.n = {1.0, 0.0, 0.0};
        port.r = 0.3;
        port.speed_ratio = 2.0;
        m.ports = {port};
        Placement p;
        p.length_cells = 48.0f;
        t.set_model(m, p);
        t.set_power(true, 1.0f);
        t.set_averaging(true);
        for (int batches = 0; batches < 2000; ++batches) {
            t.advance(100);
            const TunnelStatus st = t.status();
            if (st.averaging_active && st.avg_flow_throughs >= 3.0)
                break;
        }
        t.refresh_analysis();
        const TunnelAnalysis& a = t.analysis();
        const TunnelStatus st = t.status();
        // the injection expected: face cells (radius 0.3 x 24 cells) x rho u_j over
        // the inflow's mass flux (the tunnel's whole section at U)
        int face = 0;
        const double r = 0.3 * 24.0;
        for (int y = -12; y <= 12; ++y)
            for (int z = -12; z <= 12; ++z)
                face += (y + 0.5) * (y + 0.5) + (z + 0.5) * (z + 0.5) <= r * r ? 1 : 0;
        const double expected = 2.0 * face / (double(ts.ny) * ts.nz);
        const double err = std::abs(a.cd_wake - a.cd_balance) / std::abs(a.cd_balance);
        const double e_mass = std::abs(a.mass_imbalance - expected) / expected;
        g.check(a.wake_valid && err < 0.02 && e_mass < 0.05,
                "%lld steps, %.2f flow-throughs averaged: survey Cd %+.4f vs balance %+.4f (%.2f "
                "%%, < 2 %%); mass flux grows %.2e vs the port's %.2e (%.1f %%, < 5 %%)",
                static_cast<long long>(st.steps), a.avg_flow_throughs, a.cd_wake, a.cd_balance,
                100.0 * err, a.mass_imbalance, expected, 100.0 * e_mass);
    }
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    return gate::run("V29_jets", [](gate::Gate& g) {
        Context ctx;
        part_a(ctx, g);
        part_b(ctx, g);
    });
}
