// V31 -- rotors as actuator lines.
// Specified in THEORY 3.12 (docs/THEORY.md).
//
//   A. Actuator-disc theory: a uniformly loaded disc (radius 12 cells,
//      Gaussian eps 2) in a uniform stream at thrust coefficient
//      C_T = T / (1/2 rho U^2 pi R^2), blockage 1.2 % (the induction falls
//      with blockage: 3.6 % short at 2.2 %, 4.9 % at 4.9 %).
//      (a) Linear theory (light loading, C_T 0.1): the induction at the disc,
//          a = 1 - <u_x> / U over the inner 0.8 R, equals C_T / 4 within 3 %.
//      (b) The applied thrust (C_T 0.5) equals the momentum deficit between
//          planes ahead of and behind the disc (V25's survey) within 2 % --
//          the smeared force is the force applied.
//      Measured, not gated, at C_T 0.5: the induction against 1-D momentum
//      theory, a = (1 - sqrt(1 - C_T)) / 2, and Froude's ratio of far-wake to
//      disc induction on the axis. The induction reads 8 % short; a wider
//      section, a narrower kernel and a lower viscosity barely move it, so its
//      cause is open (THEORY 3.12).
//   B. The app's tunnel (Tunnel, fast grid): the catalogue wind turbine in
//      free air (a floor's friction would join the survey but not the
//      balance), sized to a swept area of 5 % of the section, its actuator-line rotor
//      at the design tip-speed ratio 7, averaged over 3 flow-throughs: the
//      wake survey equals the force balance of tower and rotor together
//      within 3 % (the rotor's force enters the balance); the power
//      coefficient positive (the flow drives it) and under the Betz limit
//      16/27; and the run stays healthy at the top speed.
#include <filesystem>

#include "cases.hpp"
#include "windoa/flow_stats.hpp"
#include "windoa/rotor.hpp"

using namespace windoa;

namespace {

struct Disc {
    double a_inner, a_axis, a_far, deficit, thrust;
};

Disc disc_run(Context& ctx, double ct) {
    constexpr int NX = 192, NY = 192, NZ = 192;
    constexpr float U = 0.05f, R = 12.0f;
    constexpr int RAMP = 1000, STEPS = 16'000, AVG = 4'000;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.51f;
    c.regularised = true;
    c.u_inlet = U;
    c.smagorinsky_cs = 0.0f;
    lbm::Solver s(ctx, c);
    const std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
    s.set_flags(flags);
    RotorSpec disc;
    disc.hub = {64.0f, NY / 2.0f, NZ / 2.0f};
    disc.axis = {1.0f, 0.0f, 0.0f};
    disc.r_tip = R;
    disc.disc = true;
    const double thrust = ct * 0.5 * U * U * gate::kPi * R * R;
    disc.disc_force = float(thrust);
    ActuatorLines alm(ctx, s, {disc});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < RAMP; ++k) { // the stream ramps under the loading
        s.set_inlet_velocity(U * float(k + 1) / RAMP);
        alm.step_with(s, 1);
    }
    alm.step_with(s, STEPS - RAMP - AVG);
    FlowStats st(ctx, s.cells());
    Disc d{};
    int samples = 0;
    // the mean axial velocity over the planes xs within radius frac R
    auto mean_u = [&](const std::vector<float>& v, std::initializer_list<int> xs, double frac) {
        double sum = 0.0;
        int n = 0;
        for (int x : xs)
            for (int y = 0; y < NY; ++y)
                for (int z = 0; z < NZ; ++z) {
                    const double dy = y + 0.5 - NY / 2.0, dz = z + 0.5 - NZ / 2.0;
                    if (dy * dy + dz * dz > frac * frac * R * R)
                        continue;
                    sum += v[3 * ((std::size_t(x) * NY + y) * NZ + z)];
                    ++n;
                }
        return sum / n;
    };
    for (int k = 0; k < AVG / 100; ++k) {
        alm.step_with(s, 100);
        st.add(s.macro_buffer(s.live_index()), 100.0);
        if (k % 8 == 0) {
            const auto v = s.velocity();
            d.a_inner += 1.0 - mean_u(v, {63, 64}, 0.8) / U;
            d.a_axis += 1.0 - mean_u(v, {63, 64}, 0.25) / U;
            d.a_far += 1.0 - mean_u(v, {123, 124}, 0.25) / U; // 5 R behind
            ++samples;
        }
    }
    d.a_inner /= samples;
    d.a_axis /= samples;
    d.a_far /= samples;
    const double nu = (c.tau - 0.5) / 3.0;
    d.deficit = plane_momentum_flux(st, flags, NX, NY, NZ, 24, nu).momentum -
                plane_momentum_flux(st, flags, NX, NY, NZ, 150, nu).momentum;
    d.thrust = thrust;
    return d;
}

void part_a(Context& ctx, gate::Gate& g) {
    g.section("A: a uniformly loaded actuator disc");
    {
        const double ct = 0.1;
        const Disc d = disc_run(ctx, ct);
        const double e_a = std::abs(d.a_inner - ct / 4.0) / (ct / 4.0);
        g.check(e_a < 0.03,
                "C_T %.1f: induction %.4f vs linear theory C_T / 4 = %.4f (%.1f %%, < 3 %%)", ct,
                d.a_inner, ct / 4.0, 100.0 * e_a);
    }
    {
        const double ct = 0.5;
        const Disc d = disc_run(ctx, ct);
        const double e_t = std::abs(d.deficit - d.thrust) / d.thrust;
        g.check(e_t < 0.02, "C_T %.1f: momentum deficit %.4f vs thrust %.4f (%.2f %%, < 2 %%)", ct,
                d.deficit, d.thrust, 100.0 * e_t);
        const double a_1d = (1.0 - std::sqrt(1.0 - ct)) / 2.0;
        g.note("measured, not gated: induction (inner 0.8 R) %.4f vs 1-D momentum theory %.4f "
               "(%+.1f %%); far wake / disc on the axis %.3f (Froude, for averages: 2)",
               d.a_inner, a_1d, 100.0 * (d.a_inner - a_1d) / a_1d, d.a_far / d.a_axis);
    }
}

void part_b(Context& ctx, gate::Gate& g) {
    g.section("B: the app's wind turbine, averaged");
    const auto dir = std::filesystem::temp_directory_path() / "windoa_v31_cache";
    std::filesystem::remove_all(dir);
    {
        TunnelSettings ts = tunnel_preset("fast");
        Tunnel t(ctx, ts, dir);
        const catalogue::Entry* e = catalogue::find("wind_turbine");
        Model m = model_from_catalogue(*e);
        Placement p = default_placement(*e, ts, m.mesh);
        p.length_cells = 30.0f;            // rotor radius 12 cells: 4.9 % of the 96 x 96 section
        p.ground = catalogue::Ground::Air; // no floor: its friction would join the survey
        t.set_model(std::move(m), p);
        t.set_rotors(true, 7.0f);
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
        const double err = std::abs(a.cd_wake - a.cd_balance) / std::abs(a.cd_balance);
        g.check(a.wake_valid && err < 0.03,
                "%.2f flow-throughs averaged: survey %+.3f vs balance (tower + rotor) %+.3f "
                "(%.2f %%, < 3 %%)",
                a.avg_flow_throughs, a.cd_wake, a.cd_balance, 100.0 * err);
        g.check(st.rotor_cp > 0.0 && st.rotor_cp < 16.0 / 27.0,
                "tip-speed ratio %.1f, swept area %.1f %%: C_T %.3f, C_P %.3f (0 < C_P < Betz "
                "0.593)",
                st.rotor_tsr, 100.0 * st.rotor_blockage, st.rotor_ct, st.rotor_cp);
        t.set_speed(ts.u_max);
        float peak = 0.0f;
        for (int k = 0; k < 40; ++k) {
            t.advance(200);
            peak = std::max(peak, t.status().max_speed);
        }
        const TunnelStatus top = t.status();
        g.check(top.health_note.empty() && peak < 0.3f,
                "U %.2f, 8,000 steps at tip-speed ratio 7: max |u| %.3f (< 0.3), no restart",
                ts.u_max, peak);
    }
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    return gate::run("V31_actuator_lines", [](gate::Gate& g) {
        Context ctx;
        part_a(ctx, g);
        part_b(ctx, g);
    });
}
