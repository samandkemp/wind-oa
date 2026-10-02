// V11 -- the stability safeguards.
// Specified in THEORY 9.4, 4.3, 9.8 (docs/THEORY.md).
//
//   A. Invariant: health() counts non-finite FLUID cells and reports max |u|
//      (testing the bits, because an in-kernel `x != x` NaN test is folded
//      away by fast-math). The gate poisons the distributions
//      (set_state refreshes rho / u from them): a NaN f, an inf f, a NaN
//      in another cell's rest population, and one cell at equilibrium with
//      u = 0.9. Expect 3 non-finite cells and max |u| 0.9.
//   B. Invariant: at the configured cap (TunnelSettings max_wall_speed), a
//      spinning wheel (finite cylinder, R 14, width 12) switched on in a
//      developed, ramped flow -- what the app's checkbox does -- stays
//      healthy for 8,000 steps at the default and the maximum freestream,
//      on the app's own solver (solver_config) and fast grid.
//   C. Invariant: a 10,000-step read_mean_forces() window equals the double
//      host sum of the per-step forces (rel 1e-3; a long window summed in
//      f32 per link would lose precision).
#include <array>
#include <limits>
#include <memory>

#include "cases.hpp"
#include "windoa/tunnel.hpp"

using namespace windoa;

namespace {

constexpr int kEx[lbm::Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
constexpr double kWq[lbm::Q] = {1.0 / 3,  1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
                                1.0 / 18, 1.0 / 18, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                                1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                                1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};

void part_a(Context& ctx, gate::Gate& g) {
    g.section("A: health() on a healthy and a poisoned field");
    constexpr int NX = 32, NY = 16, NZ = 16;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.6f;
    c.u_inlet = 0.05f;
    lbm::Solver s(ctx, c);
    s.set_flags(std::vector<std::uint8_t>(std::size_t(NX) * NY * NZ, lbm::FLUID));
    s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
    s.step(10);
    const auto h0 = s.health();
    auto f = s.get_state();
    const std::size_t n = std::size_t(NX) * NY * NZ;
    auto cell = [](int x, int y, int z) { return (std::size_t(x) * NY + y) * NZ + z; };
    f[1 * n + cell(5, 5, 5)] = std::numeric_limits<float>::quiet_NaN();
    f[1 * n + cell(6, 6, 6)] = std::numeric_limits<float>::infinity();
    f[0 * n + cell(8, 8, 8)] = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < lbm::Q; ++i) { // equilibrium at rho 1, u (0.9, 0, 0)
        const double eu = kEx[i] * 0.9;
        f[i * n + cell(9, 9, 9)] = float(kWq[i] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * 0.81));
    }
    s.set_state(f);
    const auto h1 = s.health();
    g.note("healthy : max|u| %.4f, non-finite cells %d", h0.max_speed, h0.bad_cells);
    g.note("poisoned: max|u| %.4f, non-finite cells %d (expect 0.9000 and 3)", h1.max_speed,
           h1.bad_cells);
    g.check(h0.bad_cells == 0 && std::abs(h0.max_speed - 0.05f) < 0.01f && h1.bad_cells == 3 &&
                std::abs(h1.max_speed - 0.9f) < 1e-3f,
            "health() detects the blow-up and the fast cell");
}

struct Wheel {
    std::vector<std::uint8_t> flags;
    std::array<float, 3> centre{};
    double rmax = 0.0;
};

Wheel make_wheel(const TunnelSettings& t) {
    Wheel w;
    w.centre = {0.3f * t.nx, 0.5f * t.ny, 0.5f * t.nz};
    constexpr double R = 14.0, HALF_W = 6.0;
    w.flags.assign(std::size_t(t.nx) * t.ny * t.nz, lbm::FLUID);
    for (int x = 0; x < t.nx; ++x)
        for (int y = 0; y < t.ny; ++y) {
            const double dx = x + 0.5 - w.centre[0], dy = y + 0.5 - w.centre[1];
            if (dx * dx + dy * dy >= R * R)
                continue;
            for (int z = 0; z < t.nz; ++z)
                if (std::abs(z + 0.5 - w.centre[2]) < HALF_W) {
                    w.flags[(std::size_t(x) * t.ny + y) * t.nz + z] = lbm::OBSTACLE;
                    w.rmax = std::max(w.rmax, std::sqrt(dx * dx + dy * dy));
                }
        }
    return w;
}

// The app's solver, the wheel voxelised in, ramped like the app and
// developed for 4,000 steps.
std::unique_ptr<lbm::Solver> app_solver(Context& ctx, const TunnelSettings& t, const Wheel& w,
                                        float U) {
    auto s = std::make_unique<lbm::Solver>(ctx, solver_config(t));
    s->set_flags(w.flags);
    s->init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < 20; ++k) {
        s->set_inlet_velocity(U * float(k + 1) / 20.0f);
        s->step(t.ramp_steps / 20);
    }
    s->step(4000);
    return s;
}

} // namespace

int main() {
    return gate::run("V11_stability", [](gate::Gate& g) {
        Context ctx;
        const TunnelSettings t = tunnel_preset("fast");
        part_a(ctx, g);

        g.section("B: spinning wheel at the wall-speed cap, switched on in a developed flow");
        const Wheel w = make_wheel(t);
        const float speeds[2] = {t.u_inlet, t.u_max};
        for (int k = 0; k < 2; ++k) {
            auto s = app_solver(ctx, t, w, speeds[k]);
            // scaled so the fastest obstacle cell moves at the cap, as the app does
            s->set_rotation(w.centre, {0.0f, 0.0f, float(t.max_wall_speed / w.rmax)});
            bool healthy = true;
            float worst = 0.0f;
            for (int i = 0; i < 16 && healthy; ++i) {
                s->step(500);
                const auto h = s->health();
                worst = std::max(worst, h.max_speed);
                healthy = h.bad_cells == 0 && h.max_speed <= t.diverge_speed;
            }
            g.check(healthy, "U %.3f: %s, worst max|u| %.3f (guard %.2f, cap %.2f)", speeds[k],
                    healthy ? "healthy" : "DIVERGED", worst, t.diverge_speed, t.max_wall_speed);
        }

        g.section("C: long force window vs the double host sum of per-step forces");
        {
            constexpr int N_LONG = 10'000;
            auto s = app_solver(ctx, t, w, t.u_inlet);
            s->read_mean_forces();
            double host = 0.0;
            for (int i = 0; i < N_LONG; ++i) {
                s->step(1);
                host += s->obstacle_force()[0];
            }
            const auto f = s->read_mean_forces();
            const double ref = host / N_LONG;
            const double err = std::abs(f.force[0] - ref) / std::abs(ref);
            g.check(f.steps == N_LONG && err < 1e-3,
                    "Fx window %.6e vs host %.6e: rel %.1e (< 1e-3), n %d", f.force[0], ref, err,
                    f.steps);
        }
    });
}
