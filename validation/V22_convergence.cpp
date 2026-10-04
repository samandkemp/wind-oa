// V22 -- the "flow is developed" detector (engine ConvergenceMonitor).
// Specified in THEORY 9.5, 9.8 (docs/THEORY.md).
//
// The app tells the user the numbers have SETTLED; that is only worth
// saying if it is true, so the detector is checked on real solver output.
//   A. Invariants on synthetic signals (flow-throughs at u 0.05, nx 256):
//      an exponential settle converges, not before ft_min; a shedding
//      oscillation about 0.7 converges on the mean (within 2 %); a steady
//      drift is released at ft_max as "gave up", never "settled"; near-zero
//      noisy Cl converges via the absolute floor; a slow monotone approach
//      (time constant 1.2 ft) is declared only once the true remaining gap
//      is < 2.5 % (or released).
//   B. Real flow, cold start: the Ahmed body from rest, the app's solver on
//      the fast grid. When the detector says SETTLED, the Cd it reports is
//      within 2 % of the long-run mean (the last 12,000 of 36,000 steps).
//   C. Real flow, warm restart AoA 0 -> 6 keeping the flow (what the app
//      does when a slider moves): the same 2 % check.
#include <random>

#include "cases.hpp"
#include "windoa/convergence.hpp"

using namespace windoa;

namespace {

constexpr int NXS = 256;

template <class F> void feed(ConvergenceMonitor& m, F fn, int n) {
    for (int i = 0; i < n && m.developing(); ++i)
        m.add(fn(m.flow_throughs()), 100, 0.05, NXS);
}

struct History {
    std::vector<std::pair<long, double>> cd; // cumulative steps, Cd
    long declared_at = -1;
};

// Step in chunks of 100, feeding the frame-mean coefficients to the monitor.
History run(lbm::Solver& s, const TunnelSettings& t, double area, long n_steps, bool ramp,
            ConvergenceMonitor& mon) {
    s.read_mean_forces();
    const double q = 0.5 * t.u_inlet * t.u_inlet * area;
    History h;
    for (long done = 0; done < n_steps;) {
        constexpr int chunk = 100;
        if (ramp)
            s.set_inlet_velocity(t.u_inlet * std::min(1.0f, float(done + chunk) / t.ramp_steps));
        s.step(chunk);
        done += chunk;
        const auto f = s.read_mean_forces();
        const double cd = f.force[0] / q, cl = f.force[1] / q;
        h.cd.emplace_back(done, cd);
        if (mon.developing()) {
            mon.add({cd, cl}, f.steps, s.inlet_velocity(), t.nx);
            if (!mon.developing())
                h.declared_at = done;
        }
    }
    return h;
}

double long_run(const History& h, long tail = 12000) {
    const long last = h.cd.back().first;
    double sum = 0.0;
    int n = 0;
    for (const auto& [step, cd] : h.cd)
        if (step > last - tail) {
            sum += cd;
            ++n;
        }
    return sum / n;
}

void judge(gate::Gate& g, const char* label, const ConvergenceMonitor& mon, const History& h) {
    const double final_cd = long_run(h);
    if (!mon.settled()) {
        g.check(false, "%s: detector never settled (gave_up %d)", label, int(mon.gave_up()));
        return;
    }
    const double got = (*mon.last_window_means())[0];
    const double err = std::abs(got - final_cd) / std::abs(final_cd);
    g.check(err < 0.02,
            "%s: SETTLED at step %ld (%.2f ft); reported Cd %.3f vs long-run %.3f (%.1f %%)", label,
            h.declared_at, mon.flow_throughs(), got, final_cd, err * 100.0);
}

} // namespace

int main() {
    return gate::run("V22_convergence", [](gate::Gate& g) {
        g.section("A: synthetic signals");
        {
            ConvergenceMonitor m;
            feed(
                m,
                [](double ft) {
                    return ConvergenceMonitor::Coeffs{0.7 + 1.5 * std::exp(-ft / 0.4), 0.2};
                },
                4000);
            g.check(m.settled() && m.flow_throughs() >= 1.5,
                    "exponential settle: settled %d at %.2f ft", int(m.settled()),
                    m.flow_throughs());
        }
        {
            ConvergenceMonitor m;
            feed(
                m,
                [](double ft) {
                    return ConvergenceMonitor::Coeffs{
                        0.7 + 0.08 * std::sin(2 * gate::kPi * ft / 0.07), 0.0};
                },
                4000);
            const double w = m.last_window_means() ? (*m.last_window_means())[0] : 0.0;
            g.check(m.settled() && std::abs(w - 0.7) < 0.02 * 0.7,
                    "shedding about 0.7: settled %d, window mean %.4f", int(m.settled()), w);
        }
        {
            ConvergenceMonitor m;
            feed(
                m, [](double ft) { return ConvergenceMonitor::Coeffs{0.7 + 0.2 * ft, 0.0}; }, 4000);
            g.check(m.gave_up() && !m.settled(), "steady drift: gave_up %d, settled %d",
                    int(m.gave_up()), int(m.settled()));
        }
        {
            std::mt19937 rng(1);
            std::normal_distribution<double> nd(0.0, 1.0);
            ConvergenceMonitor m;
            feed(m, [&](double) { return ConvergenceMonitor::Coeffs{0.7, 0.002 * nd(rng)}; }, 4000);
            g.check(m.settled(), "near-zero Cl noise: settled %d", int(m.settled()));
        }
        {
            ConvergenceMonitor m;
            feed(
                m,
                [](double ft) {
                    return ConvergenceMonitor::Coeffs{0.7 + 0.3 * std::exp(-ft / 1.2), 0.0};
                },
                4000);
            if (m.settled()) {
                const double gap = 0.3 * std::exp(-m.flow_throughs() / 1.2) / 0.7;
                g.check(gap < 0.025,
                        "slow approach: settled at %.2f ft, true remaining gap %.1f %%",
                        m.flow_throughs(), gap * 100.0);
            } else {
                g.check(m.gave_up(), "slow approach: released at ft_max (gave_up %d)",
                        int(m.gave_up()));
            }
        }

        Context ctx;
        const TunnelSettings t = tunnel_preset("fast");
        // exactly the app's solver (incl. the outlet sponge): the detector is
        // validated on the signal the app actually feeds it
        lbm::Solver s(ctx, solver_config(t));
        Voxeliser vox(ctx, t.nx, t.ny, t.nz);

        g.section("B: real flow, cold start (Ahmed, fast preset)");
        auto p0 = cases::app::ahmed(vox, t, 0.0f);
        cases::app::place(s, p0);
        s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
        ConvergenceMonitor mb;
        const auto hb = run(s, t, p0.area, 36000, true, mb);
        judge(g, "cold AoA 0", mb, hb);

        g.section("C: real flow, warm restart AoA 0 -> AoA 6");
        auto p6 = cases::app::ahmed(vox, t, 6.0f);
        cases::app::place(s, p6);
        ConvergenceMonitor mc;
        mc.restart("aoa change");
        const auto hc = run(s, t, p6.area, 34000, false, mc);
        judge(g, "warm AoA 6", mc, hc);
    });
}
