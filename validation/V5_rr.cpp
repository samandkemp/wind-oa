// V5 -- recursive-regularised (RR) collision.
// Specified in THEORY 2.4, 2.7 (docs/THEORY.md).
//
// RR rebuilds the 3rd-order non-equilibrium Hermite coefficients from the
// 2nd-order ones (Jacob, Malaspinas & Sagaut 2018).
// A. Closed form: RR must recover the same Navier-Stokes viscosity as BGK --
//    Poiseuille vs the parabola, L2 < 1 %.
// B. Documented scope, not pass / fail: the cs = 0 stability boundary of the
//    enclosed driven cavity (128^2, lid 0.1) for plain-regularised and RR,
//    scanned from tau 0.57 down to 0.51; the gate prints where the boundary
//    lies. HRR (hybrid FD stress) is deliberately not implemented.
#include "cases.hpp"

using namespace windoa;

namespace {

// Part B's tau scan, in hundredths.
constexpr int kScanTau100[] = {57, 55, 53, 52, 51};

// Stable = finite and |u| <= 0.3 at every 1,000-step check.
bool cavity_stable(Context& ctx, float tau, bool recursive, int steps = 8000) {
    cases::cavity::Setup su;
    su.tau = tau;
    su.regularised = !recursive;
    su.recursive = recursive;
    lbm::Solver s(ctx, cases::cavity::config(su));
    s.set_flags(cases::cavity::flags());
    s.set_lid_velocity({cases::cavity::U_LID, 0.0f, 0.0f});
    s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    for (int k = 0; k < steps / 1000; ++k) {
        s.step(1000);
        const auto h = s.health();
        if (h.bad_cells > 0 || !(h.max_speed <= 0.3f))
            return false;
    }
    return true;
}

} // namespace

int main() {
    return gate::run("V5_rr", [](gate::Gate& g) {
        Context ctx;
        g.section("A: RR recovers the NS viscosity (Poiseuille)");
        const auto p = cases::poiseuille::run(ctx, false, true);
        g.check(p.l2 < 0.01, "RR Poiseuille: relative L2 %.3f %% (< 1 %%)", p.l2 * 100.0);

        g.section("B: stability boundary, cs = 0 driven cavity (documentation)");
        for (const int t100 : kScanTau100) {
            const float tau = float(t100) / 100.0f;
            const double re = 0.1 * 128 / ((tau - 0.5) / 3.0);
            const bool reg = cavity_stable(ctx, tau, false);
            const bool rr = cavity_stable(ctx, tau, true);
            g.note("tau %.2f (Re ~%4.0f): plain-reg %-3s RR %s", tau, re, reg ? "ok" : "NaN",
                   rr ? "ok" : "NaN");
        }
    });
}
