// V1 -- body-force Poiseuille channel.
// Specified in THEORY 1.5, 2.7 (docs/THEORY.md).
//
// External reference: the analytic parabola between half-way bounce-back
// walls (validation/cases.hpp). Pass: relative L2 < 1 % (measured 0.074 %).
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V1_poiseuille", [](gate::Gate& g) {
        Context ctx;
        using namespace cases::poiseuille;
        g.note("%dx%dx%d, tau %.1f, g %.0e, %d steps, BGK", NX, NY, NZ, TAU, G_X, STEPS);
        const auto r = run(ctx, false, false);
        g.note("u_max sim %.7f  exact %.7f", r.u[NY / 2], r.exact[NY / 2 - 1]);
        g.check(r.l2 < 0.01, "analytic parabola: relative L2 %.3f %% (< 1 %%)", r.l2 * 100.0);
    });
}
