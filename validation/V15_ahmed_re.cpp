// V15 -- Ahmed body Cd vs Reynolds number, regularised collision.
// Specified in THEORY 4.6 (docs/THEORY.md).
//
// Published trend: the drag of a bluff road body falls as Re rises towards
// the experimental 0.285 (Re 4.3e6), as boundary layers thin and
// separation is delayed. Under BGK the tunnel was capped at Re_L ~1,250;
// the regularised operator lifts the ceiling. Re_L 1,250 / 7,500 / 20,000
// in V14's setup. Pass: every run stable and Cd(20,000) < Cd(1,250) -- a
// correct downward trend (absolute convergence needs resolution beyond a
// sandbox).
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V15_ahmed_re", [](gate::Gate& g) {
        Context ctx;
        using namespace cases::ahmed;
        const double res[3] = {1250.0, 7500.0, 20000.0};
        double cd[3] = {};
        bool finite = true;
        g.note("  Re_L     tau       Cd      Cl");
        for (int k = 0; k < 3; ++k) {
            const float tau = shapes::tau_for_reynolds(U_IN, LENGTH, float(res[k]));
            const auto r = run(ctx, tau, true);
            cd[k] = r.cd;
            finite = finite && r.finite;
            g.note("%7.0f  %.5f  %.3f  %+.3f", res[k], tau, r.cd, r.cl);
        }
        g.check(finite && cd[0] > cd[2],
                "Cd falls with Re: %.3f -> %.3f (experiment 0.285 at 4.3e6)", cd[0], cd[2]);
    });
}
