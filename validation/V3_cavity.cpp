// V3 -- lid-driven cavity at Re 1000.
// Specified in THEORY 2.7 (docs/THEORY.md).
//
// Published reference: Ghia, Ghia & Shin (1982), u_x along the vertical
// centreline. 128^2 fluid cells, BGK, 200,000 steps with the closed-domain
// mass anchor every 500 steps (without it the lid's mass source deflates
// the velocity field -- see V2). Pass: RMS deviation < 0.03 lid-speed units
// (measured 0.0064).
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V3_cavity", [](gate::Gate& g) {
        Context ctx;
        using namespace cases::cavity;
        const Setup su; // BGK, Re 1000, anchored
        g.note("Re 1000: %d^2 fluid cells, tau %.4f, %d steps, anchor every %d", N, su.tau,
               su.steps, su.anchor_every);
        const auto r = run(ctx, su);
        g.note(" y_hat    sim      Ghia");
        for (std::size_t k = 0; k < GHIA_Y.size(); ++k)
            g.note("%6.4f  %+7.4f  %+7.4f", GHIA_Y[k], r.at_ghia[k], GHIA_U[k]);
        g.note("mean fluid density %.6f (anchored)", r.mean_rho);
        g.check(r.finite && r.rms < 0.03, "Ghia centreline: RMS %.4f (< 0.03)", r.rms);
    });
}
