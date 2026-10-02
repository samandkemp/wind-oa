// V14 -- the Ahmed body, 25 deg slant.
// Specified in THEORY 4.5, 4.6 (docs/THEORY.md).
//
// Published reference, with the caveat: experimental Cd 0.285 at Re_L
// 4.3e6. At the Re this grid affords (Re_L ~1,250, BGK + cs 0.1, L 100
// cells in a 256 x 128 x 128 tunnel, ~6.5 % blockage) a much higher Cd is
// expected -- thick laminar layers and early separation lose the nose's
// streamlining, and blockage inflates it further: Reynolds number and
// blockage, not a solver defect (THEORY 4.5). The checks meaningful at
// this Re:
//   - a stable run with Cd in the low-Re bluff-body band 0.6 .. 1.8;
//   - positive lift (the slant generates lift on the real body);
//   - a recirculating near wake: reverse flow (u_x < 0) behind the base in
//     the mid-span plane.
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V14_ahmed", [](gate::Gate& g) {
        Context ctx;
        using namespace cases::ahmed;
        constexpr float TAU = 0.512f;
        const auto r = run(ctx, TAU, false);
        g.note("%zu voxels, A_ref %.0f cells^2, Re_L %.0f", r.n_solid, r.a_ref,
               U_IN * LENGTH / ((TAU - 0.5) / 3.0));
        g.check(r.finite && r.cd > 0.6 && r.cd < 1.8, "Cd %.3f in the low-Re band (0.6, 1.8)",
                r.cd);
        g.check(r.cl > 0.0, "Cl %+.3f > 0 (slant lift)", r.cl);
        g.check(r.min_ux_wake < 0.0, "near-wake recirculation: min u_x %+.4f < 0", r.min_ux_wake);
        g.note("experiment: Cd 0.285 at Re 4.3e6 -- see the header for why this Re reads high");
    });
}
