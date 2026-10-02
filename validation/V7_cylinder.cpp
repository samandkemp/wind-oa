// V7 -- vortex shedding off a circular cylinder at Re 200: the Strouhal
// number.
// Specified in THEORY 4.6 (docs/THEORY.md).
//
// Published reference: St = f D / U ~ 0.196 at Re 200 (experiment and
// simulation agree closely). Quasi-2D: cylinder along a thin periodic z,
// D 20, BGK laminar, a small transverse kick (v 0.002) so the symmetric
// wake starts shedding promptly. 20,000 warm-up + 40,000 recorded steps,
// lift sampled every 5 steps; the period from upward zero crossings of the
// demeaned lift. Pass: St within 15 % of 0.196.
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V7_cylinder", [](gate::Gate& g) {
        Context ctx;
        constexpr int NX = 256, NY = 128, NZ = 4;
        constexpr float D = 20.0f, U_IN = 0.05f, RE = 200.0f;
        constexpr double ST_REF = 0.196;
        constexpr int WARMUP = 20'000, RECORD = 40'000, SAMPLE = 5;
        lbm::Config c;
        c.nx = NX;
        c.ny = NY;
        c.nz = NZ;
        c.tau = shapes::tau_for_reynolds(U_IN, D, RE);
        c.u_inlet = U_IN;
        c.smagorinsky_cs = 0.0f;
        c.mode_z = lbm::AxisYZ::Periodic;
        lbm::Solver s(ctx, c);
        std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
        shapes::add_cylinder_z(flags, NX, NY, NZ, NX * 0.25, NY / 2.0, D / 2.0);
        s.set_flags(flags);
        s.init_equilibrium(1.0f, {U_IN, 0.002f, 0.0f});
        g.note("Re %.0f: D %.0f, tau %.4f, %d warm-up + %d recorded steps", RE, D, c.tau, WARMUP,
               RECORD);
        s.step(WARMUP);

        std::vector<double> lift;
        for (int k = 0; k < RECORD / SAMPLE; ++k) {
            s.step(SAMPLE);
            lift.push_back(s.obstacle_force()[1]);
        }
        double mean = 0.0;
        for (const double v : lift)
            mean += v;
        mean /= double(lift.size());
        double amp = 0.0;
        std::vector<std::size_t> up;
        for (std::size_t k = 0; k < lift.size(); ++k) {
            amp = std::max(amp, std::abs(lift[k] - mean));
            // the sign of the demeaned lift rises: an upward crossing between k and k+1
            if (k + 1 < lift.size()) {
                auto sgn = [](double v) { return (v > 0) - (v < 0); };
                if (sgn(lift[k + 1] - mean) - sgn(lift[k] - mean) > 0)
                    up.push_back(k);
            }
        }
        g.note("lift oscillation amplitude %.2e", amp);
        if (!g.check(up.size() >= 3, "periodic shedding detected (%zu upward crossings)",
                     up.size()))
            return;
        const double period = double(up.back() - up.front()) / double(up.size() - 1) * SAMPLE;
        const double st = D / (period * U_IN);
        const double err = std::abs(st - ST_REF) / ST_REF;
        g.check(err < 0.15, "%zu cycles, period %.0f steps -> St %.3f vs %.3f, %.1f %% (< 15 %%)",
                up.size(), period, st, ST_REF, err * 100.0);
    });
}
