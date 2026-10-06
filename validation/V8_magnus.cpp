// V8 -- moving boundaries: the spinning-cylinder Magnus effect.
// Specified in THEORY 3.7, 3.10 (docs/THEORY.md).
//
// Published / physical reference: a cylinder spinning in crossflow develops
// a transverse force (Magnus). Tunnel frame, flow +x, omega about +z: the
// -y surface moves with the flow, so the force points -y: Cl_y < 0 for
// omega > 0, growing with the spin ratio alpha = omega R / U. Re 150, D 20,
// regularised + cs 0.1 (tau ~0.52 with a fast wall is below BGK's floor).
// Pass: |Cl(0)| < 0.4; Cl(1), Cl(2) < 0; |Cl(2)| > |Cl(1)| > |Cl(0)|;
// 1.5 < |Cl(2)| < 9.
#include "cases.hpp"

using namespace windoa;

int main() {
    return gate::run("V8_magnus", [](gate::Gate& g) {
        Context ctx;
        constexpr int NX = 220, NY = 140, NZ = 8;
        constexpr float D = 20.0f, R = D / 2.0f, U = 0.05f, RE = 150.0f;
        constexpr float CX = NX * 0.32f, CY = NY * 0.5f;
        constexpr int RAMP = 2000, STEPS = 24'000, N_AVG = 6'000;
        // cylinder mask on cell centres, all z (periodic span)
        std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
        for (int x = 0; x < NX; ++x)
            for (int y = 0; y < NY; ++y) {
                const double dx = x + 0.5 - CX, dy = y + 0.5 - CY;
                if (dx * dx + dy * dy <= double(R) * R)
                    for (int z = 0; z < NZ; ++z)
                        flags[Grid{NX, NY, NZ}.index(x, y, z)] = lbm::OBSTACLE;
            }

        const double alphas[3] = {0.0, 1.0, 2.0};
        double cl[3];
        for (int k = 0; k < 3; ++k) {
            lbm::Config c;
            c.nx = NX;
            c.ny = NY;
            c.nz = NZ;
            c.tau = shapes::tau_for_reynolds(U, D, RE);
            c.u_inlet = 0.0f;
            c.smagorinsky_cs = 0.1f;
            c.regularised = true;
            c.mode_z = lbm::AxisYZ::Periodic;
            c.moving_boundaries = true;
            lbm::Solver s(ctx, c);
            s.set_flags(flags);
            const float omega = float(2.0 * alphas[k] * U / D); // omega R = alpha U
            s.set_rotation({CX, CY, NZ / 2.0f}, {0.0f, 0.0f, omega});
            s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
            for (int i = 0; i < RAMP; ++i) {
                s.set_inlet_velocity(U * float(i + 1) / RAMP);
                s.step(1);
            }
            s.step(STEPS - RAMP - N_AVG);
            s.read_mean_forces();
            s.step(N_AVG);
            const auto f = s.read_mean_forces();
            cl[k] = f.force[1] / (0.5 * U * U * D * NZ);
            g.note("alpha %.1f (omega %.4f): Cl_y %+.3f", alphas[k], omega, cl[k]);
        }
        g.check(std::abs(cl[0]) < 0.4, "no spin: |Cl| %.3f (< 0.4)", std::abs(cl[0]));
        g.check(cl[1] < 0 && cl[2] < 0, "Magnus sign: Cl(1) %+.3f, Cl(2) %+.3f (< 0)", cl[1],
                cl[2]);
        g.check(std::abs(cl[2]) > std::abs(cl[1]) && std::abs(cl[1]) > std::abs(cl[0]),
                "|Cl| grows with the spin ratio");
        g.check(std::abs(cl[2]) > 1.5 && std::abs(cl[2]) < 9.0, "|Cl(2)| %.3f in (1.5, 9)",
                std::abs(cl[2]));
    });
}
