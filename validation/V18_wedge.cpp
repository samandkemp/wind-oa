// V18 -- Mach 2 over a 15 deg wedge built from voxels.
// Specified in THEORY 8.6, 8.10 (docs/THEORY.md).
//
// Closed-form reference: the oblique-shock (theta-beta-M) relation -- the
// weak-shock angle (45.34 deg) and the post-shock wall pressure. Tests the
// solid-wall treatment on a staircase (image-point ghost cells with the
// exact signed distance; a distance estimated from the flags alone puts
// the wedge about 13 % off).
// 240 x 160 x 2, slip sides, three flow-throughs. Shock trace: the first
// cell above the wall where p < the mean of the two states, x from 40 to
// 120 cells past the apex, least-squares slope. Pass: angle within 1.5 deg,
// wall p/p_inf within 5 %.
#include "euler_cases.hpp"

using namespace windoa;
using namespace windoa::euler_cases;

int main() {
    return gate::run("V18_wedge", [](gate::Gate& g) {
        Context ctx;
        const double MB = 2.0, TH = 15.0 * kDeg;
        const double beta = theta_beta_m(MB, TH);
        const double p_ratio =
            1.0 + 2 * G / (G + 1) * (MB * MB * std::sin(beta) * std::sin(beta) - 1.0);
        constexpr int NX = 240, NY = 160, NZ = 2;
        const double X0 = 30.0;
        euler::Config c;
        c.nx = NX;
        c.ny = NY;
        c.nz = NZ;
        c.mach = float(MB);
        c.side_bc = euler::SideBC::Slip;
        euler::Solver s(ctx, c);
        std::vector<std::uint8_t> flags(s.cells(), 0);
        std::vector<float> phi(s.cells());
        for (int x = 0; x < NX; ++x)
            for (int y = 0; y < NY; ++y) {
                const double xc = x + 0.5, yc = y + 0.5;
                const bool solid = xc > X0 && yc < (xc - X0) * std::tan(TH);
                // exact distance: to the ramp downstream of the apex, the apex upstream
                const double d = xc > X0 ? (yc - (xc - X0) * std::tan(TH)) * std::cos(TH)
                                         : std::hypot(xc - X0, yc);
                for (int z = 0; z < NZ; ++z) {
                    const std::size_t k = Grid{NX, NY, NZ}.index(x, y, z);
                    flags[k] = solid ? 1 : 0;
                    phi[k] = float(d);
                }
            }
        s.set_flags(flags);
        s.set_distance(phi);
        s.init_freestream();
        while (s.time() < 3.0 * NX / MB) // three flow-throughs
            s.step(200);
        const auto w = s.primitive();
        auto p_at = [&](int x, int y) { return w[5 * ((std::size_t(x) * NY + y) * NZ) + 4] * G; };
        std::vector<double> xs, ys;
        double pw = 0.0;
        int npw = 0;
        for (int x = int(X0) + 40; x <= int(X0) + 120; ++x) {
            const int yw = int((x + 0.5 - X0) * std::tan(TH));
            for (int y = yw + 2; y < NY; ++y)
                if (p_at(x, y) < 0.5 * (1.0 + p_ratio)) {
                    xs.push_back(x + 0.5);
                    ys.push_back(y);
                    break;
                }
            pw += (p_at(x, yw + 1) + p_at(x, yw + 2) + p_at(x, yw + 3)) / 3.0;
            ++npw;
        }
        const double beta_num = std::atan(lsq_slope(xs, ys)) / kDeg;
        pw /= npw;
        g.note("%lld steps", static_cast<long long>(s.steps_taken()));
        g.check(std::abs(beta_num - beta / kDeg) < 1.5, "shock angle %.2f deg (exact %.2f, +-1.5)",
                beta_num, beta / kDeg);
        g.check(std::abs(pw / p_ratio - 1.0) < 0.05,
                "wall p/p_inf %.3f (exact %.3f, %+.1f %%; +-5 %%)", pw, p_ratio,
                (pw / p_ratio - 1) * 100);
    });
}
