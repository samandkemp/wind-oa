// V17 -- Sod shock tube.
// Specified in THEORY 8.10 (docs/THEORY.md).
//
// Closed-form reference: the exact Riemann solution (Toro ch. 4) -- all
// three wave families in one test. 400 cells x 2 x 2 (slip sides,
// transmissive ends), run to t = 0.2 of the unit tube. Pass:
// relative L1 of density < 2 %, the shock within 2 cells of the exact one,
// and transverse symmetry < 1e-6 (a 1-D problem must stay 1-D).
#include "euler_cases.hpp"

using namespace windoa;
using namespace windoa::euler_cases;

int main() {
    return gate::run("V17_sod", [](gate::Gate& g) {
        Context ctx;
        constexpr int NX = 400, NY = 2, NZ = 2;
        const double T_END = 0.2 * NX; // t = 0.2 on the unit domain, in cells
        euler::Config c;
        c.nx = NX;
        c.ny = NY;
        c.nz = NZ;
        c.x_bc = euler::XBC::Transmissive;
        c.side_bc = euler::SideBC::Slip;
        c.mach = 0.0f;
        euler::Solver s(ctx, c);
        const Prim L{1.0, 0.0, 1.0}, R{0.125, 0.0, 0.1};
        std::vector<float> w0(s.cells() * 5, 0.0f);
        for (int x = 0; x < NX; ++x)
            for (int yz = 0; yz < NY * NZ; ++yz) {
                const Prim& q = x < NX / 2 ? L : R;
                float* v = &w0[(std::size_t(x) * NY * NZ + yz) * 5];
                v[0] = float(q.rho);
                v[4] = float(q.p);
            }
        s.set_primitive(w0);
        while (s.time() < T_END - 1e-4) {
            s.set_dt_cap(float(T_END - s.time()));
            s.step(1);
        }
        const auto w = s.primitive();
        auto rho_at = [&](int x, int y, int z) {
            return double(w[5 * (Grid{NX, NY, NZ}.index(x, y, z))]);
        };
        std::vector<double> rho(NX), rho_e(NX);
        double num = 0, den = 0, sym = 0;
        int shock_ex = 0;
        for (int x = 0; x < NX; ++x) {
            rho[x] = rho_at(x, 0, 0);
            rho_e[x] = sample_exact(L, R, (x + 0.5 - NX / 2.0) / T_END).rho;
            num += std::abs(rho[x] - rho_e[x]);
            den += std::abs(rho_e[x]);
            for (int k = 0; k < 5; ++k) // all primitives, (0,0) vs (1,1)
                sym = std::max(sym, double(std::abs(w[5 * (Grid{NX, NY, NZ}.index(x, 0, 0)) + k] -
                                                    w[5 * (Grid{NX, NY, NZ}.index(x, 1, 1)) + k])));
            if (rho_e[x] > 0.125 + 1e-6)
                shock_ex = x;
        }
        const double l1 = num / den;
        // shock = the last cell above the midpoint between the undisturbed right
        // density and the exact post-shock density
        const double mid = 0.5 * (0.125 + rho_e[shock_ex]);
        int shock_num = 0;
        for (int x = 0; x < NX; ++x)
            if (rho[x] > mid)
                shock_num = x;
        const int xc = int(0.62 * NX);
        g.note("%d cells, %lld steps; rho* right of the contact %.4f (exact %.4f)", NX,
               static_cast<long long>(s.steps_taken()), rho[xc], rho_e[xc]);
        g.check(l1 < 0.02, "exact Riemann: relative L1(rho) %.2f %% (< 2 %%)", l1 * 100.0);
        g.check(std::abs(shock_num - shock_ex) <= 2, "shock at cell %d (exact %d, +-2)", shock_num,
                shock_ex);
        g.check(sym < 1e-6, "transverse symmetry %.1e (< 1e-6)", sym);
    });
}
