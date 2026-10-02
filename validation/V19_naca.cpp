// V19 -- the transonic gate: NACA 0012 at Mach 0.8, alpha 1.25 deg (the
// AGARD Euler case).
// Specified in THEORY 8.7, 8.10 (docs/THEORY.md).
//
// Published reference (AGARD-AR-211, Jameson and others, inviscid): Cl
// ~0.35, wave drag Cd ~0.022, a strong upper-surface shock near x/c 0.62
// and a weak lower one near 0.35. Here on a voxel staircase, chord 128
// cells, a 2-D section (slip z), domain 12 x 9 chords (a closer far field
// inflates lift), 60,000 steps. The bands allow for that: Cl 0.28 - 0.42
// (the voxel wall's documented lift deficit), Cd 0.012 - 0.035, upper
// shock x/c 0.50 - 0.72, and a converged lift (last-5k Cl swing < 0.01).
#include <chrono>

#include "euler_cases.hpp"

using namespace windoa;
using namespace windoa::euler_cases;

int main() {
    return gate::run("V19_naca", [](gate::Gate& g) {
        const double MC = 0.8, AL = 1.25 * kDeg, CH = 128.0;
        constexpr int NX = 1536, NY = 1152, NZ = 2;
        const double PX = 0.35 * NX, PY = 0.5 * NY; // pivot = quarter chord
        auto half = [](double xc) {
            xc = std::max(xc, 0.0);
            return 0.6 * (0.2969 * std::sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc +
                          0.2843 * xc * xc * xc - 0.1015 * xc * xc * xc * xc);
        };
        std::vector<std::uint8_t> foil(std::size_t(NX) * NY, 0);
        std::vector<double> xa_of(foil.size());
        for (int x = 0; x < NX; ++x)
            for (int y = 0; y < NY; ++y) {
                const double xg = x + 0.5 - PX, yg = y + 0.5 - PY;
                const double xa = (xg * std::cos(AL) - yg * std::sin(AL)) / CH + 0.25;
                const double ya = (xg * std::sin(AL) + yg * std::cos(AL)) / CH;
                xa_of[std::size_t(x) * NY + y] = xa;
                foil[std::size_t(x) * NY + y] =
                    (xa >= 0.0 && xa <= 1.0 && std::abs(ya) < half(xa)) ? 1 : 0;
            }
        // exact signed distance to the contour within 6 cells of the foil, +-4 beyond
        std::vector<double> cx, cy;
        for (int i = 0; i < 2000; ++i) {
            const double bt = gate::kPi * i / 1999.0, xs = 0.5 * (1 - std::cos(bt)), yt = half(xs);
            for (int sg : {1, -1}) {
                const double px = (xs - 0.25) * CH, py = sg * yt * CH;
                cx.push_back(px * std::cos(AL) + py * std::sin(AL) + PX);
                cy.push_back(-px * std::sin(AL) + py * std::cos(AL) + PY);
            }
        }
        std::vector<float> phi2(foil.size());
        for (int x = 0; x < NX; ++x)
            for (int y = 0; y < NY; ++y) {
                const std::size_t k = std::size_t(x) * NY + y;
                bool near = false;
                for (int dx = -6; dx <= 6 && !near; ++dx)
                    for (int dy = -6; dy <= 6; ++dy) {
                        const int xx = x + dx, yy = y + dy;
                        if (xx >= 0 && yy >= 0 && xx < NX && yy < NY &&
                            foil[std::size_t(xx) * NY + yy]) {
                            near = true;
                            break;
                        }
                    }
                double d = 4.0;
                if (near) {
                    d = 1e30;
                    for (std::size_t i = 0; i < cx.size(); ++i)
                        d = std::min(d, std::hypot(x + 0.5 - cx[i], y + 0.5 - cy[i]));
                }
                phi2[k] = float(foil[k] ? -d : d);
            }

        Context ctx;
        euler::Config c;
        c.nx = NX;
        c.ny = NY;
        c.nz = NZ;
        c.mach = float(MC);
        c.side_bc = euler::SideBC::Farfield;
        c.z_bc = int(euler::SideBC::Slip);
        euler::Solver s(ctx, c);
        std::vector<std::uint8_t> flags(s.cells());
        std::vector<float> phi(s.cells());
        for (std::size_t k = 0; k < foil.size(); ++k)
            for (int z = 0; z < NZ; ++z) {
                flags[k * NZ + z] = foil[k];
                phi[k * NZ + z] = phi2[k];
            }
        s.set_flags(flags);
        s.set_distance(phi);
        s.init_freestream();
        const double qc = 0.5 * MC * MC * CH * 2.0; // q * chord * span
        std::vector<std::pair<double, double>> hist;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < 60; ++k) {
            s.step(1000);
            const auto f = s.body_force();
            hist.emplace_back(f[1] / qc, f[0] / qc);
            if ((k + 1) % 20 == 0)
                g.note(
                    "%dk steps: Cl %.4f Cd %.4f (%.0f s)", k + 1, hist.back().first,
                    hist.back().second,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        double cl = 0, cd = 0, lo = 1e30, hi = -1e30;
        for (std::size_t i = hist.size() - 5; i < hist.size(); ++i) {
            cl += hist[i].first / 5;
            cd += hist[i].second / 5;
            lo = std::min(lo, hist[i].first);
            hi = std::max(hi, hist[i].first);
        }
        // surface Cp: first fluid cell above / below the foil in each column
        const auto w = s.primitive();
        std::vector<std::array<double, 3>> cols; // x/c, cp_up, cp_lo
        for (int x = 0; x < NX; ++x) {
            int ymin = NY, ymax = -1, cnt = 0;
            double xm = 0;
            for (int y = 0; y < NY; ++y)
                if (foil[std::size_t(x) * NY + y]) {
                    ymin = std::min(ymin, y);
                    ymax = std::max(ymax, y);
                    xm += xa_of[std::size_t(x) * NY + y];
                    ++cnt;
                }
            if (!cnt)
                continue;
            auto cp = [&](int y) {
                return (w[5 * ((std::size_t(x) * NY + y) * NZ) + 4] - 1.0 / G) / (0.5 * MC * MC);
            };
            cols.push_back({xm / cnt, cp(ymax + 1), cp(ymin - 1)});
        }
        std::sort(cols.begin(), cols.end());
        // x/c of the steepest compression aft of 0.2c on a 3-column smoothed
        // distribution (zero-padded ends)
        auto shock_at = [&](int which) {
            std::vector<double> sm(cols.size());
            for (std::size_t i = 0; i < cols.size(); ++i) {
                double a = 0;
                for (int d = -1; d <= 1; ++d) {
                    const long j = long(i) + d;
                    if (j >= 0 && j < long(cols.size()))
                        a += cols[std::size_t(j)][std::size_t(which)];
                }
                sm[i] = a / 3.0;
            }
            double best = -1e30, xbest = 0;
            for (std::size_t i = 1; i < cols.size(); ++i) {
                if (cols[i][0] <= 0.2 || cols[i][0] >= 0.95)
                    continue;
                if (sm[i] - sm[i - 1] > best) {
                    best = sm[i] - sm[i - 1];
                    xbest = cols[i][0];
                }
            }
            return xbest;
        };
        const double x_up = shock_at(1), x_lo = shock_at(2);
        double cp_min = 1e30;
        for (const auto& col : cols)
            cp_min = std::min(cp_min, col[1]);
        const double cp_star =
            (std::pow((2 + (G - 1) * MC * MC) / (G + 1), G / (G - 1)) - 1) / (0.7 * MC * MC);
        g.note(
            "%lld steps; min Cp upper %.2f (supersonic pocket: Cp* %.2f); lower shock %.2f (~0.35)",
            static_cast<long long>(s.steps_taken()), cp_min, cp_star, x_lo);
        g.check(cl >= 0.28 && cl <= 0.42, "Cl %.3f in [0.28, 0.42] (published Euler ~0.35)", cl);
        g.check(cd >= 0.012 && cd <= 0.035, "Cd %.4f in [0.012, 0.035] (wave drag ~0.022)", cd);
        g.check(x_up >= 0.50 && x_up <= 0.72, "upper shock x/c %.2f in [0.50, 0.72] (~0.62)", x_up);
        g.check(hi - lo < 0.01, "converged: last-5k Cl swing %.4f (< 0.01)", hi - lo);
    });
}
