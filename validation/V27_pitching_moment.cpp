// V27 -- the transonic regime's lift and pitching moment are consistent:
// a NACA 0012 section at +-alpha.
// Specified in THEORY 8.11 (docs/THEORY.md).
//
// References: (a) a section and its mirror image carry opposite lift and
// moment (an invariant of the sums on a mirrored grid); (b) the moment about
// any point follows from the moment about another plus r x F (statics), so
// moving the reference to the leading edge must agree with the transfer;
// (c) the Kutta-Joukowski theorem, L = rho U Gamma: the surface-pressure
// lift equals the lift implied by the flow's circulation on a contour round
// the section (Anderson, Fundamentals of Aerodynamics, chapter 3).
//
// Case: Mach 0.5 (subcritical), alpha +2 and -2 deg, chord 96 cells, a 2-D
// section (slip z) in a domain 12 x 9 chords, 40,000 steps.
//
// Measured, not gated: the aerodynamic centre. Thin-aerofoil theory, with
// Prandtl-Glauert, puts it at the quarter chord; the voxel wall reads 0.16 -
// 0.23 c depending on the resolution, because the circulation it develops
// falls short of the potential-flow value and the shortfall sits aft (the
// lift deficit of V19; THEORY 8.9).
#include <array>
#include <chrono>

#include "euler_cases.hpp"

using namespace windoa;
using namespace windoa::euler_cases;

namespace {

constexpr double kMach = 0.5, kAlpha = 2.0 * kDeg, kChord = 96.0;
constexpr int kNx = 1152, kNy = 864, kNz = 2, kBatches = 40, kBatchSteps = 1000;

struct Result {
    double cl, cd, cm;   // Cm about the quarter chord, nose up positive
    double swing;        // Cl range over the last 5 batches
    double transfer_err; // (b): |M_le - (M_c4 + r x F)| / |M_le|
    double cl_kj;        // (c): Kutta-Joukowski lift from the circulation
};

Result run(gate::Gate& g, Context& ctx, double alpha) {
    const double px = 0.35 * kNx, py = 0.5 * kNy; // pivot = quarter chord
    const NacaSection sec = naca0012_section(kNx, kNy, kChord, alpha, px, py);
    euler::Config c;
    c.nx = kNx;
    c.ny = kNy;
    c.nz = kNz;
    c.mach = float(kMach);
    c.side_bc = euler::SideBC::Farfield;
    c.z_bc = int(euler::SideBC::Slip);
    euler::Solver s(ctx, c);
    set_section(s, sec, kNz);
    s.init_freestream();
    const std::array<float, 3> c4 = {float(px), float(py), 0.0f};
    const double q = 0.5 * kMach * kMach, span = kNz;
    std::vector<std::array<double, 3>> hist; // Cl, Cd, Cm
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < kBatches; ++k) {
        s.step(kBatchSteps);
        const euler::Solver::Loads l = s.body_loads(c4);
        hist.push_back({l.force[1] / (q * kChord * span), l.force[0] / (q * kChord * span),
                        -l.moment[2] / (q * kChord * kChord * span)});
        if ((k + 1) % 20 == 0)
            g.note("  alpha %+.0f deg, %dk steps: Cl %+.4f Cd %+.4f Cm_c/4 %+.5f (%.0f s)",
                   alpha / kDeg, k + 1, hist.back()[0], hist.back()[1], hist.back()[2],
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    Result r{};
    double lo = 1e30, hi = -1e30;
    for (std::size_t i = hist.size() - 5; i < hist.size(); ++i) {
        r.cl += hist[i][0] / 5;
        r.cd += hist[i][1] / 5;
        r.cm += hist[i][2] / 5;
        lo = std::min(lo, hist[i][0]);
        hi = std::max(hi, hist[i][0]);
    }
    r.swing = hi - lo;

    // (b) the final state about the leading edge, c/4 ahead of the pivot along the chord
    const double lex = px - 0.25 * kChord * std::cos(alpha);
    const double ley = py + 0.25 * kChord * std::sin(alpha);
    const euler::Solver::Loads a = s.body_loads(c4);
    const euler::Solver::Loads b = s.body_loads({float(lex), float(ley), 0.0f});
    const double dx = double(float(px)) - double(float(lex));
    const double dy = double(float(py)) - double(float(ley));
    const double predicted = a.moment[2] + dx * a.force[1] - dy * a.force[0];
    r.transfer_err = std::abs(b.moment[2] - predicted) / std::max(std::abs(b.moment[2]), 1e-12);

    // (c) circulation, counter-clockwise, on a rectangle from 0.5 c ahead of
    // the pivot to 1 c behind it (the trailing edge lies 0.75 c behind) and
    // 0.5 c above and below; lift = -rho U Gamma_ccw, rho = 1
    const std::vector<float> w = s.primitive();
    auto u_at = [&](int x, int y, int k) {
        return double(w[5 * ((std::size_t(x) * kNy + std::size_t(y)) * kNz) + 1 + std::size_t(k)]);
    };
    const int x0 = int(px - 0.5 * kChord), x1 = int(px + 1.0 * kChord);
    const int y0 = int(py - 0.5 * kChord), y1 = kNy - 1 - y0; // mirror rows
    double gamma = 0.0; // trapezoidal on each edge, so a mirrored flow gives -gamma exactly
    for (int x = x0; x <= x1; ++x)
        gamma += (x == x0 || x == x1 ? 0.5 : 1.0) * (u_at(x, y0, 0) - u_at(x, y1, 0));
    for (int y = y0; y <= y1; ++y)
        gamma += (y == y0 || y == y1 ? 0.5 : 1.0) * (u_at(x1, y, 1) - u_at(x0, y, 1));
    r.cl_kj = -2.0 * gamma / (kMach * kChord);
    return r;
}

} // namespace

int main() {
    return gate::run("V27_pitching_moment", [](gate::Gate& g) {
        Context ctx;
        const Result up = run(g, ctx, +kAlpha);
        const Result dn = run(g, ctx, -kAlpha);
        g.note("alpha +2: Cl %+.4f Cd %+.4f Cm_c/4 %+.5f; alpha -2: Cl %+.4f Cd %+.4f Cm_c/4 %+.5f",
               up.cl, up.cd, up.cm, dn.cl, dn.cd, dn.cm);
        const double x_ac = 0.25 - (up.cm - dn.cm) / (up.cl - dn.cl);
        g.note("measured, not gated: aerodynamic centre x/c %.3f (thin aerofoil: 0.25); lift "
               "slope %.2f per rad (thin aerofoil with Prandtl-Glauert: %.2f)",
               x_ac, (up.cl - dn.cl) / (2 * kAlpha), 2 * gate::kPi / std::sqrt(1 - kMach * kMach));
        g.check(std::abs(up.cl + dn.cl) < 1e-3 * std::abs(up.cl) &&
                    std::abs(up.cm + dn.cm) < 1e-3 * std::abs(up.cm),
                "(a) antisymmetric: Cl sum %.1e, Cm sum %.1e (< 1e-3 of each)", up.cl + dn.cl,
                up.cm + dn.cm);
        g.check(up.transfer_err < 1e-5 && dn.transfer_err < 1e-5,
                "(b) moment transfer to the leading edge exact to %.1e / %.1e (< 1e-5)",
                up.transfer_err, dn.transfer_err);
        const double kj_up = std::abs(up.cl_kj / up.cl - 1), kj_dn = std::abs(dn.cl_kj / dn.cl - 1);
        g.check(kj_up < 0.05 && kj_dn < 0.05,
                "(c) Kutta-Joukowski: circulation lift %+.4f / %+.4f vs surface %+.4f / %+.4f "
                "(%.1f / %.1f %%, < 5 %%)",
                up.cl_kj, dn.cl_kj, up.cl, dn.cl, 100 * kj_up, 100 * kj_dn);
        g.check(up.swing < 0.005 && dn.swing < 0.005,
                "converged: last-5k Cl swing %.4f / %.4f (< 0.005)", up.swing, dn.swing);
    });
}
