// V26 -- spectra and the Strouhal number (engine spectrum, Solver::macro_at).
// Specified in THEORY 12.3, 12.4, 12.6 (docs/THEORY.md).
//
//   A. Closed form: a tone sampled at uneven intervals (as the tunnel's
//      batches are), in noise, is found at its frequency within 0.5 %; two
//      tones are both found; a flat signal reports no peak.
//   B. Published data: the lift of the cylinder of V7 (Re 200, St 0.196),
//      sampled per uneven batch as the tunnel samples it, gives St within
//      15 % of 0.196 and within 2 % of the zero-crossing estimate V7 uses.
//   C. Published structure of the vortex street: a probe on the wake
//      centreline sees the transverse velocity oscillate at the shedding
//      frequency and the streamwise velocity at twice it (each vortex, of
//      either sign, passes once per half cycle).
//   D. Closed form: the free-slip side faces reflect sound perfectly, so a
//      transverse standing wave between them rings at f = c_s / (2 n_y) --
//      the acoustic modes the tunnel's analysis marks in its spectra
//      (THEORY 12.3). Within 0.5 %; a face that reflected one row inside
//      the domain would ring about 1.5 % sharp at n_y = 64.
#include <cmath>
#include <random>

#include "cases.hpp"
#include "windoa/spectrum.hpp"

using namespace windoa;

namespace {

void part_a(gate::Gate& g) {
    g.section("A: tones at uneven sampling");
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> step(60.0, 140.0);
    std::normal_distribution<double> noise(0.0, 0.3);
    const double f1 = 1.0 / 377.0, f2 = 2.3 / 377.0;
    std::vector<double> t, one, two;
    double tt = 0.0;
    for (int i = 0; i < 400; ++i) {
        t.push_back(tt);
        one.push_back(std::sin(2 * gate::kPi * f1 * tt) + noise(rng));
        two.push_back(std::sin(2 * gate::kPi * f1 * tt) + 0.5 * std::sin(2 * gate::kPi * f2 * tt));
        tt += step(rng) * 0.25; // mean spacing 25 steps: ~15 samples per cycle
    }
    const Spectrum s1 = spectrum(t, one);
    const double e1 = std::abs(s1.peak_freq - f1) / f1;
    g.check(e1 < 0.005, "one tone in noise: f %.6f vs %.6f, %.2f %% (< 0.5 %%), amplitude %.2f (1)",
            s1.peak_freq, f1, e1 * 100.0, s1.peak_amp);
    const auto pk = spectral_peaks(spectrum(t, two), 2);
    const bool both =
        pk.size() == 2 && std::abs(pk[0] - f1) / f1 < 0.005 && std::abs(pk[1] - f2) / f2 < 0.005;
    g.check(both, "two tones: %.6f and %.6f found as %.6f and %.6f", f1, f2,
            pk.size() > 0 ? pk[0] : 0.0, pk.size() > 1 ? pk[1] : 0.0);
    const std::vector<double> flat(t.size(), 0.7);
    const Spectrum s0 = spectrum(t, flat);
    g.check(s0.peak_amp == 0.0, "a flat signal: no peak (amplitude %.1e)", s0.peak_amp);
}

void part_d(Context& ctx, gate::Gate& g) {
    g.section("D: a transverse standing sound wave between the free-slip faces");
    constexpr int NX = 4, NY = 64, NZ = 4;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.55f;
    c.smagorinsky_cs = 0.0f;
    c.mode_x = lbm::AxisX::Periodic;
    c.mode_z = lbm::AxisYZ::Periodic; // y: free-slip faces
    lbm::Solver s(ctx, c);
    s.set_flags(std::vector<std::uint8_t>(s.cells(), lbm::FLUID));
    // rest equilibrium with rho = 1 + 1e-4 cos(pi (y + 1/2) / n_y): the
    // fundamental mode, pressure antinodes on the faces
    constexpr double kW[3] = {1.0 / 3, 1.0 / 18, 1.0 / 36};
    const std::size_t n = s.cells();
    std::vector<float> f(lbm::Q * n);
    for (int i = 0; i < lbm::Q; ++i) {
        const double w = i == 0 ? kW[0] : i <= 6 ? kW[1] : kW[2];
        for (int x = 0; x < NX; ++x)
            for (int y = 0; y < NY; ++y)
                for (int z = 0; z < NZ; ++z) {
                    const double rho = 1.0 + 1e-4 * std::cos(gate::kPi * (y + 0.5) / NY);
                    f[std::size_t(i) * n + Grid{NX, NY, NZ}.index(x, y, z)] = float(w * rho);
                }
    }
    s.set_state(f);
    const std::size_t wall = 0; // cell (0, 0, 0), on a face
    std::vector<double> t, rho;
    for (int k = 1; k <= 4000; ++k) {
        s.step(1);
        t.push_back(double(k));
        rho.push_back(s.macro_at({&wall, 1})[0][3]);
    }
    const double f_exact = std::sqrt(1.0 / 3.0) / (2.0 * NY);
    const double f_num = spectrum(t, rho, 0, 2.0 / t.back()).peak_freq;
    const double e = std::abs(f_num - f_exact) / f_exact;
    g.check(e < 0.005, "f %.4e vs c_s / (2 n_y) %.4e: %.2f %% (< 0.5 %%)", f_num, f_exact,
            e * 100.0);
}

} // namespace

int main() {
    return gate::run("V26_spectra", [](gate::Gate& g) {
        part_a(g);
        {
            Context ctx_d;
            part_d(ctx_d, g);
        }

        g.section("B + C: the shedding cylinder (V7's setup)");
        Context ctx;
        constexpr int NX = 256, NY = 128, NZ = 4;
        constexpr float D = 20.0f, U_IN = 0.05f, RE = 200.0f;
        constexpr double ST_REF = 0.196;
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
        s.step(20'000);

        // the wake centreline three diameters behind the cylinder (index
        // convention: the cylinder is centred on cell 64 in y)
        const int xp = int(NX * 0.25 + 3 * D), yp = NY / 2;
        const std::size_t probe = (std::size_t(xp) * NY + yp) * NZ + NZ / 2;
        std::vector<double> t, cl, pu, pv;
        std::mt19937 rng(5);
        std::uniform_int_distribution<int> len(10, 30); // uneven batches, as the tunnel's
        std::int64_t step = 0;
        s.read_mean_forces();
        while (step < 40'000) {
            const int n = len(rng);
            s.step(n);
            step += n;
            const auto f = s.read_mean_forces();
            const auto m = s.macro_at({&probe, 1});
            t.push_back(double(step));
            cl.push_back(f.force[1] / (0.5 * U_IN * U_IN * D * NZ));
            pu.push_back(m[0][0]);
            pv.push_back(m[0][1]);
        }
        const double fmin = 2.0 / (t.back() - t.front());
        const Spectrum sl = spectrum(t, cl, 0, fmin);
        const double st = sl.peak_freq * D / U_IN;
        // V7's estimate on the same samples: the mean period between upward
        // zero crossings of the demeaned lift
        double mean = 0.0;
        for (const double v : cl)
            mean += v;
        mean /= double(cl.size());
        std::vector<double> up;
        for (std::size_t i = 0; i + 1 < cl.size(); ++i)
            if (cl[i] - mean <= 0.0 && cl[i + 1] - mean > 0.0) // interpolated crossing time
                up.push_back(t[i] + (t[i + 1] - t[i]) * (mean - cl[i]) / (cl[i + 1] - cl[i]));
        const double period = (up.back() - up.front()) / double(up.size() - 1);
        const double st_zc = D / (period * U_IN);
        g.note("%zu samples over %.0f steps; bin width %.2e per step", t.size(),
               t.back() - t.front(), sl.df);
        g.check(std::abs(st - ST_REF) / ST_REF < 0.15 && std::abs(st - st_zc) / st_zc < 0.02,
                "lift spectrum: St %.4f vs 0.196 (%.1f %%, < 15 %%) and the zero-crossing %.4f "
                "(%.2f %%, < 2 %%)",
                st, std::abs(st - ST_REF) / ST_REF * 100.0, st_zc,
                std::abs(st - st_zc) / st_zc * 100.0);
        const double fv = spectrum(t, pv, 0, fmin).peak_freq;
        const double fu = spectrum(t, pu, 0, fmin).peak_freq;
        const double ev = std::abs(fv - sl.peak_freq) / sl.peak_freq;
        const double eu = std::abs(fu - 2.0 * sl.peak_freq) / (2.0 * sl.peak_freq);
        g.check(ev < 0.02, "centreline probe, v: f %.3e vs the lift's %.3e (%.2f %%, < 2 %%)", fv,
                sl.peak_freq, ev * 100.0);
        g.check(eu < 0.03, "centreline probe, u: f %.3e vs twice the lift's %.3e (%.2f %%, < 3 %%)",
                fu, 2.0 * sl.peak_freq, eu * 100.0);
    });
}
