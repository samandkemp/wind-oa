// V24 -- time-averaged flow statistics (engine FlowStats).
// Specified in THEORY 12.1, 12.6 (docs/THEORY.md).
//
//   A. Invariant (exactness): the GPU accumulator equals a double-precision
//      host accumulation of the same samples -- the step-weighted mean and
//      variance of u and rho, with uneven batch lengths -- on an unsteady
//      flow (the shedding cylinder of part C).
//   B. Identity: on a steady flow (the Poiseuille channel) the mean equals
//      the instantaneous field and the variance vanishes -- to the floor of
//      the steady state's own f32 jitter, measured at an rms of ~5e-6 u_max
//      (its per-step Guo increment is only ~30 ulps of f): the bands
//      are 1e-5 u_max and (1e-5 u_max)^2.
//   C. Published structure of a Karman vortex street (cylinder, Re 200,
//      quasi-2-D, symmetric about the channel centre): averaged over ~30
//      shedding cycles, the mean wake is symmetric (<u> even, <v> odd about
//      the centreline) although every instantaneous field is not; the
//      transverse fluctuation v_rms peaks on the centreline, and the
//      streamwise fluctuation u_rms has twin peaks off it with a local
//      minimum on it (the alternate vortices pass either side).
#include <memory>
#include <random>

#include "cases.hpp"
#include "windoa/flow_stats.hpp"

using namespace windoa;

namespace {

constexpr int NX = 256, NY = 128, NZ = 4;
constexpr float D = 20.0f, U = 0.05f, RE = 200.0f;
constexpr double CX = 64.0, CY = 64.0; // centre on the cell boundary: symmetric about the walls

std::size_t idx(int x, int y, int z) {
    return (std::size_t(x) * NY + y) * NZ + z;
}

// Host reference: weighted Welford in double.
struct HostStats {
    std::vector<double> mean, m2;
    double w = 0.0;
    void add(const std::vector<float>& vel, const std::vector<float>& rho, double weight) {
        if (mean.empty()) {
            mean.assign(rho.size() * 4, 0.0);
            m2.assign(rho.size() * 4, 0.0);
        }
        w += weight;
        const double a = weight / w;
        for (std::size_t c = 0; c < rho.size(); ++c)
            for (int k = 0; k < 4; ++k) {
                const double x = k < 3 ? vel[c * 3 + k] : rho[c] - 1.0;
                const double d = x - mean[c * 4 + k];
                mean[c * 4 + k] += a * d;
                m2[c * 4 + k] += weight * d * (x - mean[c * 4 + k]);
            }
    }
};

std::unique_ptr<lbm::Solver> make_cylinder(Context& ctx, std::vector<std::uint8_t>& flags) {
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = shapes::tau_for_reynolds(U, D, RE);
    c.u_inlet = U;
    c.smagorinsky_cs = 0.0f;
    c.mode_z = lbm::AxisYZ::Periodic;
    auto s = std::make_unique<lbm::Solver>(ctx, c);
    flags.assign(std::size_t(NX) * NY * NZ, lbm::FLUID);
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y) {
            const double dx = x + 0.5 - CX, dy = y + 0.5 - CY;
            if (dx * dx + dy * dy <= double(D) * D / 4.0)
                for (int z = 0; z < NZ; ++z)
                    flags[idx(x, y, z)] = lbm::OBSTACLE;
        }
    s->set_flags(flags);
    s->init_equilibrium(1.0f, {U, 0.002f, 0.0f}); // a transverse kick starts the shedding
    return s;
}

} // namespace

int main() {
    return gate::run("V24_flow_stats", [](gate::Gate& g) {
        Context ctx;

        // -- B: steady channel ------------------------------------------------
        g.section("B: steady Poiseuille channel -- mean = instantaneous, variance 0");
        {
            namespace P = cases::poiseuille;
            lbm::Config c;
            c.nx = P::NX;
            c.ny = P::NY;
            c.nz = P::NZ;
            c.tau = P::TAU;
            c.smagorinsky_cs = 0.0f;
            c.mode_x = lbm::AxisX::Periodic;
            c.mode_z = lbm::AxisYZ::Periodic;
            lbm::Solver s(ctx, c);
            std::vector<std::uint8_t> flags(s.cells(), lbm::FLUID);
            for (int x = 0; x < P::NX; ++x)
                for (int z = 0; z < P::NZ; ++z) {
                    flags[(std::size_t(x) * P::NY + 0) * P::NZ + z] = lbm::WALL;
                    flags[(std::size_t(x) * P::NY + P::NY - 1) * P::NZ + z] = lbm::WALL;
                }
            s.set_flags(flags);
            s.set_body_force({P::G_X, 0.0f, 0.0f});
            s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
            s.step(P::STEPS);
            FlowStats st(ctx, s.cells());
            for (int k = 0; k < 10; ++k) {
                s.step(100);
                st.add(s.macro_buffer(s.live_index()), 100.0);
            }
            const auto vel = s.velocity();
            std::vector<float> mean, var;
            st.read(0, s.cells(), mean, var);
            double dmax = 0.0, vmax = 0.0, umax = 0.0;
            for (std::size_t c2 = 0; c2 < s.cells(); ++c2) {
                if (flags[c2] != lbm::FLUID)
                    continue;
                umax = std::max(umax, double(std::abs(vel[c2 * 3])));
                dmax = std::max(dmax, double(std::abs(mean[c2 * 4] - vel[c2 * 3])));
                vmax = std::max(vmax, double(var[c2 * 4]));
            }
            g.check(dmax < 1e-5 * umax && vmax < 1e-10 * umax * umax,
                    "max |<u_x> - u_x| %.2e (< 1e-5 u_max), max var u_x %.2e (< 1e-10 u_max^2)",
                    dmax, vmax);
        }

        // -- A + C: the shedding cylinder ---------------------------------------
        std::vector<std::uint8_t> flags;
        auto sp = make_cylinder(ctx, flags);
        lbm::Solver& s = *sp;
        g.note("cylinder Re %.0f: D %.0f, tau %.4f, %d x %d x %d", RE, D,
               shapes::tau_for_reynolds(U, D, RE), NX, NY, NZ);
        s.step(20'000); // developed shedding

        g.section("A: GPU accumulator vs a double-precision host accumulation");
        FlowStats st(ctx, s.cells());
        HostStats host;
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> len(37, 63); // uneven batches
        for (int k = 0; k < 40; ++k) {
            const int n = len(rng);
            s.step(n);
            st.add(s.macro_buffer(s.live_index()), double(n));
            host.add(s.velocity(), s.density(), double(n));
        }
        {
            std::vector<float> mean, var;
            st.read(0, s.cells(), mean, var);
            double du = 0.0, dr = 0.0, dv = 0.0, vref = 0.0;
            for (std::size_t c = 0; c < s.cells(); ++c) {
                if (flags[c] != lbm::FLUID)
                    continue;
                for (int k = 0; k < 3; ++k) {
                    du = std::max(du, std::abs(mean[c * 4 + k] - host.mean[c * 4 + k]));
                    vref = std::max(vref, host.m2[c * 4 + k] / host.w);
                    dv = std::max(dv, std::abs(var[c * 4 + k] - host.m2[c * 4 + k] / host.w));
                }
                dr = std::max(dr, std::abs(mean[c * 4 + 3] - host.mean[c * 4 + 3]));
            }
            g.note("%d samples, %.0f steps", st.samples(), st.weight());
            g.check(du < 1e-6 * U && dr < 1e-8 && dv < 1e-4 * vref,
                    "max |mean u| error %.2e (< 1e-6 U), mean rho %.2e (< 1e-8), variance %.2e "
                    "(< 1e-4 of the largest, %.2e)",
                    du, dr, dv, vref);
        }

        g.section("C: the mean wake of the vortex street");
        st.reset();
        for (int k = 0; k < 600; ++k) { // 60,000 steps, ~30 shedding cycles
            s.step(100);
            st.add(s.macro_buffer(s.live_index()), 100.0);
        }
        const int xs = int(CX + 3.0 * D); // three diameters behind the centre
        const std::size_t plane = std::size_t(NY) * NZ;
        std::vector<float> mean, var;
        st.read(std::size_t(xs) * plane, plane, mean, var);
        const auto vel = s.velocity();
        double asym_mean = 0.0, asym_inst = 0.0;
        for (int k = 0; k < NY / 2; ++k) {
            const int ya = NY / 2 + k, yb = NY / 2 - 1 - k; // mirror pairs about y = 64
            const std::size_t a = std::size_t(ya) * NZ, b = std::size_t(yb) * NZ;
            asym_mean = std::max(asym_mean, double(std::abs(mean[a * 4] - mean[b * 4]) +
                                                   std::abs(mean[a * 4 + 1] + mean[b * 4 + 1])));
            const std::size_t ia = idx(xs, ya, 0), ib = idx(xs, yb, 0);
            asym_inst = std::max(asym_inst, double(std::abs(vel[ia * 3] - vel[ib * 3]) +
                                                   std::abs(vel[ia * 3 + 1] + vel[ib * 3 + 1])));
        }
        g.check(asym_mean < 0.02 * U && asym_mean < 0.1 * asym_inst,
                "mean-wake asymmetry %.4f U (< 0.02 U), instantaneous %.3f U (mean < 10 %%)",
                asym_mean / U, asym_inst / U);
        int y_vmax = 0, y_umax = 0;
        double vmax = -1.0, umax = -1.0;
        for (int y = 0; y < NY; ++y) {
            const double vr = var[std::size_t(y) * NZ * 4 + 1], ur = var[std::size_t(y) * NZ * 4];
            if (vr > vmax) {
                vmax = vr;
                y_vmax = y;
            }
            if (ur > umax) {
                umax = ur;
                y_umax = y;
            }
        }
        const double u_centre =
            0.5 * (var[std::size_t(NY / 2) * NZ * 4] + var[std::size_t(NY / 2 - 1) * NZ * 4]);
        const double off = std::abs(y_umax + 0.5 - CY) / D;
        g.check(std::abs(y_vmax + 0.5 - CY) <= 1.0,
                "v_rms peaks at y = %.1f (centreline %.1f, +-1 cell): %.3f U", y_vmax + 0.5, CY,
                std::sqrt(vmax) / U);
        g.check(
            off > 0.2 && u_centre < 0.8 * umax,
            "u_rms peaks off the centreline (%.2f D away), centre %.3f U < 80 %% of peak %.3f U",
            off, std::sqrt(u_centre) / U, std::sqrt(umax) / U);
        // informational: the mean recirculation length behind the cylinder
        std::vector<float> line_m, line_v;
        int x_end = -1;
        for (int x = int(CX + D / 2) + 1; x < NX - 1; ++x) {
            st.read(idx(x, NY / 2, 0), 1, line_m, line_v);
            if (line_m[0] >= 0.0f) {
                x_end = x;
                break;
            }
        }
        if (x_end > 0)
            g.note("mean recirculation length behind the cylinder: %.2f D",
                   (x_end + 0.5 - (CX + D / 2)) / D);
    });
}
