// P4 safety net: every LBM performance switch against the reference path, in
// the same binary. The switches that claim to be exact identities must agree
// to the last bit (distributions, rho / u, the force window); f16 storage is
// an approximation, so it reports its deviation instead (and is judged by
// the physics gates). Registered with CTest as P4_equiv. THEORY 11.
//
//   lbm_equiv            the identity checks (exit 1 on any bit difference)
//   lbm_equiv --f16      also report the f16-storage deviations
//   lbm_equiv --save DIR / --check DIR
//                        save the reference path's results / compare them bit
//                        for bit against a saved baseline (across builds: a
//                        refactor of the shared code must not change a bit)
//
// Configurations cover every kernel path: inlet / outlet / sponge, spinning
// walls, BGK + periodic + body force, RR + interpolated bounce-back, a lid-
// driven cavity, inlet turbulence, and the per-step hook (dye).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/dye.hpp"
#include "windoa/lbm.hpp"
#include "windoa/shapes.hpp"

using namespace windoa;

namespace {

constexpr int NX = 64, NY = 32, NZ = 32;

struct Case {
    std::string name;
    std::function<void(lbm::Config&)> config;
    std::function<void(lbm::Solver&)> setup;
    bool dye = false;
};

struct Result {
    std::vector<float> f, u, rho;
    lbm::MeanForces forces;
    std::vector<float> dye;
};

Result run(Context& ctx, const Case& k, bool opt, bool f16) {
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.56f;
    c.u_inlet = 0.05f;
    k.config(c);
    c.opt_lazy_macro = opt;
    c.opt_sparse_forces = opt;
    c.storage_f16 = f16;
    lbm::Solver s(ctx, c);
    k.setup(s);
    Result r;
    if (k.dye) {
        Dye d(ctx, s);
        std::vector<float> src(s.cells(), 0.0f);
        for (int y = 12; y < 20; ++y)
            for (int z = 12; z < 20; ++z)
                src[Grid{NX, NY, NZ}.index(4, y, z)] = 0.5f;
        d.set_sources(src);
        // reference: the separate dye pass; optimised: fused into the step
        auto steps = [&](int n) { opt ? d.step_with(s, n) : d.step_with_reference(s, n); };
        steps(150);
        s.read_mean_forces();
        steps(151);
        r.dye = d.concentration();
    } else {
        // odd batch sizes: parity changes and submit boundaries mid-run
        s.step(37);
        s.read_mean_forces();
        s.step(129);
        s.step(135);
    }
    r.forces = s.read_mean_forces();
    r.f = s.get_state();
    r.u = s.velocity();
    r.rho = s.density();
    return r;
}

std::size_t bit_diffs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size())
        return std::size_t(-1);
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0)
            ++n;
    return n;
}

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, double(std::abs(a[i] - b[i])));
    return m;
}

std::vector<std::uint8_t> sphere(float cx, float r) {
    std::vector<std::uint8_t> f(std::size_t(NX) * NY * NZ, lbm::FLUID);
    shapes::add_sphere(f, NX, NY, NZ, cx, NY / 2.0f, NZ / 2.0f, r);
    return f;
}

} // namespace

const char* const kUsage =
    R"(usage: lbm_equiv               the identity checks (exit 1 on any bit difference)
       lbm_equiv --f16         also report the f16-storage deviations
       lbm_equiv --save DIR    save the reference path's results
       lbm_equiv --check DIR   compare them bit for bit against a saved baseline
       lbm_equiv -h | --help   this text
)";

int main(int argc, char** argv) {
    bool f16 = false;
    std::string save_dir, check_dir;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--f16")
            f16 = true;
        else if (s == "--save" && a + 1 < argc)
            save_dir = argv[++a];
        else if (s == "--check" && a + 1 < argc)
            check_dir = argv[++a];
        else if (s == "-h" || s == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n\n%s", s.c_str(), kUsage);
            return 2;
        }
    }
    int case_no = 0;
    if (!save_dir.empty())
        std::filesystem::create_directories(save_dir);
    const std::vector<Case> cases = {
        {"tunnel: regularised + LES + sponge + spinning sphere",
         [](lbm::Config& c) {
             c.regularised = true;
             c.smagorinsky_cs = 0.1f;
             c.outlet_sponge = 12;
             c.sponge_target = 1;
             c.moving_boundaries = true;
             c.tau = 0.51f;
         },
         [](lbm::Solver& s) {
             s.set_flags(sphere(20.0f, 6.0f));
             s.set_torque_ref({20.0f, 16.0f, 16.0f});
             s.set_rotation({20.0f, 16.0f, 16.0f}, {0.0f, 0.0f, -0.008f});
             s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
         }},
        {"channel: BGK, periodic x / z, body force, walls",
         [](lbm::Config& c) {
             c.mode_x = lbm::AxisX::Periodic;
             c.mode_z = lbm::AxisYZ::Periodic;
             c.smagorinsky_cs = 0.0f;
             c.u_inlet = 0.0f;
         },
         [](lbm::Solver& s) {
             std::vector<std::uint8_t> f(s.cells(), lbm::FLUID);
             for (int x = 0; x < NX; ++x)
                 for (int z = 0; z < NZ; ++z) {
                     f[(std::size_t(x) * NY + 0) * NZ + z] = lbm::WALL;
                     f[(std::size_t(x) * NY + NY - 1) * NZ + z] = lbm::WALL;
                 }
             s.set_flags(f);
             s.set_body_force({1e-5f, 0.0f, 0.0f});
             s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
         }},
        {"sphere: recursive-regularised + interpolated bounce-back",
         [](lbm::Config& c) {
             c.recursive = true;
             c.use_ibb = true;
         },
         [](lbm::Solver& s) {
             const std::vector<std::uint8_t> f = sphere(22.0f, 7.3f);
             s.set_flags(f);
             s.set_link_q(shapes::sphere_link_fractions(f, NX, NY, NZ, 22.0f, 16.0f, 16.0f, 7.3f));
             s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
         }},
        {"cavity: moving lid + walls (closed)",
         [](lbm::Config& c) {
             c.mode_x = lbm::AxisX::Periodic;
             c.u_inlet = 0.0f;
         },
         [](lbm::Solver& s) {
             std::vector<std::uint8_t> f(s.cells(), lbm::FLUID);
             for (int x = 0; x < NX; ++x)
                 for (int y = 0; y < NY; ++y)
                     for (int z = 0; z < NZ; ++z)
                         if (x == 0 || x == NX - 1 || y == 0)
                             f[Grid{NX, NY, NZ}.index(x, y, z)] = lbm::WALL;
                         else if (y == NY - 1)
                             f[Grid{NX, NY, NZ}.index(x, y, z)] = lbm::LID;
             s.set_flags(f);
             s.set_lid_velocity({0.08f, 0.0f, 0.0f});
             s.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
         }},
        {"inlet turbulence (post-step inlet overwrite)",
         [](lbm::Config& c) { c.regularised = true; },
         [](lbm::Solver& s) {
             s.set_flags(sphere(20.0f, 5.0f));
             s.set_inlet_turbulence(0.02f, 4.0f, 64, 0.05f);
             s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
         }},
        {"dye: fused vs the separate pass", [](lbm::Config& c) { c.regularised = true; },
         [](lbm::Solver& s) {
             s.set_flags(sphere(24.0f, 5.0f));
             s.init_equilibrium(1.0f, {0.05f, 0.0f, 0.0f});
         },
         true},
    };

    try {
        Context ctx;
        int failed = 0;
        std::printf("P4 equivalence (%d x %d x %d, 301 steps)\n", NX, NY, NZ);
        for (const Case& k : cases) {
            const Result ref = run(ctx, k, false, false);
            const Result opt = run(ctx, k, true, false);
            const std::size_t df = bit_diffs(ref.f, opt.f), du = bit_diffs(ref.u, opt.u),
                              dr = bit_diffs(ref.rho, opt.rho), dd = bit_diffs(ref.dye, opt.dye);
            const bool forces_same =
                std::memcmp(&ref.forces, &opt.forces, sizeof(lbm::MeanForces)) == 0;
            const bool ok = df == 0 && du == 0 && dr == 0 && dd == 0 && forces_same;
            failed += ok ? 0 : 1;
            std::printf("  %-56s %s", k.name.c_str(), ok ? "bit-identical" : "DIFFERS");
            if (!ok)
                std::printf(" (f %zu, u %zu, rho %zu, dye %zu words; forces %s)", df, du, dr, dd,
                            forces_same ? "same" : "differ");
            std::printf("\n");
            // Cross-build check: the reference path against a saved baseline,
            // bit for bit. Catches a refactor that changes both paths alike
            // (the in-binary comparison above cannot).
            if (!save_dir.empty() || !check_dir.empty()) {
                ++case_no;
                std::vector<float> blob = ref.f;
                blob.insert(blob.end(), ref.u.begin(), ref.u.end());
                blob.insert(blob.end(), ref.rho.begin(), ref.rho.end());
                blob.insert(blob.end(), ref.dye.begin(), ref.dye.end());
                const float* fw = reinterpret_cast<const float*>(&ref.forces);
                blob.insert(blob.end(), fw, fw + sizeof(lbm::MeanForces) / sizeof(float));
                const std::string path = (save_dir.empty() ? check_dir : save_dir) + "/case" +
                                         std::to_string(case_no) + ".bin";
                if (!save_dir.empty()) {
                    std::ofstream(path, std::ios::binary)
                        .write(reinterpret_cast<const char*>(blob.data()),
                               std::streamsize(blob.size() * sizeof(float)));
                } else {
                    std::vector<float> base(blob.size());
                    std::ifstream in(path, std::ios::binary);
                    in.read(reinterpret_cast<char*>(base.data()),
                            std::streamsize(base.size() * sizeof(float)));
                    const std::size_t d = in ? bit_diffs(base, blob) : std::size_t(-1);
                    std::printf("      vs baseline: %s\n", d == 0 ? "bit-identical" : "DIFFERS");
                    failed += d == 0 ? 0 : 1;
                }
            }
            if (f16) {
                const Result h = run(ctx, k, true, true);
                double umax = 0.0;
                for (float v : ref.u)
                    umax = std::max(umax, double(std::abs(v)));
                std::printf(
                    "      f16: max|du| %.2e (%.3f %% of max|u|), max|drho| %.2e, force x %+.5e "
                    "vs %+.5e (%.3f %%)\n",
                    max_abs(ref.u, h.u), 100.0 * max_abs(ref.u, h.u) / std::max(umax, 1e-12),
                    max_abs(ref.rho, h.rho), h.forces.force[0], ref.forces.force[0],
                    100.0 * std::abs(h.forces.force[0] - ref.forces.force[0]) /
                        std::max(1e-12, double(std::abs(ref.forces.force[0]))));
            }
        }
        std::printf(failed ? "FAIL: %d case(s) differ\n" : "PASS\n", failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
