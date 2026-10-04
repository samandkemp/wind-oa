// Euler solver development runs (the physics checks are the gates V17-V19
// under validation/: Sod, the voxel wedge, the NACA 0012). kUsage below is
// the reference for the command line.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/euler.hpp"
#include "windoa/mesh.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

namespace {

int sphere(float mach) {
    Context ctx;
    euler::Config c;
    c.nx = 256;
    c.ny = 96;
    c.nz = 96;
    c.mach = mach;
    euler::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(s.cells(), 0);
    Voxeliser vox(ctx, c.nx, c.ny, c.nz);
    const geometry::Mesh m =
        geometry::fit_to_box(catalogue::make_sphere(), {90.0f, 48.0f, 48.0f}, 24.0f);
    vox.voxelise(m, flags);
    s.set_flags(flags);
    s.set_distance(vox.signed_distance(flags));
    s.init_freestream();
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 10; ++k) {
        s.step(100);
        const auto f = s.body_force();
        const auto h = s.health();
        const double q = 0.5 * mach * mach * 3.14159265 * 12.0 * 12.0;
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf(
            "step %5lld  t %7.1f  Cd %+.4f  Cl %+.4f  peak M %.3f  bad %d  (%.0f Mcell-steps/s)\n",
            static_cast<long long>(s.steps_taken()), s.time(), f[0] / q, f[1] / q, h.max_mach,
            h.bad_cells, s.cells() * double(s.steps_taken()) / sec / 1e6);
        if (h.bad_cells)
            return 1;
    }
    return 0;
}

} // namespace

const char* const kUsage =
    R"(usage: euler_run sphere [MACH]           a sphere at Mach MACH (default 0.8):
                                         forces, peak local Mach, speed
       euler_run profile                 GPU ms per kernel category
       euler_run equiv --save DIR        save three short cases' results
       euler_run equiv --check DIR       compare them bit for bit (the safety
                                         net for performance work)
       euler_run -h | --help             this text
)";

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "-h" || mode == "--help") {
        std::fputs(kUsage, stdout);
        return 0;
    }
    try {
        if (mode == "sphere")
            return sphere(argc > 2 ? float(std::atof(argv[2])) : 0.8f);
        if (mode == "equiv") {
            // P4 safety net for the Euler solver: short runs of three cases
            // (1-D Riemann, a voxel wedge with image walls, a 3-D sphere with
            // curvature-corrected walls), saved or compared bit for bit.
            //   euler_run equiv --save DIR | --check DIR
            const std::string how = argc > 2 ? argv[2] : "", dir = argc > 3 ? argv[3] : "";
            if ((how != "--save" && how != "--check") || dir.empty()) {
                std::fputs(kUsage, stderr);
                return 2;
            }
            Context ctx;
            std::vector<float> blob;
            auto run_case = [&](euler::Config c, const std::vector<std::uint8_t>& flags,
                                const std::vector<float>* phi, const std::vector<float>* w0,
                                int steps) {
                euler::Solver s(ctx, c);
                s.set_flags(flags);
                if (phi)
                    s.set_distance(*phi);
                if (w0)
                    s.set_primitive(*w0);
                else
                    s.init_freestream();
                s.step(steps);
                const auto f = s.body_force();
                std::vector<float> w = s.primitive();
                w.push_back(float(f[0]));
                w.push_back(float(f[1]));
                w.push_back(float(f[2]));
                w.push_back(float(s.time()));
                return w;
            };
            std::vector<std::vector<float>> results;
            { // Sod
                euler::Config c;
                c.nx = 200;
                c.ny = 2;
                c.nz = 2;
                c.x_bc = euler::XBC::Transmissive;
                c.side_bc = euler::SideBC::Slip;
                c.mach = 0.0f;
                std::vector<float> w0(std::size_t(200) * 4 * 5, 0.0f);
                for (int x = 0; x < 200; ++x)
                    for (int k = 0; k < 4; ++k) {
                        float* q = &w0[(std::size_t(x) * 4 + k) * 5];
                        q[0] = x < 100 ? 1.0f : 0.125f;
                        q[4] = x < 100 ? 1.0f : 0.1f;
                    }
                results.push_back(
                    run_case(c, std::vector<std::uint8_t>(800, 0), nullptr, &w0, 150));
            }
            { // Mach 2 voxel wedge, exact distance
                euler::Config c;
                c.nx = 120;
                c.ny = 80;
                c.nz = 2;
                c.mach = 2.0f;
                c.side_bc = euler::SideBC::Slip;
                std::vector<std::uint8_t> fl(std::size_t(120) * 80 * 2, 0);
                std::vector<float> phi(fl.size());
                const double th = 15.0 * 3.14159265358979 / 180.0;
                for (int x = 0; x < 120; ++x)
                    for (int y = 0; y < 80; ++y) {
                        const double xc = x + 0.5, yc = y + 0.5;
                        const bool solid = xc > 15.0 && yc < (xc - 15.0) * std::tan(th);
                        const double d = xc > 15.0
                                             ? (yc - (xc - 15.0) * std::tan(th)) * std::cos(th)
                                             : std::hypot(xc - 15.0, yc);
                        for (int z = 0; z < 2; ++z) {
                            fl[(std::size_t(x) * 80 + y) * 2 + z] = solid ? 1 : 0;
                            phi[(std::size_t(x) * 80 + y) * 2 + z] = float(d);
                        }
                    }
                results.push_back(run_case(c, fl, &phi, nullptr, 300));
            }
            { // 3-D sphere, Mach 0.8, voxeliser distance
                euler::Config c;
                c.nx = 96;
                c.ny = 48;
                c.nz = 48;
                c.mach = 0.8f;
                std::vector<std::uint8_t> fl(std::size_t(96) * 48 * 48, 0);
                Voxeliser vox(ctx, 96, 48, 48);
                vox.voxelise(
                    geometry::fit_to_box(catalogue::make_sphere(), {36.0f, 24.0f, 24.0f}, 16.0f),
                    fl);
                const std::vector<float> phi = vox.signed_distance(fl);
                results.push_back(run_case(c, fl, &phi, nullptr, 200));
            }
            int failed = 0;
            for (std::size_t k = 0; k < results.size(); ++k) {
                const std::string path = dir + "/euler" + std::to_string(k + 1) + ".bin";
                const std::vector<float>& r = results[k];
                if (how == "--save") {
                    std::filesystem::create_directories(dir);
                    std::ofstream(path, std::ios::binary)
                        .write(reinterpret_cast<const char*>(r.data()),
                               std::streamsize(r.size() * 4));
                    continue;
                }
                std::vector<float> base(r.size());
                std::ifstream in(path, std::ios::binary);
                in.read(reinterpret_cast<char*>(base.data()), std::streamsize(base.size() * 4));
                const bool same = in && std::memcmp(base.data(), r.data(), r.size() * 4) == 0;
                double dmax = 0.0, vmax = 0.0;
                std::size_t ndiff = 0;
                for (std::size_t i = 0; i < r.size(); ++i) {
                    dmax = std::max(dmax, double(std::abs(r[i] - base[i])));
                    vmax = std::max(vmax, double(std::abs(base[i])));
                    ndiff += std::memcmp(&r[i], &base[i], 4) != 0;
                }
                std::printf("  euler case %zu: %s", k + 1, same ? "bit-identical" : "DIFFERS");
                if (!same)
                    std::printf(" (%zu of %zu words; max |diff| %.3e, max |value| %.3e; forces "
                                "%+.6e vs %+.6e)",
                                ndiff, r.size(), dmax, vmax, r[r.size() - 4],
                                base[base.size() - 4]);
                std::printf("\n");
                failed += same ? 0 : 1;
            }
            std::printf(how == "--save" ? "saved\n" : (failed ? "FAIL\n" : "PASS\n"));
            return failed ? 1 : 0;
        }
        if (mode == "profile") {
            Context ctx;
            euler::Config c;
            c.nx = 256;
            c.ny = 96;
            c.nz = 96;
            euler::Solver s(ctx, c);
            std::vector<std::uint8_t> flags(s.cells(), 0);
            Voxeliser vox(ctx, c.nx, c.ny, c.nz);
            vox.voxelise(
                geometry::fit_to_box(catalogue::make_sphere(), {90.0f, 48.0f, 48.0f}, 24.0f),
                flags);
            s.set_flags(flags);
            s.set_distance(vox.signed_distance(flags));
            s.init_freestream();
            s.step(20);
            const auto ms = s.profile(50);
            const double total = ms[0] + ms[1] + ms[2] + ms[3];
            std::printf(
                "Euler, 2.36 M cells, per step: dt %.3f  ghost %.3f  update %.3f  (other %.3f)  "
                "= %.3f ms (%.0f Mcell-steps/s)\n",
                ms[0] / 50, ms[1] / 50, ms[2] / 50, ms[3] / 50, total / 50,
                s.cells() * 50.0 / (total * 1e-3) / 1e6);
            return 0;
        }
        std::fputs(kUsage, stderr);
        return 2;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
