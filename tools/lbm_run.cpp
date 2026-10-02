// Headless LBM run: a sphere in the tunnel. Prints throughput (MLUPS),
// health, the drag coefficient and a mass check as it goes. A development
// tool, not a validation gate; the gates are under validation/
// (docs/VALIDATION.md).
//
//   lbm_run [nx ny nz] [--steps N] [--bgk] [--rr] [--ibb] [--spin S]
//           [--u U] [--tau T] [--no-sponge] [--stl PATH [--size CELLS]]
//
// Defaults follow the app's solver: regularised collision,
// tau 0.504, Smagorinsky 0.1, u 0.05, outlet sponge 24 (full freestream).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/lbm.hpp"
#include "windoa/mesh.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

int main(int argc, char** argv) {
    lbm::Config cfg;
    cfg.nx = 256;
    cfg.ny = 96;
    cfg.nz = 96;
    cfg.tau = 0.504f;
    cfg.u_inlet = 0.05f;
    cfg.smagorinsky_cs = 0.1f;
    cfg.regularised = true;
    cfg.outlet_sponge = 24;
    cfg.sponge_target = 1;
    int steps = 4000;
    float spin = 0.0f;
    std::string stl_path;
    float stl_size = 0.0f;

    int positional = 0;
    int* const dims[3] = {&cfg.nx, &cfg.ny, &cfg.nz};
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--steps" && a + 1 < argc) {
            steps = std::atoi(argv[++a]);
        } else if (s == "--u" && a + 1 < argc) {
            cfg.u_inlet = static_cast<float>(std::atof(argv[++a]));
        } else if (s == "--tau" && a + 1 < argc) {
            cfg.tau = static_cast<float>(std::atof(argv[++a]));
        } else if (s == "--bgk") {
            cfg.regularised = false;
        } else if (s == "--no-sponge") {
            cfg.outlet_sponge = 0;
        } else if (s == "--rr") {
            cfg.recursive = true;
        } else if (s == "--ibb") {
            cfg.use_ibb = true; // link fractions left at 128 = half-way
        } else if (s == "--spin" && a + 1 < argc) {
            cfg.moving_boundaries = true;
            spin = static_cast<float>(std::atof(argv[++a])); // surface speed / u_inlet
        } else if (s == "--stl" && a + 1 < argc) {
            stl_path = argv[++a];
        } else if (s == "--size" && a + 1 < argc) {
            stl_size = static_cast<float>(std::atof(argv[++a])); // longest axis, cells
        } else if (!s.empty() && s[0] != '-' && positional < 3) {
            *dims[positional++] = std::atoi(s.c_str());
        } else {
            std::printf("unknown argument: %s\n", s.c_str());
            return 2;
        }
    }

    try {
        Context ctx;
        std::printf("device: %s\n", ctx.gpu().name.c_str());
        std::printf("grid %d x %d x %d (%.2f M cells), tau %.4f, u %.3f, %s, sponge %d\n", cfg.nx,
                    cfg.ny, cfg.nz, cfg.nx * double(cfg.ny) * cfg.nz / 1e6, cfg.tau, cfg.u_inlet,
                    cfg.regularised ? "regularised" : "BGK", cfg.outlet_sponge);

        lbm::Solver solver(ctx, cfg);

        // Default body: a sphere of diameter ny / 6 at x = nx / 4, centred in
        // y and z. With --stl: the mesh fitted to `size` cells there instead.
        const float d = cfg.ny / 6.0f;
        const float cx = cfg.nx / 4.0f, cy = cfg.ny / 2.0f, cz = cfg.nz / 2.0f;
        std::vector<std::uint8_t> flags(solver.cells(), lbm::FLUID);
        double area = 3.14159265358979 * 0.25 * d * d;
        std::size_t solid = 0;
        if (!stl_path.empty()) {
            const geometry::Mesh raw = geometry::load_stl(stl_path);
            const geometry::Mesh placed =
                geometry::fit_to_box(raw, {cx, cy, cz}, stl_size > 0 ? stl_size : 2.0f * d);
            Voxeliser vox(ctx, cfg.nx, cfg.ny, cfg.nz);
            const auto t0 = std::chrono::steady_clock::now();
            const Voxeliser::Stats vs = vox.voxelise(placed, flags);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            std::printf("%s: %zu triangles -> %zu solid cells (%zu from the thin shell), %.0f ms\n",
                        stl_path.c_str(), vs.n_tris, vs.n_solid, vs.n_thin, ms);
            solid = vs.n_solid;
            // Reference area: the flags' projected (frontal) area.
            std::vector<std::uint8_t> shadow(std::size_t(cfg.ny) * cfg.nz, 0);
            for (int x = 0; x < cfg.nx; ++x)
                for (int y = 0; y < cfg.ny; ++y)
                    for (int z = 0; z < cfg.nz; ++z)
                        if (flags[(std::size_t(x) * cfg.ny + y) * cfg.nz + z] == lbm::OBSTACLE)
                            shadow[std::size_t(y) * cfg.nz + z] = 1;
            area = 0.0;
            for (std::uint8_t s : shadow)
                area += s;
        } else {
            for (int x = 0; x < cfg.nx; ++x)
                for (int y = 0; y < cfg.ny; ++y)
                    for (int z = 0; z < cfg.nz; ++z) {
                        const float rx = x + 0.5f - cx, ry = y + 0.5f - cy, rz = z + 0.5f - cz;
                        if (rx * rx + ry * ry + rz * rz <= 0.25f * d * d) {
                            flags[(static_cast<std::size_t>(x) * cfg.ny + y) * cfg.nz + z] =
                                lbm::OBSTACLE;
                            ++solid;
                        }
                    }
        }
        solver.set_flags(flags);
        solver.set_torque_ref({cx, cy, cz});
        if (cfg.moving_boundaries) {
            // Spin about z: omega = spin * u_inlet / r. Backspin (top surface
            // moving downstream with the air) gives +y Magnus lift.
            const float omega_z = -spin * cfg.u_inlet / (0.5f * d);
            const int n = solver.set_rotation({cx, cy, cz}, {0.0f, 0.0f, omega_z});
            std::printf("spinning %d cells, surface speed %.2f x u_inlet\n", n, spin);
        }
        solver.init_equilibrium(1.0f, {cfg.u_inlet, 0.0f, 0.0f});
        const double q = 0.5 * cfg.u_inlet * cfg.u_inlet * area;
        const double nu = (cfg.tau - 0.5) / 3.0;
        std::printf("body %zu solid cells, reference area %.1f cells^2, Re_D %.0f (molecular "
                    "nu, D = %.1f)\n",
                    solid, area, cfg.u_inlet * d / nu, d);

        solver.step(10); // warm up: pipelines, clocks
        solver.read_mean_forces();

        const int report = std::max(1, steps / 8);
        double gpu_seconds = 0.0;
        int done = 0;
        while (done < steps) {
            const int n = std::min(report, steps - done);
            const auto t0 = std::chrono::steady_clock::now();
            solver.step(n);
            const auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            gpu_seconds += sec;
            done += n;
            const lbm::MeanForces mf = solver.read_mean_forces();
            const lbm::Health h = solver.health();
            std::printf("step %6d  %7.0f MLUPS  Cd %.4f  Cl_y %+.4f  max|u| %.4f  bad %d\n", done,
                        solver.cells() * double(n) / sec / 1e6, mf.force[0] / q, mf.force[1] / q,
                        h.max_speed, h.bad_cells);
            if (h.bad_cells > 0 || h.max_speed > 0.5f) {
                std::printf("DIVERGED\n");
                return 1;
            }
        }
        std::printf("mean %.0f MLUPS over %d steps; mean fluid rho %.6f; upstream-plane rho %.6f\n",
                    solver.cells() * double(steps) / gpu_seconds / 1e6, steps,
                    solver.mean_fluid_density(), solver.plane_mean_density(4));
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
