// Throughput benchmark (a measurement, not a gate): the LBM at each grid
// preset -- plain, with the app's moving boundaries, with dye -- and the
// Euler solver. The baseline for performance work. `bench controls` times
// instead what a slider costs the app's worker: one speed change. kUsage
// below is the reference for the command line.
//
// Timing: wall clock around blocking submits of 64 steps (so it includes
// submit overhead, like the app), after a second of warm-up -- the GPU's
// clocks take about that long to rise from idle, and a short run once read
// 2,964 MLUPS where a long one read 3,450 -- and over at least `--steps`
// steps and two seconds. Run it twice before believing a difference.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/dye.hpp"
#include "windoa/euler.hpp"
#include "windoa/lbm.hpp"
#include "windoa/shapes.hpp"
#include "windoa/tunnel.hpp"

using namespace windoa;
using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Million cell updates a second: `run(k)` advances k steps (blocking).
template <class Run> double timed_rate(double cells, int steps, Run run) {
    constexpr int kChunk = 64;
    const auto w0 = Clock::now();
    while (seconds_since(w0) < 1.0) // pipelines, caches, clocks
        run(kChunk);
    const auto t0 = Clock::now();
    long long done = 0;
    while (done < steps || seconds_since(t0) < 2.0) {
        run(kChunk);
        done += kChunk;
    }
    return cells * double(done) / seconds_since(t0) / 1e6;
}

// A sphere of diameter ny / 6 at x = nx / 4 (lbm_run's body).
std::vector<std::uint8_t> sphere_flags(int nx, int ny, int nz) {
    std::vector<std::uint8_t> f(std::size_t(nx) * ny * nz, lbm::FLUID);
    shapes::add_sphere(f, nx, ny, nz, nx / 4.0f, ny / 2.0f, nz / 2.0f, ny / 12.0f);
    return f;
}

double lbm_mlups(Context& ctx, const TunnelSettings& t, bool moving, bool dye, int steps,
                 bool f16 = false) {
    lbm::Config c;
    c.nx = t.nx;
    c.ny = t.ny;
    c.nz = t.nz;
    c.tau = t.tau;
    c.u_inlet = t.u_inlet;
    c.smagorinsky_cs = t.smagorinsky_cs;
    c.regularised = true;
    c.moving_boundaries = moving;
    c.storage_f16 = f16;
    c.outlet_sponge = t.outlet_sponge;
    c.sponge_target = t.sponge_target;
    lbm::Solver s(ctx, c);
    s.set_flags(sphere_flags(t.nx, t.ny, t.nz));
    s.init_equilibrium(1.0f, {c.u_inlet, 0.0f, 0.0f});
    Dye d(ctx, s);
    auto run = [&](int n) {
        if (dye)
            d.step_with(s, n);
        else
            s.step(n);
    };
    return timed_rate(double(s.cells()), steps, run);
}

double euler_mcells(Context& ctx, const TunnelSettings& t, int steps) {
    euler::Config c;
    c.nx = t.nx;
    c.ny = t.ny;
    c.nz = t.nz;
    c.mach = 0.8f;
    euler::Solver s(ctx, c);
    s.set_flags(sphere_flags(t.nx, t.ny, t.nz));
    s.init_freestream();
    return timed_rate(double(s.cells()), steps, [&](int n) { s.step(n); });
}

// The worker's cost of one speed change (Tunnel::set_speed re-requests every
// wall velocity, since spinning parts and engine faces scale with the wind):
// a model with nothing moving, the saloon with its wheels spinning, and the
// open-wheel car with wheels spinning and engines on. Blocking host time,
// uploads included; the mean over alternating changes.
void controls(Context& ctx, const TunnelSettings& t) {
    const auto cache = std::filesystem::temp_directory_path() / "windoa_bench_controls";
    struct Case {
        const char* id;
        bool spin, power;
    };
    for (const Case& k : {Case{"ahmed_25deg", false, false}, Case{"car_saloon", true, false},
                          Case{"car_open_wheel", true, true}}) {
        std::filesystem::remove_all(cache);
        Tunnel tn(ctx, t, cache);
        const catalogue::Entry* e = catalogue::find(k.id);
        Model m = model_from_catalogue(*e);
        const Placement pl = default_placement(*e, t, m.mesh);
        tn.set_model(std::move(m), pl);
        if (k.spin)
            tn.set_spin(true, 1.0f);
        if (k.power)
            tn.set_power(true, 1.0f);
        constexpr int n = 20;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i)
            tn.set_speed(i % 2 ? 0.055f : 0.05f);
        std::printf("  %-16s %8.2f ms per speed change\n", k.id, 1000.0 * seconds_since(t0) / n);
    }
    std::filesystem::remove_all(cache);
}

} // namespace

const char* const kUsage =
    R"(usage: bench [fast|balanced|fine|ultra|all] [--steps N]
       bench controls [fast|balanced|fine|ultra]

  fast|balanced|fine|ultra|all
                           the preset to measure (default all: every one but ultra)
  --steps N                LBM steps per measurement (default 1000; Euler a quarter)
  controls                 the worker's cost of one speed change instead (default fast)
  -h, --help               this text
)";

int main(int argc, char** argv) {
    std::string which = "all";
    int steps = 1000;
    bool control_costs = false;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--steps" && a + 1 < argc)
            steps = std::atoi(argv[++a]);
        else if (s == "controls")
            control_costs = true;
        else if (s == "fast" || s == "balanced" || s == "fine" || s == "ultra" || s == "all")
            which = s;
        else if (s == "-h" || s == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n\n%s", s.c_str(), kUsage);
            return 2;
        }
    }
    try {
        Context ctx;
        std::printf("device: %s\n", ctx.gpu().name.c_str());
        if (control_costs) {
            const std::string p = which == "all" ? "fast" : which;
            std::printf("speed changes, %s preset:\n", p.c_str());
            controls(ctx, tunnel_preset(p));
            return 0;
        }
        std::printf("%-9s %8s  %9s %9s %9s %9s  %11s\n", "preset", "Mcells", "LBM", "+moving",
                    "+dye", "f16", "Euler");
        std::printf("%-9s %8s  %9s %9s %9s %9s  %11s\n", "", "", "MLUPS", "MLUPS", "MLUPS", "MLUPS",
                    "Mcell-st/s");
        for (const char* p : {"fast", "balanced", "fine", "ultra"}) {
            if (which != p && (which != "all" || std::string(p) == "ultra"))
                continue;
            const TunnelSettings t = tunnel_preset(p);
            const double cells = double(t.nx) * t.ny * t.nz / 1e6;
            const double a = lbm_mlups(ctx, t, false, false, steps);
            const double b = lbm_mlups(ctx, t, true, false, steps);
            const double c = lbm_mlups(ctx, t, true, true, steps);
            const double h = lbm_mlups(ctx, t, true, false, steps, true);
            const double e = euler_mcells(ctx, t, std::max(steps / 4, 50));
            std::printf("%-9s %8.2f  %9.0f %9.0f %9.0f %9.0f  %11.0f\n", p, cells, a, b, c, h, e);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
