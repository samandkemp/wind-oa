// V23 -- a restored flow must be the settled flow (engine FlowCache).
// Specified in THEORY 9.6, 9.8 (docs/THEORY.md).
//
// Identity reference, judged against a never-restored control rather than
// a fixed bar, because the wake is unsteady. A short check is not enough:
// restoring f without rho / u leaves the pressure outlet imposing a flow at
// rest for one step, which launches a pulse that swings Cd between -1.0
// and +1.8 about 250 steps later.
//   1. develop the Ahmed body (the app's solver, fast grid, 22,000 steps)
//      and save it through the real FlowCache (disk round trip, f16 of
//      f - w);
//   2. restore into a separate solver in the startup state (at rest);
//   3. run both 2,000 steps (> 5 acoustic transits) in the app's growing
//      steps-per-frame schedule, reading frame-mean forces.
// Pass: means within 1.5 %; the restored run's largest frame excursion
// <= 1.5x the control's (+0.01); frame-by-frame within 5 % over the first
// 450 steps (the window a broken restore's pulse arrives in).
#include <filesystem>

#include "cases.hpp"
#include "windoa/flow_cache.hpp"

using namespace windoa;

namespace {

struct Frame {
    int total, n;
    double cd;
};

std::vector<Frame> trace(lbm::Solver& s, double q) {
    std::vector<int> schedule(20, 1);
    for (const int n : {2, 4, 7, 11})
        schedule.push_back(n);
    schedule.insert(schedule.end(), 200, 17);
    std::vector<Frame> out;
    int total = 0;
    for (const int n : schedule) {
        if (total >= 2000)
            break;
        s.step(n);
        total += n;
        out.push_back({total, n, s.read_mean_forces().force[0] / q});
    }
    return out;
}

std::pair<double, double> stats(const std::vector<Frame>& tr) { // weighted mean, max excursion
    double sw = 0, s = 0;
    for (const auto& f : tr) {
        s += f.cd * f.n;
        sw += f.n;
    }
    const double mean = s / sw;
    double ex = 0.0;
    for (const auto& f : tr)
        ex = std::max(ex, std::abs(f.cd - mean));
    return {mean, ex};
}

} // namespace

int main() {
    return gate::run("V23_flow_cache", [](gate::Gate& g) {
        Context ctx;
        const TunnelSettings t = tunnel_preset("fast");
        Voxeliser vox(ctx, t.nx, t.ny, t.nz);
        const auto placed = cases::app::ahmed(vox, t);
        const double q = 0.5 * t.u_inlet * t.u_inlet * placed.area;

        // 1. develop and save through the real cache
        lbm::Solver a(ctx, solver_config(t));
        cases::app::place(a, placed);
        a.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
        for (int done = 0; done < 20000; done += 500) {
            a.set_inlet_velocity(t.u_inlet * std::min(1.0f, float(done + 500) / t.ramp_steps));
            a.step(500);
        }
        a.read_mean_forces();
        a.step(2000);
        const double cd_saved = a.read_mean_forces().force[0] / q;
        const auto dir = std::filesystem::temp_directory_path() / "windoa_v23_cache";
        std::filesystem::remove_all(dir);
        FlowCache cache(dir, 512.0);
        const std::string key = cache.key("V23 gate: ahmed_25deg fast preset, app solver");
        const double dt_save = cache.save(key, a.get_state(), {});
        g.note("developed 22,000 steps; Cd over the last 2,000 = %.4f; saved in %.2f s (%.0f MiB)",
               cd_saved, dt_save, cache.usage_mb());

        // 2. restore into a solver at rest (the app's startup state)
        lbm::Solver b(ctx, solver_config(t));
        cases::app::place(b, placed);
        b.init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
        b.set_inlet_velocity(t.u_inlet);
        const auto entry = cache.load(key);
        if (!g.check(entry.has_value(), "cache entry loads back from disk"))
            return;
        b.set_state(entry->first);
        b.read_mean_forces();

        // 3. the same schedule on the control (never restored) and the restored solver
        const auto ctl = trace(a, q);
        const auto res = trace(b, q);
        std::filesystem::remove_all(dir);
        const auto [m_c, spread_c] = stats(ctl);
        const auto [m_r, spread_r] = stats(res);
        double early = 0.0;
        for (std::size_t i = 0; i < ctl.size() && i < res.size(); ++i)
            if (ctl[i].total <= 450)
                early = std::max(early, std::abs(res[i].cd - ctl[i].cd) / std::abs(ctl[i].cd));
        g.note("control (never restored): mean Cd %.4f, max frame excursion %.4f", m_c, spread_c);
        g.note("restored over rest state: mean Cd %.4f, max frame excursion %.4f", m_r, spread_r);
        g.check(std::abs(m_r - m_c) / std::abs(m_c) < 0.015, "means differ by %.2f %% (< 1.5 %%)",
                std::abs(m_r - m_c) / std::abs(m_c) * 100.0);
        g.check(spread_r < 1.5 * spread_c + 0.01, "restored excursion %.4f <= 1.5 x control + 0.01",
                spread_r);
        g.check(early < 0.05, "frame-by-frame difference over the first 450 steps %.1f %% (< 5 %%)",
                early * 100.0);
    });
}
