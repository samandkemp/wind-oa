// Headless sandbox run: the app's Tunnel without a window. A development
// tool (not a gate).
//
//   tunnel_run [--preset fast|balanced|fine] [--model ID] [--batches N]
//              [--steps S] [--spin R] [--dye] [--tu PCT] [--aoa DEG]
//   tunnel_run --catalogue    voxelise every catalogue model at its default
//                           placement: triangles, cells, areas, bounds
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/tunnel.hpp"

using namespace windoa;

int main(int argc, char** argv) {
    std::string preset = "fast", model_id = "ahmed_25deg";
    int batches = 40, steps = 50;
    float spin = -1.0f, tu = 0.0f, aoa = 0.0f;
    bool dye = false, catalogue_check = false, f16 = false;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        auto next = [&]() { return a + 1 < argc ? argv[++a] : ""; };
        if (s == "--preset")
            preset = next();
        else if (s == "--model")
            model_id = next();
        else if (s == "--batches")
            batches = std::atoi(next());
        else if (s == "--steps")
            steps = std::atoi(next());
        else if (s == "--spin")
            spin = float(std::atof(next()));
        else if (s == "--tu")
            tu = float(std::atof(next()));
        else if (s == "--aoa")
            aoa = float(std::atof(next()));
        else if (s == "--dye")
            dye = true;
        else if (s == "--f16")
            f16 = true;
        else if (s == "--catalogue")
            catalogue_check = true;
        else {
            std::printf("unknown argument: %s\n", s.c_str());
            return 2;
        }
    }
    try {
        Context ctx;
        TunnelSettings ts = tunnel_preset(preset);
        ts.storage_f16 = f16;
        Tunnel tunnel(ctx, ts, "cache/flow");
        std::printf("device: %s, grid %d x %d x %d\n", ctx.gpu().name.c_str(), ts.nx, ts.ny, ts.nz);

        if (catalogue_check) {
            int bad = 0;
            for (const catalogue::Entry& e : catalogue::entries()) {
                const auto t0 = std::chrono::steady_clock::now();
                Model m = model_from_catalogue(e);
                const double build_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
                const Placement pl = default_placement(e, ts, m.mesh);
                tunnel.set_model(std::move(m), pl);
                const TunnelStatus st = tunnel.status();
                const bool ok = st.n_solid > 0 && !st.out_of_bounds;
                bad += ok ? 0 : 1;
                std::printf(
                    "%-22s %-18s %7zu tris %8zu cells  front %6.0f  plan %6.0f  %5.0f+%4.0f ms%s\n",
                    e.id.c_str(), e.group.c_str(), st.n_tris, st.n_solid, st.a_frontal,
                    st.a_planform, build_ms, st.vox_ms, ok ? "" : "  <-- CHECK");
            }
            std::printf("%zu models, %d flagged\n", catalogue::entries().size(), bad);
            return bad ? 1 : 0;
        }

        const catalogue::Entry* e = catalogue::find(model_id);
        if (!e) {
            std::printf("no catalogue model '%s'\n", model_id.c_str());
            return 2;
        }
        Model m = model_from_catalogue(*e);
        Placement p = default_placement(*e, ts, m.mesh);
        p.aoa_deg = aoa;
        tunnel.set_model(std::move(m), p);
        if (spin >= 0.0f)
            tunnel.set_spin(true, spin);
        if (tu > 0.0f)
            tunnel.set_turbulence(tu);
        if (dye) {
            const TunnelStatus st = tunnel.status();
            tunnel.set_dye(true);
            tunnel.set_dye_rake(std::max(4.0f, st.placed_centre[0] - 0.18f * ts.nx),
                                st.placed_centre[1], st.placed_centre[2], 0.1f * ts.ny,
                                0.1f * ts.nz);
        }
        const auto t0 = std::chrono::steady_clock::now();
        // time averages over the second half of the run (unsteady wakes)
        double sum_cd = 0, sum_cl = 0, sq_cd = 0, sq_cl = 0;
        int n_avg = 0;
        for (int b = 1; b <= batches; ++b) {
            tunnel.advance(steps);
            if (b > batches / 2) {
                const TunnelStatus sa = tunnel.status();
                sum_cd += sa.cd;
                sum_cl += sa.cl;
                sq_cd += sa.cd * sa.cd;
                sq_cl += sa.cl * sa.cl;
                ++n_avg;
            }
            if (b % std::max(1, batches / 10) == 0 || b == batches) {
                const TunnelStatus st = tunnel.status();
                const double sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("step %7lld  u %.4f  Cd %+.4f  Cl %+.4f  Cm %+.4f  max|u| %.3f  %s  "
                            "(%.0f MLUPS)\n",
                            static_cast<long long>(st.steps), st.u_applied, st.cd, st.cl, st.cm,
                            st.max_speed, st.phase.c_str(),
                            double(ts.nx) * ts.ny * ts.nz * double(b) * steps / sec / 1e6);
                if (!st.health_note.empty())
                    std::printf("  note: %s\n", st.health_note.c_str());
            }
        }
        if (dye) {
            const std::vector<float> c = tunnel.dye()->concentration();
            double sum = 0.0, mx = 0.0;
            for (float v : c) {
                sum += v;
                mx = std::max(mx, double(v));
            }
            std::printf("dye: total %.1f, max C %.4f\n", sum, mx);
            if (mx > 1.2) { // where is it? (diagnostics for an unbounded dye)
                const std::vector<float> u = tunnel.solver().velocity();
                const auto& fl = tunnel.solver().flags();
                std::size_t k = 0;
                for (std::size_t i = 0; i < c.size(); ++i)
                    if (c[i] > c[k])
                        k = i;
                const int z = int(k % ts.nz), y = int((k / ts.nz) % ts.ny),
                          x = int(k / (std::size_t(ts.ny) * ts.nz));
                auto at = [&](int xx, int yy, int zz) {
                    return int(fl[(std::size_t(xx) * ts.ny + yy) * ts.nz + zz]);
                };
                std::printf("  max at (%d,%d,%d) flag %d: u (%.4f %.4f %.4f); face neighbours "
                            "%d %d %d %d %d %d\n",
                            x, y, z, at(x, y, z), u[3 * k], u[3 * k + 1], u[3 * k + 2],
                            at(x - 1, y, z), at(x + 1, y, z), at(x, y - 1, z), at(x, y + 1, z),
                            at(x, y, z - 1), at(x, y, z + 1));
            }
        }
        const TunnelStatus st = tunnel.status();
        if (n_avg > 1) {
            const double mcd = sum_cd / n_avg, mcl = sum_cl / n_avg;
            std::printf("second-half mean: Cd %.4f (sd %.4f)  Cl %.4f (sd %.4f)  over %d batches\n",
                        mcd, std::sqrt(std::max(sq_cd / n_avg - mcd * mcd, 0.0)), mcl,
                        std::sqrt(std::max(sq_cl / n_avg - mcl * mcl, 0.0)), n_avg);
        }
        std::printf("spin scale %.3f, cache: %s (%d entries)\n", st.spin_scale,
                    st.cache_note.c_str(), st.cache_entries);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
