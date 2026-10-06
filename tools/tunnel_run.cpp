// Headless sandbox run: the app's Tunnel without a window. A development
// tool (not a gate). kUsage below is the reference for the command line.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "windoa/airspeed.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/tunnel.hpp"

using namespace windoa;

namespace {

const char* const kUsage =
    R"(usage: tunnel_run [options]     the app's sandbox, headless
       tunnel_run --list          the catalogue's model IDs
       tunnel_run --catalogue     voxelise every catalogue model at its default
                                  placement: triangles, cells, areas, bounds

  --preset fast|balanced|fine|ultra
                                grid preset (default fast)
  --model ID                    catalogue model (default ahmed_25deg)
  --stl FILE                    your own model instead: a binary or ASCII STL,
                                nose towards -x (scaled to --size, default 64)
  --ground air|road|fixed       free air, a rolling road or a fixed floor
                                (default: the catalogue model's own, air for --stl)
  --batches N                   batches to run (default 40)
  --steps S                     steps per batch (default 50)
  --spin R                      spin ratio, rim speed / U (models with spinners)
  --dye                         dye smoke on
  --tu PCT                      inlet turbulence intensity, percent
  --aoa DEG                     pitch, degrees
  --yaw DEG                     yaw, degrees
  --size CELLS                  the model's longest axis, cells (default: its own)
  --speed U                     the freestream, lattice units (default 0.05; at most 0.11)
  --f16                         f16 distribution storage (an ungated approximation)
  --average                     time averaging once settled; reports the wake survey
  --probe X,Y,Z                 a probe at (X, Y, Z) cells (up to 4); reports its spectrum
  --no-cache                    never restore or save a settled flow
  --walls sub|half              sub-cell walls (default) or half-way bounce-back
  --rotors L                    rotors turning at tip-speed ratio L (models with rotors)
  --power T                     engines on at throttle T (models with jets / intakes)
  --csv FILE                    append the run's summary as one row (a header
                                when the file is new): runs of several models,
                                sizes or angles become one comparison table
  -h, --help                    this text

The summary gives the second-half means and the lift spectrum's peak as a
Strouhal number on the body's height (THEORY 12.3).
)";

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0); // progress survives a crash
    std::string preset = "fast", model_id = "ahmed_25deg", stl, ground, csv;
    int batches = 40, steps = 50;
    float spin = -1.0f, tu = 0.0f, aoa = 0.0f, yaw = 0.0f, rotors = -1.0f, power = -1.0f,
          size = 0.0f, speed = 0.0f;
    bool dye = false, catalogue_check = false, f16 = false, average = false, no_cache = false,
         half_walls = false;
    std::vector<std::array<float, 3>> probes;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        auto next = [&]() { return a + 1 < argc ? argv[++a] : ""; };
        if (s == "--preset")
            preset = next();
        else if (s == "--model")
            model_id = next();
        else if (s == "--stl")
            stl = next();
        else if (s == "--csv")
            csv = next();
        else if (s == "--yaw")
            yaw = float(std::atof(next()));
        else if (s == "--ground") {
            ground = next();
            if (ground != "air" && ground != "road" && ground != "fixed") {
                std::fprintf(stderr, "--ground takes air, road or fixed\n");
                return 2;
            }
        } else if (s == "--batches")
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
        else if (s == "--list") {
            for (const catalogue::Entry& e : catalogue::entries())
                std::printf("%-20s %-24s %s\n", e.id.c_str(), e.group.c_str(), e.label.c_str());
            return 0;
        } else if (s == "--average")
            average = true;
        else if (s == "--no-cache")
            no_cache = true;
        else if (s == "--speed")
            speed = float(std::atof(next()));
        else if (s == "--size")
            size = float(std::atof(next()));
        else if (s == "--rotors")
            rotors = float(std::atof(next()));
        else if (s == "--power")
            power = float(std::atof(next()));
        else if (s == "--walls") {
            const std::string w = next();
            if (w != "sub" && w != "half") {
                std::fprintf(stderr, "--walls takes sub or half\n");
                return 2;
            }
            half_walls = w == "half";
        } else if (s == "--probe") {
            std::array<float, 3> q{};
            if (sscanf_s(next(), "%f,%f,%f", &q[0], &q[1], &q[2]) != 3) {
                std::fprintf(stderr, "--probe needs X,Y,Z (cells)\n");
                return 2;
            }
            probes.push_back(q);
        } else if (s == "-h" || s == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n\n%s", s.c_str(), kUsage);
            return 2;
        }
    }
    try {
        Context ctx;
        TunnelSettings ts = tunnel_preset(preset);
        ts.storage_f16 = f16;
        ts.sub_cell_walls = !half_walls;
        const std::filesystem::path cache_dir =
            no_cache ? std::filesystem::temp_directory_path() / "windoa_tunnel_run_nocache"
                     : std::filesystem::path("cache/flow");
        if (no_cache)
            std::filesystem::remove_all(cache_dir);
        Tunnel tunnel(ctx, ts, cache_dir);
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
                char full[24] = "    -";
                if (st.full_scale_length_m > 0.0)
                    std::snprintf(full, sizeof(full), "%7.3g m", st.full_scale_length_m);
                std::printf("%-22s %-18s %7zu tris %8zu cells  front %6.0f  plan %6.0f  full size "
                            "%s  %5.0f+%4.0f ms%s\n",
                            e.id.c_str(), e.group.c_str(), st.n_tris, st.n_solid, st.a_frontal,
                            st.a_planform, full, build_ms, st.vox_ms, ok ? "" : "  <-- CHECK");
            }
            std::printf("%zu models, %d flagged\n", catalogue::entries().size(), bad);
            return bad ? 1 : 0;
        }

        Model m;
        Placement p;
        std::string name = model_id;
        if (!stl.empty()) { // your own model: free air, centred, unless told otherwise
            m = model_from_stl(stl);
            name = std::filesystem::path(stl).filename().string();
        } else {
            const catalogue::Entry* e = catalogue::find(model_id);
            if (!e) {
                std::printf("no catalogue model '%s' (tunnel_run --list)\n", model_id.c_str());
                return 2;
            }
            m = model_from_catalogue(*e);
            p = default_placement(*e, ts, m.mesh);
        }
        if (size > 0.0f)
            p.length_cells = size;
        if (!ground.empty()) { // as the app's placement control sets it
            p.ground = ground == "road"    ? catalogue::Ground::Road
                       : ground == "fixed" ? catalogue::Ground::Fixed
                                           : catalogue::Ground::Air;
            p.ride_height =
                p.ground == catalogue::Ground::Road ? std::max(4.0f, 0.06f * p.length_cells) : 0.0f;
            if (p.ground == catalogue::Ground::Air)
                p.pos_frac[1] = 0.5f;
        }
        p.aoa_deg = aoa;
        p.yaw_deg = yaw;
        tunnel.set_model(std::move(m), p);
        if (spin >= 0.0f)
            tunnel.set_spin(true, spin);
        if (speed > 0.0f)
            tunnel.set_speed(speed);
        if (rotors >= 0.0f)
            tunnel.set_rotors(true, rotors);
        if (power >= 0.0f)
            tunnel.set_power(true, power);
        if (tu > 0.0f)
            tunnel.set_turbulence(tu);
        if (average)
            tunnel.set_averaging(true);
        if (!probes.empty())
            tunnel.set_probes(probes);
        if (dye) {
            const TunnelStatus st = tunnel.status();
            tunnel.set_dye(true);
            tunnel.set_dye_rake(ts.rake_x, st.placed_centre[1], st.placed_centre[2], 0.1f * ts.ny,
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
        const double run_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double mean_cd = st.cd, mean_cl = st.cl, sd_cd = 0.0, sd_cl = 0.0;
        if (n_avg > 1) {
            mean_cd = sum_cd / n_avg;
            mean_cl = sum_cl / n_avg;
            sd_cd = std::sqrt(std::max(sq_cd / n_avg - mean_cd * mean_cd, 0.0));
            sd_cl = std::sqrt(std::max(sq_cl / n_avg - mean_cl * mean_cl, 0.0));
        }
        if (n_avg > 1) {
            const double mcd = sum_cd / n_avg, mcl = sum_cl / n_avg;
            std::printf("second-half mean: Cd %.4f (sd %.4f)  Cl %.4f (sd %.4f)  over %d batches\n",
                        mcd, std::sqrt(std::max(sq_cd / n_avg - mcd * mcd, 0.0)), mcl,
                        std::sqrt(std::max(sq_cl / n_avg - mcl * mcl, 0.0)), n_avg);
        }
        tunnel.refresh_analysis();
        const TunnelAnalysis& a = tunnel.analysis();
        if (a.lift.peak_amp > 0.0) {
            const auto pk = spectral_peaks(a.lift, 3, 2.0 / std::max(a.window_steps, 1.0));
            std::printf("lift spectrum over %.0f steps: peak St %.3f (period %.0f steps, "
                        "amplitude %.3g); next peaks at St",
                        a.window_steps, a.st_lift, 1.0 / a.lift.peak_freq, a.lift.peak_amp);
            for (std::size_t k = 1; k < pk.size(); ++k)
                std::printf(" %.3f", pk[k] * a.l_ref / a.u_ref);
            std::printf("  (h %.0f cells; acoustic modes every St %.3f)\n", a.l_ref,
                        a.f_acoustic * a.l_ref / a.u_ref);
            if (a.st_shedding > 0.0)
                std::printf("shedding (strongest peak below the first acoustic mode): St %.3f\n",
                            a.st_shedding);
        }
        for (int i = 0; i < a.n_probes; ++i) {
            static const char* kComp[] = {"u_x", "u_y", "u_z", "rho"};
            std::printf("probe %d:", i + 1);
            for (int k = 0; k < 4; ++k)
                std::printf("  %s St %.3f (amp %.2g)", kComp[k],
                            a.probe_spec[i][k].peak_freq * a.l_ref / a.u_ref,
                            a.probe_spec[i][k].peak_amp);
            std::printf("\n");
        }
        if (average)
            std::printf("averaged %.2f flow-throughs (%d samples); wake survey %s: Cd %.4f vs the "
                        "force balance's %.4f (%+.2f %%), mass flux change %.1e\n",
                        a.avg_flow_throughs, a.avg_samples, a.wake_valid ? "valid" : "not ready",
                        a.cd_wake, a.cd_balance,
                        a.wake_valid ? (a.cd_wake / a.cd_balance - 1.0) * 100.0 : 0.0,
                        a.mass_imbalance);
        if (st.has_rotors && st.rotor_coeffs_ready)
            std::printf("rotors %s, tip-speed ratio %.2f: C_T %+.4f  C_P %+.4f (swept area %.1f %% "
                        "of the section)\n",
                        st.rotors_turning ? "turning" : "parked", st.rotor_tsr, st.rotor_ct,
                        st.rotor_cp, 100.0 * st.rotor_blockage);
        else if (st.has_rotors)
            std::printf("rotors %s, tip-speed ratio %.2f: the wind is not yet at speed (swept "
                        "area %.1f %% of the section)\n",
                        st.rotors_turning ? "turning" : "parked", st.rotor_tsr,
                        100.0 * st.rotor_blockage);
        if (st.has_ports)
            std::printf("engines %s, jet scale %.3f\n", st.power_on ? "on" : "off", st.jet_scale);
        if (st.full_scale_length_m > 0.0)
            std::printf("wind %s: full size (%.3g m) at that speed Re %.2g, this run Re %.0f\n",
                        airspeed::describe(st.airspeed_mach).c_str(), st.full_scale_length_m,
                        st.full_scale_re, st.re_sim);
        else
            std::printf("wind %s: this run Re %.0f\n", airspeed::describe(st.airspeed_mach).c_str(),
                        st.re_sim);
        std::printf("spin scale %.3f, cache: %s (%d entries)\n", st.spin_scale,
                    st.cache_note.c_str(), st.cache_entries);
        std::printf("voxelised %zu triangles to %zu cells in %.0f ms (walls included)\n", st.n_tris,
                    st.n_solid, st.vox_ms);
        if (!csv.empty()) { // one row per run, a header for a new file
            std::error_code ec;
            const bool fresh =
                !std::filesystem::exists(csv, ec) || std::filesystem::file_size(csv, ec) == 0;
            std::ofstream out(csv, std::ios::app);
            if (!out) {
                std::printf("FAIL: cannot write %s\n", csv.c_str());
                return 1;
            }
            if (fresh)
                out << "model,preset,size_cells,aoa_deg,yaw_deg,ground,u_lattice,speed_mps,"
                       "re_sim,re_full_size,steps,settled,flow_throughs,cd_mean,cd_sd,cl_mean,"
                       "cl_sd,cs,cm,st_shedding,a_ref_cells,mlups\n";
            static const char* kGround[] = {"air", "road", "fixed"};
            char row[512];
            std::snprintf(row, sizeof(row),
                          "\"%s\",%s,%.0f,%.2f,%.2f,%s,%.4f,%.2f,%.0f,%.4g,%lld,%d,%.2f,%.5f,%.5f,"
                          "%.5f,%.5f,%.5f,%.5f,%.4f,%.1f,%.0f\n",
                          name.c_str(), preset.c_str(), p.length_cells, aoa, yaw,
                          kGround[std::clamp(int(p.ground), 0, 2)], st.u_applied,
                          airspeed::metres_per_second(st.airspeed_mach), st.re_sim,
                          st.full_scale_re, static_cast<long long>(st.steps), st.settled ? 1 : 0,
                          st.flow_throughs, mean_cd, sd_cd, mean_cl, sd_cl, st.cs, st.cm,
                          a.st_shedding, st.a_ref,
                          double(ts.nx) * ts.ny * ts.nz * double(batches) * steps / run_sec / 1e6);
            out << row;
            std::printf("appended the summary to %s\n", csv.c_str());
        }
        if (no_cache)
            std::filesystem::remove_all(cache_dir);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
