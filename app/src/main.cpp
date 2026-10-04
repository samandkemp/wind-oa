// wind-oa: the interactive tunnel. kUsage below is the reference for the
// command line (--help prints it).
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iterator>
#include <string>

#include <windows.h>

#include "app.hpp"
#include "windoa/catalogue.hpp"

namespace {

const char* const kUsage =
    R"(usage: windoa_app [fast|balanced|fine|ultra] [subsonic|transonic] [options]

  fast|balanced|fine  grid preset (tunnel_preset); default fast
  ultra               512 x 192 x 192, for patient studies (~160 steps / s)
  transonic           start in the compressible Euler regime (Mach 0.3-1.6)
  --model ID          start with this catalogue model (default car_saloon);
                      tunnel_run --list lists them
  --f16               f16 distribution storage: about a third faster per
                      step, an ungated approximation (THEORY 11.3)
  --no-vsync          present without waiting for the display
  --frames N          exit after N frames (a smoke test)
  --shot FILE         with --frames: save the last frame, UI included
  --warmup STEPS      run the solver this many steps before the first frame
  --show LIST         view toggles, comma-separated: q, dye, lines, nosmoke,
                      slice, hslice, xslice, nohaze, voxel, avg, recirc,
                      lic, arrows, timelines, nopaint, wallspeed,
                      reversed, oil, analysis, probes, help, noui,
                      noplots, nobox, nosurface
  --field NAME        speed|pressure|vorticity|vortx|mach|schlieren|mean|turb
  --spin R            spin on at this ratio (models with spinners)
  --power T           engines on at throttle T (models with jets / intakes)
  --rotors L          rotors turning at tip-speed ratio L (models with rotors)
  --aoa D             pitch at start, degrees
  --size CELLS        the model's longest axis at start, cells
  --mach M            transonic Mach number at start (default 0.8)
  --zoom F            camera distance x F, aimed at the model (F < 1 nearer)
  --view AZ,EL        camera azimuth and elevation, degrees (start: 35,18)
  -h, --help          this text

Controls: RMB orbit, MMB pan, wheel zoom, WASD pan, Q / E zoom, F focus on
the model, Space pause, H hide panels, P screenshot, F1 help, Esc quit.
)";

const char* const kFields[] = {"speed", "pressure",  "vorticity", "vortx",
                               "mach",  "schlieren", "mean",      "turb"};

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0); // progress survives a crash
    windoa::app::Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        auto next = [&]() -> std::string { return a + 1 < argc ? argv[++a] : ""; };
        if (s == "fast" || s == "balanced" || s == "fine" || s == "ultra")
            o.preset = s;
        else if (s == "transonic")
            o.transonic = true;
        else if (s == "--f16")
            o.f16 = true;
        else if (s == "subsonic")
            o.transonic = false;
        else if (s == "--model")
            o.model = next();
        else if (s == "--frames")
            o.max_frames = std::atoi(next().c_str());
        else if (s == "--shot")
            o.shot = next();
        else if (s == "--warmup")
            o.warmup_steps = std::atoi(next().c_str());
        else if (s == "--no-vsync")
            o.no_vsync = true;
        else if (s == "--show")
            o.show = next();
        else if (s == "--field")
            o.field = next();
        else if (s == "--spin")
            o.spin = float(std::atof(next().c_str()));
        else if (s == "--power")
            o.power = float(std::atof(next().c_str()));
        else if (s == "--rotors")
            o.rotors = float(std::atof(next().c_str()));
        else if (s == "--aoa")
            o.aoa = float(std::atof(next().c_str()));
        else if (s == "--size")
            o.size = float(std::atof(next().c_str()));
        else if (s == "--mach")
            o.mach = float(std::atof(next().c_str()));
        else if (s == "--zoom")
            o.zoom = float(std::atof(next().c_str()));
        else if (s == "--view") {
            std::array<float, 2> v{};
            if (sscanf_s(next().c_str(), "%f,%f", &v[0], &v[1]) != 2) {
                std::fprintf(stderr, "--view needs AZ,EL (degrees)\n");
                return 2;
            }
            o.view = v;
        } else if (s == "-h" || s == "--help") {
            std::fputs(kUsage, stdout);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n\n%s", s.c_str(), kUsage);
            return 2;
        }
    }
    if (!windoa::catalogue::find(o.model)) {
        std::fprintf(stderr, "no catalogue model '%s'; the models are:\n", o.model.c_str());
        for (const auto& e : windoa::catalogue::entries())
            std::fprintf(stderr, "  %s\n", e.id.c_str());
        return 2;
    }
    if (!o.field.empty() &&
        std::find(std::begin(kFields), std::end(kFields), o.field) == std::end(kFields)) {
        std::fprintf(stderr, "unknown field '%s'\n\n%s", o.field.c_str(), kUsage);
        return 2;
    }
    // Work from the repository root wherever the exe was started (a double-
    // click starts it in build\app\<config>): cache/, screenshots/, models/
    // and the ImGui layout then always land in the same place.
    std::error_code ec;
    if (!o.shot.empty())
        o.shot = std::filesystem::absolute(o.shot, ec).string();
    wchar_t exe[MAX_PATH];
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0) {
        std::filesystem::path p = std::filesystem::path(exe).parent_path();
        for (int up = 0; up < 5 && !p.empty(); ++up, p = p.parent_path()) {
            if (std::filesystem::exists(p / "engine", ec) &&
                std::filesystem::exists(p / "CMakeLists.txt", ec)) {
                std::filesystem::current_path(p, ec);
                break;
            }
            if (p == p.parent_path())
                break;
        }
    }
    try {
        windoa::app::App app(o);
        return app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
