// wind-oa: the interactive tunnel.
//
//   windoa_app [fast|balanced|fine] [subsonic|transonic] [--model ID] [--no-vsync]
//              [--frames N] [--shot FILE.png] [--warmup STEPS]
//
//   fast|balanced|fine  grid preset (tunnel_preset); default fast
//   transonic           start in the compressible Euler regime (Mach 0.3-1.6)
//   --model ID          start with this catalogue model (default car_saloon)
//   --f16               f16 distribution storage: ~1.3x faster LBM, an
//                       approximation (THEORY 11.3)
//   --frames N          exit after N frames (a smoke test)
//   --shot FILE         with --frames: save the last frame, UI included
//   --warmup STEPS      run the solver this many steps before the first frame
//   --show LIST         view toggles, comma-separated: q, dye, lines, nosmoke,
//                       slice, nohaze, voxel (for scripted screenshots)
//   --field NAME        speed|pressure|vorticity|vortx|mach|schlieren
//   --spin R / --aoa D  spin ratio (models with spinners) / pitch at start
//
// Controls: RMB orbit, MMB pan, wheel zoom, WASD pan, Q / E zoom, F focus on
// the model, Space pause, H hide panels, P screenshot, Esc quit.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>

#include <windows.h>

#include "app.hpp"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0); // progress survives a crash
    windoa::app::Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        auto next = [&]() -> std::string { return a + 1 < argc ? argv[++a] : ""; };
        if (s == "fast" || s == "balanced" || s == "fine")
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
        else if (s == "--aoa")
            o.aoa = float(std::atof(next().c_str()));
        else {
            std::fprintf(stderr, "unknown argument: %s\n", s.c_str());
            return 2;
        }
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
