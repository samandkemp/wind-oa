// Helpers the app's UI files share (app.cpp, panels.cpp, play.cpp,
// overlays.cpp): the wind in familiar units, the looks' names, the colour
// ramps of the legends. Internal to the app.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "imgui.h"
#include "windoa/airspeed.hpp"

namespace windoa::app {

using Clock = std::chrono::steady_clock;

// Display: the user's UI scale (times the monitor's DPI) and the panels a
// scale change resizes.
inline constexpr float kUiScaleMin = 0.6f, kUiScaleMax = 2.5f;
inline const char* const kPanelNames[] = {"Tunnel", "Model", "Compare", "View", "Analysis"};
// A length in multiples of the current font size (sizes scale with the text).
inline float em(float n) {
    return n * ImGui::GetFontSize();
}

// The wind in familiar units: lattice speed or Mach <-> m/s (sea-level air,
// airspeed.hpp) and the display unit chosen on the quick bar.
inline const char* const kUnitNames[] = {"mph", "km/h", "m/s"};
inline double unit_per_mps(int unit) {
    return unit == 0 ? 1.0 / airspeed::kMetresPerSecondPerMph : unit == 1 ? 3.6 : 1.0;
}
inline double lattice_to_mps(double u) {
    return airspeed::metres_per_second(airspeed::mach_from_lattice(u));
}
inline double mps_to_lattice(double v) {
    return v / (airspeed::kSeaLevelSoundSpeed * std::sqrt(3.0));
}
// "9.4 million" / "2,400": a Reynolds number to read at a glance.
inline std::string readable(double v) {
    char b[48];
    if (v >= 1e6)
        std::snprintf(b, sizeof(b), "%.1f million", v / 1e6);
    else if (v >= 1e4)
        std::snprintf(b, sizeof(b), "%.0f thousand", v / 1e3);
    else
        std::snprintf(b, sizeof(b), "%.0f", v);
    return b;
}

// The looks (keys 1 - 9): named combinations of what is drawn.
inline constexpr int kLooks = 9;
inline const char* const kLookNames[kLooks] = {"tunnel", "smoke", "pressure", "vortices", "texture",
                                               "oil",    "dye",   "wake",     "schlieren"};
inline const char* const kLookHints[kLooks] = {
    "Smoke from the wand, the speed haze and surface pressure: the start look.",
    "Smoke streaks only, over a plain body, as in a smoke tunnel.",
    "Surface pressure and a pressure slice through the model.",
    "Vortex cores (Q criterion) in a haze of streamwise vorticity.",
    "A moving flow texture on a slice through the model.",
    "Oil-flow streaks on the surface, as in an oil-film test.",
    "Dye smoke filling the wake, coloured by speed.",
    "The time-averaged wake: mean speed on a slice and the recirculation\n"
    "shells (switches averaging on; the flow must settle first).",
    "Density gradients on a slice: shocks in transonic mode, sound waves below."};
inline const char* const kViewNames[] = {"front", "side", "top", "3/4", "rear", "tunnel"};
inline constexpr int kViews = 6;

// Colour stops for the legends -- the same tables as the shaders' maps.
struct Stop {
    float t, r, g, b;
};
inline const std::vector<Stop> kCoolwarm = {{0.0f, 0, 0, 1}, {0.5f, 1, 1, 1}, {1.0f, 1, 0, 0}};
inline const std::vector<Stop> kVort = {{0.0f, 40 / 255.f, 130 / 255.f, 1.0f},
                                        {0.3f, 10 / 255.f, 30 / 255.f, 90 / 255.f},
                                        {0.5f, 3 / 255.f, 4 / 255.f, 8 / 255.f},
                                        {0.7f, 110 / 255.f, 25 / 255.f, 20 / 255.f},
                                        {1.0f, 1.0f, 140 / 255.f, 40 / 255.f}};
inline const std::vector<Stop> kDye = {{0.0f, 0.20f, 0.07f, 0.42f},
                                       {0.35f, 0.66f, 0.13f, 0.46f},
                                       {0.65f, 0.98f, 0.45f, 0.18f},
                                       {1.0f, 1.00f, 0.93f, 0.62f}};
inline const std::vector<Stop> kQcore = {{0.0f, 0.30f, 0.95f, 0.50f}, {1.0f, 0.85f, 0.55f, 0.30f}};
inline const std::vector<Stop> kGrey = {{0.0f, 0, 0, 0}, {1.0f, 0.88f, 0.92f, 1.0f}};
// the View panel's colour maps (vol_march.comp seqmap: after matplotlib's viridis)
inline const std::vector<Stop> kSeq = {{0.0f, 0.267f, 0.005f, 0.329f},
                                       {0.25f, 0.229f, 0.322f, 0.546f},
                                       {0.5f, 0.128f, 0.567f, 0.551f},
                                       {0.75f, 0.369f, 0.789f, 0.383f},
                                       {1.0f, 0.993f, 0.906f, 0.144f}};
inline const char* const kPaletteNames[] = {"the field's own", "sequential", "diverging",
                                            "greyscale"};

inline ImU32 ramp(const std::vector<Stop>& s, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    for (std::size_t i = 1; i < s.size(); ++i) {
        if (t <= s[i].t) {
            const float u = (t - s[i - 1].t) / std::max(s[i].t - s[i - 1].t, 1e-6f);
            return ImGui::ColorConvertFloat4ToU32({s[i - 1].r + u * (s[i].r - s[i - 1].r),
                                                   s[i - 1].g + u * (s[i].g - s[i - 1].g),
                                                   s[i - 1].b + u * (s[i].b - s[i - 1].b), 1.0f});
        }
    }
    return ImGui::ColorConvertFloat4ToU32({s.back().r, s.back().g, s.back().b, 1.0f});
}

inline const char* const kFieldNames[] = {
    "speed vs freestream",   "pressure (Cp)",
    "|vorticity|",           "streamwise vorticity",
    "local Mach number",     "schlieren (shocks)",
    "mean speed (averaged)", "turbulence intensity (averaged)"};
inline const std::vector<Stop> kRecirc = {{0.0f, 0.25f, 0.75f, 0.85f}, {1.0f, 0.10f, 0.35f, 0.95f}};

} // namespace windoa::app
