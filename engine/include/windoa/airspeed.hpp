// Equivalent airspeeds (THEORY 1.2). The lattice's one absolute speed scale is
// its sound speed, c_s = 1 / sqrt(3), so a lattice speed u is the Mach number
// u sqrt(3), and the compressible solver works in Mach numbers directly. In
// sea-level air (ISA, 15 C) a Mach number is a definite speed. The mapping
// matches compressibility, not the Reynolds number: the sandbox runs near
// Re 10^3 whatever the speed reads (THEORY 4.5).
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace windoa::airspeed {

inline constexpr double kSeaLevelSoundSpeed = 340.29; // m/s, ISA at 15 C
inline constexpr double kMetresPerSecondPerMph = 0.44704;

inline double mach_from_lattice(double u) {
    return u * std::sqrt(3.0);
}
inline double metres_per_second(double mach) {
    return mach * kSeaLevelSoundSpeed;
}
inline double mph(double metres_per_second) {
    return metres_per_second / kMetresPerSecondPerMph;
}

// "29.5 m/s (66 mph)" for a Mach number: one decimal where it still matters.
inline std::string describe(double mach) {
    const double v = metres_per_second(mach), m = mph(v);
    char buf[48];
    std::snprintf(buf, sizeof(buf), v < 100.0 ? "%.1f m/s (%.*f mph)" : "%.0f m/s (%.*f mph)", v,
                  m < 10.0 ? 1 : 0, m);
    return buf;
}

} // namespace windoa::airspeed
