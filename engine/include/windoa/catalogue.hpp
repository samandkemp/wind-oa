// The procedural model catalogue: 30 generated meshes with their metadata
// (spinning parts, suggested size, ground preset).
//
// Every model is built in memory from primitives (boxes, cylinders, lofts,
// NACA sections, bodies of revolution): nothing is downloaded or shipped,
// and imported STLs (F35, f1, ...) never are. Stylised shapes with the
// right proportions, not replicas.
//
// Axis convention (the tunnel's): +x downstream, +y up, +z lateral. Every
// model is authored front at low x, so placed unrotated it faces the wind.
// Model units are the generator's own (mm, m, or unit); placement rescales.
#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "windoa/mesh.hpp"

namespace windoa::catalogue {

enum class Ground { Air, Road, Fixed }; // aviation / rolling road / fixed floor

// A spinning sub-part in model units: centre, spin axis, rim radius and
// half-length (the capture cylinder; the rim normalises the spin rate),
// sense (+/-1), and an optional lower wall-speed cap (thin rotor blades).
struct Spinner {
    std::array<double, 3> c{};
    std::array<double, 3> axis{};
    double r = 0.0, hl = 0.0;
    int sense = 1;
    double max_wall = 0.0; // 0: the global cap
};

struct Entry {
    std::string id;    // stable key (--model ID, cache keys)
    std::string label; // menu text
    std::string group; // menu group
    geometry::Mesh (*build)() = nullptr;
    double size_frac = 0.0; // suggested length / nx (0: 64 cells)
    Ground ground = Ground::Air;
    std::vector<Spinner> spinners;
};

// In menu order (groups contiguous).
const std::vector<Entry>& entries();
const Entry* find(std::string_view id);

// The generators (exposed for tests and tools).
// An axis-aligned closed box, outward-oriented.
geometry::Mesh make_box(double cx, double cy, double cz, double sx, double sy, double sz);
geometry::Mesh make_sphere(int n_lat = 32, int n_lon = 64);
geometry::Mesh make_wing(double chord = 1.0, double span = 3.0, int n_chord = 48, int n_span = 24);
geometry::Mesh make_ahmed(double slant_deg = 25.0, int n_sub = 8);

} // namespace windoa::catalogue
