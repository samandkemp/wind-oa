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

// An engine port in model units: an exhaust that blows or an intake that
// draws, a disc of radius r on the model's surface at c with outward normal
// n (an exhaust's jet runs along n, an intake's flow against it).
// Subsonic: the face moves at speed_ratio x U along the normal, a velocity
// boundary (THEORY 3.11). Transonic exhaust: the face emits the exit state,
// Mach exit_mach along n at p_ratio and t_ratio times the freestream's
// static pressure and temperature (THEORY 8.12); a transonic intake is a
// wall.
struct Port {
    enum class Kind { Exhaust, Intake };
    Kind kind = Kind::Exhaust;
    std::array<double, 3> c{}, n{};
    double r = 0.0;
    double speed_ratio = 2.0;
    double exit_mach = 1.0, p_ratio = 1.0, t_ratio = 1.0;
};

// A rotor as an actuator line (THEORY 3.12), in model units: hub c; axis,
// the through-flow direction (a turbine slows the air along it, a propeller
// drives it along it); tip and hub radii; blades; chord and twist (degrees,
// from the rotor plane) evenly spaced root to tip; the design tip-speed
// ratio omega R / U; the sense of rotation about the axis; a thin-aerofoil
// section polar.
struct Rotor {
    std::array<double, 3> c{}, axis{1.0, 0.0, 0.0};
    double r_tip = 1.0, r_hub = 0.1;
    int blades = 3;
    std::vector<double> chord, twist_deg;
    double tsr = 6.0;
    int sense = 1;
    double cl_alpha = 6.2832, alpha_stall_deg = 12.0, cd0 = 0.012;
};

struct Entry {
    std::string id;    // stable key (--model ID, cache keys)
    std::string label; // menu text
    std::string group; // menu group
    geometry::Mesh (*build)() = nullptr;
    double size_frac = 0.0; // suggested length / nx (0: 64 cells)
    Ground ground = Ground::Air;
    std::vector<Spinner> spinners;
    std::vector<Port> ports;
    std::vector<Rotor> rotors;
    // Metres per model unit (most models are built in metres, the munitions
    // and a few others in millimetres); 0 for a shape with no real size. It
    // gives the full-size Reynolds number beside the simulated one.
    double metres_per_unit = 1.0;
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
