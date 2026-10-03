// V2 -- mass conservation and the closed-domain anchor.
// Specified in THEORY 3.8, 3.10 (docs/THEORY.md).
//
// Identity / invariant references:
//   A. the open tunnel (inlet + pressure outlet) holds its density level
//      unaided: |drift| < 1e-8 per step; and conserves mass locally: in the
//      steady flow the exact mass flux through every x-face is the same, to
//      1e-4 (the free-slip side faces neither create nor destroy mass);
//   B. enforce_mass() pins the mean density to 1 and leaves the velocity
//      field exactly unchanged (rho and rho u scale together);
//   C. with the anchor every 500 steps the closed cavity stays bounded,
//      against the raw (unanchored) moving-wall drift -- a tracked
//      diagnostic, not a gate. The raw drift (measured +2.5e-8 per step)
//      is a property of the lid boundary: a deterministic per-link mass
//      source.
#include <memory>

#include "cases.hpp"

using namespace windoa;

namespace {

// e_x per D3Q19 direction (engine/shaders/lattice.glsl order).
constexpr int kEx[lbm::Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};

void part_a(Context& ctx, gate::Gate& g) {
    g.section("A: open tunnel (inlet + pressure outlet), sphere obstacle");
    constexpr int NX = 96, NY = 48, NZ = 48;
    constexpr float U_IN = 0.05f;
    lbm::Config c;
    c.nx = NX;
    c.ny = NY;
    c.nz = NZ;
    c.tau = 0.6f;
    c.u_inlet = U_IN;
    c.smagorinsky_cs = 0.1f;
    lbm::Solver s(ctx, c);
    std::vector<std::uint8_t> flags(std::size_t(NX) * NY * NZ, lbm::FLUID);
    for (int x = 0; x < NX; ++x)
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const double r =
                    std::sqrt(double((x - 32) * (x - 32)) + (y - NY / 2.0) * (y - NY / 2.0) +
                              (z - NZ / 2.0) * (z - NZ / 2.0));
                if (r < 8.0)
                    flags[(std::size_t(x) * NY + y) * NZ + z] = lbm::OBSTACLE;
            }
    s.set_flags(flags);
    s.init_equilibrium(1.0f, {U_IN, 0.0f, 0.0f});
    s.step(2000);
    const double rho0 = s.mean_fluid_density();
    s.step(20000);
    const double rho1 = s.mean_fluid_density();
    const double drift = (rho1 - rho0) / 20000.0;
    g.note("mean rho after 2,000 steps %.6f, after 22,000 %.6f", rho0, rho1);
    g.check(std::abs(drift) < 1e-8, "open tunnel drift %+.3e per step (|.| < 1e-8)", drift);

    // Local conservation: in the steady flow the mass crossing every x-face
    // per step -- the post-collision populations that stream across it --
    // is the same at each face. A side face that took its reflected
    // population from the wrong row would create and destroy mass along the
    // walls; the inlet and outlet would hide that from the drift above.
    const auto f = s.get_state();
    const std::size_t n = s.cells();
    std::vector<double> flux;
    for (int x = 4; x < NX - 4; x += 4) {
        if (x >= 22 && x <= 42)
            continue; // faces through the sphere (x 24 - 40) hold solid cells
        double fx = 0.0;
        for (int y = 0; y < NY; ++y)
            for (int z = 0; z < NZ; ++z) {
                const std::size_t cell = (std::size_t(x) * NY + y) * NZ + z;
                const std::size_t next = cell + std::size_t(NY) * NZ; // the cell at x + 1
                for (int i = 0; i < lbm::Q; ++i) {
                    if (kEx[i] == 1)
                        fx += f[std::size_t(i) * n + cell];
                    else if (kEx[i] == -1)
                        fx -= f[std::size_t(i) * n + next];
                }
            }
        flux.push_back(fx);
    }
    double lo = flux[0], hi = flux[0], mean = 0.0;
    for (const double v : flux) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        mean += v / double(flux.size());
    }
    g.check((hi - lo) / mean < 1e-4,
            "mass flux through %zu x-faces: %.4f, spread %.1e (< 1e-4): the side faces conserve "
            "mass",
            flux.size(), mean, (hi - lo) / mean);
}

// The closed cavity: 48^2 fluid cells, tau 0.6, laminar, lid 0.1
constexpr int CN = 48, CX = CN + 2, CY = CN + 2, CZ = 4;

std::vector<std::uint8_t> cavity_flags() {
    std::vector<std::uint8_t> f(std::size_t(CX) * CY * CZ, lbm::FLUID);
    auto at = [&](int x, int y, int z) -> std::uint8_t& {
        return f[(std::size_t(x) * CY + y) * CZ + z];
    };
    for (int y = 0; y < CY; ++y)
        for (int z = 0; z < CZ; ++z)
            at(0, y, z) = at(CX - 1, y, z) = lbm::WALL;
    for (int x = 0; x < CX; ++x)
        for (int z = 0; z < CZ; ++z) {
            at(x, 0, z) = lbm::WALL;
            at(x, CY - 1, z) = lbm::LID;
        }
    return f;
}

std::unique_ptr<lbm::Solver> closed_cavity(Context& ctx) {
    lbm::Config c;
    c.nx = CX;
    c.ny = CY;
    c.nz = CZ;
    c.tau = 0.6f;
    c.smagorinsky_cs = 0.0f;
    c.mode_x = lbm::AxisX::Periodic;
    c.mode_y = lbm::AxisYZ::FreeSlip;
    c.mode_z = lbm::AxisYZ::Periodic;
    auto s = std::make_unique<lbm::Solver>(ctx, c);
    s->set_flags(cavity_flags());
    s->set_lid_velocity({0.1f, 0.0f, 0.0f});
    s->init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    return s;
}

void part_b(Context& ctx, gate::Gate& g) {
    g.section("B: closed cavity -- enforce_mass pins rho, preserves u");
    auto s = closed_cavity(ctx);
    s->step(5000);
    const auto before = s->velocity();
    const double rho_before = s->mean_fluid_density();
    s->enforce_mass(1.0);
    const auto after = s->velocity(); // no step in between: must be bit-identical
    const double rho_after = s->mean_fluid_density();
    double dv = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i)
        dv = std::max(dv, double(std::abs(after[i] - before[i])));
    g.note("mean rho before the anchor %.6f", rho_before);
    g.check(std::abs(rho_after - 1.0) < 1e-4, "mean rho after the anchor %.6f (target 1, +-1e-4)",
            rho_after);
    g.check(dv < 1e-9, "max |change in u| %.3e (< 1e-9: the rescale leaves u exact)", dv);
}

void part_c(Context& ctx, gate::Gate& g) {
    constexpr int STEPS = 40000, EVERY = 500;
    g.section("C: closed cavity over 40,000 steps, anchored vs raw");
    auto anchored = closed_cavity(ctx);
    for (int done = 0; done < STEPS; done += EVERY) {
        anchored->step(EVERY);
        anchored->enforce_mass(1.0);
    }
    const double rho_anchored = anchored->mean_fluid_density();
    auto raw = closed_cavity(ctx);
    raw->step(STEPS);
    const double rho_raw = raw->mean_fluid_density();
    const double drift_raw = (rho_raw - 1.0) / STEPS;
    g.note("UNanchored (diagnostic): mean rho %.6f, drift %+.3e per step", rho_raw, drift_raw);
    // The anchor fires every `EVERY` steps: the density must stay within about
    // one interval's drift of the target -- bounded, not growing.
    const double bound = std::max(4.0 * EVERY * std::abs(drift_raw), 1e-4);
    const double dev = std::abs(rho_anchored - 1.0);
    g.check(dev < bound && dev < 0.1 * std::abs(rho_raw - 1.0),
            "anchored |rho - 1| %.2e (< %.2e and < 10 %% of the raw excursion)", dev, bound);
}

} // namespace

int main() {
    return gate::run("V2_mass", [](gate::Gate& g) {
        Context ctx;
        part_a(ctx, g);
        part_b(ctx, g);
        part_c(ctx, g);
    });
}
