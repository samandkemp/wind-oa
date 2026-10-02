// V16 -- marching cubes on an analytic sphere (engine marching_cubes).
// Specified in THEORY 10.5, 10.6 (docs/THEORY.md).
//
// Closed-form reference: the iso-surface f = 0 of the signed distance
// f = R - |p - c|, sampled at cell centres, is the sphere of radius R. Two
// spheres in a 64^3 grid: R 20 about (32, 32, 32), the centre on a cell
// corner so the field is symmetric about the grid, and R 12.3 about the
// off-grid centre (31.7, 32.2, 32.4), which reaches the generic cube cases.
// Pass, for each:
//   - closed and consistently oriented: after welding, every edge is used
//     by exactly two triangles, once in each direction, and no triangle
//     collapses to a repeated vertex;
//   - a topological sphere: V - E + F = 2;
//   - every vertex within 0.02 cells of the sphere (linear interpolation of
//     the distance along a cell edge errs by about h^2 / (8 R) <= 0.01);
//   - every triangle's normal points outward;
//   - area and enclosed volume within 1 % of 4 pi R^2 and 4/3 pi R^3;
//   - a plausible triangle count: between 1 and 4 per unit of area.
#include <array>
#include <cstring>
#include <map>
#include <utility>

#include "gate.hpp"
#include "windoa/isosurface.hpp"

using namespace windoa;

namespace {

constexpr int N = 64;

struct Sphere {
    const char* name;
    double cx, cy, cz, r;
};

void check(gate::Gate& g, const Sphere& s) {
    g.section(s.name);
    std::vector<float> f(std::size_t(N) * N * N);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                const double dx = x + 0.5 - s.cx, dy = y + 0.5 - s.cy, dz = z + 0.5 - s.cz;
                f[(std::size_t(x) * N + y) * N + z] =
                    float(s.r - std::sqrt(dx * dx + dy * dy + dz * dz));
            }
    const auto mesh = marching_cubes(f, N, N, N, 0.0f);
    const std::size_t nt = mesh.triangles();

    // Weld vertices by their exact bits.
    std::map<std::array<std::uint32_t, 3>, int> index;
    std::vector<std::array<double, 3>> verts;
    std::vector<std::array<int, 3>> tris(nt);
    for (std::size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) {
            std::array<std::uint32_t, 3> key;
            std::memcpy(key.data(), &mesh.xyz[t * 9 + k * 3], sizeof key);
            const auto [it, fresh] = index.emplace(key, int(verts.size()));
            if (fresh)
                verts.push_back({mesh.xyz[t * 9 + k * 3], mesh.xyz[t * 9 + k * 3 + 1],
                                 mesh.xyz[t * 9 + k * 3 + 2]});
            tris[t][k] = it->second;
        }

    // Topology: each directed edge once, and its reverse once.
    std::map<std::pair<int, int>, int> directed;
    std::size_t collapsed = 0;
    for (const auto& t : tris) {
        if (t[0] == t[1] || t[1] == t[2] || t[2] == t[0]) {
            ++collapsed;
            continue;
        }
        for (int k = 0; k < 3; ++k)
            ++directed[{t[k], t[(k + 1) % 3]}];
    }
    std::size_t bad_edges = 0;
    for (const auto& [e, n] : directed) {
        const auto rev = directed.find({e.second, e.first});
        if (n != 1 || rev == directed.end() || rev->second != 1)
            ++bad_edges;
    }
    const long euler = long(verts.size()) - long(directed.size() / 2) + long(nt);
    g.check(nt > 0 && collapsed == 0 && bad_edges == 0 && euler == 2,
            "%zu triangles, %zu vertices: closed and oriented (%zu bad edges, %zu collapsed), "
            "V - E + F = %ld (2)",
            nt, verts.size(), bad_edges, collapsed, euler);

    // Geometry against the sphere.
    double r_err = 0.0, area = 0.0, volume = 0.0;
    std::size_t inward = 0;
    for (const auto& v : verts)
        r_err = std::max(r_err, std::abs(std::hypot(v[0] - s.cx, v[1] - s.cy, v[2] - s.cz) - s.r));
    for (const auto& t : tris) {
        double p[3][3];
        for (int k = 0; k < 3; ++k) {
            p[k][0] = verts[t[k]][0] - s.cx;
            p[k][1] = verts[t[k]][1] - s.cy;
            p[k][2] = verts[t[k]][2] - s.cz;
        }
        const double e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
        const double e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
        const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                             e1[0] * e2[1] - e1[1] * e2[0]};
        const double mid[3] = {p[0][0] + p[1][0] + p[2][0], p[0][1] + p[1][1] + p[2][1],
                               p[0][2] + p[1][2] + p[2][2]};
        if (n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2] <= 0.0)
            ++inward;
        area += 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        // signed tetrahedron volume from the centre (divergence theorem)
        volume += (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) +
                   p[0][1] * (p[1][2] * p[2][0] - p[1][0] * p[2][2]) +
                   p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0])) /
                  6.0;
    }
    const double a_exact = 4.0 * gate::kPi * s.r * s.r;
    const double v_exact = 4.0 / 3.0 * gate::kPi * s.r * s.r * s.r;
    const double e_a = area / a_exact - 1.0, e_v = volume / v_exact - 1.0;
    const double per_area = double(nt) / a_exact;
    g.check(r_err < 0.02, "max radial error %.4f cells (< 0.02)", r_err);
    g.check(inward == 0, "outward normals: %zu of %zu triangles point inward", inward, nt);
    g.check(std::abs(e_a) < 0.01 && std::abs(e_v) < 0.01,
            "area %.1f vs %.1f (%+.3f %%), volume %.1f vs %.1f (%+.3f %%) (each < 1 %%)", area,
            a_exact, e_a * 100.0, volume, v_exact, e_v * 100.0);
    g.check(per_area > 1.0 && per_area < 4.0, "%.2f triangles per unit area (1 - 4)", per_area);
}

} // namespace

int main() {
    return gate::run("V16_marching_cubes", [](gate::Gate& g) {
        check(g, {"R 20, centre on a cell corner", 32.0, 32.0, 32.0, 20.0});
        check(g, {"R 12.3, off-grid centre", 31.7, 32.2, 32.4, 12.3});
    });
}
