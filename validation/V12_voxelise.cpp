// V12 -- the voxeliser's inside / outside rules.
// Specified in THEORY 5.7 (docs/THEORY.md).
//
//   A. Closed form: a box rotated 30 / 20 / 10 deg voxelises to exactly the
//      cells whose centres are inside it (the half-way bounce-back
//      convention), and the thin-feature pass adds nothing.
//   B. Invariant: a plate 0.3 cells thick, tilted, spanning the grid,
//      becomes a continuous sheet -- every surface sample has solid within
//      one cell, and no D3Q19 fluid path leads through it (the outer x / z
//      ring is closed: the plate runs off the grid there). Surface samples:
//      200,000, area-weighted, fixed seed.
//   C. Invariant: two overlapping closed boxes fill their union (parity
//      punched a hole at every junction); the only extra cells allowed are
//      a 1-cell fillet in the concave corner where the parts meet.
#include <random>

#include "cases.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/mesh.hpp"
#include "windoa/voxeliser.hpp"

using namespace windoa;

namespace {

constexpr int N = 64;
using Mask = std::vector<std::uint8_t>; // 1 = solid, [x][y][z]

std::size_t at(int x, int y, int z) {
    return (std::size_t(x) * N + y) * N + z;
}

struct Vox {
    Mask solid;
    Voxeliser::Stats st;
};

Vox voxelise(Voxeliser& v, const geometry::Mesh& m) {
    std::vector<std::uint8_t> flags(std::size_t(N) * N * N, lbm::FLUID);
    Vox r;
    r.st = v.voxelise(m, flags);
    r.solid.resize(flags.size());
    for (std::size_t i = 0; i < flags.size(); ++i)
        r.solid[i] = flags[i] == lbm::OBSTACLE;
    return r;
}

std::size_t count(const Mask& m) {
    std::size_t n = 0;
    for (const auto v : m)
        n += v;
    return n;
}

// a moved by integer offset d, zero-filled (no wrap)
Mask shift(const Mask& a, int dx, int dy, int dz) {
    Mask out(a.size(), 0);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                const int sx = x - dx, sy = y - dy, sz = z - dz;
                if (sx >= 0 && sy >= 0 && sz >= 0 && sx < N && sy < N && sz < N)
                    out[at(x, y, z)] = a[at(sx, sy, sz)];
            }
    return out;
}

} // namespace

int main() {
    return gate::run("V12_voxelise", [](gate::Gate& g) {
        Context ctx;
        Voxeliser vox(ctx, N, N, N);

        // -- A: rotated box vs its exact centre-inside set
        g.section("A: rotated box vs its exact centre-inside set");
        const geometry::Vec3 C{31.7f, 32.2f, 32.4f};
        const double EXT[3] = {30.3, 14.6, 9.8};
        const float ANG[3] = {30.0f, 20.0f, 10.0f};
        const auto box =
            geometry::transform(catalogue::make_box(C[0], C[1], C[2], EXT[0], EXT[1], EXT[2]),
                                ANG[0], ANG[1], ANG[2], C);
        // exact reference: the same rotation applied to the three unit axes
        geometry::Mesh unit;
        unit.add({1, 0, 0}, {0, 1, 0}, {0, 0, 1});
        const auto axes =
            geometry::transform(unit, ANG[0], ANG[1], ANG[2], geometry::Vec3{0, 0, 0});
        Mask inside(std::size_t(N) * N * N, 0);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    const double rel[3] = {x + 0.5 - C[0], y + 0.5 - C[1], z + 0.5 - C[2]};
                    bool in = true;
                    for (int k = 0; k < 3; ++k) {
                        const float* a = &axes.xyz[std::size_t(k) * 3];
                        in = in &&
                             std::abs(rel[0] * a[0] + rel[1] * a[1] + rel[2] * a[2]) < EXT[k] / 2;
                    }
                    inside[at(x, y, z)] = in;
                }
        const auto a = voxelise(vox, box);
        std::size_t n_diff = 0;
        for (std::size_t i = 0; i < inside.size(); ++i)
            n_diff += a.solid[i] != inside[i];
        g.check(n_diff == 0 && a.st.n_thin == 0,
                "%zu cells expected, %zu voxelised, %zu differ; thin pass added %zu", count(inside),
                count(a.solid), n_diff, a.st.n_thin);

        // -- B: a 0.3-cell plate becomes a continuous sheet
        g.section("B: a plate 0.3 cells thick spanning the box (tilted)");
        const geometry::Vec3 PC{32.0f, 32.3f, 32.0f};
        const auto plate = geometry::transform(
            catalogue::make_box(PC[0], PC[1], PC[2], 90.0, 0.3, 90.0), 0.0f, 12.0f, 7.0f, PC);
        const auto b = voxelise(vox, plate);
        // area-weighted surface samples inside the grid: solid within 1 cell
        const std::size_t nt = plate.triangles();
        std::vector<double> cum(nt);
        auto vtx = [&](std::size_t t, int v, int k) {
            return double(plate.xyz[t * 9 + v * 3 + k]);
        };
        double total = 0.0;
        for (std::size_t t = 0; t < nt; ++t) {
            double e1[3], e2[3];
            for (int k = 0; k < 3; ++k) {
                e1[k] = vtx(t, 1, k) - vtx(t, 0, k);
                e2[k] = vtx(t, 2, k) - vtx(t, 0, k);
            }
            const double cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2],
                         cz = e1[0] * e2[1] - e1[1] * e2[0];
            total += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
            cum[t] = total;
        }
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::size_t samples = 0, near = 0;
        for (int s = 0; s < 200'000; ++s) {
            const auto t = std::size_t(std::lower_bound(cum.begin(), cum.end(), u01(rng) * total) -
                                       cum.begin());
            const double sq = std::sqrt(u01(rng)), r2 = u01(rng);
            double p[3];
            for (int k = 0; k < 3; ++k)
                p[k] =
                    (1 - sq) * vtx(t, 0, k) + sq * (1 - r2) * vtx(t, 1, k) + sq * r2 * vtx(t, 2, k);
            if (p[0] < 1.0 || p[1] < 1.0 || p[2] < 1.0 || p[0] >= N - 1.0 || p[1] >= N - 1.0 ||
                p[2] >= N - 1.0)
                continue;
            ++samples;
            const int c[3] = {int(std::floor(p[0])), int(std::floor(p[1])), int(std::floor(p[2]))};
            bool hit = false;
            for (int dx = -1; dx <= 1 && !hit; ++dx)
                for (int dy = -1; dy <= 1 && !hit; ++dy)
                    for (int dz = -1; dz <= 1 && !hit; ++dz)
                        hit = b.solid[at(std::clamp(c[0] + dx, 0, N - 1),
                                         std::clamp(c[1] + dy, 0, N - 1),
                                         std::clamp(c[2] + dz, 0, N - 1))] != 0;
            near += hit;
        }
        // flood fill from the top face along D3Q19 links, the x / z ring closed
        Mask blocked = b.solid;
        for (int y = 0; y < N; ++y)
            for (int k = 0; k < N; ++k) {
                blocked[at(0, y, k)] = blocked[at(N - 1, y, k)] = 1;
                blocked[at(k, y, 0)] = blocked[at(k, y, N - 1)] = 1;
            }
        Mask reach(blocked.size(), 0);
        std::vector<std::size_t> front;
        for (int x = 0; x < N; ++x)
            for (int z = 0; z < N; ++z)
                if (!blocked[at(x, N - 1, z)]) {
                    reach[at(x, N - 1, z)] = 1;
                    front.push_back(at(x, N - 1, z));
                }
        while (!front.empty()) {
            std::vector<std::size_t> next;
            for (const std::size_t c : front) {
                const int x = int(c / (std::size_t(N) * N)), y = int(c / N % N), z = int(c % N);
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz) {
                            const int l1 = std::abs(dx) + std::abs(dy) + std::abs(dz);
                            if (l1 == 0 || l1 > 2)
                                continue; // D3Q19: faces and edges, no corners
                            const int sx = x + dx, sy = y + dy, sz = z + dz;
                            if (sx < 0 || sy < 0 || sz < 0 || sx >= N || sy >= N || sz >= N)
                                continue;
                            const std::size_t s = at(sx, sy, sz);
                            if (!blocked[s] && !reach[s]) {
                                reach[s] = 1;
                                next.push_back(s);
                            }
                        }
            }
            front.swap(next);
        }
        std::size_t leak = 0;
        for (int x = 0; x < N; ++x)
            for (int z = 0; z < N; ++z)
                leak += reach[at(x, 0, z)];
        g.check(near == samples && b.st.n_thin > 0 && leak == 0,
                "thin pass added %zu; surface samples with solid within 1 cell %zu / %zu; far-side "
                "cells reachable along D3Q19 links %zu",
                b.st.n_thin, near, samples, leak);

        // -- C: overlapping boxes fill their union
        g.section("C: two overlapping closed boxes");
        geometry::Mesh two = catalogue::make_box(28.0, 32.0, 32.0, 20.0, 10.0, 10.0);
        two.append(catalogue::make_box(36.0, 34.0, 32.0, 20.0, 10.0, 6.0));
        const auto c = voxelise(vox, two);
        auto in_box = [&](double cx, double cy, double cz, double sx, double sy, double sz) {
            Mask m(std::size_t(N) * N * N, 0);
            for (int x = 0; x < N; ++x)
                for (int y = 0; y < N; ++y)
                    for (int z = 0; z < N; ++z)
                        m[at(x, y, z)] = std::abs(x + 0.5 - cx) < sx / 2 &&
                                         std::abs(y + 0.5 - cy) < sy / 2 &&
                                         std::abs(z + 0.5 - cz) < sz / 2;
            return m;
        };
        const Mask b1 = in_box(28, 32, 32, 20, 10, 10), b2 = in_box(36, 34, 32, 20, 10, 6);
        auto touches = [&](const Mask& m) {
            Mask out(m.size(), 0);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        const Mask s = shift(m, dx, dy, dz);
                        for (std::size_t i = 0; i < out.size(); ++i)
                            out[i] |= s[i];
                    }
            return out;
        };
        const Mask t1 = touches(b1), t2 = touches(b2);
        std::size_t uni = 0, overlap = 0, holes = 0, extra = 0, stray = 0;
        for (std::size_t i = 0; i < c.solid.size(); ++i) {
            const bool un = b1[i] || b2[i];
            uni += un;
            overlap += b1[i] && b2[i];
            holes += un && !c.solid[i];
            const bool ex = c.solid[i] && !un;
            extra += ex;
            stray += ex && !(t1[i] && t2[i]);
        }
        g.check(
            holes == 0 && stray == 0,
            "union %zu cells (overlap %zu); holes %zu; extra %zu, outside the junction fillet %zu",
            uni, overlap, holes, extra, stray);
    });
}
