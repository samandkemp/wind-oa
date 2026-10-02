#include "windoa/shapes.hpp"

#include <algorithm>
#include <cmath>

namespace windoa::shapes {

namespace {

// D3Q19 directions, in the order of engine/shaders/lattice.glsl.
constexpr int kE[19][3] = {{0, 0, 0},  {1, 0, 0},   {-1, 0, 0},  {0, 1, 0},   {0, -1, 0},
                           {0, 0, 1},  {0, 0, -1},  {1, 1, 0},   {-1, -1, 0}, {1, -1, 0},
                           {-1, 1, 0}, {1, 0, 1},   {-1, 0, -1}, {1, 0, -1},  {-1, 0, 1},
                           {0, 1, 1},  {0, -1, -1}, {0, 1, -1},  {0, -1, 1}};

std::size_t idx(int x, int y, int z, int ny, int nz) {
    return (std::size_t(x) * ny + y) * nz + z;
}

} // namespace

int add_sphere(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, double cx, double cy,
               double cz, double radius) {
    int n = 0;
    const double r2 = radius * radius;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                const double dx = x - cx, dy = y - cy, dz = z - cz;
                if (dx * dx + dy * dy + dz * dz <= r2) {
                    flags[idx(x, y, z, ny, nz)] = lbm::OBSTACLE;
                    ++n;
                }
            }
    return n;
}

int add_cylinder_z(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, double cx, double cy,
                   double radius) {
    int n = 0;
    const double r2 = radius * radius;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y) {
            const double dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy > r2)
                continue;
            for (int z = 0; z < nz; ++z) {
                flags[idx(x, y, z, ny, nz)] = lbm::OBSTACLE;
                ++n;
            }
        }
    return n;
}

int add_box(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, int x0, int x1, int y0,
            int y1, int z0, int z1) {
    (void)nx;
    for (int x = x0; x <= x1; ++x)
        for (int y = y0; y <= y1; ++y)
            for (int z = z0; z <= z1; ++z)
                flags[idx(x, y, z, ny, nz)] = lbm::OBSTACLE;
    return (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1);
}

std::vector<std::uint8_t> sphere_link_fractions(const std::vector<std::uint8_t>& flags, int nx,
                                                int ny, int nz, double cx, double cy, double cz,
                                                double radius, int* filled) {
    const std::size_t cells = std::size_t(nx) * ny * nz;
    std::vector<std::uint8_t> q(19 * cells, 128);
    int count = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                if (flags[idx(i, j, k, ny, nz)] != lbm::FLUID)
                    continue;
                const double p[3] = {i + 0.5, j + 0.5, k + 0.5};
                const double oc[3] = {p[0] - cx, p[1] - cy, p[2] - cz};
                // only cells near the sphere can have boundary links
                if (std::abs(std::sqrt(oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]) - radius) >=
                    2.5)
                    continue;
                for (int d = 1; d < 19; ++d) {
                    const int si = i + kE[d][0], sj = j + kE[d][1], sk = k + kE[d][2];
                    if (si < 0 || sj < 0 || sk < 0 || si >= nx || sj >= ny || sk >= nz)
                        continue;
                    if (flags[idx(si, sj, sk, ny, nz)] != lbm::OBSTACLE)
                        continue;
                    // |p0 + t e - c|^2 = r^2: a quadratic in t
                    const double a =
                        kE[d][0] * kE[d][0] + kE[d][1] * kE[d][1] + kE[d][2] * kE[d][2];
                    const double b = 2.0 * (oc[0] * kE[d][0] + oc[1] * kE[d][1] + oc[2] * kE[d][2]);
                    const double c =
                        oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
                    const double disc = b * b - 4 * a * c;
                    if (disc < 0)
                        continue; // grazing: keep half-way
                    const double t = (-b - std::sqrt(disc)) / (2 * a);
                    if (t >= 0.0 && t <= 1.0) {
                        // round half to even: std::nearbyint in the default rounding mode
                        q[std::size_t(d) * cells + idx(i, j, k, ny, nz)] =
                            std::uint8_t(std::clamp(std::nearbyint(t * 255.0), 1.0, 255.0));
                        ++count;
                    }
                }
            }
    if (filled)
        *filled = count;
    return q;
}

} // namespace windoa::shapes
