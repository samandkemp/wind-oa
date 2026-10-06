#include "windoa/shapes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <thread>

namespace windoa::shapes {

namespace {

// D3Q19 directions, in the order of engine/shaders/lattice.glsl.

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
                    flags[Grid{nx, ny, nz}.index(x, y, z)] = lbm::OBSTACLE;
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
                flags[Grid{nx, ny, nz}.index(x, y, z)] = lbm::OBSTACLE;
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
                flags[Grid{nx, ny, nz}.index(x, y, z)] = lbm::OBSTACLE;
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
                if (flags[Grid{nx, ny, nz}.index(i, j, k)] != lbm::FLUID)
                    continue;
                const double p[3] = {i + 0.5, j + 0.5, k + 0.5};
                const double oc[3] = {p[0] - cx, p[1] - cy, p[2] - cz};
                // only cells near the sphere can have boundary links
                if (std::abs(std::sqrt(oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]) - radius) >=
                    2.5)
                    continue;
                for (int d = 1; d < 19; ++d) {
                    const int si = i + lattice::E[std::size_t(d)][0],
                              sj = j + lattice::E[std::size_t(d)][1],
                              sk = k + lattice::E[std::size_t(d)][2];
                    if (si < 0 || sj < 0 || sk < 0 || si >= nx || sj >= ny || sk >= nz)
                        continue;
                    if (flags[Grid{nx, ny, nz}.index(si, sj, sk)] != lbm::OBSTACLE)
                        continue;
                    // |p0 + t e - c|^2 = r^2: a quadratic in t
                    const double a = lattice::E[std::size_t(d)][0] * lattice::E[std::size_t(d)][0] +
                                     lattice::E[std::size_t(d)][1] * lattice::E[std::size_t(d)][1] +
                                     lattice::E[std::size_t(d)][2] * lattice::E[std::size_t(d)][2];
                    const double b = 2.0 * (oc[0] * lattice::E[std::size_t(d)][0] +
                                            oc[1] * lattice::E[std::size_t(d)][1] +
                                            oc[2] * lattice::E[std::size_t(d)][2]);
                    const double c =
                        oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
                    const double disc = b * b - 4 * a * c;
                    if (disc < 0)
                        continue; // grazing: keep half-way
                    const double t = (-b - std::sqrt(disc)) / (2 * a);
                    if (t >= 0.0 && t <= 1.0) {
                        // round half to even: std::nearbyint in the default rounding mode
                        q[std::size_t(d) * cells + Grid{nx, ny, nz}.index(i, j, k)] =
                            std::uint8_t(std::clamp(std::nearbyint(t * 255.0), 1.0, 255.0));
                        ++count;
                    }
                }
            }
    if (filled)
        *filled = count;
    return q;
}

namespace {

// The triangles binned on a coarse grid (kBin cells a side), so a link only
// tests the few triangles near it.
struct TriangleBins {
    static constexpr int kBin = 4;
    int bx, by, bz;
    std::vector<std::vector<int>> bins;
    const geometry::Mesh& mesh;

    TriangleBins(const geometry::Mesh& m, int nx, int ny, int nz)
        : bx(nx / kBin + 1), by(ny / kBin + 1), bz(nz / kBin + 1), bins(std::size_t(bx) * by * bz),
          mesh(m) {
        for (std::size_t t = 0; t < m.triangles(); ++t) {
            const float* v = &m.xyz[9 * t];
            int lo[3], hi[3];
            const int dims[3] = {bx, by, bz};
            for (int a = 0; a < 3; ++a) {
                const float mn = std::min({v[a], v[3 + a], v[6 + a]});
                const float mx = std::max({v[a], v[3 + a], v[6 + a]});
                lo[a] = std::clamp(int(std::floor(mn)) / kBin, 0, dims[a] - 1);
                hi[a] = std::clamp(int(std::floor(mx)) / kBin, 0, dims[a] - 1);
            }
            for (int i = lo[0]; i <= hi[0]; ++i)
                for (int j = lo[1]; j <= hi[1]; ++j)
                    for (int k = lo[2]; k <= hi[2]; ++k)
                        bins[(std::size_t(i) * by + j) * bz + k].push_back(int(t));
        }
    }

    // Smallest t in [0, 1] where p + t d crosses a triangle (Moller-Trumbore),
    // or -1.
    double first_crossing(const std::array<double, 3>& p, const std::array<double, 3>& d) const {
        int lo[3], hi[3];
        const int dims[3] = {bx, by, bz};
        for (int a = 0; a < 3; ++a) {
            const double e = std::min(p[a], p[a] + d[a]), f = std::max(p[a], p[a] + d[a]);
            lo[a] = std::clamp(int(std::floor(e)) / kBin, 0, dims[a] - 1);
            hi[a] = std::clamp(int(std::floor(f)) / kBin, 0, dims[a] - 1);
        }
        double best = 2.0;
        for (int i = lo[0]; i <= hi[0]; ++i)
            for (int j = lo[1]; j <= hi[1]; ++j)
                for (int k = lo[2]; k <= hi[2]; ++k)
                    for (const int t : bins[(std::size_t(i) * by + j) * bz + k]) {
                        const float* v = &mesh.xyz[9 * std::size_t(t)];
                        const std::array<double, 3> a{v[0], v[1], v[2]};
                        const std::array<double, 3> e1{v[3] - a[0], v[4] - a[1], v[5] - a[2]};
                        const std::array<double, 3> e2{v[6] - a[0], v[7] - a[1], v[8] - a[2]};
                        const std::array<double, 3> h{d[1] * e2[2] - d[2] * e2[1],
                                                      d[2] * e2[0] - d[0] * e2[2],
                                                      d[0] * e2[1] - d[1] * e2[0]};
                        const double det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
                        if (std::abs(det) < 1e-14)
                            continue; // the link runs along the triangle's plane
                        const double inv = 1.0 / det;
                        const std::array<double, 3> s{p[0] - a[0], p[1] - a[1], p[2] - a[2]};
                        const double u = inv * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
                        if (u < 0.0 || u > 1.0)
                            continue;
                        const std::array<double, 3> qv{s[1] * e1[2] - s[2] * e1[1],
                                                       s[2] * e1[0] - s[0] * e1[2],
                                                       s[0] * e1[1] - s[1] * e1[0]};
                        const double w = inv * (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]);
                        if (w < 0.0 || u + w > 1.0)
                            continue;
                        const double tt = inv * (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]);
                        if (tt >= 0.0 && tt <= 1.0)
                            best = std::min(best, tt);
                    }
        return best <= 1.0 ? best : -1.0;
    }
};

} // namespace

LinkSet link_set(const std::vector<std::uint8_t>& flags, const std::vector<float>& phi, int nx,
                 int ny, int nz, const geometry::Mesh* mesh, int* filled, int* exact) {
    const Grid grid{nx, ny, nz};
    const std::size_t plane = std::size_t(ny) * nz;
    // The OBSTACLE cells' x-range: a wall link starts in a fluid cell at most
    // one plane off it.
    int xlo = nx, xhi = -1;
    for (int i = 0; i < nx; ++i) {
        const auto first = flags.begin() + std::ptrdiff_t(i * plane);
        if (std::find(first, first + std::ptrdiff_t(plane), std::uint8_t(lbm::OBSTACLE)) !=
            first + std::ptrdiff_t(plane)) {
            xlo = std::min(xlo, i);
            xhi = i;
        }
    }
    LinkSet slab;
    if (xhi < 0) { // nothing solid: every link half-way
        if (filled)
            *filled = 0;
        if (exact)
            *exact = 0;
        return slab;
    }
    slab.x0 = std::max(0, xlo - 1);
    slab.x1 = std::min(nx, xhi + 2);
    const std::size_t cells = std::size_t(nx) * plane;
    std::unique_ptr<TriangleBins> bins;
    if (mesh && !mesh->empty())
        bins = std::make_unique<TriangleBins>(*mesh, nx, ny, nz);
    // The planes split among threads (the crossing tests are independent and
    // read-only); each part keeps the sequential order, and the parts are
    // joined in plane order, so the list is the single-threaded one.
    struct Part {
        std::vector<std::uint32_t> index;
        std::vector<std::uint8_t> q;
        int count = 0, crossed = 0;
    };
    const int planes = slab.x1 - slab.x0;
    const int n_parts =
        std::clamp(int(std::thread::hardware_concurrency()), 1, std::min(16, planes));
    std::vector<Part> parts(static_cast<std::size_t>(n_parts));
    auto work = [&](int part) {
        Part& out = parts[std::size_t(part)];
        const int i0 = slab.x0 + planes * part / n_parts;
        const int i1 = slab.x0 + planes * (part + 1) / n_parts;
        int& count = out.count;
        int& crossed = out.crossed;
        for (int i = i0; i < i1; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k) {
                    const std::size_t c = grid.index(i, j, k);
                    if (flags[c] != lbm::FLUID || phi[c] > 2.5f) // boundary links lie within 1.8
                        continue;
                    for (int d = 1; d < 19; ++d) {
                        const int si = i + lattice::E[std::size_t(d)][0],
                                  sj = j + lattice::E[std::size_t(d)][1],
                                  sk = k + lattice::E[std::size_t(d)][2];
                        if (si < 0 || sj < 0 || sk < 0 || si >= nx || sj >= ny || sk >= nz)
                            continue;
                        const std::size_t s = grid.index(si, sj, sk);
                        if (flags[s] != lbm::OBSTACLE)
                            continue;
                        double t = -1.0;
                        if (bins)
                            t = bins->first_crossing({i + 0.5, j + 0.5, k + 0.5},
                                                     {double(lattice::E[std::size_t(d)][0]),
                                                      double(lattice::E[std::size_t(d)][1]),
                                                      double(lattice::E[std::size_t(d)][2])});
                        if (t >= 0.0) {
                            ++crossed;
                        } else {
                            const double pf = phi[c], ps = phi[s];
                            if (!(pf >= 0.0 && ps < 0.0))
                                continue; // inconsistent with the flags: keep half-way
                            t = pf / (pf - ps);
                        }
                        const auto v =
                            std::uint8_t(std::clamp(std::nearbyint(t * 255.0), 1.0, 255.0));
                        if (v != 128) { // 128 is half-way already
                            out.index.push_back(std::uint32_t(std::size_t(d) * cells + c));
                            out.q.push_back(v);
                        }
                        ++count;
                    }
                }
    };
    std::vector<std::thread> pool;
    for (int part = 1; part < n_parts; ++part)
        pool.emplace_back(work, part);
    work(0);
    for (std::thread& t : pool)
        t.join();
    int count = 0, crossed = 0;
    for (const Part& p : parts) {
        slab.index.insert(slab.index.end(), p.index.begin(), p.index.end());
        slab.q.insert(slab.q.end(), p.q.begin(), p.q.end());
        count += p.count;
        crossed += p.crossed;
    }
    if (filled)
        *filled = count;
    if (exact)
        *exact = crossed;
    return slab;
}

std::vector<std::uint8_t> link_fractions(const std::vector<std::uint8_t>& flags,
                                         const std::vector<float>& phi, int nx, int ny, int nz,
                                         const geometry::Mesh* mesh, int* filled, int* exact) {
    const std::size_t cells = std::size_t(nx) * ny * nz;
    std::vector<std::uint8_t> q(19 * cells, 128);
    const LinkSet set = link_set(flags, phi, nx, ny, nz, mesh, filled, exact);
    for (std::size_t k = 0; k < set.index.size(); ++k)
        q[set.index[k]] = set.q[k];
    return q;
}

} // namespace windoa::shapes
