#include "windoa/isosurface.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "mc_tables.hpp"

namespace windoa {

namespace {

// Corner offsets and edge endpoints, in the tables' numbering (mc_tables.hpp).
constexpr int kCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
constexpr int kEdge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                              {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

int corner_sum(int c) {
    return kCorner[c][0] + kCorner[c][1] + kCorner[c][2];
}

} // namespace

geometry::Mesh marching_cubes(const std::vector<float>& f, int nx, int ny, int nz, float iso) {
    if (nx < 2 || ny < 2 || nz < 2 || f.size() != std::size_t(nx) * ny * nz)
        throw std::invalid_argument("marching_cubes: the field does not match the grid");
    auto at = [&](int x, int y, int z) { return f[(std::size_t(x) * ny + y) * nz + z]; };

    geometry::Mesh mesh;
    for (int x = 0; x + 1 < nx; ++x)
        for (int y = 0; y + 1 < ny; ++y)
            for (int z = 0; z + 1 < nz; ++z) {
                const int base[3] = {x, y, z};
                float v[8];
                int cube = 0;
                for (int c = 0; c < 8; ++c) {
                    v[c] = at(x + kCorner[c][0], y + kCorner[c][1], z + kCorner[c][2]);
                    if (v[c] < iso)
                        cube |= 1 << c;
                }
                const std::uint16_t crossed = mc::kEdgeTable[cube];
                if (crossed == 0)
                    continue;

                geometry::Vec3 p[12];
                for (int e = 0; e < 12; ++e) {
                    if (!(crossed >> e & 1))
                        continue;
                    int a = kEdge[e][0], b = kEdge[e][1];
                    // From the lower corner: the neighbour sharing this edge
                    // then computes the same bits, and the soup welds exactly.
                    if (corner_sum(a) > corner_sum(b))
                        std::swap(a, b);
                    // v[a] and v[b] straddle iso, so the denominator is not zero.
                    const float t = std::clamp((iso - v[a]) / (v[b] - v[a]), 0.0f, 1.0f);
                    for (int d = 0; d < 3; ++d) {
                        const float pa = float(base[d] + kCorner[a][d]) + 0.5f;
                        const float pb = float(base[d] + kCorner[b][d]) + 0.5f;
                        p[e][d] = pa + t * (pb - pa);
                    }
                }
                const std::int8_t* tri = mc::kTriTable[cube];
                for (int k = 0; tri[k] >= 0; k += 3)
                    mesh.add(p[tri[k]], p[tri[k + 1]], p[tri[k + 2]]);
            }
    return mesh;
}

} // namespace windoa
