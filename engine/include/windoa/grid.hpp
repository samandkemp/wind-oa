// Cell indexing in the engine's order, [x][y][z] with z fastest (THEORY,
// "Conventions"): the one place the host-side arithmetic lives.
#pragma once

#include <array>
#include <cstddef>

namespace windoa {

struct Grid {
    int nx = 0, ny = 0, nz = 0;

    std::size_t cells() const { return std::size_t(nx) * ny * nz; }
    std::size_t index(int x, int y, int z) const { return (std::size_t(x) * ny + y) * nz + z; }
    std::array<int, 3> coords(std::size_t c) const {
        const std::size_t plane = std::size_t(ny) * nz;
        return {int(c / plane), int((c / std::size_t(nz)) % std::size_t(ny)),
                int(c % std::size_t(nz))};
    }
    bool contains(int x, int y, int z) const {
        return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
    }
};

} // namespace windoa
