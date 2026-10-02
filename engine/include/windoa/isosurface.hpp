// Marching cubes: the iso-surface of a scalar field sampled at cell centres,
// as a triangle soup in lattice coordinates (the inverse of the voxeliser's
// job). CPU, deterministic: a gate and the tools call it on a field read
// back from the GPU. Specification: THEORY 10.5.
#pragma once

#include <vector>

#include "windoa/mesh.hpp"

namespace windoa {

// The surface f = iso of f[x][y][z] (C order, nx * ny * nz values; sample
// (i, j, k) sits at the cell centre (i + 0.5, j + 0.5, k + 0.5)). Each
// triangle's right-hand normal points towards decreasing f, out of the
// region f > iso. A vertex shared by neighbouring cubes is computed to the
// same bits in each (every edge is interpolated from its lower corner), so
// the soup welds exactly into a closed surface wherever the region f > iso
// stays clear of the grid boundary. Throws if f does not match the grid.
geometry::Mesh marching_cubes(const std::vector<float>& f, int nx, int ny, int nz, float iso);

} // namespace windoa
