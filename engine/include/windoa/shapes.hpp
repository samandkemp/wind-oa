// Parametric obstacle shapes, voxelised analytically straight into a flag
// array, for the validation gates (the sandbox uses STL / catalogue meshes
// through the Voxeliser).
//
// Flags are [x][y][z] C order (lbm::Flag values); coordinates and sizes in
// cells. Note the convention: the shapes test the cell index (i, j, k),
// not the centre (i + 0.5, ...); the link fractions use centres.
// Specification: THEORY 5.5.
//
// Centres and radii are double: pass nx * 0.3, not nx * 0.3f. Cells at
// exactly r from an integer centre are on the boundary, so a float32
// centre (200 * 0.3f = 60.0000038) drops them: 5 of the 2,109 cells of
// V9's sphere, which moves its Cd by 1 %.
#pragma once

#include <cstdint>
#include <vector>

#include "windoa/lbm.hpp"

namespace windoa::shapes {

// Each returns the number of cells marked OBSTACLE.
int add_sphere(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, double cx, double cy,
               double cz, double radius);
int add_cylinder_z(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, double cx, double cy,
                   double radius); // spans the full z extent
int add_box(std::vector<std::uint8_t>& flags, int nx, int ny, int nz, int x0, int x1, int y0,
            int y1, int z0, int z1); // inclusive ranges

// Exact per-link wall distances for a sphere (interpolated bounce-back):
// for every FLUID cell with an OBSTACLE neighbour along e_d, the smallest t
// in [0, 1] with |p0 + t e_d - c| = r, quantised to u8 (t 255, clamped to
// 1..255; grazing links keep 128 = half-way). Returns [Q][x][y][z] for
// Solver::set_link_q and the number of links filled.
std::vector<std::uint8_t> sphere_link_fractions(const std::vector<std::uint8_t>& flags, int nx,
                                                int ny, int nz, double cx, double cy, double cz,
                                                double radius, int* filled = nullptr);

// The relaxation time giving Reynolds number `re` at lattice speed u over a
// length of L cells: nu = u L / Re, tau = nu / cs^2 + 1/2.
inline float tau_for_reynolds(float u, float length, float re) {
    return u * length / re * 3.0f + 0.5f;
}

} // namespace windoa::shapes
