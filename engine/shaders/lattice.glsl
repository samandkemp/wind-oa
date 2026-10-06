// D3Q19 lattice tables, shared by every LBM kernel (THEORY 1.1). The host's
// copy (include/windoa/lattice.hpp) uses the same direction order, since f
// is stored direction-major: keep the two in lock-step.
//
//   0      rest                1/3
//   1-6    +-x, +-y, +-z       1/18
//   7-18   the 12 edges        1/36

const int Q = 19;
const float CS2 = 1.0 / 3.0;

const ivec3 E[19] = ivec3[](
    ivec3( 0,  0,  0),
    ivec3( 1,  0,  0), ivec3(-1,  0,  0),
    ivec3( 0,  1,  0), ivec3( 0, -1,  0),
    ivec3( 0,  0,  1), ivec3( 0,  0, -1),
    ivec3( 1,  1,  0), ivec3(-1, -1,  0),
    ivec3( 1, -1,  0), ivec3(-1,  1,  0),
    ivec3( 1,  0,  1), ivec3(-1,  0, -1),
    ivec3( 1,  0, -1), ivec3(-1,  0,  1),
    ivec3( 0,  1,  1), ivec3( 0, -1, -1),
    ivec3( 0,  1, -1), ivec3( 0, -1,  1));

const float W[19] = float[](
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0);

// Opposite direction (bounce-back).
const int OPP[19] = int[](0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17);

// Specular reflections for free-slip walls: e_y (resp. e_z) sign-flipped.
const int SPEC_Y[19] = int[](0, 1, 2, 4, 3, 5, 6, 9, 10, 7, 8, 11, 12, 13, 14, 18, 17, 16, 15);
const int SPEC_Z[19] = int[](0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16);

// Cell flags (stored one uint per cell -- no u8 alignment games).
const uint FLUID = 0u;
const uint OBSTACLE = 1u;
const uint WALL = 2u;
const uint LID = 3u;

// D3Q19 equilibrium.
float feq(int i, float rho, vec3 u) {
    float eu = dot(vec3(E[i]), u);
    return W[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * dot(u, u));
}

// Linear cell index, C order [x][y][z].
uint cell_index(int x, int y, int z, ivec3 n) {
    return uint((x * n.y + y) * n.z + z);
}

ivec3 cell_coords(uint c, ivec3 n) {
    int ci = int(c);
    return ivec3(ci / (n.y * n.z), (ci / n.z) % n.y, ci % n.z);
}

// Linear workgroup id for dispatches split over x and y (see groups_for).
uint linear_group() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}
