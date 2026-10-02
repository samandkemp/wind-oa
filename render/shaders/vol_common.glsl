// Shared by the volume renderer's kernels. Cell layout matches the engine
// (engine/shaders/lattice.glsl): C order [x][y][z], one uint flag per cell,
// macro = vec4(u.xyz, rho).

const uint FLUID = 0u;

uint cell_index(int x, int y, int z, ivec3 n) {
    return uint((x * n.y + y) * n.z + z);
}

ivec3 cell_coords(uint c, ivec3 n) {
    int ci = int(c);
    return ivec3(ci / (n.y * n.z), (ci / n.z) % n.y, ci % n.z);
}

// Linear workgroup id for dispatches split over x and y (engine groups_for).
uint linear_group() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// Field modes (render::Field) and colour modes.
const int MODE_SPEED = 0;
const int MODE_PRESSURE = 1;
const int MODE_VORTICITY = 2;
const int MODE_VORT_X = 3;
const int MODE_MACH = 4;
const int MODE_SCHLIEREN = 5;

const int CMODE_COOLWARM = 0;
const int CMODE_VORT = 1;
const int CMODE_GREY = 2;
const int CMODE_MACH = 3;

// Solid occupancy block edge, cells (must match render/volume.cpp).
const int OCC_BLOCK = 4;
