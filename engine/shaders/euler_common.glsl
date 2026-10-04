// Compressible Euler solver: shared device code. The comments record why
// each piece is the way it is.
//
// State per cell: 5 floats U = (rho, rho u, rho v, rho w, E), stored
// cell-major (U[5 c + k]). Units: freestream rho = 1, c = 1 (p = 1/gamma),
// cell = 1, so the freestream speed is the Mach number. A 5-vector is an S
// = (vec4 v = components 0..3, float e = component 4): for a primitive W,
// v = (rho, u, v, w) and e = p; the velocity component on axis ax is
// v[1 + ax].

const float GAMMA = 1.4;
const float PHI_FAR = 4.0;
const uint FLUID = 0u;
const uint OBSTACLE = 1u;
const uint PORT0 = 8u;  // flags 8 + k: engine port k (an exhaust face; THEORY 8.12)

struct S {
    vec4 v;
    float e;
};

S s_add(S a, S b) { return S(a.v + b.v, a.e + b.e); }
S s_sub(S a, S b) { return S(a.v - b.v, a.e - b.e); }
S s_mul(S a, float k) { return S(a.v * k, a.e * k); }

// Push constants of every Euler kernel (96 bytes).
layout(push_constant) uniform Params {
    ivec4 dims;   // nx, ny, nz, axis (per-kernel)
    ivec4 bc;     // x_bc (0 tunnel, 1 transmissive), side_bc (0 farfield, 1 slip), z_bc, wall (0 mirror, 1 image)
    ivec4 opt;    // limiter (0 minmod, 1 van Leer), first (residual), mode (stage), -
    vec4 flow;    // mach, flow angle (rad), cfl, dt_cap
    vec4 aux;     // per-kernel
    vec4 aux2;
} P;

ivec3 N() { return P.dims.xyz; }
uint cidx(ivec3 c) { return uint((c.x * P.dims.y + c.y) * P.dims.z + c.z); }
bool in_grid(ivec3 c) { return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, N())); }
ivec3 e3(int ax) { return ivec3(ax == 0 ? 1 : 0, ax == 1 ? 1 : 0, ax == 2 ? 1 : 0); }
uint linear_group() { return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x; }
ivec3 coords(uint c) {
    int ci = int(c);
    return ivec3(ci / (P.dims.y * P.dims.z), (ci / P.dims.z) % P.dims.y, ci % P.dims.z);
}

// conserved <-> primitive
S prim(S u) {
    const float rho = max(u.v.x, 1e-8);
    const vec3 vel = u.v.yzw / rho;
    const float p = (GAMMA - 1.0) * (u.e - 0.5 * rho * dot(vel, vel));
    return S(vec4(rho, vel), max(p, 1e-8));
}
S cons(S w) {
    const float rho = w.v.x;
    const float ke = 0.5 * rho * dot(w.v.yzw, w.v.yzw);
    return S(vec4(rho, rho * w.v.yzw), w.e / (GAMMA - 1.0) + ke);
}
S freestream() {
    const float m = P.flow.x, a = P.flow.y;
    return S(vec4(1.0, m * cos(a), m * sin(a), 0.0), 1.0 / GAMMA);
}
float vn(S w, int ax) { return w.v[1 + ax]; }
S mirror(S w, int ax) {  // w with the ax-velocity negated
    w.v[1 + ax] = -w.v[1 + ax];
    return w;
}
