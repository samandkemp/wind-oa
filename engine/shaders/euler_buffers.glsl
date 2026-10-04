// The one descriptor layout every Euler kernel shares (bound once per
// solver; each kernel touches what it needs).
layout(std430, binding = 0) buffer BU { float U[]; };        // 5 per cell, conserved
layout(std430, binding = 1) buffer BU0 { float U0[]; };      // RK stage copy
layout(std430, binding = 2) buffer BR { float R[]; };        // residual
layout(std430, binding = 3) readonly buffer BFlags { uint flags[]; };
layout(std430, binding = 4) readonly buffer BPhi { float phi[]; };  // signed distance, - in solid
layout(std430, binding = 5) buffer BGs { float gs[]; };      // 5 per cell: ghost primitive state
layout(std430, binding = 6) buffer BGn { vec4 gn[]; };       // ghost normal xyz, w = ok (1 / 0)
layout(std430, binding = 7) buffer BGpw { float gpw[]; };    // wall pressure at the true wall
layout(std430, binding = 8) buffer BDt { vec4 dtbuf; };      // dt, time, -, -
layout(std430, binding = 9) buffer BPart { vec4 partials[]; };  // per-workgroup reductions
layout(std430, binding = 10) buffer BMacro { vec4 macro[]; };   // renderer: u.xyz, p
layout(std430, binding = 11) buffer BRho { float rho_out[]; };  // renderer: rho
layout(std430, binding = 12) readonly buffer BPorts { float port_w[]; };  // 5 per port: rho, u, v, w, p

S load_u(uint c) { return S(vec4(U[5 * c], U[5 * c + 1], U[5 * c + 2], U[5 * c + 3]), U[5 * c + 4]); }
S load_u0(uint c) { return S(vec4(U0[5 * c], U0[5 * c + 1], U0[5 * c + 2], U0[5 * c + 3]), U0[5 * c + 4]); }
S load_r(uint c) { return S(vec4(R[5 * c], R[5 * c + 1], R[5 * c + 2], R[5 * c + 3]), R[5 * c + 4]); }
S load_gs(uint c) { return S(vec4(gs[5 * c], gs[5 * c + 1], gs[5 * c + 2], gs[5 * c + 3]), gs[5 * c + 4]); }
void store_u(uint c, S s) {
    U[5 * c] = s.v.x; U[5 * c + 1] = s.v.y; U[5 * c + 2] = s.v.z; U[5 * c + 3] = s.v.w; U[5 * c + 4] = s.e;
}
void store_u0(uint c, S s) {
    U0[5 * c] = s.v.x; U0[5 * c + 1] = s.v.y; U0[5 * c + 2] = s.v.z; U0[5 * c + 3] = s.v.w; U0[5 * c + 4] = s.e;
}
void store_r(uint c, S s) {
    R[5 * c] = s.v.x; R[5 * c + 1] = s.v.y; R[5 * c + 2] = s.v.z; R[5 * c + 3] = s.v.w; R[5 * c + 4] = s.e;
}
void store_gs(uint c, S s) {
    gs[5 * c] = s.v.x; gs[5 * c + 1] = s.v.y; gs[5 * c + 2] = s.v.z; gs[5 * c + 3] = s.v.w; gs[5 * c + 4] = s.e;
}
