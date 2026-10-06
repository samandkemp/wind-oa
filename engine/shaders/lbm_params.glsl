// Push constants shared by every LBM kernel (96 bytes; Vulkan guarantees
// 128). Must match windoa::lbm::Params in engine/src/lbm.cpp -- all vec4 /
// ivec4 so the std430 layout has no padding surprises.
layout(push_constant) uniform Params {
    ivec4 dims;        // nx, ny, nz, step: the partials slice's first group
    vec4 flow;         // u_inlet, tau0, smag_pref = 18 sqrt(2) Cs^2, (unused)
    vec4 g;            // body force per unit volume (Guo), xyz
    vec4 u_lid;        // LID wall velocity, xyz; w: the fused dye's omega_minus
    vec4 torque_ref;   // moment reference point, xyz; w: the fused dye's omega_plus
    vec4 aux;          // per-kernel: init (u.xyz, rho0); stats (plane x or -1);
                       // scale (k)
} P;
