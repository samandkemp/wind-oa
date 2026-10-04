// Walls, reconstruction and the HLLC flux (THEORY 8.2, 8.3, 8.5 - 8.7).
// Include after euler_common.glsl and euler_buffers.glsl.

bool solid(ivec3 c) { return in_grid(c) && flags[cidx(c)] != FLUID; }
bool is_port(ivec3 c) { return in_grid(c) && flags[cidx(c)] >= PORT0; }
// the model: its walls and its engine ports (whose momentum flux it feels)
bool model_cell(ivec3 c) { return in_grid(c) && (flags[cidx(c)] == OBSTACLE || is_port(c)); }
S port_state(ivec3 c) {
    const uint k = 5u * (flags[cidx(c)] - PORT0);
    return S(vec4(port_w[k], port_w[k + 1u], port_w[k + 2u], port_w[k + 3u]), port_w[k + 4u]);
}

// Primitive state at cell c, which may lie outside the grid: the domain
// boundary conditions live here, as ghost states.
S w_at(ivec3 c) {
    const ivec3 cc = clamp(c, ivec3(0), N() - 1);
    S w = prim(load_u(cidx(cc)));
    if (c.x < 0) {
        if (P.bc.x == 0) w = freestream();
    } else if (c.x >= N().x) {
        if (P.bc.x == 0 && P.flow.x < 1.0) w.e = 1.0 / GAMMA;  // subsonic outflow: hold p_inf
    }
    for (int a = 1; a <= 2; ++a) {
        if (c[a] < 0 || c[a] >= N()[a]) {
            const int bcv = a == 1 ? P.bc.y : P.bc.z;
            if (bcv == 0) w = freestream();
            else w.v[1 + a] = -w.v[1 + a];
        }
    }
    return w;
}

// -- wall geometry ------------------------------------------------------------------

float phi_at(ivec3 c) { return phi[cidx(clamp(c, ivec3(0), N() - 1))]; }

vec3 grad_phi(ivec3 c) {
    vec3 g;
    for (int ax = 0; ax < 3; ++ax) {
        const ivec3 e = e3(ax);
        const ivec3 lo = max(c - e, ivec3(0)), hi = min(c + e, N() - 1);
        g[ax] = (phi[cidx(hi)] - phi[cidx(lo)]) / max(float(hi[ax] - lo[ax]), 1.0);
    }
    return g;
}

// t . Hess(phi) . t at cell c: for a signed distance, the surface's normal
// curvature along t (positive on a convex body). Clamped to +-1 / cell.
float curv_along(ivec3 c, vec3 t) {
    mat3 h;
    const float f0 = phi_at(c);
    for (int a = 0; a < 3; ++a) {
        const ivec3 ea = e3(a);
        h[a][a] = phi_at(c + ea) - 2.0 * f0 + phi_at(c - ea);
        for (int b = a + 1; b < 3; ++b) {
            const ivec3 eb = e3(b);
            const float v = 0.25 * (phi_at(c + ea + eb) - phi_at(c + ea - eb) - phi_at(c - ea + eb) +
                                    phi_at(c - ea - eb));
            h[a][b] = v;
            h[b][a] = v;
        }
    }
    return clamp(dot(t, h * t), -1.0, 1.0);
}

// Trilinear primitive state at x over FLUID cells only (weights
// renormalised). Returns the weighted sum; ws = total weight.
S interp_fluid(vec3 x, out float ws) {
    const vec3 gx = x - 0.5;
    const ivec3 b = ivec3(floor(gx));
    const vec3 t = gx - vec3(b);
    S acc = S(vec4(0.0), 0.0);
    ws = 0.0;
    for (int dx = 0; dx < 2; ++dx)
        for (int dy = 0; dy < 2; ++dy)
            for (int dz = 0; dz < 2; ++dz) {
                const ivec3 c = b + ivec3(dx, dy, dz);
                if (in_grid(c) && !solid(c)) {
                    const float w = (dx == 1 ? t.x : 1.0 - t.x) * (dy == 1 ? t.y : 1.0 - t.y) *
                                    (dz == 1 ? t.z : 1.0 - t.z);
                    acc = s_add(acc, s_mul(prim(load_u(cidx(c))), w));
                    ws += w;
                }
            }
    return acc;
}

// Ghost state from the image state w (THEORY 8.7): the curvature-corrected symmetry
// technique (Dadone & Grossman, AIAA J. 2004). A plain mirror assumes
// dp/dn = 0; on a curved wall dp/dn = rho u_t^2 kappa_t, so the ghost (dn
// inside the image) gets p_I - rho u_t^2 kappa_t dn; density and speed then
// follow from constant entropy and total enthalpy; the normal velocity is
// reversed. (Without it the NACA 0012 lift converged like sqrt(h): 25 % low
// at 64 cells a chord.) kappa_t from the image point's cell (inside a thin
// body phi has a ridge whose Hessian over-corrected the nose into thrust),
// carried to the wall: a level set at distance phi has k / (1 + phi k).
S ccst(S w, vec3 n, vec3 xi, float dn) {
    const vec3 u = w.v.yzw;
    const float un = dot(u, n);
    const vec3 ut = u - un * n;
    const float utn = length(ut);
    float kt = 0.0;
    if (utn > 1e-6) {
        const vec3 t = ut / utn;
        const ivec3 ci = clamp(ivec3(floor(xi)), ivec3(0), N() - 1);
        const float ki = curv_along(ci, t);
        const float phi_i = phi[cidx(ci)];
        kt = ki / max(1.0 - phi_i * ki, 0.2);
    }
    const float rho_i = w.v.x, p_i = w.e;
    const float p_g = max(p_i - rho_i * utn * utn * kt * dn, 0.3 * p_i);
    const float rho_g = rho_i * pow(p_g / p_i, 1.0 / GAMMA);  // isentropic
    const float h_i = GAMMA / (GAMMA - 1.0) * p_i / rho_i + 0.5 * dot(u, u);
    const float q2 = max(2.0 * (h_i - GAMMA / (GAMMA - 1.0) * p_g / rho_g), 0.0);
    vec3 v = ut - un * n;  // reflected
    const float vnorm = length(v);
    if (vnorm > 1e-9) v *= sqrt(q2) / vnorm;  // same total enthalpy
    return S(vec4(rho_g, v), p_g);
}

// Does the face (fluid cf | solid cs) use cs's image ghost? Only if the
// ghost is valid, cf is on the fluid side of its normal, and the solid is
// more than one cell thick across the face: a cell with fluid on both sides
// along the axis is a 1-cell sheet (trailing edge, fin) whose distance
// normal points along the sheet -- its image reversed the streamwise
// velocity and let flow through the TE (NACA 0012 M 0.8: Cl 0.26 vs 0.35).
bool image_used(ivec3 cf, ivec3 cs) {
    if (!in_grid(cs)) return false;
    const vec4 g = gn[cidx(cs)];
    if (g.w < 0.5) return false;
    if (dot(vec3(cf - cs), g.xyz) <= 0.0) return false;
    const ivec3 co = cs + (cs - cf);  // across the solid cell
    return !(in_grid(co) && !solid(co));
}

// The state in solid cell cs for a face whose fluid side is cf (state wf):
// the image ghost if usable, else the impermeable grid-axis mirror.
// An engine port's face emits the port's exit state.
S wall_state(S wf, ivec3 cf, ivec3 cs, int ax) {
    if (is_port(cs)) return port_state(cs);
    return image_used(cf, cs) ? load_gs(cidx(cs)) : mirror(wf, ax);
}

// -- limiter (THEORY 8.2) / HLLC (THEORY 8.3) --------------------------------------------

float lim1(float a, float b) {
    if (a * b <= 0.0) return 0.0;
    if (P.opt.x == 0) return abs(a) < abs(b) ? a : b;  // minmod
    return 2.0 * a * b / (a + b);                      // van Leer
}
S limit(S a, S b) {
    return S(vec4(lim1(a.v.x, b.v.x), lim1(a.v.y, b.v.y), lim1(a.v.z, b.v.z), lim1(a.v.w, b.v.w)),
             lim1(a.e, b.e));
}

S phys_flux(S w, S u, int ax) {
    const float un = vn(w, ax);
    S f = S(vec4(w.v.x * un, w.v.x * w.v.y * un, w.v.x * w.v.z * un, w.v.x * w.v.w * un), (u.e + w.e) * un);
    f.v[1 + ax] += w.e;
    return f;
}

S hllc(S wl, S wr, int ax) {
    const S ul = cons(wl), ur = cons(wr);
    const float cl = sqrt(GAMMA * wl.e / wl.v.x), cr = sqrt(GAMMA * wr.e / wr.v.x);
    const float unl = vn(wl, ax), unr = vn(wr, ax);
    const float sl = min(unl - cl, unr - cr), sr = max(unl + cl, unr + cr);
    const float den = wl.v.x * (sl - unl) - wr.v.x * (sr - unr);
    float ss = 0.0;
    if (abs(den) > 1e-12)
        ss = (wr.e - wl.e + wl.v.x * unl * (sl - unl) - wr.v.x * unr * (sr - unr)) / den;
    if (0.0 <= sl) return phys_flux(wl, ul, ax);
    if (0.0 >= sr) return phys_flux(wr, ur, ax);
    if (0.0 <= ss) {
        const float k = wl.v.x * (sl - unl) / (sl - ss);
        S us = S(vec4(k, k * wl.v.y, k * wl.v.z, k * wl.v.w),
                 k * (ul.e / wl.v.x + (ss - unl) * (ss + wl.e / (wl.v.x * (sl - unl)))));
        us.v[1 + ax] = k * ss;  // normal momentum -> k S*
        return s_add(phys_flux(wl, ul, ax), s_mul(s_sub(us, ul), sl));
    }
    const float k = wr.v.x * (sr - unr) / (sr - ss);
    S us = S(vec4(k, k * wr.v.y, k * wr.v.z, k * wr.v.w),
             k * (ur.e / wr.v.x + (ss - unr) * (ss + wr.e / (wr.v.x * (sr - unr)))));
    us.v[1 + ax] = k * ss;
    return s_add(phys_flux(wr, ur, ax), s_mul(s_sub(us, ur), sr));
}

// HLLC flux through the face between cell I - e and cell I, walls
// included, in one code path: the four stencil states as seen from the
// fluid side (a solid becomes its wall state), one MUSCL reconstruction,
// one HLLC solve.
S flux_any(ivec3 I, int ax) {
    const ivec3 e = e3(ax);
    const ivec3 L = I - e;
    const bool sl = solid(L), sr = solid(I);
    if (sl && sr) return S(vec4(0.0), 0.0);
    const S rl = w_at(L), rr = w_at(I);
    S wl = rl, wr = rr;
    if (sl) wl = wall_state(rr, I, L, ax);
    if (sr) wr = wall_state(rl, L, I, ax);
    S wll = w_at(L - e);
    if (sl) wll = wl;
    else if (solid(L - e)) wll = wall_state(wl, L, L - e, ax);
    S wrr = w_at(I + e);
    if (sr) wrr = wr;
    else if (solid(I + e)) wrr = wall_state(wr, I, I + e, ax);
    S fl = s_add(wl, s_mul(limit(s_sub(wl, wll), s_sub(wr, wl)), 0.5));
    S fr = s_sub(wr, s_mul(limit(s_sub(wr, wl), s_sub(wrr, wr)), 0.5));
    if (fl.v.x <= 0.0 || fl.e <= 0.0 || fr.v.x <= 0.0 || fr.e <= 0.0) {
        fl = wl;  // positivity: first order
        fr = wr;
    }
    return hllc(fl, fr, ax);
}
