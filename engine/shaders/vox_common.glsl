// Shared by the voxeliser kernels.
// Triangles are in lattice coordinates; cell (i, j, k) has its centre at
// (i + 0.5, j + 0.5, k + 0.5).

layout(push_constant) uniform VoxParams {
    ivec4 dims;   // nx, ny, nz, n_tris
    ivec4 misc;   // x: casting axis (winding kernel)
    vec4 fparam;  // x: distance band (signed distance kernels)
} V;

layout(std430, binding = 0) readonly buffer Tris { vec4 tri[]; };  // 3 per triangle, xyz
layout(std430, binding = 1) buffer Vote { int vote[]; };           // 0..3 axis votes
layout(std430, binding = 2) buffer Shell { int shell[]; };         // packed keys / distances
layout(std430, binding = 3) buffer Flags { uint flags[]; };
layout(std430, binding = 4) buffer Phi { float phi[]; };
layout(std430, binding = 5) buffer Counter { uint counter[]; };

const uint FLUID = 0u;       // same values as lattice.glsl / lbm::Flag
const uint OBSTACLE = 1u;

const int SHELL_EMPTY = 0x7FFFFFFF;
const int TRI_BITS = 20;               // low bits of a packed shell key: triangle index
const int MAX_HITS = 64;               // surface crossings kept per grid column

// Global invocation id for dispatches split over x and y (groups_for).
int linear_id() {
    const uint group = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    return int(group * gl_WorkGroupSize.x + gl_LocalInvocationID.x);
}

vec3 tri_vertex(int t, int k) {
    return tri[3 * t + k].xyz;
}

int cell_of(ivec3 p) {
    return (p.x * V.dims.y + p.y) * V.dims.z + p.z;
}

ivec3 clamp_cell(ivec3 p) {
    return clamp(p, ivec3(0), V.dims.xyz - 1);
}

// Closest point to p on triangle abc (Ericson, Real-Time Collision
// Detection, 5.1.5) -- the Voronoi-region walk.
vec3 closest_on_tri(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 ab = b - a;
    const vec3 ac = c - a;
    const vec3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;                        // vertex a
    const vec3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;                         // vertex b
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));  // edge ab
    const vec3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;                         // vertex c
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));  // edge ac
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));  // edge bc
    }
    const float den = 1.0 / max(va + vb + vc, 1e-30);
    return a + ab * (vb * den) + ac * (vc * den);                // face
}

// Does triangle (a0, b0, c0) intersect the unit cube centred at c?
// Separating-axis test (Akenine-Moller 2001): the 9 edge x axis directions,
// the 3 box normals, and the triangle normal.
bool tri_box_overlap(vec3 c, vec3 a0, vec3 b0, vec3 c0) {
    const float h = 0.5;
    const vec3 v0 = a0 - c;
    const vec3 v1 = b0 - c;
    const vec3 v2 = c0 - c;
    const vec3 edges[3] = vec3[](v1 - v0, v2 - v1, v0 - v2);
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            vec3 unit = vec3(0.0);
            unit[k] = 1.0;
            const vec3 ax = cross(edges[i], unit);
            const float p0 = dot(ax, v0);
            const float p1 = dot(ax, v1);
            const float p2 = dot(ax, v2);
            const float r = h * (abs(ax.x) + abs(ax.y) + abs(ax.z));
            if (min(p0, min(p1, p2)) > r || max(p0, max(p1, p2)) < -r) return false;
        }
    }
    for (int k = 0; k < 3; ++k) {
        if (min(v0[k], min(v1[k], v2[k])) > h || max(v0[k], max(v1[k], v2[k])) < -h) {
            return false;
        }
    }
    const vec3 n = cross(edges[0], edges[1]);
    const float r = h * (abs(n.x) + abs(n.y) + abs(n.z));
    return abs(dot(n, v0)) <= r;
}

// Unit normal of the triangle packed in a shell key (0 if none / degenerate).
vec3 key_normal(int key) {
    if (key == SHELL_EMPTY) return vec3(0.0);
    const int t = key & ((1 << TRI_BITS) - 1);
    const vec3 a = tri_vertex(t, 0);
    const vec3 m = cross(tri_vertex(t, 1) - a, tri_vertex(t, 2) - a);
    const float ln = length(m);
    return ln > 1e-12 ? m / ln : vec3(0.0);
}

// Cell-index bounding box of a triangle grown by `grow` cells, clamped.
void tri_cell_box(int t, float grow, out ivec3 lo_c, out ivec3 hi_c) {
    const vec3 a = tri_vertex(t, 0), b = tri_vertex(t, 1), c = tri_vertex(t, 2);
    const vec3 lo = min(a, min(b, c)) - grow;
    const vec3 hi = max(a, max(b, c)) + grow;
    lo_c = max(ivec3(floor(lo - 0.5)), ivec3(0));
    hi_c = min(ivec3(floor(hi + 0.5)), V.dims.xyz - 1);
}
