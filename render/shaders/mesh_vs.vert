#version 450
// The model's triangles for the march (mesh_raster.cpp, THEORY 10.10). Each
// corner is pulled from a storage buffer (no vertex input state) and
// projected with the march's own camera basis, so a pixel here is the same
// ray there: NDC x = (r . right) / (z tan(fov / 2) aspect), y the same with
// up (Vulkan's y points down), z = r . fwd.
layout(std430, binding = 0) readonly buffer Corners { vec4 corners[]; };  // position, normal

layout(push_constant) uniform Params {
    vec4 eye;    // xyz, tan(fov / 2)
    vec4 fwd;    // xyz, aspect
    vec4 right;  // xyz, -
    vec4 up;     // xyz, -
} P;

layout(location = 0) out vec3 wpos;
layout(location = 1) out vec3 wnrm;

const float NEAR = 0.05;  // cells

void main() {
    const vec3 pos = corners[2 * gl_VertexIndex].xyz;
    wpos = pos;
    wnrm = corners[2 * gl_VertexIndex + 1].xyz;
    const vec3 r = pos - P.eye.xyz;
    const float z = dot(r, P.fwd.xyz);
    const float x = dot(r, P.right.xyz) / (P.eye.w * P.fwd.w);
    const float y = -dot(r, P.up.xyz) / P.eye.w;
    // Reversed depth, NEAR / z: even precision at every distance (the exact
    // distance goes out in the colour; the depth only orders the triangles).
    gl_Position = vec4(x, y, NEAR, z);
}
