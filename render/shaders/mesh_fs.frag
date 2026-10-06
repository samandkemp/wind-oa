#version 450
// Per pixel: the shading normal and the distance along the pixel's ray to
// the nearest triangle (mesh_raster.cpp, THEORY 10.10). The normal is turned
// to the viewer's side of the triangle: a triangle soup may wind its faces
// either way, and the march samples the flow out along it.
layout(push_constant) uniform Params {
    vec4 eye;
    vec4 fwd;
    vec4 right;
    vec4 up;
} P;

layout(location = 0) in vec3 wpos;
layout(location = 1) in vec3 wnrm;
layout(location = 0) out vec4 o;  // normal xyz, distance (cells)

void main() {
    const vec3 to_eye = P.eye.xyz - wpos;
    vec3 face = cross(dFdx(wpos), dFdy(wpos));  // the triangle's own normal
    if (dot(face, face) < 1e-30) face = to_eye;
    if (dot(face, to_eye) < 0.0) face = -face;
    vec3 n = dot(wnrm, wnrm) > 1e-12 ? normalize(wnrm) : normalize(face);
    if (dot(n, face) < 0.0) n = -n;
    o = vec4(n, length(to_eye));
}
