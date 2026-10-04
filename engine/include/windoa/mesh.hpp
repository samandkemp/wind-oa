// Triangle-soup meshes and STL I/O.
// Hand-rolled (no mesh library). STL carries no shared-vertex topology and
// none is needed: voxelisation and flat-shaded rendering both work on the
// soup directly.
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace windoa::geometry {

using Vec3 = std::array<float, 3>;

struct Mesh {
    // 9 floats per triangle: v0 xyz, v1 xyz, v2 xyz (right-hand winding =
    // outward normal for a well-formed closed mesh).
    std::vector<float> xyz;

    std::size_t triangles() const { return xyz.size() / 9; }
    bool empty() const { return xyz.empty(); }
    void add(const Vec3& a, const Vec3& b, const Vec3& c);
    void append(const Mesh& other);
};

struct Bounds {
    Vec3 lo{}, hi{};
    Vec3 centre() const;
    float longest() const;
};

// Binary STL (validated by its declared size), else ASCII. Throws on failure.
Mesh load_stl(const std::string& path);
void save_stl(const std::string& path, const Mesh& mesh);

Bounds bounds(const Mesh& mesh);

// Uniformly scale + translate (THEORY 5.1) so the longest axis equals `length` and the
// bounding-box centre sits at `centre` (lattice-unit placement).
// frame: the box to fit (default: the mesh's own bounds); a model whose
// rotors are actuator lines is fitted by its mesh and their swept discs.
Mesh fit_to_box(const Mesh& mesh, const Vec3& centre, float length, const Bounds* frame = nullptr);

// Rotate about `about` (default: bounding-box centre). Yaw about +y, pitch
// about +z (nose-up positive for a +x-pointing model), roll about +x;
// applied roll -> pitch -> yaw.
Mesh transform(const Mesh& mesh, float yaw_deg, float pitch_deg, float roll_deg,
               std::optional<Vec3> about = {});

} // namespace windoa::geometry
