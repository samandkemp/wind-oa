#include "windoa/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace windoa::geometry {

void Mesh::add(const Vec3& a, const Vec3& b, const Vec3& c) {
    xyz.insert(xyz.end(), {a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]});
}

void Mesh::append(const Mesh& other) {
    xyz.insert(xyz.end(), other.xyz.begin(), other.xyz.end());
}

Vec3 Bounds::centre() const {
    return {(lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f};
}

float Bounds::longest() const {
    return std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
}

namespace {

Mesh parse_ascii(const std::string& text) {
    Mesh m;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string word;
        float v[3];
        if ((ls >> word) && word == "vertex" && (ls >> v[0] >> v[1] >> v[2])) {
            m.xyz.insert(m.xyz.end(), {v[0], v[1], v[2]});
        }
    }
    if (m.xyz.empty() || m.xyz.size() % 9 != 0) {
        throw std::runtime_error("unparseable STL (not binary; bad ASCII vertex count)");
    }
    return m;
}

} // namespace

Mesh load_stl(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    const std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Binary sanity check: the declared size must match the actual size.
    if (raw.size() >= 84) {
        std::uint32_t count = 0;
        std::memcpy(&count, raw.data() + 80, 4);
        if (84 + std::uint64_t{count} * 50 == raw.size()) {
            Mesh m;
            m.xyz.resize(std::size_t{count} * 9);
            for (std::uint32_t t = 0; t < count; ++t) {
                // record: normal (12 bytes), 3 vertices (36), attribute (2)
                std::memcpy(&m.xyz[std::size_t{t} * 9], raw.data() + 84 + std::size_t{t} * 50 + 12,
                            36);
            }
            return m;
        }
    }
    return parse_ascii(raw);
}

void save_stl(const std::string& path, const Mesh& mesh) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    char header[80] = {};
    const char tag[] = "wind-oa binary STL";
    std::memcpy(header, tag, sizeof(tag));
    f.write(header, 80);
    const auto count = static_cast<std::uint32_t>(mesh.triangles());
    f.write(reinterpret_cast<const char*>(&count), 4);
    for (std::size_t t = 0; t < mesh.triangles(); ++t) {
        const float* v = &mesh.xyz[t * 9];
        const float e1[3] = {v[3] - v[0], v[4] - v[1], v[5] - v[2]};
        const float e2[3] = {v[6] - v[0], v[7] - v[1], v[8] - v[2]};
        float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                      e1[0] * e2[1] - e1[1] * e2[0]};
        const float len = std::max(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1e-20f);
        for (float& c : n)
            c /= len;
        const std::uint16_t attr = 0;
        f.write(reinterpret_cast<const char*>(n), 12);
        f.write(reinterpret_cast<const char*>(v), 36);
        f.write(reinterpret_cast<const char*>(&attr), 2);
    }
}

Bounds bounds(const Mesh& mesh) {
    Bounds b;
    if (mesh.empty())
        return b;
    b.lo = {mesh.xyz[0], mesh.xyz[1], mesh.xyz[2]};
    b.hi = b.lo;
    for (std::size_t i = 0; i < mesh.xyz.size(); i += 3) {
        for (int k = 0; k < 3; ++k) {
            b.lo[k] = std::min(b.lo[k], mesh.xyz[i + k]);
            b.hi[k] = std::max(b.hi[k], mesh.xyz[i + k]);
        }
    }
    return b;
}

Mesh fit_to_box(const Mesh& mesh, const Vec3& centre, float length, const Bounds* frame) {
    const Bounds b = frame ? *frame : bounds(mesh);
    const float scale = length / std::max(b.longest(), 1e-20f);
    const Vec3 mid = b.centre();
    Mesh out = mesh;
    for (std::size_t i = 0; i < out.xyz.size(); i += 3) {
        for (int k = 0; k < 3; ++k)
            out.xyz[i + k] = (mesh.xyz[i + k] - mid[k]) * scale + centre[k];
    }
    return out;
}

Mesh transform(const Mesh& mesh, float yaw_deg, float pitch_deg, float roll_deg,
               std::optional<Vec3> about) {
    const Vec3 p = about.value_or(bounds(mesh).centre());
    const double d2r = 3.14159265358979323846 / 180.0;
    const double cy = std::cos(yaw_deg * d2r), sy = std::sin(yaw_deg * d2r);
    const double cp = std::cos(pitch_deg * d2r), sp = std::sin(pitch_deg * d2r);
    const double cr = std::cos(roll_deg * d2r), sr = std::sin(roll_deg * d2r);
    // M = Ry(yaw) * Rz(pitch) * Rx(roll)
    const double ry[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
    const double rz[3][3] = {{cp, -sp, 0}, {sp, cp, 0}, {0, 0, 1}};
    const double rx[3][3] = {{1, 0, 0}, {0, cr, -sr}, {0, sr, cr}};
    double t[3][3] = {}, m[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                t[i][j] += rz[i][k] * rx[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                m[i][j] += ry[i][k] * t[k][j];

    Mesh out = mesh;
    for (std::size_t i = 0; i < out.xyz.size(); i += 3) {
        const double v[3] = {mesh.xyz[i] - p[0], mesh.xyz[i + 1] - p[1], mesh.xyz[i + 2] - p[2]};
        for (int r = 0; r < 3; ++r) {
            out.xyz[i + r] =
                static_cast<float>(m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] + p[r]);
        }
    }
    return out;
}

} // namespace windoa::geometry
