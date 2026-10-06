// GPU voxeliser: triangle soup (lattice coordinates) -> cell flags, plus the
// exact signed distance the Euler walls need:
//   - solid fill by the nonzero winding rule, cast along all three axes
//     with a 2-of-3 vote (tolerant of imperfect, non-watertight STLs)
//   - a thin-feature supercover shell kept only where the feature is thin
//   - signed_distance(): exact point-to-triangle distance in a band
// Specification: THEORY 5.2 - 5.4.
// Flat-field interface: flags in / flags out, [x][y][z] C order, the same
// values as lbm::Flag. Not implemented: a boundary-cell -> nearest-
// triangle map (only per-triangle Cp painting would need it).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/mesh.hpp"

namespace windoa {

class Voxeliser {
  public:
    // The shell pass packs a triangle index into 20 bits of its key.
    static constexpr std::size_t kMaxTriangles = std::size_t{1} << 20;

    Voxeliser(Context& ctx, int nx, int ny, int nz);
    ~Voxeliser();
    Voxeliser(const Voxeliser&) = delete;
    Voxeliser& operator=(const Voxeliser&) = delete;

    struct Stats {
        std::size_t n_tris = 0;
        std::size_t n_solid = 0; // OBSTACLE cells after the pass
        std::size_t n_thin = 0;  // of those, added by the thin-feature shell
    };

    // Clears previous OBSTACLE cells in `flags`, marks the mesh (lattice
    // coordinates) as OBSTACLE; WALL / LID cells are preserved.
    Stats voxelise(const geometry::Mesh& mesh, std::vector<std::uint8_t>& flags);

    // Signed distance to the mesh of the last voxelise(), in cells:
    // negative where `flags` is solid, clamped to +-4, band + 1 where no
    // triangle is within `band`. [x][y][z].
    std::vector<float> signed_distance(const std::vector<std::uint8_t>& flags, float band = 3.0f);

  private:
    void upload_flags(const std::vector<std::uint8_t>& flags);
    // What flags_ holds on the device (empty: unknown), so an upload sends
    // only what differs; and the last mesh's x-extent, beyond which no pass
    // changes a cell (the downloads take that slab only).
    std::vector<std::uint8_t> device_flags_;
    float tri_xmin_ = 0.0f, tri_xmax_ = -1.0f;
    std::pair<int, int> slab(float margin) const; // planes [x0, x1)
    void run(const ComputeKernel& k, const void* params, std::uint64_t items, std::uint32_t local);

    Context& ctx_;
    int nx_, ny_, nz_;
    std::size_t n_;
    std::size_t n_tris_ = 0;

    std::unique_ptr<Buffer> tris_; // grows to the largest mesh seen
    Buffer vote_;
    Buffer shell_;
    Buffer flags_;
    Buffer phi_;
    Buffer counter_;

    std::unique_ptr<ComputeKernel> winding_, votes_, shell_mark_, shell_apply_, dist_band_,
        phi_kernel_;
    void bind_all();
};

} // namespace windoa
