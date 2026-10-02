// Lagrangian flow visuals on the renderer's snapshots:
//   smoke        22k massless particles from a rectangular rake (the smoke
//                wand): continuous emission draws streaklines
//   streamlines  instantaneous field lines, RK2 from a 6 x 6 seed grid
//   wand marker  the rake's outline, so the user sees where smoke comes from
// Everything stays on the GPU; the VolumeRenderer splats the buffers
// (register them with add_splat_source).
#pragma once

#include <array>
#include <cstdint>

#include "windoa/context.hpp"
#include "windoa/volume.hpp"

namespace windoa::render {

class Tracers {
  public:
    static constexpr std::uint32_t kSmokeParticles = 22000;
    static constexpr int kRakeSide = 9;         // 81 streaklines
    static constexpr std::uint32_t kSeeds = 36; // 6 x 6
    static constexpr std::uint32_t kLineSteps = 220;
    static constexpr float kLineStep = 1.2f; // cells

    Tracers(Context& ctx, int nx, int ny, int nz);
    Tracers(const Tracers&) = delete;
    Tracers& operator=(const Tracers&) = delete;

    // Snapshot slot `slot`'s flags + macro (as VolumeRenderer::set_sources).
    void set_sources(int slot, const Buffer& flags, const Buffer& macro);

    // Aim the wand: upstream plane x, centre (cy, cz), half-extents (cells).
    void set_rake(float x, float cy, float cz, float half_y, float half_z);
    std::array<float, 5> rake() const { return {rake_x_, rake_cy_, rake_cz_, half_y_, half_z_}; }
    void request_reset() { reset_pending_ = true; }

    // Record this frame's tracer work on slot `slot`: advect the smoke by
    // `steps` lattice steps (0 = frozen, e.g. paused), retrace streamlines.
    void record(VkCommandBuffer cmd, int slot, int steps, float u_ref, bool smoke,
                bool streamlines);

    // Splat inputs
    const Buffer& smoke_pos() const { return smoke_pos_; }
    const Buffer& smoke_colours() const { return smoke_col_; }
    const Buffer& line_verts() const { return line_verts_; }
    const Buffer& line_colours() const { return line_col_; }
    const Buffer& marker_verts() const { return marker_verts_; }
    const Buffer& marker_colours() const { return marker_col_; }
    static constexpr std::uint32_t line_segments() { return kSeeds * (kLineSteps - 1); }

  private:
    Context& ctx_;
    std::array<int, 3> n_{};
    float rake_x_ = 8.0f, rake_cy_ = 0.0f, rake_cz_ = 0.0f, half_y_ = 8.0f, half_z_ = 8.0f;
    bool reset_pending_ = true;
    bool marker_dirty_ = true;
    std::uint32_t frame_ = 0;

    Buffer smoke_pos_;
    Buffer smoke_col_;
    Buffer line_verts_;
    Buffer line_col_;
    Buffer marker_verts_;
    Buffer marker_col_;
    ComputeKernel smoke_;
    ComputeKernel lines_;
};

} // namespace windoa::render
