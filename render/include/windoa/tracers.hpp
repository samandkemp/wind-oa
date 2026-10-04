// Lagrangian flow visuals on the renderer's snapshots:
//   smoke        22k massless particles from a rectangular rake (the smoke
//                wand): continuous emission draws streaklines
//   streamlines  instantaneous field lines, RK2 from a 6 x 6 seed grid
//   timelines    the smoke as a pulsed wire across the rake (THEORY 10.9)
//   wand marker  the rake's outline, so the user sees where smoke comes from,
//                with the wake-survey planes and the probes
//   arrows       velocity arrows on the slice plane
// Everything stays on the GPU; the VolumeRenderer splats the buffers
// (register them with add_splat_source).
#pragma once

#include <array>
#include <cstdint>
#include <vector>

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
    // Smoke as continuous streaklines (pulse_steps 0), or as timelines: a
    // line of particles across the rake height released every pulse_steps
    // (a hydrogen-bubble wire; THEORY 10.9).
    void set_timelines(int pulse_steps);

    // Overlay markers drawn with the wand outline: the wake-survey planes
    // (x < 0: none) and probe crosses (colour per probe).
    void set_overlay(int x_upstream, int x_survey, const std::vector<std::array<float, 3>>& probes);
    // Rotor blades (actuator lines) as segments, redrawn every frame they turn
    // without restarting the smoke.
    void set_blades(const std::vector<std::array<std::array<float, 3>, 2>>& segments);
    // The wand outline: drawn with the smoke it marks, not with the other
    // markers alone.
    void show_wand(bool on);
    static constexpr std::array<std::array<float, 4>, 4> kProbeColours = {
        {{1.0f, 0.75f, 0.25f, 1.0f},
         {1.0f, 0.40f, 0.80f, 1.0f},
         {0.60f, 1.0f, 0.35f, 1.0f},
         {0.45f, 0.80f, 1.0f, 1.0f}}};

    // Record this frame's tracer work on slot `slot`: advect the smoke by
    // `steps` lattice steps (0 = frozen, e.g. paused), retrace streamlines.
    void record(VkCommandBuffer cmd, int slot, int steps, float u_ref, bool smoke,
                bool streamlines);
    // Velocity arrows on the slice plane `axis` at `pos` (cells), one per
    // `spacing` cells; returns the segment count to splat.
    std::uint32_t record_arrows(VkCommandBuffer cmd, int slot, int axis, float pos, float spacing,
                                float u_ref);

    // Splat inputs
    const Buffer& smoke_pos() const { return smoke_pos_; }
    const Buffer& smoke_colours() const { return smoke_col_; }
    const Buffer& line_verts() const { return line_verts_; }
    const Buffer& line_colours() const { return line_col_; }
    const Buffer& marker_verts() const { return marker_verts_; }
    const Buffer& marker_colours() const { return marker_col_; }
    std::uint32_t marker_segments() const { return marker_count_; }
    const Buffer& arrow_verts() const { return arrow_verts_; }
    const Buffer& arrow_colours() const { return arrow_col_; }
    static constexpr std::uint32_t line_segments() { return kSeeds * (kLineSteps - 1); }

  private:
    Context& ctx_;
    std::array<int, 3> n_{};
    float rake_x_ = 8.0f, rake_cy_ = 0.0f, rake_cz_ = 0.0f, half_y_ = 8.0f, half_z_ = 8.0f;
    bool reset_pending_ = true;
    bool marker_dirty_ = true;
    std::uint32_t frame_ = 0;
    int pulse_steps_ = 0;    // 0 = streaklines
    std::int64_t clock_ = 0; // smoke steps since the last reset (timelines)
    int x_upstream_ = -1, x_survey_ = -1;
    std::vector<std::array<float, 3>> probes_;
    std::vector<std::array<std::array<float, 3>, 2>> blades_;
    bool show_wand_ = true;
    bool redraw_ = false; // rebuild the markers without restarting the smoke
    std::uint32_t marker_count_ = 4;
    std::size_t max_arrows_ = 0;

    Buffer smoke_pos_;
    Buffer smoke_col_;
    Buffer line_verts_;
    Buffer line_col_;
    Buffer marker_verts_;
    Buffer marker_col_;
    Buffer arrow_verts_;
    Buffer arrow_col_;
    ComputeKernel smoke_;
    ComputeKernel lines_;
    ComputeKernel arrows_;
};

} // namespace windoa::render
