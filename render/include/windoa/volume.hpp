// Ray-marched volume renderer.
// Compute passes into a packed-RGBA buffer, copied into an image the app
// blits to its swapchain:
//   prepare   the transfer-function input once per cell (the march samples)
//   Q         smoothed-velocity Q-criterion + an adaptive RMS threshold,
//             reduced on the device (no host round trip)
//   march     one ray per pixel, front to back: model surface (voxel or
//             smooth, optionally painted with Cp, near-wall speed, reversed
//             flow or oil-flow streaks), slice plane (optionally with a
//             LIC texture), vortex cores, mean reversed-flow shells, dye
//             smoke, field haze, box edges; writes a depth buffer
//   splats    depth-tested points / segments (smoke, streamlines, markers)
// Solid-derived fields (occupancy blocks, blurred indicator) are rebuilt
// whenever the caller's geometry version changes.
//
// The renderer reads snapshot buffers, not a live solver: up to kSlots
// source sets (flags, macro, ...) bound once, one chosen per frame. That is
// how a solver on another queue hands it frames without racing its steps.
// Everything is in lattice cells (the camera too). What the fields mean:
// THEORY 10.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "windoa/context.hpp"

namespace windoa::render {

inline constexpr int kSlots = 2;

// Field shown by the haze and the slice (values match the shaders).
enum class Field : int {
    Speed = 0,
    Pressure = 1,
    Vorticity = 2,
    VortX = 3,
    Mach = 4,
    Schlieren = 5,
    MeanSpeed = 6, // of the time-averaged window (needs Sources::mean)
    Turbulence = 7 // turbulence intensity of the window (needs Sources::m2)
};
// What the model surface is painted with (THEORY 10.2, 10.7).
enum class Paint : int { Cp = 0, WallSpeed = 1, Reversed = 2, OilFlow = 3 };
enum class Surface : int { Hidden = 0, Voxel = 1, Smooth = 2 };

// Noise floor below which a sample is transparent, per field, in the
// field's normalised units.
float default_floor(Field f);

// One snapshot's buffers. They must outlive the renderer.
struct Sources {
    const Buffer* flags = nullptr; // uint per cell (FLUID = 0)
    const Buffer* macro = nullptr; // vec4 per cell: u.xyz, rho (Euler: p)
    const Buffer* aux = nullptr;   // float per cell: Euler rho (optional)
    const Buffer* dye = nullptr;   // float per cell: dye concentration (optional)
    // time averages (optional): vec4 per cell, as engine FlowStats
    const Buffer* mean = nullptr; // <u>, <rho> - 1
    const Buffer* m2 = nullptr;   // sum of w (x - <x>)^2
};

struct View {
    std::array<float, 3> eye{};
    std::array<float, 3> target{};
    std::array<float, 3> up{0.0f, 1.0f, 0.0f};
    float fov_deg = 45.0f;
};

struct Settings {
    Field field = Field::Speed;
    bool haze = true;
    float haze_gain = 1.0f;
    float haze_floor = -1.0f; // < 0: default_floor(field)
    Surface surface = Surface::Smooth;
    bool paint_surface = true;
    Paint paint = Paint::Cp;
    int slice_axis = -1;    // -1 none, else 0 / 1 / 2
    float slice_pos = 0.5f; // fraction of the domain along slice_axis
    bool slice_lic = false; // LIC texture of the in-plane flow on the slice
    // time averages: 1 / their weight (0 = none in this slot), and the
    // translucent shells where the mean streamwise flow runs backwards
    float stats_inv_weight = 0.0f;
    bool recirculation = false;
    bool box = true;
    bool vortex_cores = false;
    float q_sense = 1.0f; // threshold = 3 x rms(Q > 0) x q_sense
    bool dye = false;
    float dye_gain = 1.5f;
    bool dye_by_speed = true;
    int steps = 128;            // march samples across the box
    float u_ref = 0.05f;        // freestream speed (lattice; Euler: Mach)
    float rho_ref = 1.0f;       // Cp reference density (Euler: p_ref)
    float pscale = 1.0f / 3.0f; // pressure per unit macro.w (LBM cs^2; Euler 1)
    float gamma = 0.0f;         // > 0: compressible source (macro.w = p, aux = rho)
};

// A depth-tested splat batch: `verts` as points (vec4 xyz) or segment
// pairs, `colours` one vec4 per point / segment. Register once.
struct SplatSource {
    const Buffer* verts = nullptr;
    const Buffer* colours = nullptr;
};
struct SplatDraw {
    int source = 0;          // add_splat_source() id
    std::uint32_t count = 0; // points or segments
    bool segments = false;
    float radius = 0.2f; // points: world radius in cells
    float alpha = 0.85f;
    float depth_bias = 0.0f; // cells: draw this far behind the marched surface
};

class VolumeRenderer {
  public:
    VolumeRenderer(Context& ctx, int nx, int ny, int nz);
    ~VolumeRenderer();
    VolumeRenderer(const VolumeRenderer&) = delete;
    VolumeRenderer& operator=(const VolumeRenderer&) = delete;

    // Bind snapshot slot `slot` (0..kSlots-1). Call before first use / when
    // no frame using the renderer is in flight.
    void set_sources(int slot, const Sources& s);
    // Returns an id for SplatDraw::source (at most 4 sources).
    int add_splat_source(const SplatSource& s);

    // Output size in pixels. Waits for the graphics queue when it changes.
    void resize(std::uint32_t width, std::uint32_t height);
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }

    // Record the full render of slot `slot` into `cmd`. geometry_version
    // identifies the flags it holds (a change rebuilds the solid fields).
    // Afterwards image() is in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL.
    void record(VkCommandBuffer cmd, const View& view, const Settings& s, int slot,
                std::uint64_t geometry_version, std::span<const SplatDraw> splats = {});
    VkImage image() const { return image_; }

    // The last rendered frame as packed RGBA8, row 0 at the top (blocking;
    // for screenshots and tests).
    std::vector<std::uint32_t> read_pixels();

  private:
    void create_target();
    void destroy_target();
    void bind_target_dependent();

    Context& ctx_;
    std::array<int, 3> n_{};
    std::size_t cells_ = 0;
    Groups cell_groups_;
    std::uint64_t geometry_seen_ = UINT64_MAX;
    static constexpr int kQEvery = 3; // frames between Q refreshes
    bool q_valid_ = false;
    int q_age_ = 0;
    float q_sense_seen_ = -1.0f;
    std::array<Sources, kSlots> sources_{};
    std::vector<SplatSource> splat_sources_;

    Buffer field_;
    Buffer occ_;
    Buffer solid_raw_;
    Buffer solid_blur_;
    Buffer blur_tmp_;
    Buffer vsmooth_;
    Buffer smooth_tmp_;
    Buffer qfield_;
    Buffer qpartials_;
    Buffer qstat_;
    Buffer zero_;   // stands in for an absent dye / aux source
    Buffer speed_;  // |u| per cell (dye colour), from vol_prepare
    Buffer recirc_; // <u_x> / U per cell (reversed-flow shells), from vol_prepare
    // what the march samples, as 3-D textures (hardware trilinear)
    Image3D t_field_, t_raw_, t_blur_, t_q_, t_dye_, t_speed_, t_recirc_;

    ComputeKernel prepare_;
    ComputeKernel occ_kernel_;
    ComputeKernel blur_;
    ComputeKernel smooth_;
    ComputeKernel q_;
    ComputeKernel qstat_kernel_;
    ComputeKernel march_;
    ComputeKernel splat_;

    std::uint32_t width_ = 0, height_ = 0;
    std::unique_ptr<Buffer> pixels_;
    std::unique_ptr<Buffer> depth_;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory image_memory_ = VK_NULL_HANDLE;
};

} // namespace windoa::render
