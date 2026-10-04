// The interactive tunnel: window + swapchain + Dear ImGui over the headless
// engine, with the solver on a worker thread (sim.hpp) and the renderer on
// the graphics queue. Panels: Tunnel (run status,
// forces, flow speed, turbulence), Model (catalogue menu, placement, attitude,
// reference area, spin), View (renderer, fields, overlays), Compare (A/B),
// Analysis (time averaging, the wake survey, probes and spectra),
// colour legends along the top, Cd / Cl / Cm plots along the bottom.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "camera.hpp"
#include "plots.hpp"
#include "sim.hpp"
#include "swapchain.hpp"
#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/tracers.hpp"
#include "windoa/tunnel.hpp"
#include "windoa/volume.hpp"
#include "window.hpp"

namespace windoa::app {

struct Options {
    std::string preset = "fast";
    std::string model = "car_saloon"; // the starting catalogue model
    int max_frames = 0;               // > 0: exit after N frames (smoke test)
    std::string shot;                 // save the last frame (with UI) as PNG
    int warmup_steps = 0;             // run the solver this long before the window loop
    bool no_vsync = false;
    bool transonic = false; // start in transonic mode
    bool f16 = false;       // f16 distribution storage (faster, approximate)
    std::string show;       // e.g. "q,dye,lines,nosmoke,slice" (tests / screenshots)
    std::string field;      // speed|pressure|vorticity|vortx|mach|schlieren|mean|turb
    float spin = -1.0f;     // >= 0: spin on at this ratio
    float power = -1.0f;    // >= 0: engines on at this throttle
    float rotors = -1.0f;   // >= 0: rotors turning at this tip-speed ratio
    float aoa = 0.0f;
    float size = 0.0f;                        // > 0: the model's longest axis, cells
    float mach = 0.0f;                        // > 0: transonic Mach number at start
    float zoom = 0.0f;                        // > 0: camera distance x zoom, aimed at the model
    std::optional<std::array<float, 2>> view; // camera azimuth, elevation (degrees)
};

// One model the menu can load: a catalogue entry or an STL in ./models.
struct MenuItem {
    const catalogue::Entry* entry = nullptr;
    std::filesystem::path stl;
    std::string label, group;
};

class App {
  public:
    explicit App(const Options& o);
    ~App();
    int run();

  private:
    void init_imgui();
    void choose_model(std::size_t index);
    void ui(const TunnelStatus& st);
    void panel_tunnel(const TunnelStatus& st);
    void panel_transonic(const TunnelStatus& st);
    void panel_model(const TunnelStatus& st);
    void panel_view(const TunnelStatus& st);
    void panel_compare(const TunnelStatus& st);
    void panel_analysis(const TunnelStatus& st);
    bool begin_panel(const char* name, float x, float y, float w, float h);
    void end_panel();
    float plot_strip_height() const;
    void post_probes();
    void legends(const TunnelStatus& st);
    void plot_strip(const TunnelStatus& st);
    void help_window();
    void update_title(const TunnelStatus& st);
    void handle_keys(const TunnelStatus& st);
    void update_rake(const TunnelStatus& st);
    void record_frame(const Swapchain::Frame& f, const SimWorker::Frame& sf, const TunnelStatus& st,
                      bool shot);
    void write_screenshot(std::string path);

    Options opt_;
    TunnelSettings ts_;
    std::unique_ptr<Window> window_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Swapchain> swapchain_;
    std::unique_ptr<SimWorker> sim_;
    std::unique_ptr<render::VolumeRenderer> renderer_;
    std::unique_ptr<render::Tracers> tracers_;
    VkFormat colour_format_ = VK_FORMAT_UNDEFINED;
    bool imgui_ready_ = false;

    // models
    std::vector<MenuItem> menu_;
    std::size_t model_index_ = 0;
    Placement placement_; // the UI's copy (sliders); posted after edits settle
    bool placement_dirty_ = false;
    std::chrono::steady_clock::time_point placement_edit_{};
    bool has_spinners_ = false;

    // flow controls (mirrors of what was posted)
    float u_command_ = 0.05f;
    float mach_command_ = 0.8f;
    float turb_pct_ = 0.0f;
    bool spin_on_ = false;
    float spin_ratio_ = 1.0f;
    bool power_on_ = false;
    float throttle_ = 1.0f;
    bool rotors_on_ = false;
    float rotor_tsr_ = 6.0f;
    int area_mode_ = 0;
    float a_manual_ = 100.0f;
    float sim_rate_cap_ = 0.0f; // steps / s, 0 = flat out

    // view
    render::Settings rs_;
    int paint_mode_ = 1; // 0 none, 1 Cp, 2 near-wall speed, 3 reversed flow, 4 oil flow
    bool slice_arrows_ = false;
    float arrow_spacing_ = 4.0f; // cells
    int smoke_mode_ = 0;         // 0 streaklines, 1 timelines
    int pulse_steps_ = 80;       // timeline release interval
    int slice_mode_ =
        0; // 0 off, 1 vertical (x-y at z), 2 horizontal (x-z at y), 3 cross (y-z at x)
    bool show_smoke_ = true, show_streamlines_ = false, show_plots_ = true, show_ui_ = true;
    bool show_dye_ = false;
    bool show_help_ = false;
    bool reset_layout_ = false;      // one frame: every panel back to its initial place
    bool want_reset_layout_ = false; // reset next frame (the button, or a misfit)
    std::string title_;
    float smoke_radius_ = 0.16f;
    bool rake_track_ = true, rake_autofit_ = false;
    float rake_h_frac_ = 0.10f, rake_w_frac_ = 0.10f, rake_y_ = 0.5f, rake_z_ = 0.5f;
    float render_scale_ = 1.0f, fov_deg_ = 45.0f;
    OrbitCamera camera_;
    int smoke_id_ = -1, lines_id_ = -1, marker_id_ = -1, arrows_id_ = -1;

    // analysis (THEORY 12)
    bool show_analysis_ = true, analysis_open_ = false;
    bool averaging_ = false;
    bool show_planes_ = true, wake_auto_ = true;
    int wake_x_ = 0;
    int n_probes_ = 0;
    std::array<std::array<float, 3>, kMaxProbes> probe_pos_{};
    int spec_source_ = 0; // 0 lift, 1.. a probe
    int spec_comp_ = 1;   // probe component: 0 u_x, 1 u_y, 2 u_z, 3 rho
    TunnelAnalysis analysis_;

    // bookkeeping
    std::uint64_t render_value_ = 0;
    std::int64_t last_snap_steps_ = -1;
    std::uint64_t last_epoch_ = UINT64_MAX, last_geo_ = UINT64_MAX;
    std::array<float, 5> last_rake_{};
    TimeSeries cd_{"Cd", IM_COL32(255, 102, 69, 255)};
    TimeSeries cl_{"Cl", IM_COL32(69, 171, 255, 255)};
    TimeSeries cm_{"Cm", IM_COL32(84, 222, 135, 255)};
    std::int64_t plotted_steps_ = -1;
    bool was_settled_ = false;
    struct Snap {
        std::string model;
        float aoa, size;
        double cd, cl, cm;
    };
    std::optional<Snap> ab_[2];

    // timing
    VkQueryPool queries_ = VK_NULL_HANDLE;
    HANDLE frame_timer_ = nullptr; // --no-vsync frame cap
    double tick_ms_ = 1e-6;
    bool slot_written_[kFramesInFlight] = {};
    double render_gpu_ms_ = 0.0, frame_ms_ = 0.0;
    bool screenshot_pending_ = false;
    std::unique_ptr<Buffer> shot_buf_;
    VkExtent2D shot_extent_{};
    std::string toast_;
    std::chrono::steady_clock::time_point toast_t_{};
};

} // namespace windoa::app
