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
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
    int look = 0;                             // 1 - 9: a look (after --show)
    std::optional<std::array<float, 2>> pick; // scripted Ctrl + click at screen fractions
    std::string record;                       // record every frame to this folder
    int record_every = 1;                     // ... or every N-th
    float mach = 0.0f;                        // > 0: transonic Mach number at start
    float zoom = 0.0f;                        // > 0: camera distance x zoom, aimed at the model
    std::optional<std::array<float, 2>> view; // camera azimuth, elevation (degrees)
    std::optional<std::array<int, 2>> window; // client size; default 80 % of the screen
                                              // (scripted runs: 1600 x 900)
    float ui_scale = 0.0f;                    // > 0: the UI scale (else the saved one)
    bool scripted() const { return max_frames > 0; }
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
    // display: the UI font and scale (the monitor's DPI x the user's factor),
    // rebuilt before a frame when either changes; preferences saved in the
    // ImGui .ini file as a [windoa][prefs] entry
    void apply_style();
    void set_ui_scale(float s);
    void read_pref(const char* line);
    void write_prefs(ImGuiTextBuffer& out) const;
    void choose_model(std::size_t index);
    void ui(const TunnelStatus& st);
    void panel_tunnel(const TunnelStatus& st);
    void panel_transonic(const TunnelStatus& st);
    void worker_failure(); // the worker's last failure, with resume when it stopped
    void panel_model(const TunnelStatus& st);
    void panel_view(const TunnelStatus& st);
    void panel_compare(const TunnelStatus& st);
    void append_result(const TunnelStatus& st); // a row of results/results.csv
    void panel_analysis(const TunnelStatus& st);
    bool begin_panel(const char* name, float x, float y, float w, float h);
    void end_panel();
    float plot_strip_height() const;
    void post_probes();
    void legends(const TunnelStatus& st);
    void toast_overlay(); // the latest message, bottom centre
    // play (docs/REVIEW.md F1 - F4): the quick bar, the looks, the camera views
    void quick_bar(const TunnelStatus& st);
    void apply_look(int look, const TunnelStatus& st); // 1 .. kLooks
    void view_preset(int view, const TunnelStatus& st);
    void step_model(int delta);
    // recording (F8): numbered PNGs, written by a background thread
    void start_recording(std::string dir);
    void stop_recording();
    std::vector<std::uint8_t> capture_rgba(std::uint32_t& w, std::uint32_t& h);
    void png_writer_loop();
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
    float ui_scale_ = 1.0f;    // the user's factor (Ctrl + = / - / 0)
    int ui_font_ = 0;          // 0 the system UI font (Segoe UI), 1 Dear ImGui's own
    ImFont* fonts_[2] = {};    // fonts_[0] null when the system font is missing
    float style_dpi_ = 0.0f;   // what the current style was built for
    float style_scale_ = 0.0f; // ...
    int style_font_ = -1;      // ...
    std::chrono::steady_clock::time_point quit_armed_{}; // Esc once: a second quits
    struct Prefs {
        float ui_scale = 1.0f;
        int font = 0;
        int speed_unit = 0;
        float render_scale = 1.0f;
    };
    Prefs saved_prefs_; // as read from the .ini file
    // The window size the panels were last laid out in (Dear ImGui keeps
    // pixels): when the window is resized, each undocked panel keeps its
    // distance to its nearer edges.
    ImVec2 layout_size_{0.0f, 0.0f};
    ImVec2 relayout_from_{0.0f, 0.0f}; // this frame: the old size, or 0

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
    int speed_unit_ = 0;            // the wind in 0 mph, 1 km/h, 2 m/s
    int look_ = 1;                  // the look last applied (1 = the start look)
    bool orbit_ = false;            // turntable camera
    bool animate_textures_ = true;  // flow textures travel with the flow
    float lic_phase_ = 0.0f;        // their phase, in cycles
    float quick_bar_height_ = 0.0f; // the legends start below the bar
    // speed sweep (F5): the wind eased between two speeds and back
    bool sweep_ = false;
    float sweep_t_ = 0.0f, sweep_period_ = 20.0f;         // s
    float sweep_lo_mps_ = 13.41f, sweep_hi_mps_ = 31.29f; // 30 and 70 mph
    float sweep_lo_mach_ = 0.8f, sweep_hi_mach_ = 1.5f;   // transonic
    // recording (F8)
    bool recording_ = false;
    std::string record_dir_;
    int record_every_ = 1, record_tick_ = 0, record_count_ = 0, record_dropped_ = 0;
    struct PendingPng {
        std::string path;
        std::uint32_t w = 0, h = 0;
        std::vector<std::uint8_t> rgba;
    };
    std::deque<PendingPng> png_queue_;
    std::mutex png_mu_;
    std::condition_variable png_cv_;
    bool png_stop_ = false;
    std::thread png_writer_;
    // click to place (F10): Ctrl + click reads the depth under the cursor
    int click_mode_ = 0; // 0 focus the camera, 1 place a probe, 2 move the smoke
    int pick_mode_ = 0;  // the pending pick's (a double click always focuses)
    int next_probe_ = 0;
    std::optional<std::array<std::uint32_t, 2>> pick_request_; // render pixel, next frame
    bool pick_recorded_ = false;
    std::array<std::uint32_t, 2> pick_px_{};
    render::View pick_view_;
    std::unique_ptr<Buffer> pick_buf_;
    void request_pick(int mode);
    void resolve_pick(const TunnelStatus& st);
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
    std::uint64_t mesh_seen_ = UINT64_MAX; // the renderer's triangles' geometry version
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
    HANDLE frame_timer_ = nullptr; // the frame cap (kMaxFps)
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
