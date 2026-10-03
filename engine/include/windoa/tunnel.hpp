// The wind tunnel sandbox, headless -- everything the app does that is not
// UI: the model and its placement (voxelised into the solver), the
// freestream (startup ramp + slew limit), spinning parts under the wall-
// speed cap, the rolling road / fixed ground, force coefficients (EMA over
// solver steps), the settling monitor, the divergence guard, inlet
// turbulence, dye, and the settled-flow cache; and the measurements on top
// of the flow: time averages, the wake survey, probes and spectra.
//
// A frontend (the app, a gate, a tool) calls the setters, then advance()
// repeatedly; everything GPU-side is submitted through the Context from the
// calling thread. The app runs it on a worker thread and never touches it
// from the UI thread (commands are posted to the worker).
// Specification: docs/THEORY.md, THEORY 9 (coefficients: 4.4; statistics
// and signals: 12).
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "windoa/catalogue.hpp"
#include "windoa/context.hpp"
#include "windoa/convergence.hpp"
#include "windoa/dye.hpp"
#include "windoa/euler.hpp"
#include "windoa/flow_cache.hpp"
#include "windoa/flow_stats.hpp"
#include "windoa/lbm.hpp"
#include "windoa/mesh.hpp"
#include "windoa/spectrum.hpp"
#include "windoa/voxeliser.hpp"

namespace windoa {

// Tunables (the solver settings and the grid preset). Lattice units.
struct TunnelSettings {
    int nx = 256, ny = 96, nz = 96;
    float tau = 0.504f; // nu = (tau - 1/2) / 3; usable only with regularised
    float smagorinsky_cs = 0.1f;
    bool regularised = true;
    bool recursive = false; // RR: validated (V5), opt-in
    float u_inlet = 0.05f;  // startup command
    float u_max = 0.11f;    // Ma = u sqrt(3) stays under ~0.19
    // Inlet slew limit, per step because the worker steps in batches of
    // varying size: 1e-4 (0.0015 per 15 steps).
    float u_slew_per_step = 1.0e-4f;
    int ramp_steps = 2000; // ramp from rest at startup / after a reset
    int floor_height = 3;  // rolling road / fixed floor thickness, cells
    // Fastest a spinning surface may move: the moving-wall term destabilises
    // near 0.10-0.12 (impulsive 0.13+ went NaN within 1,000 steps).
    float max_wall_speed = 0.08f;
    // Guard: blown up above this |u| (sound speed 0.577; stable 3-D < ~0.3).
    float diverge_speed = 0.5f;
    int outlet_sponge = 24; // reflection 0.68 -> 0.01, Cd +0.2 % (V10)
    int sponge_target = 1;
    float turbulence_length = 8.0f;
    double force_ema_steps = 120.0; // dashboard smoothing, in solver steps
    int health_every = 10;          // batches between full-grid health checks
    float dye_tau = 0.53f, dye_tau_plus = 1.0f;
    float dye_nozzle_rate = 0.5f; // relaxation towards C = 1 per step
    int dye_nozzles = 6;          // per side: 6 x 6 single-cell filaments
    double flow_cache_mb = 512.0;
    // f16 distribution storage (lbm::Config::storage_f16): ~1.3x faster,
    // an approximation (tools/lbm_equiv --f16). Opt-in until the gates judge it.
    bool storage_f16 = false;
    // Transonic mode
    float mach_min = 0.3f, mach_max = 1.6f;
    float mach_slew = 0.004f;           // per batch
    double develop_flow_throughs = 2.0; // impulsive start: developing this long
    double force_ema_time = 60.0;       // sim time (cell sound-crossings)
    // Statistics and signals (THEORY 12)
    int analysis_every = 25;             // batches between wake-survey / spectrum refreshes
    double history_flow_throughs = 16.0; // signal history kept for the spectra
};

// The app's solver for these settings -- the single source of truth, so a
// gate that claims to test what the app runs constructs exactly this
// solver. u_inlet 0 (ramped).
lbm::Config solver_config(const TunnelSettings& s);

// Grid presets: fast 256x96x96, balanced 320x128x128, fine 384x160x160.
// Unknown names give fast.
TunnelSettings tunnel_preset(const std::string& name);

// A body to place: triangles in the model's own units + spinning parts.
struct Model {
    std::string id; // catalogue id, or the STL file name
    std::string label;
    geometry::Mesh mesh;
    std::vector<catalogue::Spinner> spinners;
    std::string hash; // content hash of the mesh (flow-cache key)
};
Model model_from_catalogue(const catalogue::Entry& e);
Model model_from_stl(const std::filesystem::path& path);

struct Placement {
    float length_cells = 64.0f;                       // longest axis when placed
    std::array<float, 3> pos_frac{0.35f, 0.5f, 0.5f}; // centre, fractions of the grid
    float aoa_deg = 0.0f, yaw_deg = 0.0f, roll_deg = 0.0f;
    catalogue::Ground ground = catalogue::Ground::Air;
    float ride_height = 0.0f; // above the belt, cells
    bool operator==(const Placement&) const = default;
};
// The catalogue entry's natural setting, then fitted to the grid: moved
// downstream so the nose clears the inlet, and shrunk if its cross-section
// or length would not fit. (A fixed x = 0.35 would leave the rocket, the
// APFSDS, the AIM-120 and the city block partly outside the tunnel.)
Placement default_placement(const catalogue::Entry& e, const TunnelSettings& s,
                            const geometry::Mesh& mesh);

enum class AreaMode : int { Frontal = 0, Planform = 1, Manual = 2 };

inline constexpr int kMaxProbes = 4;

// What the measurements on top of the flow found (THEORY 12), refreshed
// every TunnelSettings::analysis_every batches. Spectra are in cycles per
// solver step; probe components are u_x, u_y, u_z and rho.
struct TunnelAnalysis {
    std::uint64_t version = 0;
    // time averaging (12.1)
    int avg_samples = 0;
    double avg_flow_throughs = 0.0;
    // wake survey (12.2): a control-volume momentum balance of the mean flow
    // between the upstream plane and the survey plane, beside the force
    // balance's mean over the same window
    bool wake_valid = false;
    int x_upstream = 0, x_survey = 0;
    double cd_wake = 0.0, cd_balance = 0.0;
    double mass_imbalance = 0.0;              // relative change of the mass flux between the planes
    bool includes_floor = false;              // ground mode: the floor's shear is inside the volume
    std::vector<float> profile_y, profile_uy; // <u_x> / U along y at the survey plane
    std::vector<float> profile_z, profile_uz; // <u_x> / U along z
    // spectra of the developed signal (12.3)
    double l_ref = 1.0, u_ref = 0.05; // Strouhal scales: the body's height, the freestream
    double window_steps = 0.0;
    Spectrum lift;        // of Cl
    double st_lift = 0.0; // peak frequency x l_ref / u_ref
    // The side walls reflect sound: transverse standing waves at multiples
    // of f_acoustic = c_s / (2 max(ny, nz)) show in every spectrum. The
    // shedding peak is the strongest one below 0.8 f_acoustic.
    double f_acoustic = 0.0;  // cycles per step
    double st_shedding = 0.0; // 0 when the lift has no peak below the first mode
    int n_probes = 0;
    std::array<std::array<Spectrum, 4>, kMaxProbes> probe_spec;
    // histories for plotting (decimated): solver step, Cl, probe values
    std::vector<double> hist_t, hist_cl;
    std::array<std::array<std::vector<float>, 4>, kMaxProbes> hist_probe;
};

struct TunnelStatus {
    std::string phase; // PAUSED / RAMPING / the settling status
    std::string model_label;
    bool paused = false;
    bool developing = true;
    bool settled = false;
    double flow_throughs = 0.0;
    std::int64_t steps = 0;
    float u_applied = 0.0f, u_command = 0.0f;
    double cd = 0, cl = 0, cs = 0, cm = 0; // EMA coefficients
    double re_sim = 0.0;
    double a_ref = 1.0, a_frontal = 1.0, a_planform = 1.0, a_manual = 100.0;
    AreaMode area_mode = AreaMode::Frontal;
    double rho_ref = 1.0; // upstream reference plane mean density
    int x_ref = 0;
    float max_speed = 0.0f;
    int bad_cells = 0;
    std::string health_note;
    double health_note_age = 1e9; // seconds since the note
    std::size_t n_tris = 0, n_solid = 0;
    double vox_ms = 0.0;
    bool out_of_bounds = false;
    std::array<float, 3> placed_centre{};
    float ext_y_half = 8.0f, ext_z_half = 8.0f;
    bool has_spinners = false;
    float spin_scale = 1.0f;
    std::string cache_note;
    int cache_entries = 0;
    float inlet_turbulence_pct = 0.0f;
    bool dye_on = false;
    std::uint64_t geometry_version = 0;
    std::uint64_t flow_epoch = 0; // bumps on every reset / restore
    // statistics and signals
    bool averaging = false;        // switched on
    bool averaging_active = false; // on, and the flow is developed: sampling
    int avg_samples = 0;
    double avg_flow_throughs = 0.0;
    std::uint64_t analysis_version = 0;
    // transonic mode
    bool transonic = false;
    float mach_applied = 0.0f, mach_command = 0.8f, peak_mach = 0.0f;
    // the flags the renderer should show: the LBM's, or the Euler grid's
    std::uint64_t render_geometry = 0;
};

class Tunnel {
  public:
    Tunnel(Context& ctx, const TunnelSettings& s, std::filesystem::path cache_dir);
    ~Tunnel();
    Tunnel(const Tunnel&) = delete;
    Tunnel& operator=(const Tunnel&) = delete;

    const TunnelSettings& settings() const { return s_; }
    lbm::Solver& solver() { return *solver_; }
    Dye* dye() { return dye_.get(); }

    // -- model -------------------------------------------------------------
    // Load a body and place it: voxelised at once. A new body, or the same
    // one turned > 20 deg or resized > 25 %, restarts the flow from rest with
    // the ramp (an impulsive start can reach |u| 0.31); a small
    // nudge keeps the developed flow. Then the flow cache is consulted.
    void set_model(Model m, const Placement& p);
    void set_placement(const Placement& p);
    const Placement& placement() const { return placement_; }
    const Model& model() const { return model_; }

    // -- operating point -----------------------------------------------------
    void set_speed(float u_command);
    void set_spin(bool on, float ratio);
    void set_turbulence(float percent);
    void set_area_mode(AreaMode m, double manual = -1.0);
    void set_dye(bool on);
    // Dye nozzles follow the smoke wand: plane x, centre, half-extents.
    void set_dye_rake(float x, float cy, float cz, float half_y, float half_z);
    void set_paused(bool p) { paused_ = p; }

    // -- regime ------------------------------------------------------------------
    // Transonic mode hands the tunnel to the compressible Euler solver on the
    // same grid and model (allocated on first use; the model re-voxelised
    // into it with a slip floor and an exact wall distance). The LBM keeps
    // its state for switching back. Smoke, dye, vortex cores and spin are
    // subsonic-only. Inviscid: pressure + wave drag, no skin friction.
    void set_transonic(bool on);
    void set_mach(float m);
    // "restart flow": from freestream in transonic mode, else reset_flow().
    void restart_flow() {
        if (transonic_ && euler_)
            restart_transonic("");
        else
            reset_flow("flow reset");
    }
    bool transonic() const { return transonic_; }
    euler::Solver* euler_solver() { return euler_.get(); }
    void reset_flow(const std::string& reason); // from rest; never the cache
    void skip_develop() { develop_.finish(); }

    // -- statistics and signals (THEORY 12) -------------------------------------
    // Time averaging: while on, every batch of a developed flow adds a sample
    // weighted by its steps. A new operating point clears the window, which
    // refills once the flow has settled again.
    void set_averaging(bool on);
    void restart_averaging() { clear_statistics(false); }
    const FlowStats* stats() const { return stats_.get(); } // null until first switched on
    // Probes: up to kMaxProbes positions (cells) whose (u, rho) are recorded
    // every batch.
    void set_probes(const std::vector<std::array<float, 3>>& positions);
    // The wake-survey plane, in cells (< 0: half-way between the body and
    // the sponge). Kept behind the body and clear of the sponge.
    void set_wake_plane(int x) { wake_x_ = x; }
    int wake_plane() const;
    // The latest analysis (refreshed every analysis_every batches).
    const TunnelAnalysis& analysis() const { return analysis_; }
    void refresh_analysis();

    // -- running ---------------------------------------------------------------
    // One batch of `steps` solver steps with everything around them: ramp /
    // slew, belt speed, dye, forces, guard, settling, cache. No-op if paused.
    void advance(int steps);
    TunnelStatus status() const;

  private:
    void revoxelise();
    void load_euler();
    void advance_transonic(int steps);
    void restart_transonic(const std::string& why);
    void begin_operating_point(const std::string& reason);
    void apply_spin();
    void diverged(float umax, int n_bad);
    void maybe_save_settled_flow();
    void build_dye_nozzles();
    void clear_statistics(bool signals_too);
    void record_signals(const lbm::MeanForces& mf);
    std::string operating_point_key() const;
    geometry::Mesh placed_mesh() const;
    double q_dyn() const;

    Context& ctx_;
    TunnelSettings s_;
    std::unique_ptr<lbm::Solver> solver_;
    std::unique_ptr<Voxeliser> vox_;
    std::unique_ptr<Dye> dye_;
    FlowCache cache_;
    ConvergenceMonitor develop_;

    Model model_;
    Placement placement_;
    std::optional<Placement> placed_last_;
    std::vector<std::uint8_t> flags_;

    // freestream
    float u_command_, u_applied_ = 0.0f, u_at_develop_;
    std::int64_t steps_done_ = 0;
    bool paused_ = false;

    // forces
    std::array<double, 3> force_ema_{}, torque_ema_{};
    AreaMode area_mode_ = AreaMode::Frontal;
    double a_frontal_ = 1.0, a_planform_ = 1.0, a_manual_ = 100.0;
    double rho_ref_ = 1.0;
    int x_ref_ = 0;

    // placement results
    Voxeliser::Stats vox_stats_{};
    double vox_ms_ = 0.0;
    bool out_of_bounds_ = false;
    std::array<float, 3> placed_centre_{};
    float ext_y_half_ = 8.0f, ext_z_half_ = 8.0f;

    // spin
    bool spin_on_ = false;
    float spin_ratio_ = 1.0f, spin_scale_ = 1.0f;

    // turbulence, dye
    float turb_pct_ = 0.0f;
    bool dye_on_ = false;
    std::array<float, 5> dye_rake_{};

    // guard + health
    lbm::Health health_{};
    std::vector<std::chrono::steady_clock::time_point> divergences_;
    std::string health_note_;
    std::chrono::steady_clock::time_point health_note_t_{};
    int batch_ = 0;

    // cache
    std::string cache_key_, cache_saved_for_, cache_note_;
    int cache_entries_ = 0;
    std::uint64_t flow_epoch_ = 0;

    // statistics and signals
    struct SignalSample {
        std::int64_t step = 0;
        float cd = 0, cl = 0, cs = 0;
        std::uint32_t probe_epoch = 0;
        std::array<std::array<float, 4>, kMaxProbes> probe{};
    };
    std::unique_ptr<FlowStats> stats_;
    bool averaging_ = false;
    std::array<double, 3> force_sum_{};
    double force_w_ = 0.0, avg_ft_ = 0.0;
    std::vector<std::array<float, 3>> probes_;
    std::vector<std::size_t> probe_cells_;
    std::uint32_t probe_epoch_ = 0;
    std::deque<SignalSample> signals_;
    std::int64_t signal_start_ = -1; // the step the flow was first developed at
    int wake_x_ = -1, body_x_max_ = 0;
    TunnelAnalysis analysis_;
    std::uint64_t analysis_version_ = 0;

    // transonic
    std::unique_ptr<euler::Solver> euler_;
    bool transonic_ = false;
    bool euler_stale_ = true; // the model changed since the Euler grid was built
    float mach_command_ = 0.8f, mach_applied_ = 0.8f, peak_mach_ = 0.0f;
    std::array<double, 3> euler_force_ema_{};
    double euler_time0_ = 0.0, euler_time_ = 0.0;
};

} // namespace windoa
