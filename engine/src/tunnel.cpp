#include "windoa/tunnel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace windoa {

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kPi = 3.14159265358979323846;
constexpr double kCs2 = 1.0 / 3.0;

// R = Ry(yaw) Rz(pitch) Rx(roll), the stl.transform convention.
std::array<std::array<double, 3>, 3> rotation(double yaw, double pitch, double roll) {
    const double d = kPi / 180.0;
    const double cy = std::cos(yaw * d), sy = std::sin(yaw * d);
    const double cp = std::cos(pitch * d), sp = std::sin(pitch * d);
    const double cr = std::cos(roll * d), sr = std::sin(roll * d);
    const double ry[3][3] = {{cy, 0, sy}, {0, 1, 0}, {-sy, 0, cy}};
    const double rz[3][3] = {{cp, -sp, 0}, {sp, cp, 0}, {0, 0, 1}};
    const double rx[3][3] = {{1, 0, 0}, {0, cr, -sr}, {0, sr, cr}};
    double t[3][3] = {};
    std::array<std::array<double, 3>, 3> m{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                t[i][j] += rz[i][k] * rx[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                m[i][j] += ry[i][k] * t[k][j];
    return m;
}

std::string fmt(double v, int digits) {
    char b[48];
    std::snprintf(b, sizeof(b), "%.*f", digits, v);
    return b;
}

} // namespace

TunnelSettings tunnel_preset(const std::string& name) {
    TunnelSettings s;
    if (name == "balanced") {
        s.nx = 320;
        s.ny = s.nz = 128;
    } else if (name == "fine") {
        s.nx = 384;
        s.ny = s.nz = 160;
    }
    return s;
}

Model model_from_catalogue(const catalogue::Entry& e) {
    Model m;
    m.id = e.id;
    m.label = e.label;
    m.mesh = e.build();
    m.spinners = e.spinners;
    m.hash = content_hash(m.mesh.xyz.data(), m.mesh.xyz.size() * sizeof(float));
    return m;
}

Model model_from_stl(const std::filesystem::path& path) {
    Model m;
    m.id = path.filename().string();
    m.label = path.stem().string();
    m.mesh = geometry::load_stl(path.string());
    m.hash = content_hash(m.mesh.xyz.data(), m.mesh.xyz.size() * sizeof(float));
    return m;
}

Placement default_placement(const catalogue::Entry& e, const TunnelSettings& s,
                            const geometry::Mesh& mesh) {
    Placement p;
    p.length_cells = e.size_frac > 0.0 ? float(e.size_frac * s.nx) : 64.0f;
    p.ground = e.ground;
    if (!mesh.empty()) {
        const geometry::Bounds b = geometry::bounds(mesh);
        const float k = p.length_cells / std::max(b.longest(), 1e-12f);
        const float ex = (b.hi[0] - b.lo[0]) * k, ey = (b.hi[1] - b.lo[1]) * k,
                    ez = (b.hi[2] - b.lo[2]) * k;
        // cross-section within 90 % of the tunnel; length within inlet margin
        // + sponge + a wake allowance
        float fit = 1.0f;
        const float room_y =
            0.9f * s.ny - (e.ground != catalogue::Ground::Air ? s.floor_height + 2.0f : 0.0f);
        if (ey > room_y)
            fit = std::min(fit, room_y / ey);
        if (ez > 0.9f * s.nz)
            fit = std::min(fit, 0.9f * s.nz / ez);
        const float room_x = s.nx - 0.06f * s.nx - s.outlet_sponge - 0.1f * s.nx;
        if (ex > room_x)
            fit = std::min(fit, room_x / ex);
        p.length_cells *= fit;
        const float half_x = 0.5f * ex * fit;
        p.pos_frac[0] = std::max(p.pos_frac[0], (half_x + 0.06f * s.nx) / s.nx);
    }
    if (e.ground == catalogue::Ground::Road)
        p.ride_height = std::max(4.0f, 0.06f * p.length_cells); // a resolved under-body gap
    return p;
}

lbm::Config solver_config(const TunnelSettings& s) {
    lbm::Config c;
    c.nx = s.nx;
    c.ny = s.ny;
    c.nz = s.nz;
    c.tau = s.tau;
    c.u_inlet = 0.0f;
    c.smagorinsky_cs = s.smagorinsky_cs;
    c.regularised = s.regularised;
    c.recursive = s.recursive;
    c.moving_boundaries = true; // spinning wheels / rotors
    c.outlet_sponge = s.outlet_sponge;
    c.sponge_target = s.sponge_target;
    c.storage_f16 = s.storage_f16;
    return c;
}

Tunnel::Tunnel(Context& ctx, const TunnelSettings& s, std::filesystem::path cache_dir)
    : ctx_(ctx), s_(s), cache_(std::move(cache_dir), s.flow_cache_mb), u_command_(s.u_inlet),
      u_at_develop_(s.u_inlet) {
    solver_ = std::make_unique<lbm::Solver>(ctx, solver_config(s));
    vox_ = std::make_unique<Voxeliser>(ctx, s.nx, s.ny, s.nz);
    flags_.assign(solver_->cells(), lbm::FLUID);
    x_ref_ = s.nx / 8;
    cache_entries_ = cache_.entries();
    solver_->init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
}

Tunnel::~Tunnel() = default;

// -- model -------------------------------------------------------------------------

geometry::Mesh Tunnel::placed_mesh() const {
    const geometry::Vec3 centre{placement_.pos_frac[0] * s_.nx, placement_.pos_frac[1] * s_.ny,
                                placement_.pos_frac[2] * s_.nz};
    geometry::Mesh m = geometry::fit_to_box(model_.mesh, centre, placement_.length_cells);
    // Positive AoA = nose up for a +x-pointing model = clockwise about +z.
    m = geometry::transform(m, placement_.yaw_deg, -placement_.aoa_deg, placement_.roll_deg,
                            centre);
    if (placement_.ground != catalogue::Ground::Air && !m.empty()) {
        // Drop onto the floor: lowest point at floor + ride height. Under-
        // body gaps stay fluid, so the air runs under the model.
        float ymin = 1e30f;
        for (std::size_t i = 1; i < m.xyz.size(); i += 3)
            ymin = std::min(ymin, m.xyz[i]);
        const float shift = float(s_.floor_height) + placement_.ride_height - ymin;
        for (std::size_t i = 1; i < m.xyz.size(); i += 3)
            m.xyz[i] += shift;
    }
    return m;
}

void Tunnel::set_model(Model m, const Placement& p) {
    model_ = std::move(m);
    placement_ = p;
    placed_last_.reset(); // a new body: restart from rest
    revoxelise();
}

void Tunnel::set_placement(const Placement& p) {
    if (p == placement_ && !model_.mesh.empty())
        return;
    placement_ = p;
    revoxelise();
}

void Tunnel::revoxelise() {
    if (model_.mesh.empty())
        return;
    // New body? (a big turn / resize counts as one)
    bool new_body = !placed_last_.has_value();
    if (placed_last_) {
        auto dang = [](float a, float b) {
            const float d = std::abs(a - b);
            return std::min(d, 360.0f - d);
        };
        new_body = dang(placement_.aoa_deg, placed_last_->aoa_deg) > 20.0f ||
                   dang(placement_.yaw_deg, placed_last_->yaw_deg) > 20.0f ||
                   dang(placement_.roll_deg, placed_last_->roll_deg) > 20.0f ||
                   std::abs(placement_.length_cells - placed_last_->length_cells) >
                       0.25f * placed_last_->length_cells;
    }
    placed_last_ = placement_;

    const geometry::Mesh tris = placed_mesh();
    // The floor goes in before voxelising (the voxeliser keeps WALL / LID):
    // rolling road = LID cells moving at the wind speed; fixed ground = a
    // stationary no-slip WALL (buildings, a turbine).
    const std::uint8_t floor = placement_.ground == catalogue::Ground::Fixed ? lbm::WALL : lbm::LID;
    std::fill(flags_.begin(), flags_.end(), lbm::FLUID);
    if (placement_.ground != catalogue::Ground::Air) {
        for (int x = 0; x < s_.nx; ++x)
            for (int y = 0; y < std::min(s_.floor_height, s_.ny); ++y)
                for (int z = 0; z < s_.nz; ++z)
                    flags_[(std::size_t(x) * s_.ny + y) * s_.nz + z] = floor;
    }
    const auto t0 = Clock::now();
    vox_stats_ = vox_->voxelise(tris, flags_);
    vox_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    solver_->set_flags(flags_);

    // Reference areas: frontal (x-projection, car convention), planform
    // (y-projection, wing convention); extents; the upstream reference
    // plane half-way between the inlet and the model's front face.
    const int nx = s_.nx, ny = s_.ny, nz = s_.nz;
    std::vector<std::uint8_t> front(std::size_t(ny) * nz, 0), plan(std::size_t(nx) * nz, 0);
    int ylo = ny, yhi = -1, zlo = nz, zhi = -1, xlo = nx, xhi = -1;
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                if (flags_[(std::size_t(x) * ny + y) * nz + z] != lbm::OBSTACLE)
                    continue;
                front[std::size_t(y) * nz + z] = 1;
                plan[std::size_t(x) * nz + z] = 1;
                ylo = std::min(ylo, y);
                yhi = std::max(yhi, y);
                zlo = std::min(zlo, z);
                zhi = std::max(zhi, z);
                xlo = std::min(xlo, x);
                xhi = std::max(xhi, x);
            }
    a_frontal_ = std::max(1.0, double(std::count(front.begin(), front.end(), 1)));
    a_planform_ = std::max(1.0, double(std::count(plan.begin(), plan.end(), 1)));
    if (yhi >= 0) {
        ext_y_half_ = std::max((yhi - ylo) / 2.0f, 2.0f);
        ext_z_half_ = std::max((zhi - zlo) / 2.0f, 2.0f);
    }
    const geometry::Bounds b = geometry::bounds(tris);
    placed_centre_ = b.centre();
    out_of_bounds_ = b.lo[0] < 1.0f || b.lo[1] < 1.0f || b.lo[2] < 1.0f || b.hi[0] > nx - 1.0f ||
                     b.hi[1] > ny - 1.0f || b.hi[2] > nz - 1.0f;
    const int x_front = yhi >= 0 ? xlo : nx / 4;
    x_ref_ = std::max(3, x_front / 2);
    body_x_max_ = yhi >= 0 ? xhi : nx / 2;
    solver_->set_torque_ref({placed_centre_[0], placed_centre_[1], placed_centre_[2]});

    if (new_body)
        reset_flow("new model"); // from rest, ramped (the cache may still win)
    apply_spin();
    if (dye_)
        build_dye_nozzles();
    begin_operating_point("model placed");
    euler_stale_ = true;
    if (transonic_)
        load_euler();
}

// -- transonic ----------------------------------------------------------------------------

void Tunnel::set_transonic(bool on) {
    if (on == transonic_)
        return;
    transonic_ = on;
    if (on && (!euler_ || euler_stale_))
        load_euler();
    clear_statistics(true); // the statistics describe the lattice flow only
    ++flow_epoch_;
}

void Tunnel::set_mach(float m) {
    mach_command_ = std::clamp(m, s_.mach_min, s_.mach_max);
}

// The model into the Euler grid: a slip floor (WALL) in ground mode, the
// voxelised body, and the exact signed distance its image-point walls need
// (a distance estimated from the flags alone puts the wedge's shock 13 % off).
void Tunnel::load_euler() {
    if (model_.mesh.empty())
        return;
    if (!euler_) {
        euler::Config c;
        c.nx = s_.nx;
        c.ny = s_.ny;
        c.nz = s_.nz;
        c.mach = mach_applied_;
        c.side_bc = euler::SideBC::Farfield;
        euler_ = std::make_unique<euler::Solver>(ctx_, c);
    }
    std::vector<std::uint8_t> ef(solver_->cells(), lbm::FLUID);
    if (placement_.ground != catalogue::Ground::Air)
        for (int x = 0; x < s_.nx; ++x)
            for (int y = 0; y < std::min(s_.floor_height, s_.ny); ++y)
                for (int z = 0; z < s_.nz; ++z)
                    ef[(std::size_t(x) * s_.ny + y) * s_.nz + z] = lbm::WALL;
    vox_->voxelise(placed_mesh(), ef);
    euler_->set_flags(ef);
    euler_->set_distance(vox_->signed_distance(ef));
    euler_stale_ = false;
    restart_transonic("");
}

void Tunnel::restart_transonic(const std::string& why) {
    euler_->set_mach(mach_applied_);
    euler_->init_freestream();
    euler_force_ema_ = {};
    euler_time0_ = euler_time_ = 0.0;
    peak_mach_ = 0.0f;
    ++flow_epoch_;
    if (!why.empty()) {
        health_note_ = "transonic flow diverged (" + why + ") -- restarted from freestream";
        health_note_t_ = Clock::now();
        std::fprintf(stderr, "[guard] %s\n", health_note_.c_str());
    }
}

void Tunnel::advance_transonic(int steps) {
    // Slew the Mach number: a slider jump must not shock the whole tunnel.
    const float d = std::clamp(mach_command_ - mach_applied_, -s_.mach_slew, s_.mach_slew);
    if (d != 0.0f) {
        mach_applied_ += d;
        euler_->set_mach(mach_applied_);
    }
    const double t0 = euler_time_;
    euler_->step(steps);
    const std::array<double, 3> f = euler_->body_force();
    euler_time_ = euler_->time();
    ++batch_;
    if (!std::isfinite(f[0]) || !std::isfinite(f[1]) || !std::isfinite(f[2])) {
        restart_transonic("non-finite force");
        return;
    }
    const double a = 1.0 - std::exp(-std::max(euler_time_ - t0, 1e-9) / s_.force_ema_time);
    for (int k = 0; k < 3; ++k)
        euler_force_ema_[k] += a * (f[k] - euler_force_ema_[k]);
    if (batch_ % std::max(s_.health_every, 1) == 0) {
        const euler::Solver::Health h = euler_->health();
        peak_mach_ = h.max_mach;
        if (h.bad_cells > 0 || !std::isfinite(h.max_mach) || h.max_mach > 5.0f)
            restart_transonic("max Mach " + fmt(h.max_mach, 2) + ", " +
                              std::to_string(h.bad_cells) + " bad cells");
    } else {
        euler_->refresh(); // render fields for the snapshot
    }
}

// -- operating point ------------------------------------------------------------------

void Tunnel::set_speed(float u) {
    u_command_ = std::clamp(u, 0.005f, s_.u_max);
    apply_spin(); // omega follows the speed (spin ratio = rim / U)
}

void Tunnel::set_spin(bool on, float ratio) {
    const bool changed = on != spin_on_ || std::abs(ratio - spin_ratio_) > 0.005f;
    spin_on_ = on;
    spin_ratio_ = ratio;
    if (!changed)
        return;
    apply_spin();
    if (!model_.spinners.empty())
        begin_operating_point("spin change");
}

void Tunnel::set_turbulence(float pct) {
    pct = std::round(pct * 10.0f) / 10.0f;
    if (pct == turb_pct_)
        return;
    turb_pct_ = pct;
    solver_->set_inlet_turbulence(pct / 100.0f, s_.turbulence_length, 512,
                                  std::max(u_applied_, 1e-3f));
    begin_operating_point("inlet turbulence");
}

void Tunnel::set_area_mode(AreaMode m, double manual) {
    area_mode_ = m;
    if (manual > 0.0)
        a_manual_ = manual;
}

void Tunnel::set_dye(bool on) {
    dye_on_ = on;
    if (on && !dye_) { // ~2 x 7 floats per cell: allocated on first use
        dye_ = std::make_unique<Dye>(ctx_, *solver_, s_.dye_tau, s_.dye_tau_plus);
        build_dye_nozzles();
    }
}

void Tunnel::set_dye_rake(float x, float cy, float cz, float half_y, float half_z) {
    const std::array<float, 5> r{x, cy, cz, half_y, half_z};
    if (r == dye_rake_)
        return;
    dye_rake_ = r;
    if (dye_)
        build_dye_nozzles();
}

// An n x n grid of single-cell nozzles on the wand plane, like a real smoke
// rake (9 x 9 nozzles of 2 x 2 cells merged into one opaque sheet).
void Tunnel::build_dye_nozzles() {
    const int nx = s_.nx, ny = s_.ny, nz = s_.nz, n = s_.dye_nozzles;
    std::vector<float> src(solver_->cells(), 0.0f);
    const int xr = std::clamp(int(dye_rake_[0]), 2, nx - 3);
    const float cy = dye_rake_[1], cz = dye_rake_[2], hy = dye_rake_[3], hz = dye_rake_[4];
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            const float j = n > 1 ? cy - hy + 2 * hy * a / float(n - 1) : cy;
            const float k = n > 1 ? cz - hz + 2 * hz * b / float(n - 1) : cz;
            const int jj = std::clamp(int(j), 1, ny - 3), kk = std::clamp(int(k), 1, nz - 3);
            for (int x = xr - 1; x < xr + 1; ++x) {
                const std::size_t c = (std::size_t(x) * ny + jj) * nz + kk;
                if (flags_[c] == lbm::FLUID)
                    src[c] = s_.dye_nozzle_rate;
            }
        }
    dye_->set_sources(src);
}

// Refill the wall velocities of the model's spinning parts (or clear them).
// The request is scaled down by one factor for every spinner (rigid
// rotation and relative speeds kept) so that no captured cell moves faster
// than the cap: past ~0.10-0.12 the moving-wall term destabilises, and the
// slider alone could ask for 3 x 0.11. |omega| x capture radius bounds the
// speed of any captured cell. A spinner may carry a lower cap (blades).
//
// In ground mode the model is dropped onto the belt, and the spinner
// centres are dropped with it here: left at the undropped centre, the
// capture cylinders would miss road-car wheels (~40 cells above them at
// the fast preset).
void Tunnel::apply_spin() {
    solver_->clear_wall_velocity();
    spin_scale_ = 1.0f;
    if (!spin_on_ || model_.spinners.empty() || model_.mesh.empty())
        return;
    const geometry::Bounds mb = geometry::bounds(model_.mesh);
    const double scale = placement_.length_cells / std::max(mb.longest(), 1e-12f);
    const std::array<double, 3> centre{placement_.pos_frac[0] * s_.nx,
                                       placement_.pos_frac[1] * s_.ny,
                                       placement_.pos_frac[2] * s_.nz};
    const auto R = rotation(placement_.yaw_deg, -placement_.aoa_deg, placement_.roll_deg);
    // the ground drop (placed_mesh) applied to the spinners too
    double drop = 0.0;
    if (placement_.ground != catalogue::Ground::Air) {
        const geometry::Mesh fitted = geometry::transform(
            geometry::fit_to_box(model_.mesh,
                                 {float(centre[0]), float(centre[1]), float(centre[2])},
                                 placement_.length_cells),
            placement_.yaw_deg, -placement_.aoa_deg, placement_.roll_deg,
            geometry::Vec3{float(centre[0]), float(centre[1]), float(centre[2])});
        drop = s_.floor_height + placement_.ride_height - geometry::bounds(fitted).lo[1];
    }
    struct Part {
        std::array<double, 3> point, omega;
        double capture_r, half_len;
    };
    std::vector<Part> parts;
    for (const catalogue::Spinner& sp : model_.spinners) {
        std::array<double, 3> rel, p{}, ax{};
        for (int k = 0; k < 3; ++k)
            rel[k] = (sp.c[k] - 0.5 * (mb.lo[k] + mb.hi[k])) * scale;
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                p[i] += R[i][k] * rel[k];
                ax[i] += R[i][k] * sp.axis[k];
            }
            p[i] += centre[i];
        }
        p[1] += drop;
        const double an = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
        const double r_cells = std::max(sp.r * scale, 1.0);
        const double w = sp.sense * spin_ratio_ * std::max(double(u_command_), 1e-3) / r_cells;
        Part part{p,
                  {ax[0] / an * w, ax[1] / an * w, ax[2] / an * w},
                  sp.r * scale * 1.2,
                  sp.hl * scale * 1.4};
        const double fastest = std::abs(w) * part.capture_r;
        const double cap = std::min(double(s_.max_wall_speed),
                                    sp.max_wall > 0.0 ? sp.max_wall : double(s_.max_wall_speed));
        if (fastest > cap)
            spin_scale_ = std::min(spin_scale_, float(cap / fastest));
        parts.push_back(part);
    }
    for (const Part& p : parts) {
        solver_->set_rotation({float(p.point[0]), float(p.point[1]), float(p.point[2])},
                              {float(p.omega[0] * spin_scale_), float(p.omega[1] * spin_scale_),
                               float(p.omega[2] * spin_scale_)},
                              float(p.capture_r), float(p.half_len));
    }
}

// Everything that determines the settled flow, as a canonical string.
std::string Tunnel::operating_point_key() const {
    const bool spinning = spin_on_ && !model_.spinners.empty();
    std::ostringstream k;
    k << "model=" << model_.hash << "|len=" << fmt(placement_.length_cells, 4)
      << "|aoa=" << fmt(placement_.aoa_deg, 4) << "|yaw=" << fmt(placement_.yaw_deg, 4)
      << "|roll=" << fmt(placement_.roll_deg, 4) << "|pos=" << fmt(placement_.pos_frac[0], 4) << ","
      << fmt(placement_.pos_frac[1], 4) << "," << fmt(placement_.pos_frac[2], 4)
      << "|ground=" << int(placement_.ground) << "|ride=" << fmt(placement_.ride_height, 4)
      << "|spin=" << spinning << "|ratio=" << fmt(spinning ? spin_ratio_ : 0.0, 4);
    if (spinning) // the spinner definitions themselves (a lower blade cap
                  // once restored a flow settled at the old blade speed)
        for (const catalogue::Spinner& sp : model_.spinners)
            k << "|sp=" << fmt(sp.c[0], 4) << "," << fmt(sp.c[1], 4) << "," << fmt(sp.c[2], 4)
              << "," << sp.axis[0] << sp.axis[1] << sp.axis[2] << "," << fmt(sp.r, 4) << ","
              << fmt(sp.hl, 4) << "," << sp.sense << "," << fmt(sp.max_wall, 4);
    k << "|cap=" << fmt(s_.max_wall_speed, 4) << "|tu=" << fmt(turb_pct_, 1)
      << "|u=" << fmt(u_command_, 5) << "|grid=" << s_.nx << "x" << s_.ny << "x" << s_.nz
      << "|tau=" << fmt(s_.tau, 6) << "|cs=" << fmt(s_.smagorinsky_cs, 4)
      << "|reg=" << s_.regularised << "|rr=" << s_.recursive << "|sponge=" << s_.outlet_sponge
      << "," << s_.sponge_target << "|floor=" << s_.floor_height << "|f16=" << s_.storage_f16;
    return cache_.key(k.str());
}

void Tunnel::begin_operating_point(const std::string& reason) {
    u_at_develop_ = u_command_;
    cache_key_ = operating_point_key();
    develop_.restart(reason);
    clear_statistics(true);
    auto hit = cache_.load(cache_key_);
    if (!hit) {
        cache_note_.clear();
        return;
    }
    const auto t0 = Clock::now();
    solver_->set_state(hit->first);
    solver_->read_mean_forces(); // drop the restore step's sums
    force_ema_ = hit->second.force_ema;
    torque_ema_ = hit->second.torque_ema;
    u_applied_ = u_command_; // the cached flow is at speed
    solver_->set_inlet_velocity(u_applied_);
    steps_done_ = std::max<std::int64_t>(steps_done_, s_.ramp_steps);
    develop_.mark_restored();
    cache_saved_for_ = cache_key_;
    ++flow_epoch_;
    cache_note_ = "restored in " +
                  fmt(std::chrono::duration<double>(Clock::now() - t0).count(), 2) + " s (" +
                  std::to_string(cache_.entries()) + " cached)";
}

void Tunnel::reset_flow(const std::string& reason) {
    solver_->init_equilibrium(1.0f, {0.0f, 0.0f, 0.0f});
    solver_->read_mean_forces();
    steps_done_ = 0;
    u_applied_ = 0.0f;
    solver_->set_inlet_velocity(0.0f);
    force_ema_ = {};
    torque_ema_ = {};
    if (dye_)
        dye_->clear();
    develop_.restart(reason);
    u_at_develop_ = u_command_;
    cache_key_ = operating_point_key();
    cache_saved_for_.clear();
    clear_statistics(true);
    ++flow_epoch_;
}

void Tunnel::diverged(float umax, int n_bad) {
    const auto now = Clock::now();
    divergences_.erase(
        std::remove_if(divergences_.begin(), divergences_.end(),
                       [&](Clock::time_point t) { return now - t > std::chrono::seconds(60); }),
        divergences_.end());
    divergences_.push_back(now);
    std::vector<std::string> causes;
    if (spin_on_ && !model_.spinners.empty())
        causes.push_back("spinning parts");
    if (placement_.ground == catalogue::Ground::Road && placement_.ride_height < 4.0f)
        causes.push_back("a thin under-body gap");
    if (u_command_ > 0.09f)
        causes.push_back("high flow speed");
    std::string what = n_bad < 0   ? "forces went non-finite"
                       : n_bad > 0 ? std::to_string(n_bad) + " cells non-finite"
                                   : "max |u| " + fmt(umax, 2) + " > " + fmt(s_.diverge_speed, 2);
    std::string hint;
    for (std::size_t i = 0; i < causes.size(); ++i)
        hint += (i ? " + " : " -- likely ") + causes[i];
    reset_flow("diverged");
    if (divergences_.size() >= 2) {
        paused_ = true; // a second blow-up within a minute: stop resetting into it
        health_note_ = "DIVERGED twice in a minute (" + what + ")" + hint +
                       ". Paused: change the setup, then resume.";
    } else {
        health_note_ = "flow diverged (" + what + ")" + hint + "; reset from rest";
    }
    health_note_t_ = now;
    std::fprintf(stderr, "[guard] %s\n", health_note_.c_str());
}

void Tunnel::maybe_save_settled_flow() {
    if (!develop_.settled() || develop_.restored() || cache_saved_for_ == cache_key_ ||
        cache_key_.empty())
        return;
    if (operating_point_key() != cache_key_)
        return; // changed since it began
    FlowCache::Meta meta{force_ema_, torque_ema_, develop_.flow_throughs()};
    const std::vector<float> f = solver_->get_state();
    const double dt = cache_.save(cache_key_, f, meta);
    cache_saved_for_ = cache_key_;
    cache_entries_ = cache_.entries();
    cache_note_ = "saved to cache (" + fmt(dt, 1) + " s)";
}

double Tunnel::q_dyn() const {
    const double u = std::max({double(u_applied_), 0.25 * u_command_, 1e-4});
    return 0.5 * u * u;
}

// -- running ------------------------------------------------------------------------------

void Tunnel::advance(int steps) {
    if (paused_ || steps <= 0 || model_.mesh.empty())
        return;
    if (transonic_ && euler_) {
        advance_transonic(steps);
        return;
    }

    // A significant speed change is a new operating point: develop again.
    if (!develop_.developing() &&
        std::abs(u_command_ - u_at_develop_) > 0.1f * std::max(u_at_develop_, 1e-3f))
        begin_operating_point("speed change");

    // Ramp from rest, then slew towards the command (per step, applied per batch).
    const float desired =
        u_command_ * std::min(1.0f, float(steps_done_ + steps) / float(std::max(s_.ramp_steps, 1)));
    const float lim = s_.u_slew_per_step * float(steps);
    u_applied_ += std::clamp(desired - u_applied_, -lim, lim);
    solver_->set_inlet_velocity(u_applied_);
    // Rolling road: the belt moves at the freestream (no spurious floor BL).
    solver_->set_lid_velocity(placement_.ground == catalogue::Ground::Road
                                  ? lbm::Vec3{u_applied_, 0.0f, 0.0f}
                                  : lbm::Vec3{0.0f, 0.0f, 0.0f});
    if (turb_pct_ > 0.0f)
        solver_->set_turbulence_convection(std::max(u_applied_, 1e-3f));

    if (dye_on_ && dye_)
        dye_->step_with(*solver_, steps); // lock-step with the flow
    else
        solver_->step(steps);
    steps_done_ += steps;
    ++batch_;

    // Mean force over every step of the batch (a last-step sample aliases
    // against shedding); EMA time constant in solver steps.
    lbm::MeanForces mf = solver_->read_mean_forces();
    bool finite = true;
    for (int k = 0; k < 3; ++k)
        finite = finite && std::isfinite(mf.force[k]) && std::isfinite(mf.torque[k]);
    if (!finite) {
        diverged(0.0f, -1);
        return;
    }
    if (batch_ % std::max(s_.health_every, 1) == 0) {
        health_ = solver_->health();
        if (health_.bad_cells > 0 || health_.max_speed > s_.diverge_speed) {
            diverged(health_.max_speed, health_.bad_cells);
            return;
        }
    }
    if (batch_ % 4 == 0)
        rho_ref_ = solver_->plane_mean_density(x_ref_);
    const double a = 1.0 - std::exp(-mf.steps / s_.force_ema_steps);
    for (int k = 0; k < 3; ++k) {
        force_ema_[k] += a * (mf.force[k] - force_ema_[k]);
        torque_ema_[k] += a * (mf.torque[k] - torque_ema_[k]);
    }
    const double q = q_dyn();
    const TunnelStatus st = status();
    develop_.add({mf.force[0] / (q * st.a_ref), mf.force[1] / (q * st.a_ref)}, mf.steps, u_applied_,
                 s_.nx);
    maybe_save_settled_flow();
    record_signals(mf);
}

// -- statistics and signals -------------------------------------------------------------

void Tunnel::set_averaging(bool on) {
    if (on == averaging_)
        return;
    averaging_ = on;
    clear_statistics(false);
}

void Tunnel::clear_statistics(bool signals_too) {
    if (stats_)
        stats_->reset();
    force_sum_ = {};
    force_w_ = 0.0;
    avg_ft_ = 0.0;
    if (signals_too) {
        signals_.clear();
        signal_start_ = -1;
    }
    analysis_.avg_samples = 0;
    analysis_.avg_flow_throughs = 0.0;
    analysis_.wake_valid = false;
}

void Tunnel::set_probes(const std::vector<std::array<float, 3>>& positions) {
    const std::size_t n = std::min<std::size_t>(positions.size(), kMaxProbes);
    probes_.assign(positions.begin(), positions.begin() + std::ptrdiff_t(n));
    probe_cells_.clear();
    for (const auto& q : probes_) {
        const int x = std::clamp(int(std::floor(q[0])), 0, s_.nx - 1);
        const int y = std::clamp(int(std::floor(q[1])), 0, s_.ny - 1);
        const int z = std::clamp(int(std::floor(q[2])), 0, s_.nz - 1);
        probe_cells_.push_back((std::size_t(x) * s_.ny + y) * s_.nz + z);
    }
    ++probe_epoch_; // older samples belong to the old positions
}

int Tunnel::wake_plane() const {
    const int last = s_.nx - s_.outlet_sponge - 2; // stay clear of the sponge
    const int first = std::min(body_x_max_ + 2, last);
    if (wake_x_ >= 0)
        return std::clamp(wake_x_, first, last);
    return std::clamp(body_x_max_ + std::max(2, (last - body_x_max_) / 2), first, last);
}

void Tunnel::record_signals(const lbm::MeanForces& mf) {
    const bool developed = !develop_.developing();
    if (developed && signal_start_ < 0)
        signal_start_ = steps_done_;

    // Signal history for the spectra: per-batch mean forces (not the EMA,
    // which would filter the shedding) and the probes at the batch's end.
    const TunnelStatus st = status();
    const double qa = q_dyn() * st.a_ref;
    SignalSample smp;
    smp.step = steps_done_;
    smp.cd = float(mf.force[0] / qa);
    smp.cl = float(mf.force[1] / qa);
    smp.cs = float(mf.force[2] / qa);
    smp.probe_epoch = probe_epoch_;
    if (!probe_cells_.empty()) {
        const auto v = solver_->macro_at(probe_cells_);
        for (std::size_t i = 0; i < v.size(); ++i)
            smp.probe[i] = v[i];
    }
    signals_.push_back(smp);
    const double window = s_.history_flow_throughs * s_.nx / std::max(double(u_applied_), 1e-3);
    while (signals_.size() > 2 && double(steps_done_ - signals_.front().step) > window)
        signals_.pop_front();

    // Time averaging of the developed flow.
    if (averaging_ && developed) {
        if (!stats_)
            stats_ = std::make_unique<FlowStats>(ctx_, solver_->cells());
        stats_->add(solver_->macro_buffer(solver_->live_index()), double(mf.steps));
        for (int k = 0; k < 3; ++k)
            force_sum_[k] += mf.force[k] * mf.steps;
        force_w_ += mf.steps;
        avg_ft_ += double(mf.steps) * u_applied_ / s_.nx;
    }
    if (batch_ % std::max(s_.analysis_every, 1) == 0)
        refresh_analysis();
}

void Tunnel::refresh_analysis() {
    TunnelAnalysis a;
    a.version = ++analysis_version_;
    const int nx = s_.nx, ny = s_.ny, nz = s_.nz;
    const TunnelStatus st = status();
    const double qa = q_dyn() * st.a_ref;
    const double u_ref = std::max(double(u_applied_), 1e-6);

    // Time averages and the wake survey (THEORY 12.1, 12.2)
    a.avg_samples = stats_ ? stats_->samples() : 0;
    a.avg_flow_throughs = avg_ft_;
    a.x_upstream = x_ref_;
    a.x_survey = wake_plane();
    if (stats_ && stats_->samples() >= 2 && !transonic_ && a.x_survey > a.x_upstream &&
        a.x_upstream >= 1) {
        const double nu = kCs2 * (s_.tau - 0.5);
        const PlaneFlux f1 = plane_momentum_flux(*stats_, flags_, nx, ny, nz, a.x_upstream, nu);
        const PlaneFlux f2 = plane_momentum_flux(*stats_, flags_, nx, ny, nz, a.x_survey, nu);
        if (f1.cells > 0 && f1.cells == f2.cells) {
            a.wake_valid = true;
            a.cd_wake = (f1.momentum - f2.momentum) / qa;
            a.cd_balance = force_w_ > 0.0 ? force_sum_[0] / force_w_ / qa : 0.0;
            a.mass_imbalance = (f2.mass - f1.mass) / std::max(std::abs(f1.mass), 1e-12);
            a.includes_floor = placement_.ground != catalogue::Ground::Air;
            // mean wake profiles through the body's centre
            std::vector<float> mean, var;
            const std::size_t plane = std::size_t(ny) * nz;
            stats_->read(std::size_t(a.x_survey) * plane, plane, mean, var);
            const int yc = std::clamp(int(placed_centre_[1]), 0, ny - 1);
            const int zc = std::clamp(int(placed_centre_[2]), 0, nz - 1);
            for (int y = 0; y < ny; ++y) {
                const std::size_t c = std::size_t(y) * nz + zc;
                if (flags_[std::size_t(a.x_survey) * plane + c] != lbm::FLUID)
                    continue;
                a.profile_y.push_back(float(y) + 0.5f);
                a.profile_uy.push_back(float(mean[c * 4] / u_ref));
            }
            for (int z = 0; z < nz; ++z) {
                const std::size_t c = std::size_t(yc) * nz + z;
                if (flags_[std::size_t(a.x_survey) * plane + c] != lbm::FLUID)
                    continue;
                a.profile_z.push_back(float(z) + 0.5f);
                a.profile_uz.push_back(float(mean[c * 4] / u_ref));
            }
        }
    }

    // Spectra of the developed signal (THEORY 12.3)
    a.u_ref = u_ref;
    a.l_ref = 2.0 * ext_y_half_;
    a.n_probes = int(probes_.size());
    std::vector<double> t, cl, tp;
    std::array<std::array<std::vector<double>, 4>, kMaxProbes> pv;
    for (const SignalSample& s : signals_) {
        if (signal_start_ < 0 || s.step < signal_start_)
            continue;
        t.push_back(double(s.step));
        cl.push_back(s.cl);
        if (s.probe_epoch == probe_epoch_) {
            tp.push_back(double(s.step));
            for (int i = 0; i < a.n_probes; ++i)
                for (int k = 0; k < 4; ++k)
                    pv[i][k].push_back(s.probe[i][k]);
        }
    }
    a.f_acoustic = std::sqrt(kCs2) / (2.0 * std::max(ny, nz));
    if (t.size() >= 32) {
        a.window_steps = t.back() - t.front();
        a.lift = spectrum(t, cl, 0, 2.0 / a.window_steps);
        a.st_lift = a.lift.peak_freq * a.l_ref / u_ref;
        for (const double f : spectral_peaks(a.lift, 8, 2.0 / a.window_steps))
            if (f < 0.8 * a.f_acoustic) { // peaks come strongest first
                a.st_shedding = f * a.l_ref / u_ref;
                break;
            }
    }
    if (tp.size() >= 32)
        for (int i = 0; i < a.n_probes; ++i)
            for (int k = 0; k < 4; ++k)
                a.probe_spec[i][k] = spectrum(tp, pv[i][k], 0, 2.0 / (tp.back() - tp.front()));

    // Histories for plotting, decimated to at most ~600 points.
    const std::size_t stride = std::max<std::size_t>(1, signals_.size() / 600);
    for (std::size_t i = 0; i < signals_.size(); i += stride) {
        const SignalSample& s = signals_[i];
        a.hist_t.push_back(double(s.step));
        a.hist_cl.push_back(s.cl);
        for (int p = 0; p < a.n_probes; ++p)
            for (int k = 0; k < 4; ++k)
                a.hist_probe[p][k].push_back(s.probe_epoch == probe_epoch_ ? s.probe[p][k]
                                                                           : std::nanf(""));
    }
    analysis_ = std::move(a);
}

TunnelStatus Tunnel::status() const {
    TunnelStatus t;
    t.paused = paused_;
    t.model_label = model_.label;
    t.phase = paused_ ? "PAUSED" : steps_done_ < s_.ramp_steps ? "RAMPING" : develop_.status();
    t.developing = develop_.developing();
    t.settled = develop_.settled();
    t.flow_throughs = develop_.flow_throughs();
    t.steps = solver_->steps_taken();
    t.u_applied = u_applied_;
    t.u_command = u_command_;
    t.a_frontal = a_frontal_;
    t.a_planform = a_planform_;
    t.a_manual = a_manual_;
    t.area_mode = area_mode_;
    t.a_ref = area_mode_ == AreaMode::Planform ? a_planform_
              : area_mode_ == AreaMode::Manual ? a_manual_
                                               : a_frontal_;
    const double qa = q_dyn() * t.a_ref;
    t.cd = force_ema_[0] / qa;
    t.cl = force_ema_[1] / qa;
    t.cs = force_ema_[2] / qa;
    t.cm = torque_ema_[2] / (qa * std::max(placement_.length_cells, 1.0f));
    t.re_sim =
        std::max(double(u_applied_), 1e-6) * placement_.length_cells / (kCs2 * (s_.tau - 0.5));
    t.rho_ref = rho_ref_;
    t.x_ref = x_ref_;
    t.max_speed = health_.max_speed;
    t.bad_cells = health_.bad_cells;
    t.health_note = health_note_;
    t.health_note_age = health_note_.empty()
                            ? 1e9
                            : std::chrono::duration<double>(Clock::now() - health_note_t_).count();
    t.n_tris = vox_stats_.n_tris;
    t.n_solid = vox_stats_.n_solid;
    t.vox_ms = vox_ms_;
    t.out_of_bounds = out_of_bounds_;
    t.placed_centre = placed_centre_;
    t.ext_y_half = ext_y_half_;
    t.ext_z_half = ext_z_half_;
    t.has_spinners = !model_.spinners.empty();
    t.spin_scale = spin_scale_;
    t.cache_note = cache_note_;
    t.cache_entries = cache_entries_;
    t.inlet_turbulence_pct = turb_pct_;
    t.dye_on = dye_on_ && dye_ != nullptr;
    t.geometry_version = solver_->geometry_version();
    t.flow_epoch = flow_epoch_;
    t.render_geometry = solver_->geometry_version();
    t.mach_command = mach_command_;
    t.mach_applied = mach_applied_;
    t.averaging = averaging_;
    t.averaging_active = averaging_ && !develop_.developing() && !transonic_;
    t.avg_samples = stats_ ? stats_->samples() : 0;
    t.avg_flow_throughs = avg_ft_;
    t.analysis_version = analysis_version_;
    if (transonic_ && euler_) {
        // Euler units: rho_inf = 1, c_inf = 1, so q = M^2 / 2 per unit area.
        t.transonic = true;
        t.render_geometry = euler_->geometry_version() | (1ull << 40);
        t.peak_mach = peak_mach_;
        const double ft = euler_time_ * std::max(mach_applied_, 1e-3f) / s_.nx;
        t.flow_throughs = ft;
        t.developing = ft < s_.develop_flow_throughs;
        t.settled = !t.developing;
        t.steps = euler_->steps_taken();
        char buf[128];
        std::snprintf(buf, sizeof(buf), "TRANSONIC  M %.3f -> %.3f   %s %.1f%s flow-throughs",
                      mach_applied_, mach_command_, t.developing ? "DEVELOPING" : "", ft,
                      t.developing ? (" / " + fmt(s_.develop_flow_throughs, 0)).c_str() : "");
        t.phase = paused_ ? "PAUSED" : buf;
        const double q =
            0.5 * double(std::max(mach_applied_, 1e-3f)) * mach_applied_ * std::max(t.a_ref, 1.0);
        t.cd = euler_force_ema_[0] / q;
        t.cl = euler_force_ema_[1] / q;
        t.cs = euler_force_ema_[2] / q;
        t.cm = 0.0; // (the transonic pressure moment is not computed yet)
        t.re_sim = 0.0;
    }
    return t;
}

} // namespace windoa
