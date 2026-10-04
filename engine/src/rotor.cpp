#include "windoa/rotor.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "alm_elements_spv.hpp"
#include "alm_spread_spv.hpp"

namespace windoa {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Mirrors `Params` in shaders/alm_elements.comp.
struct ElemParams {
    float hub[4];           // xyz, angle
    float axis[4];          // xyz, omega
    float e1[4];            // xyz, r_hub
    std::int32_t counts[4]; // blades, elements per blade, mode, element offset
    float polar[4];         // cl_alpha, alpha_stall, cd0, r_tip
    float grid[4];          // nx, ny, nz, disc force per element
    std::int32_t extra[4];  // section offset
};
static_assert(sizeof(ElemParams) == 112);

// Mirrors `Params` in shaders/alm_spread.comp.
struct SpreadParams {
    std::int32_t lo[4];   // box origin, first element
    std::int32_t size[4]; // box extent, elements
    std::int32_t grid[4];
    float kern[4]; // eps, 1 / (eps^3 pi^1.5), cut-off^2
};

// table (evenly spaced root to tip) at fraction t in [0, 1]
float lerp_table(const std::vector<float>& v, double t, float fallback) {
    if (v.empty())
        return fallback;
    if (v.size() == 1)
        return v[0];
    const double x = std::clamp(t, 0.0, 1.0) * double(v.size() - 1);
    const std::size_t i = std::min(std::size_t(x), v.size() - 2);
    return float(v[i] + (x - double(i)) * (v[i + 1] - v[i]));
}

} // namespace

ActuatorLines::ActuatorLines(Context& ctx, lbm::Solver& solver, std::vector<RotorSpec> rotors,
                             float eps)
    : ctx_(ctx), solver_(solver), rotors_(std::move(rotors)), eps_(eps),
      elem_(ctx, spv::alm_elements, 3, sizeof(ElemParams), 2),
      spread_(ctx, spv::alm_spread, 2, sizeof(SpreadParams)) {
    const lbm::Config& c = solver.config();
    std::vector<float> sections;
    std::array<int, 3> lo{INT_MAX, INT_MAX, INT_MAX}, hi{INT_MIN, INT_MIN, INT_MIN};
    for (RotorSpec& r : rotors_) {
        const double an = std::sqrt(double(r.axis[0]) * r.axis[0] + double(r.axis[1]) * r.axis[1] +
                                    double(r.axis[2]) * r.axis[2]);
        for (float& v : r.axis)
            v = float(v / std::max(an, 1e-12));
        if (r.disc) {
            r.blades = r.disc_spokes;
            n_el_.push_back(r.disc_rings);
        } else {
            n_el_.push_back(r.elements > 0
                                ? r.elements
                                : std::clamp(int(std::lround(r.r_tip - r.r_hub)), 8, 64));
        }
        offset_.push_back(total_);
        total_ += r.blades * n_el_.back();
        sec_offset_.push_back(int(sections.size() / 2));
        for (int i = 0; i < n_el_.back(); ++i) {
            const double t = (i + 0.5) / n_el_.back();
            sections.push_back(lerp_table(r.chord, t, 1.0f));
            sections.push_back(float(lerp_table(r.twist_deg, t, 0.0f) * kPi / 180.0));
        }
        // e1: a unit vector normal to the axis (the blade at angle 0)
        const std::array<float, 3> a = r.axis;
        std::array<float, 3> ref = std::abs(a[1]) < 0.9f ? std::array<float, 3>{0.0f, 1.0f, 0.0f}
                                                         : std::array<float, 3>{1.0f, 0.0f, 0.0f};
        const float d = ref[0] * a[0] + ref[1] * a[1] + ref[2] * a[2];
        std::array<float, 3> e{ref[0] - d * a[0], ref[1] - d * a[1], ref[2] - d * a[2]};
        const float en = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        for (float& v : e)
            v /= en;
        e1_.push_back(e);
        angle_.push_back(0.0);
        // its reach: the disc of radius r_tip + 3 eps, 3 eps either side of the plane
        const double reach = r.r_tip + 3.0 * eps_, thick = 3.0 * eps_ + 1.0;
        for (int k = 0; k < 3; ++k) {
            const double h = reach * std::sqrt(std::max(0.0, 1.0 - double(a[k]) * a[k])) +
                             thick * std::abs(a[k]);
            lo[k] = std::min(lo[k], int(std::floor(r.hub[k] - h)));
            hi[k] = std::max(hi[k], int(std::ceil(r.hub[k] + h)));
        }
    }
    // One box around every rotor's reach, spread in one pass over all the
    // elements: a box per rotor, written in turn, would overwrite a neighbour's
    // force wherever two boxes overlap.
    const int dims[3] = {c.nx, c.ny, c.nz};
    for (int k = 0; k < 3 && !rotors_.empty(); ++k) {
        box_.lo[k] = std::max(0, lo[k]);
        box_.size[k] = std::max(0, std::min(dims[k] - 1, hi[k]) - box_.lo[k] + 1);
    }
    elements_ = std::make_unique<Buffer>(ctx, std::size_t(std::max(total_, 1)) * 32);
    sections_ = std::make_unique<Buffer>(ctx, std::max<std::size_t>(sections.size(), 2) * 4);
    if (!sections.empty())
        ctx_.upload(*sections_, sections.data(), sections.size() * 4);
    ctx_.fill_zero(*elements_);
    solver_.enable_force_field(true);
    for (int m = 0; m < 2; ++m)
        elem_.bind(std::uint32_t(m), {&solver_.macro_buffer(m), elements_.get(), sections_.get()});
    spread_.bind({elements_.get(), &solver_.force_field_buffer()});
}

ActuatorLines::~ActuatorLines() {
    solver_.enable_force_field(false);
}

void ActuatorLines::set_omega(std::size_t rotor, float omega) {
    rotors_.at(rotor).omega = omega;
}

void ActuatorLines::record_step(VkCommandBuffer cmd, int macro_index) {
    const lbm::Config& c = solver_.config();
    for (std::size_t k = 0; k < rotors_.size(); ++k) {
        const RotorSpec& r = rotors_[k];
        ElemParams p{};
        for (int i = 0; i < 3; ++i) {
            p.hub[i] = r.hub[std::size_t(i)];
            p.axis[i] = r.axis[std::size_t(i)];
            p.e1[i] = e1_[k][std::size_t(i)];
        }
        p.hub[3] = float(angle_[k]);
        p.axis[3] = r.omega;
        p.e1[3] = r.r_hub;
        p.counts[0] = r.blades;
        p.counts[1] = n_el_[k];
        p.counts[2] = r.disc ? 1 : 0;
        p.counts[3] = offset_[k];
        p.polar[0] = r.cl_alpha;
        p.polar[1] = float(r.alpha_stall_deg * kPi / 180.0);
        p.polar[2] = r.cd0;
        p.polar[3] = r.r_tip;
        p.grid[0] = float(c.nx);
        p.grid[1] = float(c.ny);
        p.grid[2] = float(c.nz);
        p.grid[3] = r.disc ? r.disc_force / float(r.blades * n_el_[k]) : 0.0f;
        p.extra[0] = sec_offset_[k];
        const std::uint32_t n = std::uint32_t(r.blades * n_el_[k]);
        elem_.record_set(cmd, std::uint32_t(macro_index), &p, (n + 63) / 64);
    }
    Context::barrier_compute_to_compute(cmd);
    const float norm = float(1.0 / (double(eps_) * eps_ * eps_ * std::pow(kPi, 1.5)));
    SpreadParams p{};
    for (int i = 0; i < 3; ++i) {
        p.lo[i] = box_.lo[std::size_t(i)];
        p.size[i] = box_.size[std::size_t(i)];
    }
    p.lo[3] = 0;
    p.size[3] = total_;
    p.grid[0] = c.nx;
    p.grid[1] = c.ny;
    p.grid[2] = c.nz;
    p.kern[0] = eps_;
    p.kern[1] = norm;
    p.kern[2] = 9.0f * eps_ * eps_;
    const std::uint32_t cells = std::uint32_t(box_.size[0] * box_.size[1] * box_.size[2]);
    if (cells > 0)
        spread_.record(cmd, &p, (cells + 63) / 64);
    Context::barrier_compute_to_compute(cmd);
    for (std::size_t k = 0; k < rotors_.size(); ++k)
        angle_[k] = std::fmod(angle_[k] + rotors_[k].omega, 2.0 * kPi);
}

void ActuatorLines::step_with(lbm::Solver& solver, int n) {
    solver.step(n, [&](VkCommandBuffer cmd, int macro_index) { record_step(cmd, macro_index); });
}

std::vector<ActuatorLines::Loads> ActuatorLines::loads() {
    std::vector<float> e(std::size_t(std::max(total_, 1)) * 8);
    ctx_.download(*elements_, e.data(), e.size() * 4);
    std::vector<Loads> out(rotors_.size());
    for (std::size_t k = 0; k < rotors_.size(); ++k) {
        const RotorSpec& r = rotors_[k];
        Loads& l = out[k];
        for (int i = 0; i < r.blades * n_el_[k]; ++i) {
            const float* x = &e[std::size_t(8 * (offset_[k] + i))];
            const float* f = x + 4;
            const double d[3] = {x[0] - r.hub[0], x[1] - r.hub[1], x[2] - r.hub[2]};
            const double t[3] = {d[1] * f[2] - d[2] * f[1], d[2] * f[0] - d[0] * f[2],
                                 d[0] * f[1] - d[1] * f[0]};
            for (int a = 0; a < 3; ++a) {
                l.force[std::size_t(a)] += f[a];
                l.thrust += f[a] * r.axis[std::size_t(a)];
                l.torque += t[a] * r.axis[std::size_t(a)];
            }
        }
        l.power = l.torque * r.omega;
    }
    return out;
}

std::vector<std::array<std::array<float, 3>, 2>> ActuatorLines::blade_lines() const {
    std::vector<std::array<std::array<float, 3>, 2>> out;
    for (std::size_t k = 0; k < rotors_.size(); ++k) {
        const RotorSpec& r = rotors_[k];
        if (r.disc)
            continue;
        const std::array<float, 3>& a = r.axis;
        const std::array<float, 3>& e1 = e1_[k];
        const std::array<float, 3> e2{a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2],
                                      a[0] * e1[1] - a[1] * e1[0]};
        for (int b = 0; b < r.blades; ++b) {
            const double psi = angle_[k] + 2.0 * kPi * b / r.blades;
            std::array<std::array<float, 3>, 2> seg{};
            for (int i = 0; i < 3; ++i) {
                const float er =
                    float(std::cos(psi) * e1[std::size_t(i)] + std::sin(psi) * e2[std::size_t(i)]);
                seg[0][std::size_t(i)] = r.hub[std::size_t(i)] + r.r_hub * er;
                seg[1][std::size_t(i)] = r.hub[std::size_t(i)] + r.r_tip * er;
            }
            out.push_back(seg);
        }
    }
    return out;
}

std::vector<std::array<std::array<float, 3>, 2>> ActuatorLines::blade_outlines() const {
    std::vector<std::array<std::array<float, 3>, 2>> out;
    for (std::size_t k = 0; k < rotors_.size(); ++k) {
        const RotorSpec& r = rotors_[k];
        if (r.disc)
            continue;
        const std::array<float, 3>& a = r.axis;
        const std::array<float, 3>& e1 = e1_[k];
        const std::array<float, 3> e2{a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2],
                                      a[0] * e1[1] - a[1] * e1[0]};
        const float c_root = lerp_table(r.chord, 0.0, 1.0f), c_tip = lerp_table(r.chord, 1.0, 1.0f);
        const float s = r.omega < 0.0f ? -1.0f : 1.0f;
        for (int b = 0; b < r.blades; ++b) {
            const double psi = angle_[k] + 2.0 * kPi * b / r.blades;
            std::array<float, 3> er{}, et{};
            for (int i = 0; i < 3; ++i)
                er[std::size_t(i)] =
                    float(std::cos(psi) * e1[std::size_t(i)] + std::sin(psi) * e2[std::size_t(i)]);
            for (int i = 0; i < 3; ++i) // et = s (a x er)
                et[std::size_t(i)] =
                    s * (a[std::size_t((i + 1) % 3)] * er[std::size_t((i + 2) % 3)] -
                         a[std::size_t((i + 2) % 3)] * er[std::size_t((i + 1) % 3)]);
            auto at = [&](float rad, float chord_frac, float chord) {
                std::array<float, 3> p{};
                for (int i = 0; i < 3; ++i)
                    p[std::size_t(i)] = r.hub[std::size_t(i)] + rad * er[std::size_t(i)] +
                                        chord_frac * chord * et[std::size_t(i)];
                return p;
            };
            const auto rl = at(r.r_hub, 0.25f, c_root), rt = at(r.r_hub, -0.75f, c_root);
            const auto tl = at(r.r_tip, 0.25f, c_tip), tt = at(r.r_tip, -0.75f, c_tip);
            out.push_back({rl, tl});
            out.push_back({tl, tt});
            out.push_back({tt, rt});
            out.push_back({rt, rl});
        }
    }
    return out;
}

} // namespace windoa
