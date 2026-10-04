// The procedural catalogue. Built in double precision as triangle soups,
// converted to float meshes at the end. The comments on each model record
// why a shape is the way it is (several were rebuilt after reading wrong
// on screen).
#include "windoa/catalogue.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace windoa::catalogue {

namespace {

constexpr double kPi = 3.14159265358979323846;

using V3 = std::array<double, 3>;
using Tri = std::array<V3, 3>;
using Soup = std::vector<Tri>;

V3 operator+(V3 a, V3 b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
V3 operator-(V3 a, V3 b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
V3 operator*(V3 a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}
double dot(V3 a, V3 b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
V3 cross(V3 a, V3 b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
V3 tri_mean(const Tri& t) {
    return (t[0] + t[1] + t[2]) * (1.0 / 3.0);
}
double rad(double deg) {
    return deg * kPi / 180.0;
}

// n evenly spaced values from a to b (b included unless endpoint is false)
std::vector<double> linspace(double a, double b, int n, bool endpoint = true) {
    std::vector<double> v(n);
    const double d = n > 1 ? (b - a) / (endpoint ? n - 1 : n) : 0.0;
    for (int i = 0; i < n; ++i)
        v[i] = a + d * i;
    if (endpoint && n > 1)
        v[n - 1] = b;
    return v;
}

// piecewise-linear interpolation (xp increasing; clamped at the ends)
double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    if (x <= xp.front())
        return fp.front();
    if (x >= xp.back())
        return fp.back();
    for (std::size_t i = 1; i < xp.size(); ++i) {
        if (x <= xp[i]) {
            const double t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
            return fp[i - 1] + t * (fp[i] - fp[i - 1]);
        }
    }
    return fp.back();
}

// sorted, de-duplicated
std::vector<double> unique(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

void append(Soup& a, const Soup& b) {
    a.insert(a.end(), b.begin(), b.end());
}

Soup translate(Soup s, V3 d) {
    for (Tri& t : s)
        for (V3& v : t)
            v = v + d;
    return s;
}

Soup scale(Soup s, double k) {
    for (Tri& t : s)
        for (V3& v : t)
            v = v * k;
    return s;
}

// Flip the triangles whose normal faces the centre (convex primitives).
Soup orient(Soup s, V3 centre) {
    for (Tri& t : s) {
        const V3 n = cross(t[1] - t[0], t[2] - t[0]);
        if (dot(tri_mean(t) - centre, n) < 0.0)
            std::swap(t[0], t[2]);
    }
    return s;
}

// stl.transform: M = Ry(yaw) Rz(pitch) Rx(roll) about `about`.
Soup rotate(Soup s, double yaw, double pitch, double roll, V3 about = {0, 0, 0}) {
    const double cy = std::cos(rad(yaw)), sy = std::sin(rad(yaw));
    const double cp = std::cos(rad(pitch)), sp = std::sin(rad(pitch));
    const double cr = std::cos(rad(roll)), sr = std::sin(rad(roll));
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
    for (Tri& tr : s)
        for (V3& v : tr) {
            const V3 r = v - about;
            v = V3{m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
                   m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
                   m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2]} +
                about;
        }
    return s;
}

geometry::Mesh to_mesh(const Soup& s) {
    geometry::Mesh m;
    m.xyz.reserve(s.size() * 9);
    for (const Tri& t : s)
        for (const V3& v : t)
            for (double c : v)
                m.xyz.push_back(static_cast<float>(c));
    return m;
}

// -- primitives ----------------------------------------------------------------

Soup box(double cx, double cy, double cz, double sx, double sy, double sz) {
    const double hx = sx / 2, hy = sy / 2, hz = sz / 2;
    V3 corner[8];
    int n = 0;
    for (int dx : {-1, 1})
        for (int dy : {-1, 1})
            for (int dz : {-1, 1})
                corner[n++] = {cx + dx * hx, cy + dy * hy, cz + dz * hz};
    const int faces[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1},
                             {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
    Soup s;
    for (const auto& f : faces) {
        s.push_back({corner[f[0]], corner[f[1]], corner[f[2]]});
        s.push_back({corner[f[0]], corner[f[2]], corner[f[3]]});
    }
    return orient(s, {cx, cy, cz});
}

Soup cyl(V3 c, int axis, double radius, double length, int nseg = 20, bool cap = true) {
    int pq[2], k = 0;
    for (int a = 0; a < 3; ++a)
        if (a != axis)
            pq[k++] = a;
    const double h = length / 2;
    auto pt = [&](double a, double s) {
        V3 x = c;
        x[axis] += s * h;
        x[pq[0]] += radius * std::cos(a);
        x[pq[1]] += radius * std::sin(a);
        return x;
    };
    const std::vector<double> ang = linspace(0.0, 2 * kPi, nseg, false);
    Soup s;
    for (int i = 0; i < nseg; ++i) {
        const double a0 = ang[i], a1 = ang[(i + 1) % nseg];
        const V3 A = pt(a0, -1), B = pt(a1, -1), C = pt(a1, 1), D = pt(a0, 1);
        s.push_back({A, B, C});
        s.push_back({A, C, D});
        if (cap) {
            V3 lo = c, hi = c;
            lo[axis] -= h;
            hi[axis] += h;
            s.push_back({lo, B, A});
            s.push_back({hi, D, C});
        }
    }
    return orient(s, c);
}

Soup disc(V3 c, int axis, double radius, double thickness = 0.12, int nseg = 24) {
    return cyl(c, axis, radius, thickness, nseg, true);
}

// Loft through closed sections (same point count), capped, each face
// oriented outward about the local section centre. Convex sections.
Soup loft_closed(const std::vector<std::vector<V3>>& secs, const std::vector<V3>& cens) {
    Soup s;
    const std::size_t n = secs[0].size();
    for (std::size_t j = 0; j + 1 < secs.size(); ++j) {
        const auto& p0 = secs[j];
        const auto& p1 = secs[j + 1];
        const V3 cen = (cens[j] + cens[j + 1]) * 0.5;
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t k1 = (k + 1) % n;
            for (Tri t : {Tri{p0[k], p1[k], p1[k1]}, Tri{p0[k], p1[k1], p0[k1]}}) {
                const V3 nrm = cross(t[1] - t[0], t[2] - t[0]);
                if (dot(nrm, tri_mean(t) - cen) < 0)
                    std::swap(t[0], t[2]);
                s.push_back(t);
            }
        }
    }
    const std::size_t last = secs.size() - 1;
    const std::array<std::array<std::size_t, 2>, 2> caps = {{{0, 1}, {last, last - 1}}};
    for (const auto& cp : caps) {
        const auto& pts = secs[cp[0]];
        const V3 cen = cens[cp[0]];
        const V3 out = cen - cens[cp[1]]; // cap faces away
        for (std::size_t k = 0; k < n; ++k) {
            Tri t{cen, pts[k], pts[(k + 1) % n]};
            const V3 nrm = cross(t[1] - t[0], t[2] - t[0]);
            if (dot(nrm, out) < 0)
                std::swap(t[0], t[2]);
            s.push_back(t);
        }
    }
    return s;
}

double naca0012_half_thickness(double x) {
    const double t = 0.12;
    return 5 * t *
           (0.2969 * std::sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x -
            0.1036 * x * x * x * x);
}

// Closed NACA 4-digit section (x, y) loop, LE at x = 0, cosine-spaced,
// pitched nose-up by incidence_deg about the LE.
std::vector<std::array<double, 2>> naca_section(double chord, double t_frac, double camber = 0.0,
                                                double camber_pos = 0.4, double incidence_deg = 0.0,
                                                int n = 14) {
    const std::vector<double> beta = linspace(0.0, kPi, n);
    std::vector<double> xc(n), yt(n), yc(n);
    const double m = camber, pp = camber_pos;
    for (int i = 0; i < n; ++i) {
        xc[i] = 0.5 * (1.0 - std::cos(beta[i]));
        yt[i] = naca0012_half_thickness(xc[i]) * (t_frac / 0.12);
        yc[i] = xc[i] < pp
                    ? m / (pp * pp) * (2 * pp * xc[i] - xc[i] * xc[i])
                    : m / ((1 - pp) * (1 - pp)) * ((1 - 2 * pp) + 2 * pp * xc[i] - xc[i] * xc[i]);
    }
    std::vector<std::array<double, 2>> loop;
    for (int i = 0; i < n; ++i)
        loop.push_back({xc[i] * chord, (yc[i] + yt[i]) * chord});
    for (int i = n - 2; i >= 1; --i)
        loop.push_back({xc[i] * chord, (yc[i] - yt[i]) * chord});
    const double a = rad(incidence_deg); // nose up = TE down
    for (auto& p : loop) {
        const double x = p[0], y = p[1];
        p = {std::cos(a) * x + std::sin(a) * y, -std::sin(a) * x + std::cos(a) * y};
    }
    return loop;
}

V3 centroid(const std::vector<V3>& pts) {
    V3 c{0, 0, 0};
    for (const V3& p : pts)
        c = c + p;
    return c * (1.0 / double(pts.size()));
}

// Tapered, swept wing / tail panel from the root LE outward. side = +1 / -1
// for +z / -z; vertical = a fin spanning +y.
Soup lifting_surface(V3 root_le, double root_chord, double tip_chord, double span, double sweep_deg,
                     double dihedral_deg, double t_frac, int side, bool vertical = false,
                     int n_span = 6, double camber = 0.0, double incidence_deg = 0.0) {
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double f : linspace(0.0, 1.0, n_span)) {
        const double c = root_chord + (tip_chord - root_chord) * f;
        const double b = span * f;
        V3 le = root_le;
        le[0] += b * std::tan(rad(sweep_deg));
        const auto loop = naca_section(c, t_frac, camber, 0.4, incidence_deg);
        std::vector<V3> pts;
        if (vertical) {
            le[1] += b;
            for (const auto& p : loop)
                pts.push_back({le[0] + p[0], le[1], le[2] + p[1]});
        } else {
            le[2] += side * b * std::cos(rad(dihedral_deg));
            le[1] += b * std::sin(rad(dihedral_deg));
            for (const auto& p : loop)
                pts.push_back({le[0] + p[0], le[1] + p[1], le[2]});
        }
        cens.push_back(centroid(pts));
        secs.push_back(std::move(pts));
    }
    return loft_closed(secs, cens);
}

Soup body_of_revolution(const std::vector<double>& xs, const std::vector<double>& radii,
                        double y0 = 0.0, double z0 = 0.0,
                        const std::vector<double>* y_off = nullptr, int nseg = 28) {
    const std::vector<double> ang = linspace(0.0, 2 * kPi, nseg, false);
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double yc = y0 + (y_off ? (*y_off)[i] : 0.0);
        const double r = std::max(radii[i], 1e-3);
        std::vector<V3> pts;
        for (double a : ang)
            pts.push_back({xs[i], yc + r * std::sin(a), z0 + r * std::cos(a)});
        secs.push_back(std::move(pts));
        cens.push_back({xs[i], yc, z0});
    }
    return loft_closed(secs, cens);
}

double tangent_ogive(double x, double length, double radius) {
    const double rho = (radius * radius + length * length) / (2.0 * radius);
    return std::sqrt(std::max(rho * rho - (length - x) * (length - x), 0.0)) + radius - rho;
}

std::vector<Soup> fins(int n, double x_le, double root_c, double tip_c, double span,
                       double sweep_deg, double t_frac, double roll0_deg, double r_body) {
    std::vector<Soup> out;
    const Soup panel = lifting_surface({x_le, 0.0, 0.0}, root_c, tip_c, r_body + span, sweep_deg,
                                       0.0, t_frac, 1, false, 3);
    for (int k = 0; k < n; ++k)
        out.push_back(rotate(panel, 0, 0, roll0_deg + k * 360.0 / n));
    return out;
}

Soup axis_y(const Soup& s) {
    return rotate(s, 0.0, 90.0, 0.0);
} // +x -> +y

Soup blade(double r_root, double r_tip, const std::function<double(double)>& chord,
           const std::function<double(double)>& twist_deg, double t_frac, int n = 10,
           double camber = 0.02) {
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double r : linspace(r_root, r_tip, n)) {
        const double c = chord(r);
        const double th = rad(twist_deg(r));
        auto loop = naca_section(c, t_frac, camber);
        const V3 d{-std::sin(th), 0.0, std::cos(th)}; // chord direction
        const V3 m{std::cos(th), 0.0, std::sin(th)};  // thickness direction
        std::vector<V3> pts;
        for (auto& p : loop) {
            p[0] -= 0.25 * c; // pitch about c / 4
            pts.push_back(d * p[0] + m * p[1] + V3{0.0, r, 0.0});
        }
        secs.push_back(std::move(pts));
        cens.push_back({0.0, r, 0.0});
    }
    return loft_closed(secs, cens);
}

std::vector<Soup> rotor(int n_blades, const Soup& one) {
    std::vector<Soup> out;
    for (int k = 0; k < n_blades; ++k)
        out.push_back(rotate(one, 0, 0, k * 360.0 / n_blades));
    return out;
}

Soup concat(const std::vector<Soup>& parts) {
    Soup s;
    for (const Soup& p : parts)
        append(s, p);
    return s;
}

// A round tube from p0 to p1 (struts, wishbones, a halo).
Soup tube(V3 p0, V3 p1, double r, int nseg = 10) {
    const V3 d = p1 - p0;
    const double len = std::sqrt(dot(d, d));
    const Soup s = body_of_revolution({0.0, len}, {r, r}, 0.0, 0.0, nullptr, nseg);
    const double yaw = std::atan2(-d[2], d[0]) * 180.0 / kPi;
    const double pitch = std::atan2(d[1], std::sqrt(d[0] * d[0] + d[2] * d[2])) * 180.0 / kPi;
    return translate(rotate(s, yaw, pitch, 0.0), p0);
}

// A tyre about the z axis at c: rounded shoulders, the sidewalls closing to
// 0.78 r at the rim.
Soup tyre(V3 c, double r, double hw, int nseg = 32) {
    const Soup s =
        body_of_revolution({-hw, -0.75 * hw, -0.35 * hw, 0.35 * hw, 0.75 * hw, hw},
                           {0.78 * r, 0.95 * r, r, r, 0.95 * r, 0.78 * r}, 0.0, 0.0, nullptr, nseg);
    return translate(rotate(s, -90.0, 0.0, 0.0), c);
}

// A loft of superellipse sections along x: half-width w(x), bottom and top
// y(x); exponent n (2 an ellipse, larger squarer).
Soup rounded_loft(const std::vector<double>& xs, const std::function<double(double)>& w,
                  const std::function<double(double)>& bottom,
                  const std::function<double(double)>& top, double n_exp, int nang = 36,
                  double zc = 0.0) {
    const std::vector<double> ang = linspace(0.0, 2 * kPi, nang, false);
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double x : xs) {
        const double yb = bottom(x), yt = top(x), hw = std::max(w(x), 1e-4);
        const double yc = 0.5 * (yb + yt), h = std::max(0.5 * (yt - yb), 1e-4);
        std::vector<V3> pts;
        for (double a : ang) {
            const double ca = std::cos(a), sa = std::sin(a);
            const double ez = (ca > 0 ? 1 : ca < 0 ? -1 : 0) * std::pow(std::abs(ca), 2.0 / n_exp);
            const double ey = (sa > 0 ? 1 : sa < 0 ? -1 : 0) * std::pow(std::abs(sa), 2.0 / n_exp);
            pts.push_back({x, yc + h * ey, zc + hw * ez});
        }
        secs.push_back(std::move(pts));
        cens.push_back({x, yc, zc});
    }
    return loft_closed(secs, cens);
}

// A closed (r, y) profile revolved about the y axis: concave shapes too (a
// cup). Each face is oriented from the profile's own outward normal, as the
// winding-number fill needs.
Soup revolve(std::vector<std::array<double, 2>> prof, int nseg = 48) {
    double area = 0.0; // the profile anticlockwise in (r, y)
    for (std::size_t i = 0; i < prof.size(); ++i) {
        const auto& a = prof[i];
        const auto& b = prof[(i + 1) % prof.size()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    if (area < 0.0)
        std::reverse(prof.begin(), prof.end());
    const std::vector<double> ang = linspace(0.0, 2 * kPi, nseg, false);
    auto at = [](const std::array<double, 2>& p, double a) {
        return V3{p[0] * std::cos(a), p[1], p[0] * std::sin(a)};
    };
    Soup s;
    for (std::size_t i = 0; i < prof.size(); ++i) {
        const auto& p = prof[i];
        const auto& q = prof[(i + 1) % prof.size()];
        const double nr = q[1] - p[1], ny = -(q[0] - p[0]); // outward, in the profile plane
        for (int k = 0; k < nseg; ++k) {
            const double a0 = ang[std::size_t(k)], a1 = ang[std::size_t((k + 1) % nseg)];
            const double am = a0 + kPi / nseg; // the segment's mid-angle
            const V3 out{nr * std::cos(am), ny, nr * std::sin(am)};
            for (Tri t :
                 {Tri{at(p, a0), at(q, a0), at(q, a1)}, Tri{at(p, a0), at(q, a1), at(p, a1)}}) {
                const V3 n = cross(t[1] - t[0], t[2] - t[0]);
                if (dot(n, n) < 1e-24)
                    continue; // degenerate on the axis
                if (dot(n, out) < 0.0)
                    std::swap(t[0], t[2]);
                s.push_back(t);
            }
        }
    }
    return s;
}

// -- models ----------------------------------------------------------------------

Soup sphere_soup(int n_lat, int n_lon) {
    Soup s;
    auto pt = [](double th, double ph) {
        return V3{0.5 * std::sin(th) * std::cos(ph), 0.5 * std::cos(th),
                  0.5 * std::sin(th) * std::sin(ph)};
    };
    for (int i = 0; i < n_lat; ++i) {
        const double t0 = kPi * i / n_lat, t1 = kPi * (i + 1) / n_lat;
        for (int j = 0; j < n_lon; ++j) {
            const double p0 = 2 * kPi * j / n_lon, p1 = 2 * kPi * (j + 1) / n_lon;
            const V3 a = pt(t0, p0), b = pt(t1, p0), c = pt(t1, p1), d = pt(t0, p1);
            if (i > 0)
                s.push_back({a, d, b});
            if (i < n_lat - 1)
                s.push_back({b, d, c});
        }
    }
    return s;
}

// Saloon (Jaguar XE proportions), mm: length 4672, width 1850, height 1416,
// wheelbase 2835, track ~1602, clearance ~115, tyres ~670 mm.
constexpr double kSaloonL = 4672.0, kSaloonH = 1416.0;
constexpr double kSaloonWheelR = 335.0, kSaloonWheelHW = 112.0;
const V3 kSaloonWheels[4] = {
    {835.0, 335.0, 801.0}, {835.0, 335.0, -801.0}, {3670.0, 335.0, 801.0}, {3670.0, 335.0, -801.0}};

// XE-style body: a loft of superellipse sections with the XE's side profile
// and plan taper, tumblehome above the beltline, and wheel arches (the
// first saloon buried its wheels in the body: it read as a shoe).
Soup saloon_body(int n_sec = 56) {
    const double L = kSaloonL, H = kSaloonH;
    const std::vector<double> tx = {0,    40,   150,  400,  900,  1500, 1720, 2150,
                                    2600, 3000, 3500, 3950, 4450, 4600, 4672};
    const std::vector<double> ty = {560,  680,  770,  830,  890,  950, 1060, 1360,
                                    1416, 1395, 1190, 1035, 1015, 960, 870};
    const std::vector<double> bx = {0, 60, 250, 700, 4050, 4450, 4672};
    const std::vector<double> by = {330, 200, 140, 115, 115, 190, 420};
    const std::vector<double> zx = {0, 50, 250, 700, 1500, 3500, 4200, 4550, 4672};
    const std::vector<double> zz = {560, 700, 850, 910, 925, 925, 905, 840, 720};
    const double belt = 990.0, n_exp = 3.4;
    const double ra = kSaloonWheelR + 45.0;            // arch radius
    const double z_in = 801.0 - kSaloonWheelHW - 25.0; // arch inner edge
    auto arch_floor = [&](double x) {
        double h = 0.0;
        for (int w : {0, 2}) {
            const double d = x - kSaloonWheels[w][0];
            if (std::abs(d) < ra)
                h = std::max(h, kSaloonWheels[w][1] + std::sqrt(ra * ra - d * d));
        }
        return h;
    };
    std::vector<double> st;
    for (double u : linspace(0.0, kPi, n_sec))
        st.push_back((0.5 - 0.5 * std::cos(u)) * L);
    st.insert(st.end(), tx.begin(), tx.end());
    for (int w : {0, 2})
        for (double v : linspace(kSaloonWheels[w][0] - ra, kSaloonWheels[w][0] + ra, 13))
            st.push_back(v);
    for (double& v : st)
        v = std::clamp(v, 0.0, L);
    st = unique(st);
    const std::vector<double> ang = linspace(0.0, 2 * kPi, 64, false);
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double x : st) {
        const double yb = interp(x, bx, by), yt = interp(x, tx, ty), w = interp(x, zx, zz);
        const double yc = 0.5 * (yb + yt), h = 0.5 * (yt - yb);
        const double af = arch_floor(x);
        std::vector<V3> pts;
        for (double a : ang) {
            const double ca = std::cos(a), sa = std::sin(a);
            const double ez = (ca > 0 ? 1 : ca < 0 ? -1 : 0) * std::pow(std::abs(ca), 2.0 / n_exp);
            const double ey = (sa > 0 ? 1 : sa < 0 ? -1 : 0) * std::pow(std::abs(sa), 2.0 / n_exp);
            double y = yc + h * ey;
            const double k = std::clamp((y - belt) / (H - belt), 0.0, 1.0);
            const double z = w * ez * (1.0 - 0.30 * k); // tumblehome
            if (af > 0.0 && std::abs(z) > z_in)
                y = std::max(y, af);
            pts.push_back({x, y, z});
        }
        secs.push_back(std::move(pts));
        cens.push_back({x, std::max(yc, af + 40.0), 0.0});
    }
    return loft_closed(secs, cens);
}

std::vector<Soup> saloon_wheels() {
    std::vector<Soup> out;
    for (const V3& c : kSaloonWheels)
        out.push_back(cyl(c, 2, kSaloonWheelR, 2 * kSaloonWheelHW, 32));
    return out;
}

Soup saloon() {
    std::vector<Soup> parts = {saloon_body()};
    for (const V3& c : kSaloonWheels)
        parts.push_back(tyre(c, kSaloonWheelR, kSaloonWheelHW));
    for (int side : {1, -1}) { // door mirrors on stalks at the A-pillar base
        parts.push_back(tube({1640.0, 985.0, side * 860.0}, {1650.0, 1005.0, side * 975.0}, 16.0));
        parts.push_back(rounded_loft(
            linspace(1585.0, 1735.0, 6), [](double) { return 62.0; }, [](double) { return 950.0; },
            [](double) { return 1070.0; }, 3.0, 20, side * 1030.0));
    }
    parts.push_back(box(330.0, 135.0, 0.0, 420.0, 20.0, 1480.0)); // front splitter lip
    for (double z : {-560.0, -280.0, 280.0, 560.0})               // rear diffuser strakes
        parts.push_back(box(4250.0, 150.0, z, 400.0, 70.0, 10.0));
    return concat(parts);
}

// The saloon with a rear wing: an inverted cambered aerofoil (300 mm chord,
// 1.5 m span, 6 deg nose-down) on two pylons above the boot, endplates --
// a downforce study (A/B against the plain saloon).
Soup saloon_wing() {
    Soup s = saloon();
    const double deck = 1015.0, y_w = deck + 190.0;
    for (int side : {1, -1}) {
        append(s, lifting_surface({4300.0, y_w, 0.0}, 300.0, 300.0, 750.0, 0.0, 0.0, 0.12, side,
                                  false, 4, -0.04, -6.0));
        append(s, box(4450.0, 0.5 * (deck + y_w) - 10.0, side * 360.0, 150.0, (y_w - deck) + 40.0,
                      22.0));                                                 // pylon
        append(s, box(4450.0, y_w - 10.0, side * 755.0, 340.0, 170.0, 12.0)); // endplate
    }
    return s;
}

// MQ-9 Reaper-style UAV: a faired fuselage, sensor ball, high-aspect
// cambered wing, down-canted V-tail and ventral fin (aerofoil sections);
// the rear pusher propeller is an actuator line. Nose at x = 0.
Soup reaper() {
    const double y0 = 3.0;
    Soup s = body_of_revolution({0.0, 0.6, 1.8, 3.5, 8.5, 10.6, 11.0},
                                {0.15, 0.45, 0.60, 0.65, 0.62, 0.35, 0.25}, y0, 0.0, nullptr, 24);
    append(s, translate(scale(sphere_soup(16, 24), 1.5), {0.9, y0 - 0.55, 0.0})); // sensor ball
    for (int side : {1, -1}) {
        append(s, lifting_surface({4.95, y0 + 0.35, 0.0}, 1.4, 0.7, 10.0, 2.0, 1.0, 0.15, side,
                                  false, 6, 0.04, 2.0)); // wing
        append(s, lifting_surface({9.4, y0 + 0.1, 0.0}, 1.6, 0.9, 3.0, 25.0, -35.0, 0.10, side,
                                  false, 3)); // V-tail
    }
    append(s, box(10.4, y0 - 0.6, 0.0, 1.4, 1.2, 0.12)); // ventral fin
    append(s, cyl({11.05, y0, 0.0}, 0, 0.25, 0.4, 16));  // prop hub (the blades: a rotor)
    return s;
}

// DJI-style X-frame quadcopter: a rounded body and canopy, streamlined arms,
// motor pods, battery, a camera gimbal at low x and tubular skids; the four
// two-blade rotors are actuator lines.
Soup dji_quad() {
    const double yb = 3.0, reach = 3.0;
    Soup s = rounded_loft(
        linspace(-0.85, 0.85, 10), [](double) { return 0.80; }, [](double) { return 3.0 - 0.35; },
        [](double) { return 3.0 + 0.35; }, 4.0, 28);
    append(s, rounded_loft(
                  linspace(-0.5, 0.55, 8), [](double) { return 0.5; },
                  [](double) { return 3.0 + 0.2; }, [](double) { return 3.0 + 0.62; }, 2.6,
                  24));                                   // canopy
    append(s, box(0.15, yb + 0.72, 0.0, 0.9, 0.18, 0.6)); // battery
    const int dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (const auto& d : dirs) {
        const double mx = d[0] * reach, mz = d[1] * reach;
        const double yaw = -std::atan2(mz, mx) * 180.0 / kPi;
        const double len = std::sqrt(mx * mx + mz * mz);
        const Soup arm = rounded_loft(
            linspace(0.0, len, 6), [](double) { return 0.16; }, [](double) { return 3.0 - 0.10; },
            [](double) { return 3.0 + 0.14; }, 2.2, 16);
        append(s, rotate(arm, yaw, 0.0, 0.0, {0.0, 0.0, 0.0}));
        append(s, cyl({mx, yb + 0.2, mz}, 1, 0.45, 0.8, 16)); // motor (the blades: a rotor)
    }
    for (double lz : {0.55, -0.55}) { // skids
        append(s, tube({0.35, yb - 0.3, lz}, {0.35, yb - 1.2, lz * 1.3}, 0.06, 8));
        append(s, tube({-0.35, yb - 0.3, lz}, {-0.35, yb - 1.2, lz * 1.3}, 0.06, 8));
        append(s, tube({-0.9, yb - 1.2, lz * 1.3}, {0.9, yb - 1.2, lz * 1.3}, 0.07, 8));
    }
    append(s, translate(scale(sphere_soup(12, 18), 0.55), {-1.0, yb - 0.45, 0.0})); // gimbal
    append(s, box(-1.25, yb - 0.45, 0.0, 0.25, 0.3, 0.35));                         // camera
    return s;
}

// Open-wheel (F1-style) car, m: one loft for the nose and tub (tip low and
// narrow, cockpit rim at 0.74), coke-bottle sidepods with an undercut, an
// engine cover rising to the airbox over the driver; two-element front and
// rear wings (inverted cambered aerofoils) with endplates and a beam wing; a
// floor with a diffuser kick; rounded tyres on double wishbones; halo and
// mirrors. Ports: the airbox and sidepod intakes, the exhaust.
constexpr double kOwAxles[2] = {1.25, 4.85}, kOwTrack = 0.80, kOwWheelR = 0.36;
Soup open_wheel_car() {
    Soup s;
    append(s,
           rounded_loft(
               linspace(0.25, 3.55, 26),
               [](double x) {
                   return interp(x, {0.25, 0.6, 1.6, 2.6, 3.55}, {0.07, 0.12, 0.22, 0.38, 0.40});
               },
               [](double x) { return interp(x, {0.25, 1.0, 1.6, 3.55}, {0.24, 0.16, 0.11, 0.10}); },
               [](double x) {
                   return interp(x, {0.25, 1.0, 1.6, 2.4, 2.75, 3.55},
                                 {0.34, 0.44, 0.55, 0.66, 0.74, 0.74});
               },
               2.6, 28));
    for (int side : {1, -1})
        append(
            s,
            rounded_loft(
                linspace(2.35, 4.7, 14),
                [](double x) {
                    return interp(x, {2.35, 2.7, 3.4, 4.2, 4.7}, {0.13, 0.20, 0.20, 0.12, 0.05});
                },
                [](double x) { return interp(x, {2.35, 2.7, 4.7}, {0.24, 0.12, 0.12}); },
                [](double x) { return interp(x, {2.35, 2.7, 3.6, 4.7}, {0.58, 0.62, 0.55, 0.32}); },
                3.0, 24, side * 0.55));
    append(s,
           rounded_loft(
               linspace(3.4, 5.45, 14),
               [](double x) { return interp(x, {3.4, 4.0, 5.0, 5.45}, {0.30, 0.26, 0.14, 0.08}); },
               [](double) { return 0.12; },
               [](double x) { return interp(x, {3.4, 3.7, 4.4, 5.45}, {1.02, 0.95, 0.70, 0.42}); },
               2.4, 24));
    append(s, box(4.5, 0.82, 0.0, 1.2, 0.30, 0.02));  // shark fin
    append(s, box(3.45, 0.06, 0.0, 3.3, 0.03, 1.40)); // floor
    append(s, translate(rotate(box(0.0, 0.0, 0.0, 0.62, 0.02, 1.0), 0.0, 22.0, 0.0),
                        {5.27, 0.17, 0.0})); // diffuser kick
    for (int side : {1, -1}) {
        // front wing: main plane and flap, endplate, pylon to the nose
        append(s, lifting_surface({0.05, 0.10, 0.0}, 0.42, 0.40, 0.95, 0.0, 0.0, 0.10, side, false,
                                  4, -0.06, -3.0));
        append(s, lifting_surface({0.42, 0.20, 0.0}, 0.22, 0.20, 0.95, 0.0, 0.0, 0.10, side, false,
                                  4, -0.08, -20.0));
        append(s, box(0.30, 0.20, side * 0.96, 0.62, 0.30, 0.02));
        append(s, box(0.36, 0.20, side * 0.08, 0.20, 0.14, 0.03));
        // rear wing: main plane and DRS flap, endplate; beam wing
        append(s, lifting_surface({5.15, 0.86, 0.0}, 0.48, 0.48, 0.50, 0.0, 0.0, 0.12, side, false,
                                  3, -0.08, -10.0));
        append(s, lifting_surface({5.55, 0.98, 0.0}, 0.24, 0.24, 0.50, 0.0, 0.0, 0.12, side, false,
                                  3, -0.08, -28.0));
        append(s, box(5.42, 0.86, side * 0.51, 0.80, 0.50, 0.02));
        append(s, lifting_surface({5.25, 0.42, 0.0}, 0.22, 0.22, 0.45, 0.0, 0.0, 0.12, side, false,
                                  3, -0.06, -8.0));
        // mirrors on stalks
        append(s, tube({2.55, 0.70, side * 0.30}, {2.55, 0.78, side * 0.48}, 0.015, 8));
        append(s, box(2.55, 0.80, side * 0.52, 0.06, 0.06, 0.14));
        // halo hoop (the centre pillar below)
        append(s, tube({2.60, 0.92, 0.0}, {2.90, 0.94, side * 0.22}, 0.03, 8));
        append(s, tube({2.90, 0.94, side * 0.22}, {3.25, 0.80, side * 0.28}, 0.03, 8));
    }
    append(s, tube({2.45, 0.66, 0.0}, {2.60, 0.92, 0.0}, 0.03, 8)); // halo pillar
    append(s, box(5.20, 0.64, 0.0, 0.30, 0.46, 0.04));              // rear wing pylon
    for (double xa : kOwAxles) {
        const double hw = xa < 3.0 ? 0.19 : 0.21;
        for (int side : {1, -1}) {
            const double zw = side * kOwTrack;
            append(s, tyre({xa, kOwWheelR, zw}, kOwWheelR, hw, 28));
            const double z_in = side * (xa < 3.0 ? 0.16 : 0.26), z_out = zw - side * (hw + 0.02);
            for (double y : {0.24, 0.44}) // double wishbones: two legs each
                for (double dx : {-0.22, 0.22})
                    append(s, tube({xa + dx, y, z_in}, {xa, y + 0.04, z_out}, 0.02, 6));
        }
    }
    return s;
}

// Narrow-body twin-jet airliner (A320-like), m: tapered nose, upswept tail
// cone; low, 25-deg swept, 5-deg dihedral cambered wing at +3.5 deg
// incidence (uncambered at 0 read Cl negative at 0 and +4 AoA) with
// sharklets and flap-track fairings; belly fairing; tailplane at -1.5; fin;
// two nacelles on pylons, whose fan faces and nozzles are ports. The
// wingtip-vortex showcase.
Soup airliner() {
    const double R = 1.98;
    Soup s;
    const std::vector<double> xs = {0.0, 0.4, 1.2, 2.5, 4.2, 6.0, 26.0, 30.0, 33.5, 36.0, 37.6};
    std::vector<double> rr = {0.05, 0.42, 0.66, 0.85, 0.97, 1.0, 1.0, 0.86, 0.55, 0.28, 0.08};
    std::vector<double> yo = {-0.15, -0.12, -0.08, -0.04, 0.0, 0.0, 0.0, 0.12, 0.35, 0.55, 0.65};
    for (double& v : rr)
        v *= R;
    for (double& v : yo)
        v *= R;
    append(s, body_of_revolution(xs, rr, 0.0, 0.0, &yo));
    append(s, body_of_revolution({10.0, 12.0, 17.0, 20.0}, {0.2, 1.1, 1.1, 0.2}, -1.5, 0.0, nullptr,
                                 20)); // belly fairing
    const double sweep = 27.0, dih = 5.0, span = 17.9, rc = 7.2, tc = 1.6;
    for (int side : {1, -1}) {
        append(s, lifting_surface({12.0, -1.1, 0.0}, rc, tc, span, sweep, dih, 0.13, side, false, 6,
                                  0.02, 3.5));
        // sharklet at the tip
        const double tip_x = 12.0 + span * std::tan(rad(sweep));
        const double tip_y = -1.1 + span * std::sin(rad(dih));
        const double tip_z = side * span * std::cos(rad(dih));
        append(s, lifting_surface({tip_x + 0.05, tip_y, tip_z - side * 0.08}, 1.5, 0.6, 2.4, 40.0,
                                  0.0, 0.09, 1, true));
        for (double f : {0.25, 0.45, 0.65}) { // flap-track fairings under the trailing edge
            const double b = f * span;
            const double x_te = 12.0 + b * std::tan(rad(sweep)) + rc + (tc - rc) * f;
            const double y = -1.1 + b * std::sin(rad(dih)) - 0.3;
            append(s, body_of_revolution({x_te - 1.6, x_te - 1.0, x_te + 0.4, x_te + 0.9},
                                         {0.05, 0.25, 0.22, 0.04}, y, side * b * std::cos(rad(dih)),
                                         nullptr, 12));
        }
        append(s, lifting_surface({31.0, 0.9, 0.0}, 3.8, 1.4, 6.3, 32.0, 4.0, 0.10, side, false, 6,
                                  0.0, -1.5));
        const double ze = side * 5.75;
        append(s, body_of_revolution({10.2, 10.6, 11.6, 13.6, 14.5}, {0.80, 1.05, 1.08, 0.95, 0.60},
                                     -2.55, ze, nullptr, 20));
        append(s, box(13.9, -1.75, ze, 2.6, 1.0, 0.35)); // pylon
    }
    append(s, lifting_surface({29.5, R * 0.6, 0.0}, 5.6, 1.8, 6.2, 36.0, 0.0, 0.10, 1, true));
    return s;
}

// Concorde-like: slender fuselage, thin ogee delta (swept strake, gothic
// tip), paired nacelles, swept fin. Leading-edge vortex lift at 10-15 deg.
Soup concorde() {
    const double R = 1.45;
    Soup s = body_of_revolution({0.0, 1.0, 3.0, 6.0, 10.0, 50.0, 56.0, 59.5, 61.7},
                                {0.02, 0.35, 0.8, 1.2, R, R, 1.2, 0.8, 0.35}, 0, 0, nullptr, 24);
    const double zt = 12.8; // semi-span
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double sf : linspace(0.0, 1.0, 14)) {
        const double z = sf * zt;
        const double x_le = 15.0 + 29.0 * std::sqrt(sf) + 8.0 * std::pow(sf, 4);
        const double x_te = 57.5 - 2.5 * sf;
        const double c = std::max(x_te - x_le, 0.8);
        std::vector<V3> pts;
        for (const auto& p : naca_section(c, 0.03, 0.0, 0.4, 0.0, 12))
            pts.push_back({x_le + p[0], -0.9 + p[1], z});
        cens.push_back(centroid(pts));
        secs.push_back(std::move(pts));
    }
    const Soup half = loft_closed(secs, cens);
    Soup mirror = half;
    for (Tri& t : mirror) {
        for (V3& v : t)
            v[2] = -v[2];
        std::swap(t[0], t[2]);
    }
    append(s, half);
    append(s, mirror);
    for (double zc : {3.9, -3.9})
        append(s, box(46.0, -2.2, zc, 12.0, 1.5, 2.6));
    append(s, lifting_surface({47.0, R * 0.7, 0.0}, 11.0, 3.0, 6.8, 58.0, 0.0, 0.05, 1, true));
    return s;
}

// Apollo-like command module, heat shield into the flow: a spherical-
// segment shield (R 4.69, 3.91 across), shoulder, 33-deg cone with four RCS
// thruster quads, the docking tunnel. Blunt body: a detached bow shock in
// transonic mode.
Soup capsule() {
    const double Rc = 4.69, Rb = 1.955;
    std::vector<double> xs = linspace(0.0, Rc - std::sqrt(Rc * Rc - Rb * Rb), 10), rs;
    for (double x : xs)
        rs.push_back(std::sqrt(std::max(Rc * Rc - (Rc - x) * (Rc - x), 0.0)));
    rs[0] = 0.02;
    const double x0 = xs.back();
    xs.insert(xs.end(), {x0 + 0.12, x0 + 0.12 + (Rb - 0.45) / std::tan(rad(33.0)), 3.4, 3.45});
    rs.insert(rs.end(), {Rb, 0.45, 0.45, 0.2});
    Soup s = body_of_revolution(xs, rs, 0, 0, nullptr, 48);
    const double xq = x0 + 0.9, rq = Rb - 0.9 * std::tan(rad(33.0));
    for (int k = 0; k < 4; ++k)
        append(s, rotate(box(xq, rq + 0.08, 0.0, 0.35, 0.22, 0.35), 0.0, 0.0, 45.0 + 90.0 * k));
    return s;
}

std::vector<Soup> grid_fin(double x0, double depth, double width, double height, int cells,
                           double bar) {
    std::vector<Soup> parts;
    for (double yy : linspace(0.0, height, cells + 1))
        parts.push_back(box(x0, yy + 0.5 * bar, 0.0, depth, bar, width));
    for (double zz : linspace(-0.5 * width, 0.5 * width, cells + 1))
        parts.push_back(box(x0, 0.5 * height, zz, depth, height, bar));
    return parts;
}

// Falcon 9-like: 70 m, 3.7 m diameter, ogive fairing, four grid fins at the
// top of the first stage, four stowed landing legs, a raceway; the engine
// section's base is a port (the plume). Flip it (yaw 180) for the booster's
// descent.
Soup rocket() {
    const double R = 1.85, Rf = 2.6;
    std::vector<double> xs = linspace(0.0, 11.0, 14), rs;
    for (double x : xs)
        rs.push_back(tangent_ogive(x, 11.0, Rf));
    rs[0] = 0.05;
    xs.insert(xs.end(), {13.2, 14.4, 70.0});
    rs.insert(rs.end(), {Rf, R, R});
    Soup s = body_of_revolution(xs, rs, 0, 0, nullptr, 32);
    std::vector<Soup> fin = grid_fin(31.5, 0.6, 1.5, 1.3, 4, 0.08);
    for (int k = 0; k < 4; ++k) {
        for (const Soup& f : fin)
            append(s, rotate(translate(f, {0.0, R, 0.0}), 0, 0, 45.0 + 90.0 * k));
        // a stowed landing leg: a shallow strake along the lower body
        append(s, rotate(box(62.0, R + 0.12, 0.0, 14.0, 0.3, 0.7), 0.0, 0.0, 90.0 * k));
    }
    append(s, box(40.0, 0.0, R + 0.08, 52.0, 0.25, 0.2));   // raceway
    append(s, cyl({70.3, 0.0, 0.0}, 0, R * 0.95, 0.6, 24)); // engine skirt
    return s;
}

Soup cube() {
    return box(0.5, 0.5, 0.5, 1.0, 1.0, 1.0);
}

// Circular cylinder in cross-flow (axis z), 2 diameters long: the vortex street.
Soup cylinder() {
    return cyl({0.0, 0.0, 0.0}, 2, 0.5, 2.0, 48);
}

// 15-deg half-angle cone, apex into the wind (a conical shock, Taylor-Maccoll).
Soup cone() {
    std::vector<double> xs = linspace(0.0, 1.0, 24), r;
    for (double x : xs)
        r.push_back(x * std::tan(rad(15.0)));
    r[0] = 1e-3;
    return body_of_revolution(xs, r, 0, 0, nullptr, 40);
}

// 5.56 mm NATO projectile (M855-style): tangent ogive with a meplat,
// bearing surface with its cannelure (the crimp groove), 9-deg boat tail.
// mm; nose at x = 0.
Soup round_556() {
    const double R = 2.85, Ln = 12.0, Lc = 7.5, Lb = 3.6;
    std::vector<double> xs = linspace(0.0, Ln, 20), rs;
    for (double x : xs)
        rs.push_back(tangent_ogive(x, Ln, R));
    rs[0] = 0.3; // meplat
    xs.insert(xs.end(), {14.0, 14.15, 14.75, 14.9});
    rs.insert(rs.end(), {R, R - 0.22, R - 0.22, R}); // cannelure
    const std::vector<double> xb = linspace(Ln + Lc, Ln + Lc + Lb, 5);
    for (double x : xb) {
        xs.push_back(x);
        rs.push_back(R - (x - xb[0]) * std::tan(rad(9.0)));
    }
    return body_of_revolution(xs, rs, 0, 0, nullptr, 36);
}

// 105 mm howitzer HE (M1-style): fuze, ogive, body, rotating band, boat tail.
Soup shell_105() {
    const double R = 52.5;
    std::vector<double> xs = {0.0, 15.0, 60.0, 105.0, 110.0}, rs = {7.0, 13.0, 24.0, 33.0, 35.5};
    const std::vector<double> xo = linspace(110.0, 270.0, 14);
    for (std::size_t i = 1; i < xo.size(); ++i) {
        xs.push_back(xo[i]);
        rs.push_back(35.5 + (R - 35.5) * std::sin(0.5 * kPi * (xo[i] - 110.0) / 160.0));
    }
    xs.insert(xs.end(), {398.0, 400.0, 425.0, 427.0, 440.0, 494.0});
    rs.insert(rs.end(), {R, 53.8, 53.8, R, R, 46.0});
    return body_of_revolution(xs, rs, 0, 0, nullptr, 40);
}

// AIM-120-style: ogive radome, mid-body wings and tail fins in an X, a
// cable raceway; the motor nozzle at the base is a port. mm.
Soup aim120() {
    const double R = 89.0, L = 3650.0, Ln = 450.0;
    std::vector<double> xs = linspace(0.0, Ln, 16), rs;
    for (double x : xs)
        rs.push_back(tangent_ogive(x, Ln, R));
    rs[0] = 5.0;
    xs.push_back(L);
    rs.push_back(R);
    Soup s = body_of_revolution(xs, rs, 0, 0, nullptr, 32);
    for (const Soup& f : fins(4, 1380.0, 400.0, 70.0, 135.0, 55.0, 0.05, 45.0, R))
        append(s, f);
    for (const Soup& f : fins(4, 3230.0, 330.0, 150.0, 230.0, 40.0, 0.05, 45.0, R))
        append(s, f);
    append(s, box(1900.0, R + 6.0, 0.0, 2600.0, 16.0, 34.0)); // raceway
    return s;
}

// Scud-B-like: ogive-conical warhead, four trapezoidal tail fins in a +. mm.
Soup ballistic_missile() {
    const double R = 440.0, L = 11250.0, Ln = 2600.0;
    std::vector<double> xs = linspace(0.0, Ln, 18), rs;
    for (double x : xs)
        rs.push_back(tangent_ogive(x, Ln, R) * 0.6 + (x / Ln) * R * 0.4);
    rs[0] = 25.0;
    xs.push_back(L);
    rs.push_back(R);
    Soup s = body_of_revolution(xs, rs, 0, 0, nullptr, 36);
    for (const Soup& f : fins(4, L - 1400.0, 1400.0, 700.0, 620.0, 30.0, 0.04, 0.0, R))
        append(s, f);
    return s;
}

// APFSDS long rod in flight (sabot gone): L/D 30, six swept tail fins. mm.
Soup apfsds() {
    const double R = 13.0, L = 780.0;
    Soup s = body_of_revolution({0.0, 60.0, 70.0, L}, {0.5, 11.0, R, R}, 0, 0, nullptr, 24);
    for (const Soup& f : fins(6, L - 85.0, 85.0, 45.0, 32.0, 40.0, 0.06, 0.0, R))
        append(s, f);
    return s;
}

// CAARC standard tall building, 30.48 x 45.72 x 182.88 m, broad face on.
Soup caarc_building() {
    return box(15.24, 91.44, 0.0, 30.48, 182.88, 45.72);
}

// A 4 x 4 block of buildings with streets (heights 12-60 m, a set-back tower
// near the middle), parapets round every roof and plant on half of them.
// The (height, width frac, depth frac) of each building are fixed pseudo-
// random draws, so the block is the same in every build.
Soup city_block() {
    static const double draws[16][3] = {
        {29.5026731, 0.97430345, 0.943921423},  {18.3058013, 0.825041571, 0.968388361},
        {12.1474285, 0.955307105, 0.949267357}, {25.1021787, 0.825758107, 0.819606403},
        {19.1363485, 0.861269076, 0.876137065}, {27.4979259, 0.998875071, 0.94816548},
        {29.4210184, 0.997240037, 0.803827175}, {16.4859369, 0.903134901, 0.760985502},
        {12.9990478, 0.878722205, 0.866551506}, {37.6806976, 0.907306564, 0.878529412},
        {25.9124562, 0.811878731, 0.752948506}, {17.38726, 0.92300803, 0.800151681},
        {22.3470167, 0.750933561, 0.957511932}, {16.3249103, 0.816899826, 0.970083038},
        {26.2741427, 0.961787562, 0.909929292}, {32.7695865, 0.772873901, 0.885285955}};
    const double lot = 30.0, street = 12.0;
    Soup s;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            const double* d = draws[i * 4 + j];
            const bool tower = i == 2 && j == 1;
            const double h = tower ? 40.0 : d[0];
            const double cx = i * (lot + street) + 0.5 * lot, cz = (j - 1.5) * (lot + street);
            const double wx = lot * d[1], wz = lot * d[2];
            append(s, box(cx, 0.5 * h, cz, wx, h, wz));
            if (tower) { // the set-back upper storeys to 60 m
                append(s, box(cx, 50.0, cz, 0.65 * wx, 20.0, 0.65 * wz));
                append(s, box(cx, 61.5, cz, 0.3 * wx, 3.0, 0.3 * wz)); // plant
                continue;
            }
            const double t = 0.6, ph = 1.2; // parapet
            append(s, box(cx, h + 0.5 * ph, cz + 0.5 * (wz - t), wx, ph, t));
            append(s, box(cx, h + 0.5 * ph, cz - 0.5 * (wz - t), wx, ph, t));
            append(s, box(cx + 0.5 * (wx - t), h + 0.5 * ph, cz, t, ph, wz));
            append(s, box(cx - 0.5 * (wx - t), h + 0.5 * ph, cz, t, ph, wz));
            if ((i + j) % 2 == 0)
                append(s, box(cx + 0.15 * wx, h + 1.5, cz - 0.1 * wz, 0.3 * wx, 3.0, 0.25 * wz));
        }
    return s;
}

// Tacoma Narrows (1940) deck: an H -- two 2.44 m plate girders at the edges
// of an 11.9 m deck. Spans z (the whole tunnel at the suggested size).
Soup bridge_deck() {
    const double W = 11.9, span = 48.0, D = 2.44, t = 0.2;
    Soup s = box(0.5 * W, 0.0, 0.0, W, t * 1.5, span);
    for (double x : {0.25, W - 0.25})
        append(s, box(x, 0.35 * D, 0.0, 0.5, D, span));
    return s;
}

std::vector<Soup> wheel_set(const std::vector<double>& xs, double z_half, double r, double hw) {
    std::vector<Soup> out;
    for (double x : xs)
        for (int sz : {1, -1})
            out.push_back(cyl({x, r, sz * z_half}, 2, r, 2 * hw, 24));
    return out;
}

// European articulated lorry: a cab-over tractor with rounded front edges,
// mirrors, a sloped roof deflector and fuel tanks, a 0.9 m gap, a 13.6 m box
// trailer with side skirts, six axles. The truck-aero study. m.
Soup lorry() {
    Soup s = rounded_loft(
        linspace(0.0, 2.3, 10),
        [](double x) { return 1.25 * (0.86 + 0.14 * std::min(1.0, std::sqrt(x / 0.3))); },
        [](double) { return 0.6; }, [](double x) { return 3.35 + 0.15 * std::min(1.0, x / 0.3); },
        7.0, 32); // cab
    append(s, translate(rotate(box(0.0, 0.0, 0.0, 1.9, 0.08, 2.4), 0.0, 16.0, 0.0),
                        {1.45, 3.8, 0.0}));               // roof deflector
    append(s, box(1.6, 3.62, 0.0, 1.2, 0.3, 2.3));        // deflector base
    append(s, box(0.12, 1.1, 0.0, 0.24, 1.0, 2.5));       // bumper
    append(s, box(3.4, 0.9, 0.0, 3.0, 0.5, 1.0));         // chassis
    append(s, box(3.2 + 6.8, 2.6, 0.0, 13.6, 2.8, 2.55)); // trailer
    append(s, box(12.5, 0.95, 0.0, 4.0, 0.3, 2.0));       // bogie frame
    for (int side : {1, -1}) {
        append(s, box(0.55, 2.35, side * 1.42, 0.12, 0.42, 0.10)); // mirror
        append(s, tube({0.5, 2.5, side * 1.2}, {0.55, 2.45, side * 1.38}, 0.03, 6));
        append(s, cyl({3.0, 0.8, side * 0.95}, 0, 0.3, 1.2, 16)); // fuel tank
        append(s, box(7.4, 0.8, side * 1.265, 6.4, 0.75, 0.025)); // side skirt
    }
    for (const Soup& w : wheel_set({1.3, 4.3}, 0.95, 0.5, 0.3))
        append(s, w);
    for (const Soup& w : wheel_set({11.3, 12.6, 13.9}, 0.95, 0.5, 0.3))
        append(s, w);
    return s;
}

// N700-like leading car: a low, wide 'aero double-wing' bill (a pointed
// cone read as a pencil) rising to the cab hump; bogies with side fairings;
// a folded pantograph on its roof fairing. m.
Soup bullet_train() {
    const double L = 31.0, W = 3.38, H = 3.6, Ln = 10.0, y0 = 0.35;
    std::vector<double> xs = linspace(0.0, Ln, 18);
    xs.insert(xs.end(), {14.0, 26.6, 27.4, L});
    xs = unique(xs);
    const std::vector<double> ang = linspace(0.0, 2 * kPi, 40, false);
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double x : xs) {
        const double f = std::min(x / Ln, 1.0);
        double top = y0 + H * (0.22 + 0.78 * std::pow(std::sin(0.5 * kPi * f), 2.2));
        double half = 0.5 * W * (0.42 + 0.58 * std::pow(std::sin(0.5 * kPi * f), 0.8));
        if (x < 0.4)
            half *= 0.5 + 0.5 * x / 0.4; // round the bill's edge
        if (x > 26.6 && x < 27.4) {      // inter-car gap
            half *= 0.9;
            top = y0 + 0.92 * H;
        }
        const double bot = y0 + 0.25 * (1.0 - f);
        const double yc = 0.5 * (top + bot), h = 0.5 * (top - bot);
        std::vector<V3> pts;
        for (double a : ang) {
            const double ca = std::cos(a), sa = std::sin(a);
            const double ez = (ca > 0 ? 1 : ca < 0 ? -1 : 0) * std::pow(std::abs(ca), 2.0 / 3.0);
            const double ey = (sa > 0 ? 1 : sa < 0 ? -1 : 0) * std::pow(std::abs(sa), 2.0 / 3.0);
            pts.push_back({x, yc + h * ey, half * ez});
        }
        secs.push_back(std::move(pts));
        cens.push_back({x, yc, 0.0});
    }
    Soup s = loft_closed(secs, cens);
    for (double xb : {5.0, 22.0}) {
        append(s, box(xb + 1.2, 0.3, 0.0, 3.2, 0.6, 2.4)); // bogie
        for (int side : {1, -1})                           // its side fairings
            append(s, box(xb + 1.2, 0.62, side * 1.62, 3.6, 0.56, 0.05));
    }
    const double roof = y0 + H; // the pantograph (folded) on its fairing
    append(s, rounded_loft(
                  linspace(16.5, 20.5, 8), [](double) { return 0.9; },
                  [roof](double) { return roof - 0.15; },
                  [roof](double x) { return roof + 0.25 * std::sin(kPi * (x - 16.5) / 4.0); }, 2.6,
                  20));
    append(s, tube({18.0, roof + 0.25, 0.0}, {19.3, roof + 0.6, 0.0}, 0.05, 8));
    append(s, tube({19.3, roof + 0.6, 0.0}, {18.2, roof + 0.95, 0.0}, 0.05, 8));
    append(s, box(18.2, roof + 0.98, 0.0, 0.25, 0.06, 1.6)); // collector head
    return s;
}

// Le Mans prototype: one lofted body whose section top is a profile across
// the width (canopy hump, two fender humps, flat deck), wheel arches cut
// into the underside, splitter, shark fin, rear wing. (Separate lozenges
// read as a pile of sausages.) m.
Soup lmp() {
    const double L = 4.65, wheel_r = 0.34, wheel_hw = 0.16, track = 0.72, yb = 0.08;
    const double axles[2] = {0.95, 3.75};
    auto w_half = [&](double x) {
        return interp(x, {0.0, 0.25, 0.9, 4.2, L}, {0.62, 0.85, 0.95, 0.95, 0.88});
    };
    auto canopy = [&](double x) {
        return interp(x, {0.0, 0.5, 1.4, 1.9, 2.4, 3.0, 3.6, L},
                      {0.26, 0.42, 0.55, 0.92, 1.02, 0.95, 0.72, 0.62});
    };
    auto fender = [&](double x) {
        return interp(x, {0.0, 0.4, 0.95, 1.6, 2.2, 3.1, 3.75, 4.4, L},
                      {0.24, 0.55, 0.78, 0.62, 0.50, 0.66, 0.82, 0.70, 0.60});
    };
    const double deck = 0.38, ra = wheel_r + 0.05;
    auto arch = [&](double x) {
        double h = 0.0;
        for (double xw : axles) {
            const double d = x - xw;
            if (std::abs(d) < ra)
                h = std::max(h, wheel_r + std::sqrt(ra * ra - d * d));
        }
        return h;
    };
    std::vector<double> st = linspace(0.0, L, 46);
    for (double xw : axles)
        for (double v : linspace(xw - ra, xw + ra, 11))
            st.push_back(v);
    st = unique(st);
    const int nz = 36;
    std::vector<std::vector<V3>> secs;
    std::vector<V3> cens;
    for (double x : st) {
        const double w = w_half(x);
        const std::vector<double> z = linspace(-w, w, nz);
        std::vector<double> top(nz), bot(nz, yb);
        const double a = arch(x);
        for (int i = 0; i < nz; ++i) {
            const double c =
                canopy(x) * std::sqrt(std::clamp(1.0 - std::pow(z[i] / 0.36, 2), 0.0, 1.0));
            const double f =
                fender(x) *
                std::sqrt(std::clamp(1.0 - std::pow((std::abs(z[i]) - track) / 0.27, 2), 0.0, 1.0));
            double t = std::max(std::max(c, f), std::min(deck, canopy(x)));
            t = std::max(t, yb + 0.05);
            const double edge = std::clamp((w - std::abs(z[i])) / 0.08, 0.0, 1.0);
            top[i] = yb + (t - yb) * (0.35 + 0.65 * edge); // soften the outer edge
            if (a > 0.0 && std::abs(z[i]) > track - wheel_hw - 0.03)
                bot[i] = std::min(a, top[i] - 0.03); // wheel arch
        }
        std::vector<V3> pts;
        for (int i = 0; i < nz; ++i)
            pts.push_back({x, top[i], z[i]});
        for (int i = nz - 1; i >= 0; --i)
            pts.push_back({x, bot[i], z[i]});
        secs.push_back(std::move(pts));
        cens.push_back({x, yb + 0.03, 0.0});
    }
    Soup s = loft_closed(secs, cens);
    append(s, box(0.3, yb - 0.01, 0.0, 0.5, 0.04, 1.8)); // splitter
    append(s, box(3.75, 0.95, 0.0, 1.5, 0.3, 0.03));     // fin
    append(s, translate(rotate(box(0.0, 0.0, 0.0, 0.7, 0.02, 1.5), 0.0, 14.0, 0.0),
                        {4.32, 0.16, 0.0})); // diffuser upsweep
    for (int side : {1, -1}) {
        append(s, translate(rotate(box(0.0, 0.0, 0.0, 0.28, 0.015, 0.14), 0.0, -14.0, 0.0),
                            {0.42, 0.40, side * 0.86})); // dive plane
        append(s, tube({1.75, 0.72, side * 0.70}, {1.78, 0.80, side * 0.80}, 0.012, 6));
        append(s, box(1.80, 0.83, side * 0.84, 0.10, 0.06, 0.12)); // mirror
    }
    for (int side : {1, -1}) {
        append(s, lifting_surface({4.2, 1.02, 0.0}, 0.36, 0.36, 0.93, 0.0, 0.0, 0.12, side, false,
                                  3, -0.05, -8.0));
        append(s, box(4.4, 0.9, side * 0.93, 0.6, 0.4, 0.03));
    }
    for (const Soup& w : wheel_set({axles[0], axles[1]}, track, wheel_r, wheel_hw))
        append(s, w);
    return s;
}

const V3 kTurbineHub = {6.0, 90.0, 0.0};
constexpr double kTurbineR = 60.0;
constexpr double kPropR = 1.0;

// Utility-scale HAWT: 90 m tapered tower, nacelle and hub; the three 60 m
// blades are an actuator line (kTurbineRotor), not geometry. m.
Soup wind_turbine() {
    const double hx = kTurbineHub[0], hy = kTurbineHub[1];
    Soup s = translate(
        axis_y(body_of_revolution(linspace(0.0, hy, 6), linspace(2.2, 1.3, 6), 0, 0, nullptr, 20)),
        {hx + 5.0, 0.0, 0.0});
    append(s, rounded_loft(
                  linspace(hx + 1.0, hx + 13.0, 8), [](double) { return 2.0; },
                  [hy](double) { return hy - 2.0; }, [hy](double) { return hy + 2.0; }, 3.0,
                  24)); // nacelle
    append(s, translate(
                  body_of_revolution({0.0, 0.8, 2.5, 4.0}, {0.1, 1.6, 2.1, 2.1}, 0, 0, nullptr, 20),
                  {hx - 2.5, hy, 0.0}));
    return s;
}

// A 2 m propeller's spinner and cowling; its three blades are an actuator
// line (the propeller entry's rotor). m.
Soup propeller() {
    return body_of_revolution({0.0, 0.08, 0.25, 0.4, 2.2}, {0.01, 0.12, 0.2, 0.25, 0.3}, 0, 0,
                              nullptr, 24);
}

// Flying disc (275 mm): a domed flight plate and a thick rolled rim round an
// open cup underneath (the cavity matters: the rim's vortex), revolved;
// spins about its vertical axis.
Soup frisbee() {
    return revolve({{0.0, 32.0},
                    {70.0, 31.0},
                    {128.0, 29.5},
                    {134.0, 27.0},
                    {137.5, 20.0},
                    {137.0, 10.0},
                    {132.0, 3.0},
                    {126.0, 0.0},
                    {120.5, 2.0},
                    {118.0, 10.0},
                    {115.0, 22.0},
                    {100.0, 26.5},
                    {0.0, 28.0}});
}

geometry::Mesh m_sphere() {
    return make_sphere();
}
geometry::Mesh m_cube() {
    return to_mesh(cube());
}
geometry::Mesh m_cylinder() {
    return to_mesh(cylinder());
}
geometry::Mesh m_cone() {
    return to_mesh(cone());
}
geometry::Mesh m_wing() {
    return make_wing();
}
geometry::Mesh m_ahmed() {
    return make_ahmed(25.0);
}
geometry::Mesh m_saloon() {
    return to_mesh(saloon());
}
geometry::Mesh m_saloon_wing() {
    return to_mesh(saloon_wing());
}
geometry::Mesh m_lmp() {
    return to_mesh(lmp());
}
geometry::Mesh m_open_wheel() {
    return to_mesh(open_wheel_car());
}
geometry::Mesh m_lorry() {
    return to_mesh(lorry());
}
geometry::Mesh m_train() {
    return to_mesh(bullet_train());
}
geometry::Mesh m_airliner() {
    return to_mesh(airliner());
}
geometry::Mesh m_concorde() {
    return to_mesh(concorde());
}
geometry::Mesh m_reaper() {
    return to_mesh(reaper());
}
geometry::Mesh m_quad() {
    return to_mesh(dji_quad());
}
geometry::Mesh m_capsule() {
    return to_mesh(capsule());
}
geometry::Mesh m_rocket() {
    return to_mesh(rocket());
}
geometry::Mesh m_556() {
    return to_mesh(round_556());
}
geometry::Mesh m_105() {
    return to_mesh(shell_105());
}
geometry::Mesh m_apfsds() {
    return to_mesh(apfsds());
}
geometry::Mesh m_aim120() {
    return to_mesh(aim120());
}
geometry::Mesh m_scud() {
    return to_mesh(ballistic_missile());
}
geometry::Mesh m_caarc() {
    return to_mesh(caarc_building());
}
geometry::Mesh m_city() {
    return to_mesh(city_block());
}
geometry::Mesh m_bridge() {
    return to_mesh(bridge_deck());
}
geometry::Mesh m_turbine() {
    return to_mesh(wind_turbine());
}
geometry::Mesh m_prop() {
    return to_mesh(propeller());
}
geometry::Mesh m_frisbee() {
    return to_mesh(frisbee());
}

std::vector<Spinner> saloon_spinners() {
    std::vector<Spinner> v;
    for (const V3& c : kSaloonWheels)
        v.push_back({c, {0, 0, 1}, kSaloonWheelR, kSaloonWheelHW, 1, 0.0});
    return v;
}

Port intake(V3 c, V3 n, double r, double speed) {
    Port p;
    p.kind = Port::Kind::Intake;
    p.c = c;
    p.n = n;
    p.r = r;
    p.speed_ratio = speed;
    return p;
}

// An exhaust: subsonic speed ratio; transonic exit Mach, pressure and
// temperature ratios to the freestream.
Port exhaust(V3 c, V3 n, double r, double speed, double mach, double p_ratio, double t_ratio) {
    Port p;
    p.kind = Port::Kind::Exhaust;
    p.c = c;
    p.n = n;
    p.r = r;
    p.speed_ratio = speed;
    p.exit_mach = mach;
    p.p_ratio = p_ratio;
    p.t_ratio = t_ratio;
    return p;
}

// Engine ports (THEORY 3.11, 8.12): turbofans draw at the fan face and blow a
// hot, near-sonic mixed jet; Concorde's afterburners a supersonic one; the
// rockets' and missiles' motors under-expanded supersonic plumes; the shell's
// base-bleed unit a slow hot gas into its base.
std::vector<Port> airliner_ports() {
    std::vector<Port> v;
    for (double ze : {5.75, -5.75}) {
        v.push_back(intake({10.2, -2.55, ze}, {-1, 0, 0}, 0.7, 0.8));
        v.push_back(exhaust({14.5, -2.55, ze}, {1, 0, 0}, 0.5, 1.6, 0.95, 1.5, 1.6));
    }
    return v;
}

std::vector<Port> concorde_ports() {
    std::vector<Port> v;
    for (double zc : {3.9, -3.9}) {
        v.push_back(intake({40.0, -2.2, zc}, {-1, 0, 0}, 0.65, 0.7));
        v.push_back(exhaust({52.0, -2.2, zc}, {1, 0, 0}, 0.6, 2.0, 1.5, 1.8, 4.0));
    }
    return v;
}

std::vector<Port> open_wheel_ports() {
    std::vector<Port> v = {intake({3.40, 0.92, 0.0}, {-1, 0, 0}, 0.09, 0.6),
                           exhaust({5.45, 0.30, 0.0}, {1, 0, 0}, 0.06, 2.0, 0.6, 1.0, 3.0)};
    for (int side : {1, -1}) // the sidepods' radiator intakes
        v.push_back(intake({2.35, 0.42, side * 0.55}, {-1, 0, 0}, 0.11, 0.5));
    return v;
}

std::vector<Spinner> open_wheel_spinners() {
    std::vector<Spinner> v;
    for (double xa : kOwAxles)
        for (int side : {1, -1})
            v.push_back({{xa, kOwWheelR, side * kOwTrack}, {0, 0, 1}, kOwWheelR, 0.24, 1, 0.0});
    return v;
}

// Rotors (actuator lines, THEORY 3.12). Turbine: a blade-element optimum for
// a symmetric section at alpha 8 deg and tip-speed ratio 7 (twist = 2/3
// atan(1 / lambda_r) - alpha, chord from the Betz condition, capped at 5 m).
// Propellers and rotors: pitched past the inflow (alpha < 0), so the lift
// drives the air along the axis (thrust against it).
Rotor turbine_rotor() {
    Rotor r;
    r.c = kTurbineHub;
    r.axis = {1.0, 0.0, 0.0};
    r.r_tip = kTurbineR;
    r.r_hub = 4.0;
    r.blades = 3;
    r.chord = {5.00, 5.00, 5.00, 5.00, 4.17, 3.45, 2.94, 2.56};
    r.twist_deg = {35.32, 15.69, 7.47, 3.35, 0.93, -0.65, -1.76, -2.58};
    r.tsr = 7.0;
    return r;
}

Rotor propeller_rotor(V3 hub, double r_tip, int blades, double tsr, int sense) {
    Rotor r;
    r.c = hub;
    r.axis = {1.0, 0.0, 0.0};
    r.r_tip = r_tip;
    r.r_hub = 0.15 * r_tip;
    r.blades = blades;
    r.chord = {0.16 * r_tip, 0.08 * r_tip};
    r.twist_deg = {61.0, 39.8, 29.4, 23.6, 20.0, 17.5}; // atan(1 / (4.5 r / R)) + 5 deg
    r.tsr = tsr;
    r.sense = sense;
    return r;
}

std::vector<Rotor> quad_rotors() {
    std::vector<Rotor> v;
    const double xz[4][2] = {{3.0, 3.0}, {3.0, -3.0}, {-3.0, 3.0}, {-3.0, -3.0}};
    const int sense[4] = {1, -1, -1, 1}; // diagonal pairs counter-rotate
    for (int k = 0; k < 4; ++k) {
        Rotor r;
        r.c = {xz[k][0], 3.55, xz[k][1]};
        r.axis = {0.0, -1.0, 0.0}; // the air is driven down
        r.r_tip = 1.5;
        r.r_hub = 0.25;
        r.blades = 2;
        r.chord = {0.30, 0.15};
        r.twist_deg = {54.0, 33.5, 24.8, 20.2, 17.4, 15.5}; // atan(1 / (6 r / R)) + 6 deg
        r.tsr = 6.0;
        r.sense = sense[k];
        v.push_back(r);
    }
    return v;
}

std::vector<Entry> build_entries() {
    using G = Ground;
    std::vector<Entry> e;
    auto add = [&](const char* group, const char* id, const char* label, geometry::Mesh (*b)(),
                   double size = 0.0, G ground = G::Air, std::vector<Spinner> sp = {},
                   std::vector<Port> po = {}, std::vector<Rotor> ro = {}) {
        e.push_back(
            {id, label, group, b, size, ground, std::move(sp), std::move(po), std::move(ro)});
    };
    // Spinners: wheels roll so the contact patch moves downstream; the ball
    // has backspin (top surface moving downstream -> Magnus lift up). A thin
    // spinning blade would need a lower cap (Spinner::max_wall: at spin ratio
    // 3, 0.067 at the tips diverged twice, 0.04 once, and 0.025 ran clean);
    // rotors are actuator lines instead.
    // Bluff shapes sized to block 10 - 13 % of the section (64 cells blocked
    // 27 - 44 % on the fast grid and diverged at the top speed).
    add("Basic shapes", "sphere", "Sphere", m_sphere, 0.15);
    add("Basic shapes", "cube", "Cube", m_cube, 0.12);
    add("Basic shapes", "cylinder", "Cylinder (cross-flow)", m_cylinder);
    add("Basic shapes", "cone", "Cone (15 deg)", m_cone);
    add("Basic shapes", "ball_spin", "Spinning ball (Magnus)", m_sphere, 0.15, G::Air,
        {{{0, 0, 0}, {0, 0, 1}, 0.5, 0.5, -1, 0.0}});
    add("Aerodynamic references", "wing_naca0012", "NACA 0012 wing", m_wing);
    add("Aerodynamic references", "ahmed_25deg", "Ahmed body (25 deg)", m_ahmed);
    add("Road vehicles", "car_saloon", "Saloon (XE-like)", m_saloon, 0.0, G::Road,
        saloon_spinners());
    add("Road vehicles", "car_saloon_wing", "Saloon + rear wing", m_saloon_wing, 0.0, G::Road,
        saloon_spinners());
    add("Road vehicles", "car_lmp", "Le Mans prototype", m_lmp, 0.0, G::Road, {},
        {exhaust({4.65, 0.25, 0.0}, {1, 0, 0}, 0.06, 2.0, 0.6, 1.0, 3.0)});
    add("Road vehicles", "car_open_wheel", "Open-wheel race car", m_open_wheel, 0.0, G::Road,
        open_wheel_spinners(), open_wheel_ports());
    add("Road vehicles", "lorry", "Lorry + trailer", m_lorry, 0.40, G::Road);
    add("Road vehicles", "bullet_train", "Bullet train (leading car)", m_train, 0.60, G::Road);
    add("Aircraft", "airliner", "Airliner (A320-like)", m_airliner, 0.0, G::Air, {},
        airliner_ports());
    add("Aircraft", "concorde", "Supersonic delta (Concorde-like)", m_concorde, 0.50, G::Air, {},
        concorde_ports());
    add("Aircraft", "uav_reaper", "UAV (MQ-9-like)", m_reaper, 0.0, G::Air, {}, {},
        {propeller_rotor({11.2, 3.0, 0.0}, 1.35, 3, 4.5, 1)});
    add("Aircraft", "uav_quad", "Quadcopter", m_quad, 0.0, G::Air, {}, {}, quad_rotors());
    add("Space", "capsule", "Re-entry capsule (Apollo-like)", m_capsule, 0.15);
    add("Space", "rocket", "Rocket with grid fins (F9-like)", m_rocket, 0.75, G::Air, {},
        {exhaust({70.6, 0.0, 0.0}, {1, 0, 0}, 1.5, 2.4, 3.0, 1.0, 6.0)});
    add("Munitions", "round_556", "5.56 mm projectile", m_556, 0.40);
    add("Munitions", "shell_105", "105 mm howitzer shell", m_105, 0.45, G::Air, {},
        {exhaust({494.0, 0.0, 0.0}, {1, 0, 0}, 25.0, 0.3, 0.25, 1.0, 3.0)});
    add("Munitions", "apfsds", "APFSDS dart", m_apfsds, 0.70);
    add("Munitions", "aim120", "AIM-120 AMRAAM-like", m_aim120, 0.70, G::Air, {},
        {exhaust({3650.0, 0.0, 0.0}, {1, 0, 0}, 60.0, 2.4, 2.5, 1.5, 5.0)});
    add("Munitions", "ballistic_missile", "Ballistic missile (Scud-like)", m_scud, 0.65, G::Air, {},
        {exhaust({11250.0, 0.0, 0.0}, {1, 0, 0}, 300.0, 2.4, 2.8, 1.0, 5.0)});
    add("Wind engineering", "caarc_building", "Tall building (CAARC standard)", m_caarc, 0.20,
        G::Fixed);
    add("Wind engineering", "city_block", "City block", m_city, 0.40, G::Fixed);
    add("Wind engineering", "bridge_deck", "Bridge deck (Tacoma Narrows)", m_bridge, 0.36);
    // Rotors sized to sweep under 5 % of the section, as V31 runs the turbine
    // (blocked harder, their coefficients pass even the Betz limit; THEORY 3.12).
    add("Rotating", "wind_turbine", "Wind turbine", m_turbine, 0.117, G::Fixed, {}, {},
        {turbine_rotor()});
    add("Rotating", "propeller", "Propeller", m_prop, 0.10, G::Air, {}, {},
        {propeller_rotor({0.3, 0.0, 0.0}, kPropR, 3, 4.5, 1)});
    add("Rotating", "frisbee", "Frisbee", m_frisbee, 0.0, G::Air,
        {{{0.0, 16.0, 0.0}, {0, 1, 0}, 137.5, 16.0, 1, 0.0}});
    return e;
}

} // namespace

geometry::Mesh make_box(double cx, double cy, double cz, double sx, double sy, double sz) {
    return to_mesh(box(cx, cy, cz, sx, sy, sz));
}

geometry::Mesh make_sphere(int n_lat, int n_lon) {
    return to_mesh(sphere_soup(n_lat, n_lon));
}

// Straight NACA 0012 wing: chord +x, span +z, thickness +y; capped tips.
// Spanwise subdivision so a per-triangle Cp paint is not chordwise bands.
geometry::Mesh make_wing(double chord, double span, int n_chord, int n_span) {
    std::vector<double> xc, yu;
    for (double b : linspace(0.0, kPi, n_chord)) {
        const double x = 0.5 * (1 - std::cos(b));
        xc.push_back(x * chord);
        yu.push_back(naca0012_half_thickness(x) * chord);
    }
    std::vector<double> px, py; // upper TE->LE, then lower LE->TE
    for (int i = n_chord - 1; i >= 0; --i) {
        px.push_back(xc[i]);
        py.push_back(yu[i]);
    }
    for (int i = 1; i < n_chord; ++i) {
        px.push_back(xc[i]);
        py.push_back(-yu[i]);
    }
    const double z0 = -span / 2, z1 = span / 2;
    const std::vector<double> zs = linspace(z0, z1, n_span + 1);
    Soup s;
    for (std::size_t k = 0; k + 1 < px.size(); ++k)
        for (int m = 0; m < n_span; ++m) {
            const V3 a{px[k], py[k], zs[m]}, b{px[k + 1], py[k + 1], zs[m]};
            const V3 c{px[k + 1], py[k + 1], zs[m + 1]}, d{px[k], py[k], zs[m + 1]};
            s.push_back({a, b, c});
            s.push_back({a, c, d});
        }
    double cx = 0, cy = 0;
    for (std::size_t k = 0; k < px.size(); ++k) {
        cx += px[k];
        cy += py[k];
    }
    cx /= double(px.size());
    cy /= double(py.size());
    for (std::size_t k = 0; k + 1 < px.size(); ++k) {
        s.push_back({V3{cx, cy, z0}, V3{px[k + 1], py[k + 1], z0}, V3{px[k], py[k], z0}});
        s.push_back({V3{cx, cy, z1}, V3{px[k], py[k], z1}, V3{px[k + 1], py[k + 1], z1}});
    }
    return to_mesh(s);
}

// Ahmed body (Ahmed et al. 1984), mm: L 1044, W 389, H 288, nose R 100,
// 222 mm rear slant at slant_deg; stilts omitted. Oriented outward about
// its centroid (convex enough everywhere, including the slant).
geometry::Mesh make_ahmed(double slant_deg, int n_sub) {
    const double L = 1044.0, W = 389.0, H = 288.0, R = 100.0;
    const double slant = rad(slant_deg);
    const double x_slant = L - 222.0 * std::cos(slant);
    auto y_bot = [&](double x) {
        return x < R ? R - std::sqrt(
                               std::max(R * R - (R - std::min(x, R)) * (R - std::min(x, R)), 0.0))
                     : 0.0;
    };
    auto y_top = [&](double x) {
        if (x < R)
            return (H - R) + std::sqrt(std::max(R * R - (R - x) * (R - x), 0.0));
        if (x > x_slant)
            return H - (x - x_slant) * std::tan(slant);
        return H;
    };
    auto z_half = [&](double x) {
        return x < R ? (W / 2 - R) + std::sqrt(std::max(R * R - (R - x) * (R - x), 0.0)) : W / 2;
    };
    std::vector<double> st;
    for (double a : linspace(0.0, kPi / 2, 12))
        st.push_back(R * (1 - std::cos(a)));
    st.insert(st.end(), {R + 1.0, x_slant - 1.0, x_slant});
    const std::vector<double> tail = linspace(x_slant, L, 8);
    st.insert(st.end(), tail.begin() + 1, tail.end());
    for (double& v : st)
        v = std::clamp(v, 0.0, L);
    st = unique(st);

    Soup s;
    auto quad = [&](V3 a, V3 b, V3 c, V3 d) {
        s.push_back({a, b, c});
        s.push_back({a, c, d});
    };
    for (std::size_t j = 0; j + 1 < st.size(); ++j) {
        const double x0 = st[j], x1 = st[j + 1];
        for (int m = 0; m < n_sub; ++m) { // top and bottom, subdivided in z
            const double f0 = double(m) / n_sub, f1 = double(m + 1) / n_sub;
            const double za0 = -z_half(x0) + 2 * z_half(x0) * f0,
                         za1 = -z_half(x0) + 2 * z_half(x0) * f1;
            const double zb0 = -z_half(x1) + 2 * z_half(x1) * f0,
                         zb1 = -z_half(x1) + 2 * z_half(x1) * f1;
            quad({x0, y_top(x0), za0}, {x1, y_top(x1), zb0}, {x1, y_top(x1), zb1},
                 {x0, y_top(x0), za1});
            quad({x0, y_bot(x0), za0}, {x1, y_bot(x1), zb0}, {x1, y_bot(x1), zb1},
                 {x0, y_bot(x0), za1});
        }
        for (int m = 0; m < n_sub; ++m) { // sides, subdivided in y
            const double f0 = double(m) / n_sub, f1 = double(m + 1) / n_sub;
            const double ya0 = y_bot(x0) + (y_top(x0) - y_bot(x0)) * f0;
            const double ya1 = y_bot(x0) + (y_top(x0) - y_bot(x0)) * f1;
            const double yb0 = y_bot(x1) + (y_top(x1) - y_bot(x1)) * f0;
            const double yb1 = y_bot(x1) + (y_top(x1) - y_bot(x1)) * f1;
            quad({x0, ya0, z_half(x0)}, {x1, yb0, z_half(x1)}, {x1, yb1, z_half(x1)},
                 {x0, ya1, z_half(x0)});
            quad({x0, ya0, -z_half(x0)}, {x1, yb0, -z_half(x1)}, {x1, yb1, -z_half(x1)},
                 {x0, ya1, -z_half(x0)});
        }
    }
    const double caps[2][4] = {{0.0, y_bot(0), y_top(0), z_half(0)}, {L, 0.0, y_top(L), z_half(L)}};
    for (const auto& c : caps) {
        const double xc = c[0], ylo = c[1], yhi = c[2], zh = c[3];
        for (int m = 0; m < n_sub; ++m)
            for (int n = 0; n < n_sub; ++n) {
                const double y0 = ylo + (yhi - ylo) * m / n_sub,
                             y1 = ylo + (yhi - ylo) * (m + 1) / n_sub;
                const double q0 = -zh + 2 * zh * n / n_sub, q1 = -zh + 2 * zh * (n + 1) / n_sub;
                quad({xc, y0, q0}, {xc, y1, q0}, {xc, y1, q1}, {xc, y0, q1});
            }
    }
    V3 centre{0, 0, 0};
    for (const Tri& t : s)
        for (const V3& v : t)
            centre = centre + v;
    centre = centre * (1.0 / (3.0 * double(s.size())));
    return to_mesh(orient(s, centre));
}

const std::vector<Entry>& entries() {
    static const std::vector<Entry> e = build_entries();
    return e;
}

const Entry* find(std::string_view id) {
    for (const Entry& e : entries())
        if (e.id == id)
            return &e;
    return nullptr;
}

} // namespace windoa::catalogue
