// The compressible Euler cases shared by V17 - V19 and V27: the exact
// Riemann solution, the setups and the measurements.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "gate.hpp"
#include "windoa/euler.hpp"

namespace windoa::euler_cases {

inline constexpr double G = 1.4;
inline constexpr double kDeg = gate::kPi / 180.0;

// -- Exact 1-D Riemann solution, ideal gas (Toro, ch. 4) ---------------------------
struct Prim {
    double rho, u, p;
};

inline double f_k(double p, const Prim& k) {
    const double c = std::sqrt(G * k.p / k.rho);
    if (p > k.p) { // shock
        const double A = 2.0 / ((G + 1) * k.rho), B = (G - 1) / (G + 1) * k.p;
        return (p - k.p) * std::sqrt(A / (p + B));
    }
    return 2 * c / (G - 1) * (std::pow(p / k.p, (G - 1) / (2 * G)) - 1); // rarefaction
}

// The solution at similarity coordinate s = x / t.
inline Prim sample_exact(const Prim& L, const Prim& R, double s) {
    double p = 0.5 * (L.p + R.p);
    for (int it = 0; it < 100; ++it) { // Newton on fL + fR + du (numerical derivative)
        const double f = f_k(p, L) + f_k(p, R) + (R.u - L.u);
        const double h = 1e-7 * std::max(p, 1e-6);
        const double df = (f_k(p + h, L) + f_k(p + h, R) - f_k(p - h, L) - f_k(p - h, R)) / (2 * h);
        const double pn = std::max(p - f / df, 1e-8);
        if (std::abs(pn - p) < 1e-12)
            break;
        p = pn;
    }
    const double u = 0.5 * (L.u + R.u) + 0.5 * (f_k(p, R) - f_k(p, L));
    const double cl = std::sqrt(G * L.p / L.rho), cr = std::sqrt(G * R.p / R.rho);
    if (s <= u) { // left of the contact
        if (p > L.p) {
            const double sl = L.u - cl * std::sqrt((G + 1) / (2 * G) * p / L.p + (G - 1) / (2 * G));
            if (s < sl)
                return L;
            return {L.rho * ((p / L.p + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * p / L.p + 1)), u,
                    p};
        }
        const double cs = cl * std::pow(p / L.p, (G - 1) / (2 * G));
        if (s < L.u - cl)
            return L;
        if (s > u - cs)
            return {L.rho * std::pow(p / L.p, 1 / G), u, p};
        const double c = 2 / (G + 1) * (cl + (G - 1) / 2 * (L.u - s));
        return {L.rho * std::pow(c / cl, 2 / (G - 1)), 2 / (G + 1) * (cl + (G - 1) / 2 * L.u + s),
                L.p * std::pow(c / cl, 2 * G / (G - 1))};
    }
    if (p > R.p) {
        const double sr = R.u + cr * std::sqrt((G + 1) / (2 * G) * p / R.p + (G - 1) / (2 * G));
        if (s > sr)
            return R;
        return {R.rho * ((p / R.p + (G - 1) / (G + 1)) / ((G - 1) / (G + 1) * p / R.p + 1)), u, p};
    }
    const double cs = cr * std::pow(p / R.p, (G - 1) / (2 * G));
    if (s > R.u + cr)
        return R;
    if (s < u + cs)
        return {R.rho * std::pow(p / R.p, 1 / G), u, p};
    const double c = 2 / (G + 1) * (cr - (G - 1) / 2 * (R.u - s));
    return {R.rho * std::pow(c / cr, 2 / (G - 1)), 2 / (G + 1) * (-cr + (G - 1) / 2 * R.u + s),
            R.p * std::pow(c / cr, 2 * G / (G - 1))};
}

// -- Oblique shock: weak-shock angle for deflection theta (theta-beta-M) -----------
inline double theta_beta_m(double mach, double theta) {
    auto f = [&](double b) {
        return 2.0 / std::tan(b) * (mach * mach * std::sin(b) * std::sin(b) - 1.0) /
                   (mach * mach * (G + std::cos(2 * b)) + 2.0) -
               std::tan(theta);
    };
    double lo = std::asin(1.0 / mach) + 1e-6, hi = 64.0 * kDeg;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (f(lo) * f(mid) <= 0)
            hi = mid;
        else
            lo = mid;
    }
    return 0.5 * (lo + hi);
}

// -- NACA 0012 section (V19, V27) ---------------------------------------------------
// A 2-D section on an nx x ny grid: chord ch cells at incidence al (radians,
// nose up), rotated about the pivot (px, py), which sits at the quarter chord.
// The foil mask and each cell's chordwise position x/c ([x * ny + y]), and the
// exact signed distance to the contour within 6 cells of the foil (+-4 beyond;
// negative inside), which the image-point walls need (THEORY 8.6).
struct NacaSection {
    std::vector<std::uint8_t> foil;
    std::vector<double> xa;
    std::vector<float> phi;
};

inline double naca0012_half(double xc) { // half-thickness / chord
    xc = std::max(xc, 0.0);
    return 0.6 * (0.2969 * std::sqrt(xc) - 0.1260 * xc - 0.3516 * xc * xc + 0.2843 * xc * xc * xc -
                  0.1015 * xc * xc * xc * xc);
}

inline NacaSection naca0012_section(int nx, int ny, double ch, double al, double px, double py) {
    NacaSection s;
    s.foil.assign(std::size_t(nx) * ny, 0);
    s.xa.assign(s.foil.size(), 0.0);
    s.phi.assign(s.foil.size(), 0.0f);
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y) {
            const double xg = x + 0.5 - px, yg = y + 0.5 - py;
            const double xa = (xg * std::cos(al) - yg * std::sin(al)) / ch + 0.25;
            const double ya = (xg * std::sin(al) + yg * std::cos(al)) / ch;
            s.xa[std::size_t(x) * ny + y] = xa;
            s.foil[std::size_t(x) * ny + y] =
                (xa >= 0.0 && xa <= 1.0 && std::abs(ya) < naca0012_half(xa)) ? 1 : 0;
        }
    std::vector<double> cx, cy;
    for (int i = 0; i < 2000; ++i) {
        const double bt = gate::kPi * i / 1999.0, xs = 0.5 * (1 - std::cos(bt));
        const double yt = naca0012_half(xs);
        for (int sg : {1, -1}) {
            const double qx = (xs - 0.25) * ch, qy = sg * yt * ch;
            cx.push_back(qx * std::cos(al) + qy * std::sin(al) + px);
            cy.push_back(-qx * std::sin(al) + qy * std::cos(al) + py);
        }
    }
    for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y) {
            const std::size_t k = std::size_t(x) * ny + y;
            bool near = false;
            for (int dx = -6; dx <= 6 && !near; ++dx)
                for (int dy = -6; dy <= 6; ++dy) {
                    const int xx = x + dx, yy = y + dy;
                    if (xx >= 0 && yy >= 0 && xx < nx && yy < ny &&
                        s.foil[std::size_t(xx) * ny + yy]) {
                        near = true;
                        break;
                    }
                }
            double d = 4.0;
            if (near) {
                d = 1e30;
                for (std::size_t i = 0; i < cx.size(); ++i)
                    d = std::min(d, std::hypot(x + 0.5 - cx[i], y + 0.5 - cy[i]));
            }
            s.phi[k] = float(s.foil[k] ? -d : d);
        }
    return s;
}

// The section extruded through nz cells (a 2-D case: slip z faces).
inline void set_section(euler::Solver& s, const NacaSection& sec, int nz) {
    std::vector<std::uint8_t> flags(s.cells());
    std::vector<float> phi(s.cells());
    for (std::size_t k = 0; k < sec.foil.size(); ++k)
        for (int z = 0; z < nz; ++z) {
            flags[k * std::size_t(nz) + std::size_t(z)] = sec.foil[k];
            phi[k * std::size_t(nz) + std::size_t(z)] = sec.phi[k];
        }
    s.set_flags(flags);
    s.set_distance(phi);
}

// least-squares slope of y(x)
inline double lsq_slope(const std::vector<double>& xs, const std::vector<double>& ys) {
    double mx = 0, my = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        mx += xs[i];
        my += ys[i];
    }
    mx /= double(xs.size());
    my /= double(ys.size());
    double sxy = 0, sxx = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
    }
    return sxy / sxx;
}

} // namespace windoa::euler_cases
