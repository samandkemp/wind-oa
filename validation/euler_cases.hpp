// The compressible Euler cases shared by V17 - V19: the exact Riemann
// solution, the setups and the measurements.
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
