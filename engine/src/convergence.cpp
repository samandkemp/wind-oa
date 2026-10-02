#include "windoa/convergence.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace windoa {

ConvergenceMonitor::ConvergenceMonitor(double window_ft, double tol, double abs_tol, double ft_min,
                                       double ft_max)
    : window_ft_(window_ft), tol_(tol), abs_tol_(abs_tol), ft_min_(ft_min), ft_max_(ft_max) {
    restart("init");
}

void ConvergenceMonitor::restart(const std::string& reason) {
    reason_ = reason;
    ft_ = 0.0;
    verdict_ft_ = 0.0;
    steps_ = 0;
    developing_ = true;
    settled_ = gave_up_ = restored_ = false;
    windows_.clear();
    acc_ = {};
    acc_ft_ = 0.0;
}

void ConvergenceMonitor::mark_restored() {
    developing_ = false;
    settled_ = true;
    restored_ = true;
}

void ConvergenceMonitor::add(const Coeffs& c, int n_steps, double u, int nx) {
    if (n_steps <= 0)
        return;
    const double dft = n_steps * std::max(u, 0.0) / std::max(nx, 1);
    steps_ += n_steps;
    ft_ += dft;
    // a flow-through-weighted window mean
    for (int k = 0; k < kCoeffs; ++k)
        acc_[k] += c[k] * dft;
    acc_ft_ += dft;
    if (acc_ft_ >= window_ft_) {
        Coeffs w;
        for (int k = 0; k < kCoeffs; ++k)
            w[k] = acc_[k] / acc_ft_;
        windows_.push_back(w);
        acc_ = {};
        acc_ft_ = 0.0;
    }
    if (developing_)
        check();
}

// Two consecutive windows can differ by under tol on a slow monotone
// approach while still several tol from the limit (declared 2.9 % off on
// the real Ahmed). So when the last two differences share a sign, treat the
// approach as geometric (ratio r = d2 / d1) and require the remaining
// change d2 r / (1 - r) inside tolerance too. Opposite signs: oscillation
// about a mean, where the plain difference test is right.
bool ConvergenceMonitor::coefficient_settled(double w1, double w2, double w3) const {
    const double tol = std::max(tol_ * std::abs(w3), abs_tol_);
    const double d1 = w2 - w1, d2 = w3 - w2;
    if (std::abs(d2) >= tol)
        return false;
    if (d1 * d2 > 0.0) {
        const double r = d2 / d1;
        if (r >= 0.9)
            return false; // too slow to extrapolate
        return std::abs(d2) * r / (1.0 - r) < tol;
    }
    return true;
}

void ConvergenceMonitor::check() {
    if (ft_ >= ft_min_ && windows_.size() >= 3) {
        const std::size_t n = windows_.size();
        bool all = true;
        for (int k = 0; k < kCoeffs && all; ++k)
            all = coefficient_settled(windows_[n - 3][k], windows_[n - 2][k], windows_[n - 1][k]);
        if (all) {
            settled_ = true;
            developing_ = false;
            verdict_ft_ = ft_;
            return;
        }
    }
    if (ft_ >= ft_max_) {
        gave_up_ = true;
        developing_ = false;
        verdict_ft_ = ft_;
    }
}

std::string ConvergenceMonitor::status() const {
    char buf[96];
    if (developing_)
        std::snprintf(buf, sizeof(buf),
                      "DEVELOPING  %.1f flow-throughs (typically settles by 2-4.5)", ft_);
    else if (restored_)
        std::snprintf(buf, sizeof(buf), "SETTLED (restored from flow cache)");
    else if (settled_)
        std::snprintf(buf, sizeof(buf), "SETTLED after %.1f flow-throughs", verdict_ft_);
    else if (gave_up_)
        std::snprintf(buf, sizeof(buf), "NOT SETTLED after %.1f ft (unsteady?)", verdict_ft_);
    else
        std::snprintf(buf, sizeof(buf), "RUNNING (develop skipped)");
    return buf;
}

} // namespace windoa
