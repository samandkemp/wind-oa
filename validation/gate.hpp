// Shared harness for the validation gates (docs/VALIDATION.md): a pass /
// fail ledger and the small numerics the gates share. Header-only; every
// gate is one executable that checks its external reference and exits
// non-zero on failure.
#pragma once

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "windoa/context.hpp"

namespace windoa::gate {

// -- Ledger -------------------------------------------------------------------

class Gate {
  public:
    explicit Gate(std::string id) : id_(std::move(id)) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("== %s ==\n", id_.c_str());
    }

    // One pass / fail line: "  [PASS] <what>".
    bool check(bool ok, const char* fmt, ...) {
        std::va_list args;
        va_start(args, fmt);
        std::printf("  [%s] ", ok ? "PASS" : "FAIL");
        std::vprintf(fmt, args);
        std::printf("\n");
        va_end(args);
        if (!ok)
            ++failures_;
        return ok;
    }
    // Informational line (diagnostics that are not pass / fail).
    void note(const char* fmt, ...) {
        std::va_list args;
        va_start(args, fmt);
        std::printf("  ");
        std::vprintf(fmt, args);
        std::printf("\n");
        va_end(args);
    }
    void section(const char* title) { std::printf("-- %s\n", title); }

    int finish() const {
        std::printf("%s %s\n", id_.c_str(), failures_ == 0 ? "PASS" : "FAIL");
        return failures_ == 0 ? 0 : 1;
    }

  private:
    std::string id_;
    int failures_ = 0;
};

// Runs a gate body, turning an exception (Vulkan error, missing file) into
// a failure instead of a crash.
template <class F> int run(const char* id, F&& body) {
    Gate g(id);
    try {
        body(g);
    } catch (const std::exception& e) {
        g.check(false, "exception: %s", e.what());
    }
    return g.finish();
}

// -- Numerics ------------------------------------------------------------------

// ||a - b|| / ||b|| over matching entries.
template <class A, class B> double rel_l2(const A& a, const B& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(b[i]) * double(b[i]);
    }
    return std::sqrt(num / den);
}

// Piecewise-linear interpolation through (xp, fp), xp increasing, clamped.
inline double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    if (x <= xp.front())
        return fp.front();
    if (x >= xp.back())
        return fp.back();
    std::size_t k = 1;
    while (xp[k] < x)
        ++k;
    const double t = (x - xp[k - 1]) / (xp[k] - xp[k - 1]);
    return fp[k - 1] + t * (fp[k] - fp[k - 1]);
}

inline constexpr double kPi = 3.14159265358979323846;

} // namespace windoa::gate
