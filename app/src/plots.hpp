// Scrolling time-series strip (Cd / Cl / Cm) along the window bottom,
// drawn with Dear ImGui's draw lists (no ImPlot).
// Scaling is expand-only: the range grows to fit new peaks but never
// shrinks, so small fluctuations are never zoomed into fake waves; "refit"
// re-focuses on the recent history once a signal has settled.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include "imgui.h"

namespace windoa::app {

class TimeSeries {
  public:
    TimeSeries(std::string label, ImU32 colour, std::size_t length = 400)
        : label_(std::move(label)), colour_(colour), length_(length) {}

    void append(double v) {
        if (!std::isfinite(v))
            return;
        hist_.push_back(v);
        if (hist_.size() > length_)
            hist_.pop_front();
        grow(pinned_ ? std::vector<double>{v} : std::vector<double>(hist_.begin(), hist_.end()));
    }
    void clear() {
        hist_.clear();
        has_range_ = pinned_ = false;
    }
    // Fit to the newest `frac` of the history (the startup transient -- Cd
    // ~30 while the inlet ramps -- stops flattening the settled signal).
    void refit_recent(double frac) {
        if (hist_.empty())
            return;
        const std::size_t n = std::max<std::size_t>(2, std::size_t(hist_.size() * frac));
        auto first = hist_.end() - std::ptrdiff_t(std::min(n, hist_.size()));
        lo_ = *std::min_element(first, hist_.end());
        hi_ = *std::max_element(first, hist_.end());
        has_range_ = pinned_ = true;
    }

    // Draw into the rect [a, b] of the current window's draw list.
    void draw(ImDrawList* dl, ImVec2 a, ImVec2 b) const {
        dl->AddRectFilled(a, b, IM_COL32(4, 4, 12, 220), 3.0f);
        dl->AddRect(a, b, IM_COL32(60, 60, 80, 255), 3.0f);
        char buf[64];
        const float lh = ImGui::GetTextLineHeight();
        if (hist_.empty() || !has_range_) {
            dl->AddText({a.x + 6, a.y + 2}, colour_, label_.c_str());
            return;
        }
        double lo = lo_, hi = hi_;
        const double span = std::max({hi - lo, std::abs(hi) * 0.02, 1e-4});
        lo -= 0.12 * span;
        hi += 0.12 * span;
        std::snprintf(buf, sizeof(buf), "%s  %s", label_.c_str(), fmt(hist_.back()).c_str());
        dl->AddText({a.x + 6, a.y + 2}, colour_, buf);
        const float x0 = a.x + 44, x1 = b.x - 6, y0 = a.y + lh + 6, y1 = b.y - 4;
        dl->AddText({a.x + 4, y0 - 2}, IM_COL32(150, 150, 170, 255), fmt(hi).c_str());
        dl->AddText({a.x + 4, y1 - lh + 2}, IM_COL32(150, 150, 170, 255), fmt(lo).c_str());
        auto Y = [&](double v) {
            const double t = std::clamp((v - lo) / (hi - lo), 0.0, 1.0);
            return float(y1 - t * (y1 - y0));
        };
        if (lo < 0.0 && hi > 0.0)
            dl->AddLine({x0, Y(0.0)}, {x1, Y(0.0)}, IM_COL32(90, 90, 110, 255));
        const std::size_t n = hist_.size();
        const float dx = n > 1 ? (x1 - x0) / float(length_ - 1) : 0.0f;
        const float xs = x1 - dx * float(n - 1);
        for (std::size_t i = 1; i < n; ++i)
            dl->AddLine({xs + dx * float(i - 1), Y(hist_[i - 1])},
                        {xs + dx * float(i), Y(hist_[i])}, colour_, 1.5f);
    }

  private:
    static std::string fmt(double v) {
        char b[32];
        const double a = std::abs(v);
        std::snprintf(b, sizeof(b), a >= 100.0 ? "%.0f" : a >= 1.0 ? "%.2f" : "%.3f", v);
        return b;
    }
    void grow(const std::vector<double>& vals) {
        for (double v : vals) {
            if (!has_range_) {
                lo_ = hi_ = v;
                has_range_ = true;
            }
            lo_ = std::min(lo_, v);
            hi_ = std::max(hi_, v);
        }
    }

    std::string label_;
    ImU32 colour_;
    std::size_t length_;
    std::deque<double> hist_;
    double lo_ = 0.0, hi_ = 0.0;
    bool has_range_ = false, pinned_ = false;
};

// A static x-y plot into the rect [a, b] of a draw list: series sharing
// axes fitted to their finite values (y from zero when `y_from_zero`; NaN
// breaks a line), the axis ranges at the corners, an optional highlighted
// vertical marker at x_marker and dim ones at `guides`.
struct Series {
    const std::vector<double>* x;
    const std::vector<double>* y;
    ImU32 colour;
};

inline void plot_xy(ImDrawList* dl, ImVec2 a, ImVec2 b, const std::vector<Series>& series,
                    const char* x_label, const char* y_label, bool y_from_zero = false,
                    double x_marker = std::nan(""), const std::vector<double>& guides = {}) {
    dl->AddRectFilled(a, b, IM_COL32(4, 4, 12, 220), 3.0f);
    dl->AddRect(a, b, IM_COL32(60, 60, 80, 255), 3.0f);
    double x0 = 1e300, x1 = -1e300, y0 = y_from_zero ? 0.0 : 1e300, y1 = -1e300;
    for (const Series& s : series)
        for (std::size_t i = 0; i < s.x->size() && i < s.y->size(); ++i) {
            const double x = (*s.x)[i], y = (*s.y)[i];
            if (!std::isfinite(x) || !std::isfinite(y))
                continue;
            x0 = std::min(x0, x);
            x1 = std::max(x1, x);
            y0 = std::min(y0, y);
            y1 = std::max(y1, y);
        }
    const float lh = ImGui::GetTextLineHeight();
    const ImU32 grey = IM_COL32(150, 150, 170, 255);
    dl->AddText({a.x + 4, a.y + 2}, grey, y_label);
    if (!(x1 > x0) || !(y1 >= y0)) {
        dl->AddText({a.x + 4, a.y + 2 + lh}, grey, "(no data yet)");
        return;
    }
    if (y1 - y0 < 1e-12)
        y1 = y0 + 1e-12;
    const double pad = 0.08 * (y1 - y0);
    if (!y_from_zero || y0 < 0.0)
        y0 -= pad;
    y1 += pad;
    const float px0 = a.x + 6, px1 = b.x - 6, py0 = a.y + lh + 6, py1 = b.y - lh - 4;
    auto X = [&](double x) { return float(px0 + (x - x0) / (x1 - x0) * (px1 - px0)); };
    auto Y = [&](double y) { return float(py1 - (y - y0) / (y1 - y0) * (py1 - py0)); };
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.3g", y1);
    dl->AddText({b.x - 6 - ImGui::CalcTextSize(buf).x, a.y + 2}, grey, buf);
    std::snprintf(buf, sizeof(buf), "%.3g", y0);
    dl->AddText({b.x - 6 - ImGui::CalcTextSize(buf).x, py1 - lh}, grey, buf);
    std::snprintf(buf, sizeof(buf), "%.3g", x0);
    dl->AddText({px0, b.y - lh - 2}, grey, buf);
    std::snprintf(buf, sizeof(buf), "%.3g  %s", x1, x_label);
    dl->AddText({px1 - ImGui::CalcTextSize(buf).x, b.y - lh - 2}, grey, buf);
    if (y0 < 0.0 && y1 > 0.0)
        dl->AddLine({px0, Y(0.0)}, {px1, Y(0.0)}, IM_COL32(90, 90, 110, 255));
    for (const double g : guides)
        if (g >= x0 && g <= x1)
            for (float yy = py0; yy < py1; yy += 6.0f) // dashed
                dl->AddLine({X(g), yy}, {X(g), std::min(yy + 3.0f, py1)},
                            IM_COL32(120, 160, 220, 150));
    if (std::isfinite(x_marker) && x_marker >= x0 && x_marker <= x1)
        dl->AddLine({X(x_marker), py0}, {X(x_marker), py1}, IM_COL32(255, 220, 120, 200), 1.0f);
    for (const Series& s : series) {
        bool pen = false;
        ImVec2 last{};
        for (std::size_t i = 0; i < s.x->size() && i < s.y->size(); ++i) {
            const double x = (*s.x)[i], y = (*s.y)[i];
            if (!std::isfinite(x) || !std::isfinite(y)) {
                pen = false;
                continue;
            }
            const ImVec2 q{X(x), Y(y)};
            if (pen)
                dl->AddLine(last, q, s.colour, 1.5f);
            last = q;
            pen = true;
        }
    }
}

} // namespace windoa::app
