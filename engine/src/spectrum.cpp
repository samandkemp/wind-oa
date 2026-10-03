#include "windoa/spectrum.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace windoa {

namespace {

using cplx = std::complex<double>;

// In-place iterative radix-2 Cooley-Tukey; a.size() a power of two.
void fft(std::vector<cplx>& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) { // bit-reversal permutation
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * std::numbers::pi / double(len);
        const cplx wl(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            cplx w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const cplx u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
}

// Sub-bin position of a peak at bin k: the vertex of the parabola through
// the log amplitudes of bins k - 1, k, k + 1 (exact for a Gaussian peak,
// and within a few hundredths of a bin for a Hann-windowed tone).
double refine(const std::vector<double>& amp, std::size_t k) {
    if (k == 0 || k + 1 >= amp.size())
        return double(k);
    const double a = std::log(std::max(amp[k - 1], 1e-300));
    const double b = std::log(std::max(amp[k], 1e-300));
    const double c = std::log(std::max(amp[k + 1], 1e-300));
    const double den = a - 2.0 * b + c;
    if (std::abs(den) < 1e-300)
        return double(k);
    return double(k) + std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
}

} // namespace

Spectrum spectrum(std::span<const double> t, std::span<const double> v, std::size_t n_fft,
                  double f_min) {
    if (t.size() != v.size())
        throw std::invalid_argument("spectrum: t and v differ in length");
    Spectrum s;
    if (t.size() < 4 || !(t.back() > t.front()))
        return s;
    if (n_fft == 0) {
        n_fft = 64;
        while (n_fft < t.size())
            n_fft <<= 1;
    }
    if ((n_fft & (n_fft - 1)) != 0)
        throw std::invalid_argument("spectrum: n_fft must be a power of two");

    // Uniform resampling over [t0, t1] (linear interpolation), then the mean.
    const double t0 = t.front(), t1 = t.back();
    const double dt = (t1 - t0) / double(n_fft - 1);
    std::vector<double> u(n_fft);
    std::size_t j = 0;
    double mean = 0.0;
    for (std::size_t i = 0; i < n_fft; ++i) {
        const double ti = t0 + dt * double(i);
        while (j + 2 < t.size() && t[j + 1] < ti)
            ++j;
        const double span = t[j + 1] - t[j];
        const double a = span > 0.0 ? std::clamp((ti - t[j]) / span, 0.0, 1.0) : 0.0;
        u[i] = v[j] + a * (v[j + 1] - v[j]);
        mean += u[i];
    }
    mean /= double(n_fft);
    // A flat signal has no spectrum, only the rounding of its mean.
    double dev = 0.0;
    for (const double x : u)
        dev = std::max(dev, std::abs(x - mean));
    const bool flat = dev <= 1e-12 * std::max(1.0, std::abs(mean));

    // Hann window (coherent gain 1/2), FFT, single-sided amplitude.
    std::vector<cplx> a(n_fft);
    for (std::size_t i = 0; i < n_fft; ++i) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * double(i) / double(n_fft));
        a[i] = cplx((u[i] - mean) * w, 0.0);
    }
    fft(a);
    const std::size_t half = n_fft / 2;
    s.df = 1.0 / (dt * double(n_fft));
    s.freq.resize(half + 1);
    s.amp.resize(half + 1);
    for (std::size_t k = 0; k <= half; ++k) {
        s.freq[k] = s.df * double(k);
        s.amp[k] = flat ? 0.0 : (k == 0 || k == half ? 2.0 : 4.0) * std::abs(a[k]) / double(n_fft);
    }
    const auto peaks = spectral_peaks(s, 1, f_min);
    if (!peaks.empty()) {
        s.peak_freq = peaks[0];
        const std::size_t k =
            std::min<std::size_t>(std::size_t(std::lround(peaks[0] / s.df)), half);
        s.peak_amp = s.amp[k];
    }
    return s;
}

std::vector<double> spectral_peaks(const Spectrum& s, int k, double f_min) {
    std::vector<std::pair<double, std::size_t>> maxima; // amplitude, bin
    for (std::size_t i = 1; i + 1 < s.amp.size(); ++i)
        if (s.freq[i] >= f_min && s.amp[i] > s.amp[i - 1] && s.amp[i] >= s.amp[i + 1] &&
            s.amp[i] > 0.0)
            maxima.push_back({s.amp[i], i});
    std::sort(maxima.begin(), maxima.end(), [](auto& x, auto& y) { return x.first > y.first; });
    std::vector<double> out;
    for (int i = 0; i < k && i < int(maxima.size()); ++i)
        out.push_back(refine(s.amp, maxima[std::size_t(i)].second) * s.df);
    return out;
}

} // namespace windoa
