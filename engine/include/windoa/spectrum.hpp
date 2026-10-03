// The amplitude spectrum of a signal sampled at possibly uneven times (the
// tunnel's batches vary in length): resampled onto a uniform grid by linear
// interpolation, mean removed, Hann-windowed and transformed by a radix-2
// FFT. Hand-rolled; no FFT library. Specification: THEORY 12.3.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace windoa {

struct Spectrum {
    std::vector<double> freq; // bin centres, cycles per unit of t, from 0 to Nyquist
    std::vector<double> amp;  // single-sided amplitude: a tone of amplitude A reads ~A
    double df = 0.0;          // bin width
    double peak_freq = 0.0;   // strongest peak above f_min, interpolated between bins
    double peak_amp = 0.0;    // 0 when the signal is flat or too short
};

// `t` increasing, the same length as `v`. `n_fft` is the number of uniform
// samples (a power of two; 0 picks the smallest one >= the sample count,
// at least 64). Peaks below `f_min` (e.g. the window's own drift) are
// ignored by the peak search.
Spectrum spectrum(std::span<const double> t, std::span<const double> v, std::size_t n_fft = 0,
                  double f_min = 0.0);

// The `k` strongest local maxima above f_min, strongest first, each
// interpolated between bins (parabola through the log amplitudes).
std::vector<double> spectral_peaks(const Spectrum& s, int k, double f_min = 0.0);

} // namespace windoa
