// IEEE binary16 <-> float on the host (round-to-nearest-even), hand-rolled:
// the standard library has no half type before C++23's std::float16_t.
// Used by f16 distribution storage (lbm.hpp) and the flow cache.
#pragma once

#include <bit>
#include <cstdint>

namespace windoa {

inline std::uint16_t to_half(float f) {
    const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = int((x >> 23) & 0xFF) - 127 + 15;
    std::uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF)
        return std::uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31)
        return std::uint16_t(sign | 0x7C00u); // overflow -> inf
    if (exp <= 0) {                           // subnormal / zero
        if (exp < -10)
            return std::uint16_t(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        std::uint32_t h = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u)))
            ++h;
        return std::uint16_t(sign | h);
    }
    std::uint32_t h = (std::uint32_t(exp) << 10) | (mant >> 13);
    const std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
        ++h; // may carry into the exponent: fine
    return std::uint16_t(sign | h);
}

inline float from_half(std::uint16_t h) {
    const std::uint32_t sign = (std::uint32_t(h) & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0)
            return std::bit_cast<float>(sign);
        return (sign ? -1.0f : 1.0f) * float(mant) * (1.0f / 16777216.0f); // 2^-24
    }
    const std::uint32_t bits = exp == 31 ? sign | 0x7F800000u | (mant << 13)
                                         : sign | ((exp - 15 + 127) << 23) | (mant << 13);
    return std::bit_cast<float>(bits);
}

} // namespace windoa
