// Minimal PNG writer for screenshots: RGBA8, zlib "stored" (uncompressed)
// deflate blocks -- a valid PNG any viewer opens, with no zlib dependency.
// Larger than a compressed PNG (~4 bytes per pixel) and that is fine for a
// screenshot folder.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace windoa::app {

inline std::uint32_t png_crc(const std::uint8_t* p, std::size_t n, std::uint32_t c = 0xFFFFFFFFu) {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t v = i;
            for (int k = 0; k < 8; ++k)
                v = (v & 1u) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            t[i] = v;
        }
        return t;
    }();
    for (std::size_t i = 0; i < n; ++i)
        c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c;
}

// rgba: width x height pixels, row 0 at the top, 4 bytes R G B A each.
inline bool write_png(const std::string& path, const std::uint8_t* rgba, std::uint32_t w,
                      std::uint32_t h) {
    std::vector<std::uint8_t> raw; // filter byte 0 + row, per row
    raw.reserve(std::size_t(h) * (1 + 4 * std::size_t(w)));
    for (std::uint32_t y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + std::size_t(y) * w * 4, rgba + std::size_t(y + 1) * w * 4);
    }
    std::vector<std::uint8_t> z = {0x78, 0x01}; // zlib header, no compression
    std::size_t pos = 0;
    std::uint32_t a = 1, b = 0; // Adler-32
    for (std::uint8_t v : raw) {
        a = (a + v) % 65521u;
        b = (b + a) % 65521u;
    }
    do {
        const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
        const bool last = pos + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(std::uint8_t(n & 0xFF));
        z.push_back(std::uint8_t(n >> 8));
        z.push_back(std::uint8_t(~n & 0xFF));
        z.push_back(std::uint8_t((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + n));
        pos += n;
    } while (pos < raw.size());
    const std::uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8)
        z.push_back(std::uint8_t(adler >> s));

    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    const std::uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    f.write(reinterpret_cast<const char*>(sig), 8);
    auto chunk = [&](const char type[4], const std::vector<std::uint8_t>& data) {
        const std::uint32_t n = std::uint32_t(data.size());
        const std::uint8_t len[4] = {std::uint8_t(n >> 24), std::uint8_t(n >> 16),
                                     std::uint8_t(n >> 8), std::uint8_t(n)};
        f.write(reinterpret_cast<const char*>(len), 4);
        std::vector<std::uint8_t> td(type, type + 4);
        td.insert(td.end(), data.begin(), data.end());
        f.write(reinterpret_cast<const char*>(td.data()), std::streamsize(td.size()));
        const std::uint32_t c = png_crc(td.data(), td.size()) ^ 0xFFFFFFFFu;
        const std::uint8_t cb[4] = {std::uint8_t(c >> 24), std::uint8_t(c >> 16),
                                    std::uint8_t(c >> 8), std::uint8_t(c)};
        f.write(reinterpret_cast<const char*>(cb), 4);
    };
    const std::vector<std::uint8_t> ihdr = {std::uint8_t(w >> 24),
                                            std::uint8_t(w >> 16),
                                            std::uint8_t(w >> 8),
                                            std::uint8_t(w),
                                            std::uint8_t(h >> 24),
                                            std::uint8_t(h >> 16),
                                            std::uint8_t(h >> 8),
                                            std::uint8_t(h),
                                            8,
                                            6,
                                            0,
                                            0,
                                            0}; // 8-bit RGBA
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return bool(f);
}

} // namespace windoa::app
