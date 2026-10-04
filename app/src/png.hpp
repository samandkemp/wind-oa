// PNG writer for screenshots, with no zlib dependency: 8-bit RGB (the alpha
// of a swapchain image carries nothing), each row given the PNG filter that
// minimises its sum of absolute residuals, and the filtered rows compressed
// by a small deflate encoder: LZ77 over the 32 KiB window with hash chains,
// then a dynamic Huffman code per block. A 1600 x 900 screenshot shrinks from
// 5.8 MB to 0.2 - 0.5 MB in about 0.1 s. Formats: PNG (ISO/IEC 15948), zlib
// (RFC 1950), deflate (RFC 1951).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace windoa::app {

namespace png_detail {

inline std::uint32_t crc(const std::uint8_t* p, std::size_t n, std::uint32_t c = 0xFFFFFFFFu) {
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

// Deflate packs bits least significant first; Huffman codes go in most
// significant bit first, so they are stored bit-reversed.
struct BitWriter {
    std::vector<std::uint8_t>& out;
    std::uint64_t acc = 0;
    int n = 0;
    void put(std::uint32_t bits, int count) {
        acc |= std::uint64_t(bits) << n;
        n += count;
        while (n >= 8) {
            out.push_back(std::uint8_t(acc));
            acc >>= 8;
            n -= 8;
        }
    }
    void flush() {
        if (n > 0)
            out.push_back(std::uint8_t(acc));
        acc = 0;
        n = 0;
    }
};

// Huffman code lengths for `freq`, none longer than `limit`: a plain Huffman
// tree, with the frequencies halved (every used symbol kept at >= 1) until
// the deepest leaf fits. At least two symbols are always coded, so every
// code is complete.
inline std::vector<std::uint8_t> code_lengths(std::vector<std::uint32_t> freq, int limit) {
    int used = int(std::count_if(freq.begin(), freq.end(), [](std::uint32_t f) { return f != 0; }));
    for (std::size_t i = 0; used < 2 && i < freq.size(); ++i)
        if (!freq[i]) {
            freq[i] = 1;
            ++used;
        }
    for (;;) {
        std::vector<int> parent;
        std::vector<int> leaf(freq.size(), -1);
        using Item = std::pair<std::uint64_t, int>; // weight, node
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
        for (std::size_t i = 0; i < freq.size(); ++i)
            if (freq[i]) {
                leaf[i] = int(parent.size());
                parent.push_back(-1);
                q.push({freq[i], leaf[i]});
            }
        while (q.size() > 1) {
            const Item a = q.top();
            q.pop();
            const Item b = q.top();
            q.pop();
            const int p = int(parent.size());
            parent.push_back(-1);
            parent[std::size_t(a.second)] = p;
            parent[std::size_t(b.second)] = p;
            q.push({a.first + b.first, p});
        }
        std::vector<std::uint8_t> len(freq.size(), 0);
        int deepest = 0;
        for (std::size_t i = 0; i < freq.size(); ++i) {
            if (leaf[i] < 0)
                continue;
            int d = 0;
            for (int k = leaf[i]; parent[std::size_t(k)] >= 0; k = parent[std::size_t(k)])
                ++d;
            len[i] = std::uint8_t(d);
            deepest = std::max(deepest, d);
        }
        if (deepest <= limit)
            return len;
        for (std::uint32_t& f : freq)
            if (f)
                f = (f + 1) / 2;
    }
}

// Canonical codes from code lengths (RFC 1951 3.2.2), bit-reversed.
inline std::vector<std::uint16_t> canonical_codes(const std::vector<std::uint8_t>& len) {
    std::array<int, 16> count{}, next{};
    for (std::uint8_t l : len)
        if (l)
            ++count[l];
    int code = 0;
    for (int b = 1; b < 16; ++b) {
        code = (code + (b > 1 ? count[std::size_t(b - 1)] : 0)) << 1;
        next[std::size_t(b)] = code;
    }
    std::vector<std::uint16_t> out(len.size(), 0);
    for (std::size_t i = 0; i < len.size(); ++i) {
        if (!len[i])
            continue;
        const int c = next[len[i]]++;
        int r = 0;
        for (int k = 0; k < len[i]; ++k)
            r |= ((c >> k) & 1) << (len[i] - 1 - k);
        out[i] = std::uint16_t(r);
    }
    return out;
}

constexpr int kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                              31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr int kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr int kDistBase[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                               33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                               1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

inline int length_index(int len) {
    int i = 28;
    while (kLenBase[i] > len)
        --i;
    return i;
}

inline int distance_index(int dist) {
    int i = 29;
    while (kDistBase[i] > dist)
        --i;
    return i;
}

struct Token {
    std::uint16_t value; // a literal byte, or a match length
    std::uint16_t dist;  // 0: a literal
};

// Greedy LZ77 over the 32 KiB window: 3-byte hashes, chains of earlier
// positions searched newest first, every position of a match hashed.
inline std::vector<Token> lz77(const std::vector<std::uint8_t>& d) {
    constexpr int kWindow = 32768, kMaxMatch = 258, kChain = 48, kHashBits = 15;
    std::vector<int> head(std::size_t(1) << kHashBits, -1), prev(kWindow, -1);
    const int n = int(d.size());
    auto hash = [&](int p) {
        const std::uint32_t v =
            d[std::size_t(p)] | (d[std::size_t(p) + 1] << 8) | (d[std::size_t(p) + 2] << 16);
        return int((v * 2654435761u) >> (32 - kHashBits));
    };
    auto insert = [&](int p) {
        if (p + 2 >= n)
            return;
        const int h = hash(p);
        prev[std::size_t(p & (kWindow - 1))] = head[std::size_t(h)];
        head[std::size_t(h)] = p;
    };
    std::vector<Token> out;
    out.reserve(d.size() / 2);
    int p = 0;
    while (p < n) {
        int best = 0, best_dist = 0;
        if (p + 2 < n) {
            const int limit = std::min(kMaxMatch, n - p);
            int cand = head[std::size_t(hash(p))];
            for (int chain = 0; chain < kChain && cand >= 0 && p - cand <= kWindow - 1; ++chain) {
                if (d[std::size_t(cand + best)] == d[std::size_t(p + best)]) {
                    int l = 0;
                    while (l < limit && d[std::size_t(cand + l)] == d[std::size_t(p + l)])
                        ++l;
                    if (l > best) {
                        best = l;
                        best_dist = p - cand;
                        if (l == limit)
                            break;
                    }
                }
                const int next = prev[std::size_t(cand & (kWindow - 1))];
                if (next >= cand)
                    break; // the ring slot was reused by a newer position
                cand = next;
            }
        }
        if (best >= 3) {
            out.push_back({std::uint16_t(best), std::uint16_t(best_dist)});
            for (int k = 0; k < best; ++k)
                insert(p + k);
            p += best;
        } else {
            out.push_back({d[std::size_t(p)], 0});
            insert(p);
            ++p;
        }
    }
    return out;
}

// One dynamic-Huffman block (RFC 1951 3.2.7).
inline void write_block(BitWriter& bw, const Token* t, std::size_t count, bool last) {
    std::vector<std::uint32_t> lit_f(286, 0), dist_f(30, 0);
    for (std::size_t i = 0; i < count; ++i) {
        if (t[i].dist == 0) {
            ++lit_f[t[i].value];
        } else {
            ++lit_f[std::size_t(257 + length_index(t[i].value))];
            ++dist_f[std::size_t(distance_index(t[i].dist))];
        }
    }
    lit_f[256] = 1; // end of block
    const std::vector<std::uint8_t> lit_len = code_lengths(lit_f, 15);
    const std::vector<std::uint8_t> dist_len = code_lengths(dist_f, 15);
    const std::vector<std::uint16_t> lit_code = canonical_codes(lit_len);
    const std::vector<std::uint16_t> dist_code = canonical_codes(dist_len);
    int hlit = 286, hdist = 30;
    while (hlit > 257 && !lit_len[std::size_t(hlit - 1)])
        --hlit;
    while (hdist > 1 && !dist_len[std::size_t(hdist - 1)])
        --hdist;

    // The two length sequences, run-length coded with symbols 16 / 17 / 18.
    std::vector<std::uint8_t> seq(lit_len.begin(), lit_len.begin() + hlit);
    seq.insert(seq.end(), dist_len.begin(), dist_len.begin() + hdist);
    struct Rle {
        int sym, extra, bits;
    };
    std::vector<Rle> rle;
    for (std::size_t i = 0; i < seq.size();) {
        const std::uint8_t l = seq[i];
        std::size_t run = 1;
        while (i + run < seq.size() && seq[i + run] == l)
            ++run;
        i += run;
        if (l == 0) {
            while (run >= 11) {
                const std::size_t r = std::min<std::size_t>(run, 138);
                rle.push_back({18, int(r - 11), 7});
                run -= r;
            }
            if (run >= 3) {
                rle.push_back({17, int(run - 3), 3});
                run = 0;
            }
        } else {
            rle.push_back({l, 0, 0});
            --run;
            while (run >= 3) {
                const std::size_t r = std::min<std::size_t>(run, 6);
                rle.push_back({16, int(r - 3), 2});
                run -= r;
            }
        }
        for (; run > 0; --run)
            rle.push_back({l, 0, 0});
    }
    std::vector<std::uint32_t> cl_f(19, 0);
    for (const Rle& r : rle)
        ++cl_f[std::size_t(r.sym)];
    const std::vector<std::uint8_t> cl_len = code_lengths(cl_f, 7);
    const std::vector<std::uint16_t> cl_code = canonical_codes(cl_len);
    static constexpr int kOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                       11, 4,  12, 3, 13, 2, 14, 1, 15};
    int hclen = 19;
    while (hclen > 4 && !cl_len[std::size_t(kOrder[hclen - 1])])
        --hclen;

    bw.put(last ? 1 : 0, 1);
    bw.put(2, 2); // dynamic Huffman
    bw.put(std::uint32_t(hlit - 257), 5);
    bw.put(std::uint32_t(hdist - 1), 5);
    bw.put(std::uint32_t(hclen - 4), 4);
    for (int i = 0; i < hclen; ++i)
        bw.put(cl_len[std::size_t(kOrder[i])], 3);
    for (const Rle& r : rle) {
        bw.put(cl_code[std::size_t(r.sym)], cl_len[std::size_t(r.sym)]);
        if (r.bits)
            bw.put(std::uint32_t(r.extra), r.bits);
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (t[i].dist == 0) {
            bw.put(lit_code[t[i].value], lit_len[t[i].value]);
            continue;
        }
        const int li = length_index(t[i].value), di = distance_index(t[i].dist);
        bw.put(lit_code[std::size_t(257 + li)], lit_len[std::size_t(257 + li)]);
        if (kLenExtra[li])
            bw.put(std::uint32_t(t[i].value - kLenBase[li]), kLenExtra[li]);
        bw.put(dist_code[std::size_t(di)], dist_len[std::size_t(di)]);
        if (kDistExtra[di])
            bw.put(std::uint32_t(t[i].dist - kDistBase[di]), kDistExtra[di]);
    }
    bw.put(lit_code[256], lit_len[256]);
}

// A zlib stream of `raw`: header, deflate blocks of up to 64 Ki tokens, Adler-32.
inline std::vector<std::uint8_t> zlib_compress(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> z = {0x78, 0x9C}; // deflate, 32 KiB window
    const std::vector<Token> tokens = lz77(raw);
    BitWriter bw{z};
    constexpr std::size_t kBlock = 65536;
    std::size_t i = 0;
    do {
        const std::size_t n = std::min(kBlock, tokens.size() - i);
        write_block(bw, tokens.data() + i, n, i + n == tokens.size());
        i += n;
    } while (i < tokens.size());
    bw.flush();
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t v : raw) {
        a = (a + v) % 65521u;
        b = (b + a) % 65521u;
    }
    const std::uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8)
        z.push_back(std::uint8_t(adler >> s));
    return z;
}

inline std::uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    return std::uint8_t(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

} // namespace png_detail

// rgba: width x height pixels, row 0 at the top, 4 bytes R G B A each; the
// alpha is dropped.
inline bool write_png(const std::string& path, const std::uint8_t* rgba, std::uint32_t w,
                      std::uint32_t h) {
    using namespace png_detail;
    const std::size_t stride = 3 * std::size_t(w);
    std::vector<std::uint8_t> raw; // per row: filter type, filtered bytes
    raw.reserve(std::size_t(h) * (1 + stride));
    std::vector<std::uint8_t> row(stride), above(stride, 0), cand(stride), best(stride);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                row[3 * x + std::size_t(c)] = rgba[(std::size_t(y) * w + x) * 4 + std::size_t(c)];
        long best_cost = -1;
        std::uint8_t best_type = 0;
        for (std::uint8_t type = 0; type < 5; ++type) {
            long cost = 0;
            for (std::size_t i = 0; i < stride; ++i) {
                const int a = i >= 3 ? row[i - 3] : 0, b = above[i], c = i >= 3 ? above[i - 3] : 0;
                const int pred = type == 0   ? 0
                                 : type == 1 ? a
                                 : type == 2 ? b
                                 : type == 3 ? (a + b) / 2
                                             : paeth(a, b, c);
                cand[i] = std::uint8_t(row[i] - pred);
                cost += std::abs(int(std::int8_t(cand[i])));
            }
            if (best_cost < 0 || cost < best_cost) {
                best_cost = cost;
                best_type = type;
                best.swap(cand);
            }
        }
        raw.push_back(best_type);
        raw.insert(raw.end(), best.begin(), best.end());
        above.swap(row);
    }
    const std::vector<std::uint8_t> z = zlib_compress(raw);

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
        const std::uint32_t c = crc(td.data(), td.size()) ^ 0xFFFFFFFFu;
        const std::uint8_t cb[4] = {std::uint8_t(c >> 24), std::uint8_t(c >> 16),
                                    std::uint8_t(c >> 8), std::uint8_t(c)};
        f.write(reinterpret_cast<const char*>(cb), 4);
    };
    const std::vector<std::uint8_t> ihdr = {
        std::uint8_t(w >> 24),
        std::uint8_t(w >> 16),
        std::uint8_t(w >> 8),
        std::uint8_t(w),
        std::uint8_t(h >> 24),
        std::uint8_t(h >> 16),
        std::uint8_t(h >> 8),
        std::uint8_t(h),
        8,
        2,
        0,
        0,
        0}; // 8-bit RGB, deflate, adaptive filtering, no interlace
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return bool(f);
}

} // namespace windoa::app
