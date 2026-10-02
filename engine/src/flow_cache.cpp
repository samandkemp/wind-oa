#include "windoa/flow_cache.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>

#include "physics_hash.hpp"
#include "windoa/half.hpp"

namespace windoa {

namespace {

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'W', 'O', 'A', 'F', 'L', 'O', 'W', '1'};
constexpr int kQ = 19;
constexpr float kW[kQ] = {1.0f / 3,  1.0f / 18, 1.0f / 18, 1.0f / 18, 1.0f / 18,
                          1.0f / 18, 1.0f / 18, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                          1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                          1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36};

} // namespace

std::string content_hash(const void* data, std::size_t bytes) {
    // Two independent FNV-1a 64 streams (different offset bases) -> 128 bits.
    std::uint64_t a = 1469598103934665603ull, b = 0x84222325CBF29CE4ull;
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        a = (a ^ p[i]) * 1099511628211ull;
        b = (b ^ p[i]) * 0x100000001B3ull;
        b ^= b >> 29;
    }
    char buf[33];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx", static_cast<unsigned long long>(a),
                  static_cast<unsigned long long>(b));
    return buf;
}

FlowCache::FlowCache(fs::path dir, double budget_mb)
    : dir_(std::move(dir)), budget_bytes_(std::uintmax_t(budget_mb * 1024.0 * 1024.0)) {
    std::error_code ec;
    fs::create_directories(dir_, ec);
}

std::string FlowCache::key(const std::string& description) const {
    const std::string blob =
        description + "|physics=" + kPhysicsHash + "|format=" + std::to_string(kFormat);
    return content_hash(blob.data(), blob.size()).substr(0, 24);
}

fs::path FlowCache::path(const std::string& key) const {
    return dir_ / (key + ".flow");
}

std::optional<std::pair<std::vector<float>, FlowCache::Meta>>
FlowCache::load(const std::string& key) {
    const fs::path p = path(key);
    std::ifstream in(p, std::ios::binary);
    if (!in)
        return std::nullopt;
    char magic[8];
    std::uint64_t n = 0;
    Meta meta;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&meta), sizeof(meta));
    if (!in || std::memcmp(magic, kMagic, 8) != 0 || n % kQ != 0) {
        in.close();
        std::error_code ec;
        fs::remove(p, ec); // corrupt entry: drop it
        return std::nullopt;
    }
    std::vector<std::uint16_t> h(n);
    in.read(reinterpret_cast<char*>(h.data()), std::streamsize(n * sizeof(std::uint16_t)));
    if (!in)
        return std::nullopt;
    in.close();
    const std::size_t cells = n / kQ;
    std::vector<float> f(n);
    for (int i = 0; i < kQ; ++i)
        for (std::size_t c = 0; c < cells; ++c)
            f[i * cells + c] = from_half(h[i * cells + c]) + kW[i];
    std::error_code ec;
    fs::last_write_time(p, fs::file_time_type::clock::now(), ec); // recently used
    return std::make_pair(std::move(f), meta);
}

double FlowCache::save(const std::string& key, std::span<const float> f, const Meta& meta) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t cells = f.size() / kQ;
    std::vector<std::uint16_t> h(f.size());
    for (int i = 0; i < kQ; ++i)
        for (std::size_t c = 0; c < cells; ++c)
            h[i * cells + c] = to_half(f[i * cells + c] - kW[i]);
    const fs::path p = path(key), tmp = path(key + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary);
        const std::uint64_t n = f.size();
        out.write(kMagic, 8);
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
        out.write(reinterpret_cast<const char*>(h.data()), std::streamsize(h.size() * 2));
    }
    std::error_code ec;
    fs::rename(tmp, p, ec); // atomic on the same volume
    evict();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void FlowCache::evict() {
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    std::uintmax_t total = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir_, ec)) {
        if (e.path().extension() != ".flow")
            continue;
        files.emplace_back(e.last_write_time(ec), e.path());
        total += e.file_size(ec);
    }
    std::sort(files.begin(), files.end());
    for (const auto& [t, p] : files) {
        if (total <= budget_bytes_)
            break;
        total -= fs::file_size(p, ec);
        fs::remove(p, ec);
    }
}

int FlowCache::entries() const {
    int n = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir_, ec))
        if (e.path().extension() == ".flow")
            ++n;
    return n;
}

double FlowCache::usage_mb() const {
    std::uintmax_t total = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir_, ec))
        if (e.path().extension() == ".flow")
            total += e.file_size(ec);
    return double(total) / (1024.0 * 1024.0);
}

} // namespace windoa
