// Exact-match settled-flow cache.
//
// A developed flow takes ~2-4.5 flow-throughs to settle after a change, and
// that is physics: nothing precomputed shortcuts a configuration never run
// (starting from a neighbouring cached state saves only about 18 %). But an
// exact repeat -- flipping back to model A, returning a slider -- restores
// its settled distributions in a fraction of a second, and the restored
// flow matches a never-restored control (V23).
//
// Keyed by everything that determines the flow: the model's content hash,
// placement, spin (incl. the spinner definitions), speed, turbulence, grid,
// every solver setting, and a hash of the physics source (kPhysicsHash) --
// any change invalidates every entry. Stored as f16 of (f - w_i): the
// deviation from rest is ~100x smaller than f, so f16 keeps ~32x more of it
// (max error 3.8e-6 on a developed Ahmed flow). Oldest-first eviction
// beyond a disk budget. Specification: THEORY 9.6.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace windoa {

class FlowCache {
  public:
    // Bump when the on-disk layout changes (part of every key).
    static constexpr int kFormat = 1;

    struct Meta {
        std::array<double, 3> force_ema{};
        std::array<double, 3> torque_ema{};
        double flow_throughs = 0.0;
    };

    FlowCache(std::filesystem::path dir, double budget_mb = 512.0);

    // Build a key from a canonical description of the operating point (the
    // caller writes every flow-determining setting into `description`);
    // the physics hash and format are added here.
    std::string key(const std::string& description) const;

    // f32 distributions [Q][cells] + metadata, or nothing.
    std::optional<std::pair<std::vector<float>, Meta>> load(const std::string& key);
    // Seconds taken; -1 (nothing written) for an entry larger than the whole
    // budget, which would otherwise evict every entry, itself included.
    double save(const std::string& key, std::span<const float> f, const Meta& meta);

    int entries() const;
    double usage_mb() const;

  private:
    std::filesystem::path path(const std::string& key) const;
    void evict();

    std::filesystem::path dir_;
    std::uintmax_t budget_bytes_;
};

// 128-bit FNV-1a, hex (cache keys and mesh content hashes; not security).
std::string content_hash(const void* data, std::size_t bytes);

} // namespace windoa
