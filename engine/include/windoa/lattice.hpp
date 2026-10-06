// The D3Q19 lattice on the host (THEORY 1.1), in the direction order of
// shaders/lattice.glsl, which the distributions are stored in: the one
// definition every host-side use takes (the flow cache's f16 offsets, the
// f16 state round trip, the link fractions, the gates). It must stay in
// lock-step with the shader's tables; the gates would fail first if it did
// not (V28 for the velocities, P4_equiv and V23 for the weights).
#pragma once

#include <array>

namespace windoa::lattice {

inline constexpr int Q = 19;

// 0 rest, 1 - 6 the faces (+-x, +-y, +-z), 7 - 18 the twelve edges
inline constexpr std::array<std::array<int, 3>, Q> E = {{{0, 0, 0},
                                                         {1, 0, 0},
                                                         {-1, 0, 0},
                                                         {0, 1, 0},
                                                         {0, -1, 0},
                                                         {0, 0, 1},
                                                         {0, 0, -1},
                                                         {1, 1, 0},
                                                         {-1, -1, 0},
                                                         {1, -1, 0},
                                                         {-1, 1, 0},
                                                         {1, 0, 1},
                                                         {-1, 0, -1},
                                                         {1, 0, -1},
                                                         {-1, 0, 1},
                                                         {0, 1, 1},
                                                         {0, -1, -1},
                                                         {0, 1, -1},
                                                         {0, -1, 1}}};

inline constexpr std::array<float, Q> W = {1.0f / 3,  1.0f / 18, 1.0f / 18, 1.0f / 18, 1.0f / 18,
                                           1.0f / 18, 1.0f / 18, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                                           1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36,
                                           1.0f / 36, 1.0f / 36, 1.0f / 36, 1.0f / 36};

// The opposite direction (bounce-back).
inline constexpr std::array<int, Q> OPP = {0, 2,  1,  4,  3,  6,  5,  8,  7, 10,
                                           9, 12, 11, 14, 13, 16, 15, 18, 17};

static_assert(
    [] {
        for (int i = 0; i < Q; ++i)
            for (int k = 0; k < 3; ++k)
                if (E[std::size_t(OPP[std::size_t(i)])][std::size_t(k)] !=
                    -E[std::size_t(i)][std::size_t(k)])
                    return false;
        return true;
    }(),
    "OPP must reverse every direction of E");

} // namespace windoa::lattice
