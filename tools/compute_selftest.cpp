// P0 gate: the whole compute path works end to end -- GLSL compiled to
// SPIR-V at build time, embedded, turned into a pipeline, dispatched over
// storage buffers with push constants, and read back. Reference: the same
// y = a * x + y computed on the CPU.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "saxpy_spv.hpp"
#include "windoa/context.hpp"

namespace {

struct Push { // must match `Push` in shaders/saxpy.comp (std430 push block)
    float a;
    std::uint32_t n;
};
static_assert(sizeof(Push) == 8);

} // namespace

int main() {
    try {
        windoa::Context ctx; // declared first -> destroyed last
        std::printf("device: %s\n", ctx.gpu().name.c_str());

        const std::uint32_t n = (1u << 20) + 37; // not a multiple of 256: tests the bound
        const float a = 2.5f;
        // Device-local buffers, filled and read through staging copies: the
        // same path the solvers use.
        windoa::Buffer x(ctx, n * sizeof(float));
        windoa::Buffer y(ctx, n * sizeof(float));
        std::vector<float> xs(n), ys(n), fused(n), unfused(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            xs[i] = static_cast<float>(i % 1000) * 0.001f;
            ys[i] = 1.0f - static_cast<float>(i % 7) * 0.25f;
            // The driver may or may not fuse a*x+y into an FMA; accept either.
            fused[i] = std::fma(a, xs[i], ys[i]);
            unfused[i] = a * xs[i] + ys[i];
        }
        ctx.upload(x, xs.data(), n * sizeof(float));
        ctx.upload(y, ys.data(), n * sizeof(float));

        windoa::ComputeKernel saxpy(ctx, windoa::spv::saxpy, 2, sizeof(Push));
        saxpy.bind({&x, &y});
        const Push push{a, n};
        ctx.submit_and_wait(
            [&](VkCommandBuffer cmd) { saxpy.record(cmd, &push, (n + 255) / 256); });
        ctx.download(y, ys.data(), n * sizeof(float));

        std::uint32_t bad = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (ys[i] != fused[i] && ys[i] != unfused[i]) {
                if (bad < 5) {
                    std::printf("  mismatch at %u: gpu %.9g, cpu %.9g / %.9g\n", i, ys[i], fused[i],
                                unfused[i]);
                }
                ++bad;
            }
        }
        if (bad) {
            std::printf("FAIL: %u of %u elements differ\n", bad, n);
            return 1;
        }
        std::printf("PASS: %u elements, bit-exact against the CPU (fused or unfused)\n", n);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
