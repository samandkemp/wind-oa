// Prints every Vulkan GPU the engine can see and the capabilities the plan
// depends on (async compute queue, f16, subgroups).
//   device_info                    report
//   device_info --require-compute  also exit 1 unless a discrete GPU with a
//                                  compute queue exists (the P0 CTest gate)
#include <cstdio>
#include <cstring>
#include <exception>

#include <vulkan/vulkan.h>

#include "windoa/device.hpp"

int main(int argc, char** argv) {
    const bool help =
        argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0);
    if (argc > 2 || (argc > 1 && (help || std::strcmp(argv[1], "--require-compute") != 0))) {
        std::fputs("usage: device_info [--require-compute]\n", help ? stdout : stderr);
        return help ? 0 : 2;
    }
    const bool require = argc > 1;
    try {
        const auto gpus = windoa::enumerate_gpus();
        bool ok = false;
        for (const auto& g : gpus) {
            std::printf("%s\n", g.name.c_str());
            std::printf("  Vulkan %u.%u.%u  %s  %.1f GiB device-local\n",
                        VK_API_VERSION_MAJOR(g.api_version), VK_API_VERSION_MINOR(g.api_version),
                        VK_API_VERSION_PATCH(g.api_version), g.discrete ? "discrete" : "other",
                        g.device_local_bytes / (1024.0 * 1024.0 * 1024.0));
            std::printf("  compute queue family %d, async compute family %d\n",
                        static_cast<int>(g.compute_queue_family),
                        static_cast<int>(g.async_compute_family));
            std::printf("  shaderFloat16 %s, 16-bit storage %s, subgroup size %u\n",
                        g.shader_float16 ? "yes" : "no", g.storage_16bit ? "yes" : "no",
                        g.subgroup_size);
            std::printf("  R32F linear filtering (renderer 3-D textures): %s\n",
                        g.r32f_linear_filter ? "yes" : "no");
            if (g.discrete && g.compute_queue_family != UINT32_MAX)
                ok = true;
        }
        if (gpus.empty())
            std::printf("no Vulkan devices\n");
        if (require && !ok) {
            std::printf("FAIL: no discrete GPU with a compute queue\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
