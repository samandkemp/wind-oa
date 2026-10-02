// The window's Vulkan surface + swapchain, and the per-frame objects for
// kFramesInFlight frames recorded while the GPU still works on the previous
// ones. Draws with dynamic rendering (Vulkan 1.3): no VkRenderPass or
// VkFramebuffer objects to rebuild on every resize.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <windows.h>

#include <vulkan/vulkan.h>

#include "windoa/context.hpp"

namespace windoa::app {

inline constexpr std::uint32_t kFramesInFlight = 2;

class Swapchain {
  public:
    // vsync: FIFO; otherwise MAILBOX (or IMMEDIATE) when the surface has it.
    Swapchain(Context& ctx, HINSTANCE hinst, HWND hwnd, bool vsync = true);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // (Re)create for a client area of w x h. False while it is zero-sized
    // (minimised): skip drawing until it is not.
    bool rebuild(std::uint32_t w, std::uint32_t h);
    bool valid() const { return swapchain_ != VK_NULL_HANDLE; }

    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    std::uint32_t min_image_count() const { return min_images_; }
    std::uint32_t image_count() const { return static_cast<std::uint32_t>(images_.size()); }

    // One frame: begin() waits for this slot's previous use, acquires an
    // image and opens its command buffer (null when the swapchain is out of
    // date -- rebuild and try again). end() submits and presents; false
    // when the swapchain must be rebuilt.
    struct Frame {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        std::uint32_t image_index = 0;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t slot = 0; // 0..kFramesInFlight-1; its previous use is complete
    };
    Frame begin();
    // Extra timeline semaphores for the frame's submit: waits (the solver's
    // published snapshot) and signals (the frame's completion).
    struct Timeline {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::uint64_t value = 0;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    };
    bool end(const Frame& f, std::span<const Timeline> waits = {},
             std::span<const Timeline> signals = {});

  private:
    void destroy_swapchain();

    Context& ctx_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colour_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent_{};
    std::uint32_t min_images_ = 2;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    // Signalled when image i's rendering is done; presentation waits on it.
    // Per image, not per frame: a present still holds it until that image
    // comes back from vkAcquireNextImageKHR.
    std::vector<VkSemaphore> render_done_;

    struct Slot {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;        // this slot's last submit finished
        VkSemaphore acquired = VK_NULL_HANDLE; // the acquired image is ready
    };
    Slot slots_[kFramesInFlight];
    std::uint32_t slot_ = 0;
};

} // namespace windoa::app
