#include "swapchain.hpp"

#include <algorithm>
#include <stdexcept>

#include <vulkan/vulkan_win32.h>

namespace windoa::app {

Swapchain::Swapchain(Context& ctx, HINSTANCE hinst, HWND hwnd, bool vsync) : ctx_(ctx) {
    VkWin32SurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = hinst;
    sci.hwnd = hwnd;
    vk_check(vkCreateWin32SurfaceKHR(ctx_.instance(), &sci, nullptr, &surface_),
             "vkCreateWin32SurfaceKHR");

    VkBool32 can_present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(ctx_.physical(), ctx_.graphics_family(), surface_,
                                         &can_present);
    if (!can_present) {
        vkDestroySurfaceKHR(ctx_.instance(), surface_, nullptr);
        throw std::runtime_error("the engine's queue family cannot present to this window");
    }

    // B8G8R8A8_UNORM: a UNORM (not SRGB) target, because Dear ImGui's colours
    // and the renderer's colour maps are authored as display values already.
    std::uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical(), surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.physical(), surface_, &n, fmts.data());
    format_ = fmts.at(0).format;
    colour_space_ = fmts.at(0).colorSpace;
    for (const VkSurfaceFormatKHR& f : fmts) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format_ = f.format;
            colour_space_ = f.colorSpace;
            break;
        }
    }

    if (!vsync) {
        std::uint32_t nm = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical(), surface_, &nm, nullptr);
        std::vector<VkPresentModeKHR> modes(nm);
        vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.physical(), surface_, &nm, modes.data());
        for (VkPresentModeKHR want : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
            if (std::find(modes.begin(), modes.end(), want) != modes.end()) {
                present_mode_ = want;
                break;
            }
    }

    for (Slot& s : slots_) {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pci.queueFamilyIndex = ctx_.graphics_family();
        vk_check(vkCreateCommandPool(ctx_.device(), &pci, nullptr, &s.pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = s.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(ctx_.device(), &cai, &s.cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // the first begin() must not block
        vk_check(vkCreateFence(ctx_.device(), &fci, nullptr, &s.fence), "vkCreateFence");
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vk_check(vkCreateSemaphore(ctx_.device(), &si, nullptr, &s.acquired), "vkCreateSemaphore");
    }
}

Swapchain::~Swapchain() {
    vkQueueWaitIdle(ctx_.graphics_queue());
    for (Slot& s : slots_) {
        vkDestroySemaphore(ctx_.device(), s.acquired, nullptr);
        vkDestroyFence(ctx_.device(), s.fence, nullptr);
        vkDestroyCommandPool(ctx_.device(), s.pool, nullptr);
    }
    destroy_swapchain();
    vkDestroySurfaceKHR(ctx_.instance(), surface_, nullptr);
}

void Swapchain::destroy_swapchain() {
    for (VkSemaphore s : render_done_)
        vkDestroySemaphore(ctx_.device(), s, nullptr);
    for (VkImageView v : views_)
        vkDestroyImageView(ctx_.device(), v, nullptr);
    render_done_.clear();
    views_.clear();
    images_.clear();
    if (swapchain_ != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(ctx_.device(), swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

bool Swapchain::rebuild(std::uint32_t w, std::uint32_t h) {
    vkQueueWaitIdle(ctx_.graphics_queue()); // nothing may still use the old images
    destroy_swapchain();
    if (w == 0 || h == 0)
        return false;

    VkSurfaceCapabilitiesKHR caps{};
    vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.physical(), surface_, &caps),
             "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent; // the surface dictates the size
    } else {
        extent_.width = std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    if (extent_.width == 0 || extent_.height == 0)
        return false;
    min_images_ = std::max(caps.minImageCount + 1, 2u);
    if (caps.maxImageCount > 0)
        min_images_ = std::min(min_images_, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = min_images_;
    ci.imageFormat = format_;
    ci.imageColorSpace = colour_space_;
    ci.imageExtent = extent_;
    ci.imageArrayLayers = 1;
    // Colour attachment for ImGui; transfer dst for the renderer's blit;
    // transfer src for screenshots.
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present_mode_; // FIFO (vsync) is always supported
    ci.clipped = VK_TRUE;
    vk_check(vkCreateSwapchainKHR(ctx_.device(), &ci, nullptr, &swapchain_),
             "vkCreateSwapchainKHR");

    std::uint32_t n = 0;
    vkGetSwapchainImagesKHR(ctx_.device(), swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(ctx_.device(), swapchain_, &n, images_.data());
    views_.resize(n);
    render_done_.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk_check(vkCreateImageView(ctx_.device(), &vi, nullptr, &views_[i]), "vkCreateImageView");
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vk_check(vkCreateSemaphore(ctx_.device(), &si, nullptr, &render_done_[i]),
                 "vkCreateSemaphore");
    }
    return true;
}

Swapchain::Frame Swapchain::begin() {
    Slot& s = slots_[slot_];
    vk_check(vkWaitForFences(ctx_.device(), 1, &s.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    Frame f;
    const VkResult r = vkAcquireNextImageKHR(ctx_.device(), swapchain_, UINT64_MAX, s.acquired,
                                             VK_NULL_HANDLE, &f.image_index);
    if (r == VK_ERROR_OUT_OF_DATE_KHR)
        return {};
    if (r != VK_SUBOPTIMAL_KHR)
        vk_check(r, "vkAcquireNextImageKHR");

    // Reset the fence only once a submit is certain; otherwise the next wait
    // on it would never return.
    vk_check(vkResetFences(ctx_.device(), 1, &s.fence), "vkResetFences");
    vk_check(vkResetCommandPool(ctx_.device(), s.pool, 0), "vkResetCommandPool");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(s.cmd, &bi), "vkBeginCommandBuffer");

    f.cmd = s.cmd;
    f.image = images_[f.image_index];
    f.view = views_[f.image_index];
    f.slot = slot_;
    return f;
}

bool Swapchain::end(const Frame& f, std::span<const Timeline> waits,
                    std::span<const Timeline> signals) {
    Slot& s = slots_[slot_];
    slot_ = (slot_ + 1) % kFramesInFlight;
    vk_check(vkEndCommandBuffer(f.cmd), "vkEndCommandBuffer");

    // synchronization2 submit: the acquire wait names the stages that must
    // not start before the image is acquired (all: the first use may be a
    // blit or a colour write, and the layout transition itself).
    std::vector<VkSemaphoreSubmitInfo> wait(1 + waits.size()), signal(1 + signals.size());
    wait[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait[0].semaphore = s.acquired;
    wait[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signal[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal[0].semaphore = render_done_[f.image_index];
    signal[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    for (std::size_t i = 0; i < waits.size(); ++i) {
        wait[i + 1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wait[i + 1].semaphore = waits[i].semaphore;
        wait[i + 1].value = waits[i].value;
        wait[i + 1].stageMask = waits[i].stage;
    }
    for (std::size_t i = 0; i < signals.size(); ++i) {
        signal[i + 1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signal[i + 1].semaphore = signals[i].semaphore;
        signal[i + 1].value = signals[i].value;
        signal[i + 1].stageMask = signals[i].stage;
    }
    VkCommandBufferSubmitInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cbi.commandBuffer = f.cmd;
    VkSubmitInfo2 si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si.waitSemaphoreInfoCount = static_cast<std::uint32_t>(wait.size());
    si.pWaitSemaphoreInfos = wait.data();
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signal.size());
    si.pSignalSemaphoreInfos = signal.data();
    vk_check(vkQueueSubmit2(ctx_.graphics_queue(), 1, &si, s.fence), "vkQueueSubmit2");

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &render_done_[f.image_index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &f.image_index;
    const VkResult r = vkQueuePresentKHR(ctx_.graphics_queue(), &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
        return false;
    vk_check(r, "vkQueuePresentKHR");
    return true;
}

} // namespace windoa::app
