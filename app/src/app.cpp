#include "app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <thread>

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_win32.h"

#include <vulkan/vulkan_win32.h>

#include "png.hpp"
#include "windoa/airspeed.hpp"

namespace windoa::app {

namespace {

using Clock = std::chrono::steady_clock;
constexpr float kRevoxSettle = 0.5f; // s after the last placement edit
// --no-vsync: frames at most this often. Uncapped, the window drew ~900 frames a
// second and the graphics queue starved the solver's (3,040 -> 1,280 MLUPS).
constexpr double kMaxFps = 240.0;

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

void imgui_check(VkResult r) {
    vk_check(r, "Dear ImGui Vulkan backend");
}

// The docking branch's Vulkan backend asks the platform layer to make a
// surface for any ImGui window dragged out of the main one. The Win32
// backend does not provide that hook, so the app supplies it here.
int imgui_create_surface(ImGuiViewport* vp, ImU64 instance, const void* allocator,
                         ImU64* out_surface) {
    VkWin32SurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd = static_cast<HWND>(vp->PlatformHandleRaw);
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult r =
        vkCreateWin32SurfaceKHR(reinterpret_cast<VkInstance>(instance), &sci,
                                static_cast<const VkAllocationCallbacks*>(allocator), &surface);
    *out_surface = reinterpret_cast<ImU64>(surface);
    return static_cast<int>(r);
}

// Colour stops for the legends -- the same tables as the shaders' maps.
struct Stop {
    float t, r, g, b;
};
const std::vector<Stop> kCoolwarm = {{0.0f, 0, 0, 1}, {0.5f, 1, 1, 1}, {1.0f, 1, 0, 0}};
const std::vector<Stop> kVort = {{0.0f, 40 / 255.f, 130 / 255.f, 1.0f},
                                 {0.3f, 10 / 255.f, 30 / 255.f, 90 / 255.f},
                                 {0.5f, 3 / 255.f, 4 / 255.f, 8 / 255.f},
                                 {0.7f, 110 / 255.f, 25 / 255.f, 20 / 255.f},
                                 {1.0f, 1.0f, 140 / 255.f, 40 / 255.f}};
const std::vector<Stop> kDye = {{0.0f, 0.20f, 0.07f, 0.42f},
                                {0.35f, 0.66f, 0.13f, 0.46f},
                                {0.65f, 0.98f, 0.45f, 0.18f},
                                {1.0f, 1.00f, 0.93f, 0.62f}};
const std::vector<Stop> kQcore = {{0.0f, 0.30f, 0.95f, 0.50f}, {1.0f, 0.85f, 0.55f, 0.30f}};
const std::vector<Stop> kGrey = {{0.0f, 0, 0, 0}, {1.0f, 0.88f, 0.92f, 1.0f}};

ImU32 ramp(const std::vector<Stop>& s, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    for (std::size_t i = 1; i < s.size(); ++i) {
        if (t <= s[i].t) {
            const float u = (t - s[i - 1].t) / std::max(s[i].t - s[i - 1].t, 1e-6f);
            return ImGui::ColorConvertFloat4ToU32({s[i - 1].r + u * (s[i].r - s[i - 1].r),
                                                   s[i - 1].g + u * (s[i].g - s[i - 1].g),
                                                   s[i - 1].b + u * (s[i].b - s[i - 1].b), 1.0f});
        }
    }
    return ImGui::ColorConvertFloat4ToU32({s.back().r, s.back().g, s.back().b, 1.0f});
}

const char* kFieldNames[] = {"speed vs freestream",   "pressure (Cp)",
                             "|vorticity|",           "streamwise vorticity",
                             "local Mach number",     "schlieren (shocks)",
                             "mean speed (averaged)", "turbulence intensity (averaged)"};
const std::vector<Stop> kRecirc = {{0.0f, 0.25f, 0.75f, 0.85f}, {1.0f, 0.10f, 0.35f, 0.95f}};

} // namespace

// -- construction ----------------------------------------------------------------

App::App(const Options& o) : opt_(o), ts_(tunnel_preset(o.preset)) {
    ts_.storage_f16 = o.f16;
    ImGui_ImplWin32_EnableDpiAwareness();
    window_ = std::make_unique<Window>(L"wind-oa -- LBM wind tunnel", 1600, 900);

    ContextOptions co;
    co.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
    co.device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    co.graphics = true;
    co.async_compute = true; // the solver on its own queue (P3d)
    ctx_ = std::make_unique<Context>(co);
    std::printf("device: %s (graphics family %u, engine family %u)\n", ctx_->gpu().name.c_str(),
                ctx_->graphics_family(), ctx_->queue_family());
    std::printf("tunnel %d x %d x %d (%.2f M cells), tau %.4f, %s storage\n", ts_.nx, ts_.ny,
                ts_.nz, ts_.nx * double(ts_.ny) * ts_.nz / 1e6, ts_.tau,
                ts_.storage_f16 ? "f16" : "f32");

    swapchain_ =
        std::make_unique<Swapchain>(*ctx_, window_->hinstance(), window_->hwnd(), !o.no_vsync);
    std::uint32_t w, h;
    window_->client_size(w, h);
    swapchain_->rebuild(w, h);
    colour_format_ = swapchain_->format();
    init_imgui();

    // Worker + renderer on its snapshots
    sim_ = std::make_unique<SimWorker>(*ctx_, ts_, std::filesystem::path("cache") / "flow");
    renderer_ = std::make_unique<render::VolumeRenderer>(*ctx_, ts_.nx, ts_.ny, ts_.nz);
    tracers_ = std::make_unique<render::Tracers>(*ctx_, ts_.nx, ts_.ny, ts_.nz);
    for (int k = 0; k < SimWorker::kSlots; ++k) {
        const SimWorker::Snapshot& sn = sim_->snapshot(k);
        renderer_->set_sources(k, {sn.flags.get(), sn.macro.get(), sn.aux.get(), sn.dye.get(),
                                   sn.mean.get(), sn.m2.get()});
        tracers_->set_sources(k, *sn.flags, *sn.macro);
    }
    smoke_id_ = renderer_->add_splat_source({&tracers_->smoke_pos(), &tracers_->smoke_colours()});
    lines_id_ = renderer_->add_splat_source({&tracers_->line_verts(), &tracers_->line_colours()});
    marker_id_ =
        renderer_->add_splat_source({&tracers_->marker_verts(), &tracers_->marker_colours()});
    arrows_id_ =
        renderer_->add_splat_source({&tracers_->arrow_verts(), &tracers_->arrow_colours()});

    VkQueryPoolCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2 * kFramesInFlight;
    vk_check(vkCreateQueryPool(ctx_->device(), &qi, nullptr, &queries_), "vkCreateQueryPool");
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_->physical(), &props);
    tick_ms_ = props.limits.timestampPeriod * 1e-6;
    if (o.no_vsync) // the frame cap's timer (Windows 10 1803 or later; else a plain one)
        frame_timer_ = CreateWaitableTimerExW(
            nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (o.no_vsync && !frame_timer_)
        frame_timer_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);

    // Model menu: the catalogue, then any STL dropped into ./models ("Imported";
    // imported STLs are never shipped -- *.stl is gitignored).
    for (const catalogue::Entry& e : catalogue::entries())
        menu_.push_back({&e, {}, e.label, e.group});
    std::error_code ec;
    if (std::filesystem::is_directory("models", ec)) {
        std::vector<std::filesystem::path> stls;
        for (const auto& f : std::filesystem::directory_iterator("models", ec))
            if (f.path().extension() == ".stl" || f.path().extension() == ".STL")
                stls.push_back(f.path());
        std::sort(stls.begin(), stls.end());
        for (const auto& p : stls)
            menu_.push_back({nullptr, p, p.filename().string(), "Imported"});
    }
    std::size_t start = 0;
    for (std::size_t i = 0; i < menu_.size(); ++i)
        if (menu_[i].entry && menu_[i].entry->id == o.model)
            start = i;
    camera_.frame_start(ts_.nx, ts_.ny, ts_.nz);
    rs_.surface = render::Surface::Smooth;
    u_command_ = ts_.u_inlet;
    // Scripted view (tests / screenshots)
    auto has = [&](const char* w) {
        return ("," + o.show + ",").find(std::string(",") + w + ",") != std::string::npos;
    };
    if (has("q"))
        rs_.vortex_cores = true;
    if (has("dye"))
        show_dye_ = true;
    if (has("lines"))
        show_streamlines_ = true;
    if (has("nosmoke"))
        show_smoke_ = false;
    if (has("slice"))
        slice_mode_ = 1;
    if (has("hslice"))
        slice_mode_ = 2;
    if (has("xslice"))
        slice_mode_ = 3;
    if (has("nobox"))
        rs_.box = false;
    if (has("nosurface"))
        rs_.surface = render::Surface::Hidden;
    if (has("nohaze"))
        rs_.haze = false;
    if (has("help"))
        show_help_ = true;
    if (has("noui"))
        show_ui_ = false;
    if (has("noplots"))
        show_plots_ = false;
    if (has("voxel"))
        rs_.surface = render::Surface::Voxel;
    if (has("avg"))
        averaging_ = true;
    if (has("recirc"))
        rs_.recirculation = true;
    if (has("lic"))
        rs_.slice_lic = true;
    if (has("arrows"))
        slice_arrows_ = true;
    if (has("timelines"))
        smoke_mode_ = 1;
    if (has("nopaint"))
        paint_mode_ = 0;
    if (has("wallspeed"))
        paint_mode_ = 2;
    if (has("reversed"))
        paint_mode_ = 3;
    if (has("oil"))
        paint_mode_ = 4;
    if (has("analysis"))
        analysis_open_ = true;
    if (has("probes"))
        n_probes_ = 2; // placed beside the model once it is known
    static const char* kFieldIds[] = {"speed", "pressure",  "vorticity", "vortx",
                                      "mach",  "schlieren", "mean",      "turb"};
    for (int k = 0; k < 8; ++k)
        if (o.field == kFieldIds[k])
            rs_.field = static_cast<render::Field>(k);
    choose_model(start);
    if (o.aoa != 0.0f || o.size > 0.0f) {
        if (o.aoa != 0.0f)
            placement_.aoa_deg = o.aoa;
        if (o.size > 0.0f) // the size slider's range
            placement_.length_cells = std::clamp(o.size, 12.0f, 0.8f * float(ts_.nx));
        const Placement p = placement_;
        sim_->post([p](Tunnel& t) { t.set_placement(p); });
    }
    if (o.spin >= 0.0f) {
        spin_on_ = true;
        spin_ratio_ = o.spin;
        const float r = o.spin;
        sim_->post([r](Tunnel& t) { t.set_spin(true, r); });
    }
    if (o.rotors >= 0.0f) {
        rotors_on_ = true;
        rotor_tsr_ = o.rotors;
        const float t = o.rotors;
        sim_->post([t](Tunnel& tn) { tn.set_rotors(true, t); });
    }
    if (o.power >= 0.0f) {
        power_on_ = true;
        throttle_ = o.power;
        const float t = o.power;
        sim_->post([t](Tunnel& tn) { tn.set_power(true, t); });
    }
    if (show_dye_)
        sim_->post([](Tunnel& t) { t.set_dye(true); });
    if (averaging_)
        sim_->post([](Tunnel& t) { t.set_averaging(true); });
    if (o.transonic) { // start in transonic mode (its natural view)
        sim_->post([](Tunnel& t) { t.set_transonic(true); });
        if (o.field.empty())
            rs_.field = render::Field::Mach;
        show_smoke_ = false;
    }
    if (o.mach > 0.0f) {
        mach_command_ = o.mach;
        const float m = o.mach;
        sim_->post([m](Tunnel& t) { t.set_mach(m); });
    }
    if (o.warmup_steps > 0) { // tests: develop the flow before the first frame
        const int n = o.warmup_steps;
        sim_->set_running(false);
        sim_->post([n](Tunnel& t) {
            for (int done = 0; done < n; done += 200)
                t.advance(std::min(200, n - done));
        });
        sim_->flush();
        sim_->set_running(true);
    }
    sim_->flush();
    if (o.zoom > 0.0f) { // scripted close-ups
        camera_.target = sim_->status().placed_centre;
        camera_.distance *= o.zoom;
    }
    if (o.view) { // scripted viewpoint
        camera_.azimuth = (*o.view)[0] * 3.14159265f / 180.0f;
        camera_.elevation = (*o.view)[1] * 3.14159265f / 180.0f;
    }
    if (n_probes_ > 0) { // scripted: probes in the wake (centreline, shear layer)
        const TunnelStatus st = sim_->status();
        const float L = placement_.length_cells;
        probe_pos_[0] = {st.placed_centre[0] + 0.8f * L, st.placed_centre[1], st.placed_centre[2]};
        probe_pos_[1] = {st.placed_centre[0] + 0.8f * L, st.placed_centre[1] + st.ext_y_half,
                         st.placed_centre[2]};
        post_probes();
        sim_->flush();
    }
}

App::~App() {
    if (ctx_)
        vkQueueWaitIdle(ctx_->graphics_queue());
    sim_.reset(); // stop the worker before anything it hands frames to
    if (ctx_)
        vkDeviceWaitIdle(ctx_->device());
    if (queries_)
        vkDestroyQueryPool(ctx_->device(), queries_, nullptr);
    if (frame_timer_)
        CloseHandle(frame_timer_);
    shot_buf_.reset(); // every Vulkan object goes before the Context
    tracers_.reset();
    renderer_.reset();
    if (imgui_ready_) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }
    swapchain_.reset();
    ctx_.reset();
}

void App::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = "windoa_imgui.ini";
    const float dpi = ImGui_ImplWin32_GetDpiScaleForHwnd(window_->hwnd());
    ImGui::StyleColorsDark();
    ImGuiStyle& st = ImGui::GetStyle();
    st.WindowRounding = 4.0f;
    st.FrameRounding = 3.0f;
    st.Colors[ImGuiCol_WindowBg].w = 0.86f;
    st.ScaleAllSizes(dpi);
    st.FontScaleDpi = dpi;
    ImGui_ImplWin32_Init(window_->hwnd());
    ImGui::GetPlatformIO().Platform_CreateVkSurface = imgui_create_surface;

    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = ctx_->instance();
    ii.PhysicalDevice = ctx_->physical();
    ii.Device = ctx_->device();
    ii.QueueFamily = ctx_->graphics_family();
    ii.Queue = ctx_->graphics_queue();
    ii.DescriptorPoolSize = 16;
    ii.MinImageCount = swapchain_->min_image_count();
    ii.ImageCount = std::max(swapchain_->min_image_count(), swapchain_->image_count());
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colour_format_;
    ii.CheckVkResultFn = imgui_check;
    ImGui_ImplVulkan_Init(&ii);
    imgui_ready_ = true;
}

// -- model -------------------------------------------------------------------------------

void App::choose_model(std::size_t index) {
    model_index_ = index;
    const MenuItem& it = menu_[index];
    Model m;
    try {
        m = it.entry ? model_from_catalogue(*it.entry) : model_from_stl(it.stl);
    } catch (const std::exception& e) {
        toast_ = std::string("could not load: ") + e.what();
        toast_t_ = Clock::now();
        return;
    }
    if (it.entry) {
        placement_ = default_placement(*it.entry, ts_, m.mesh);
    } else {
        placement_ = Placement{};
    }
    has_spinners_ = !m.spinners.empty();
    const bool has_rotors = !m.rotors.empty();
    if (has_rotors) // each rotor model starts at its own design point
        rotor_tsr_ = float(m.rotors.front().tsr);
    placement_dirty_ = false;
    const Placement p = placement_;
    auto shared = std::make_shared<Model>(std::move(m));
    sim_->post([shared, p](Tunnel& t) { t.set_model(std::move(*shared), p); });
    if (has_rotors) {
        const bool on = rotors_on_;
        const float tsr = rotor_tsr_;
        sim_->post([on, tsr](Tunnel& t) { t.set_rotors(on, tsr); });
    }
    cd_.clear();
    cl_.clear();
    cm_.clear();
    was_settled_ = false;
}

// -- the frame loop ------------------------------------------------------------------------

int App::run() {
    int frames = 0;
    auto last = Clock::now();
    while (window_->pump() && (opt_.max_frames <= 0 || frames < opt_.max_frames)) {
        std::uint32_t w, h;
        window_->client_size(w, h);
        if (w == 0 || h == 0) {
            Sleep(16);
            continue;
        }
        if (window_->take_resized() || !swapchain_->valid()) {
            if (!swapchain_->rebuild(w, h))
                continue;
            ImGui_ImplVulkan_SetMinImageCount(swapchain_->min_image_count());
        }
        const TunnelStatus st = sim_->status();

        // Placement edits re-voxelise once they settle (~0.5 s)
        if (placement_dirty_ &&
            std::chrono::duration<float>(Clock::now() - placement_edit_).count() > kRevoxSettle) {
            const Placement p = placement_;
            sim_->post([p](Tunnel& t) { t.set_placement(p); });
            placement_dirty_ = false;
        }
        update_rake(st);

        // -- UI --
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
        handle_keys(st);
        ui(st);
        camera_.update(ImGui::GetIO());
        ImGui::Render();

        // -- draw --
        const Swapchain::Frame f = swapchain_->begin();
        if (!f.cmd) {
            swapchain_->rebuild(w, h);
            continue;
        }
        const std::uint64_t rv = ++render_value_;
        const SimWorker::Frame sf = sim_->acquire(rv);
        const bool last_frame = opt_.max_frames > 0 && frames + 1 == opt_.max_frames;
        const bool shot = screenshot_pending_ || (!opt_.shot.empty() && last_frame);
        record_frame(f, sf, st, shot);
        std::vector<Swapchain::Timeline> waits, signals;
        if (sf.slot >= 0)
            waits.push_back(
                {sim_->sim_timeline(), sf.value, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
        signals.push_back({sim_->render_timeline(), rv, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
        if (!swapchain_->end(f, waits, signals))
            swapchain_->rebuild(w, h);
        if (shot) {
            write_screenshot(screenshot_pending_ ? std::string() : opt_.shot);
            screenshot_pending_ = false;
        }

        if (opt_.no_vsync) { // wait out the frame, then yield up to the deadline
            const auto due = last + std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(1.0 / kMaxFps));
            const double left_ms =
                std::chrono::duration<double, std::milli>(due - Clock::now()).count();
            // A high-resolution timer: Sleep(1) lasts a whole 15.6 ms scheduler tick.
            if (frame_timer_ && left_ms > 0.75) {
                LARGE_INTEGER t;
                t.QuadPart = -LONGLONG((left_ms - 0.5) * 1e4); // relative, 100 ns units
                if (SetWaitableTimer(frame_timer_, &t, 0, nullptr, nullptr, FALSE))
                    WaitForSingleObject(frame_timer_, INFINITE);
            }
            while (Clock::now() < due)
                std::this_thread::yield();
        }
        const auto now = Clock::now();
        frame_ms_ +=
            0.1 * (std::chrono::duration<double, std::milli>(now - last).count() - frame_ms_);
        last = now;
        ++frames;
    }
    const TunnelStatus st = sim_->status();
    std::printf("exit after %d frames: step %lld, %.0f MLUPS (%.2f ms/step, %d steps/batch), "
                "render %.2f ms GPU, frame %.1f ms; %s\n",
                frames, static_cast<long long>(st.steps), sim_->mlups(), sim_->ms_per_step(),
                sim_->last_batch_steps(), render_gpu_ms_, frame_ms_, st.phase.c_str());
    return 0;
}

void App::update_rake(const TunnelStatus& st) {
    // The smoke wand tracks the model: upstream of it, centred on its y, z.
    const float nx = float(ts_.nx), ny = float(ts_.ny), nz = float(ts_.nz);
    float half_y = rake_autofit_ ? st.ext_y_half * 1.1f : rake_h_frac_ * ny;
    float half_z = rake_autofit_ ? st.ext_z_half * 1.1f : rake_w_frac_ * nz;
    float x, cy, cz;
    if (rake_track_) {
        x = std::max(4.0f, st.placed_centre[0] - 0.18f * nx);
        cy = st.placed_centre[1];
        cz = st.placed_centre[2];
        rake_y_ = cy / ny;
        rake_z_ = cz / nz;
    } else {
        x = tracers_->rake()[0];
        cy = rake_y_ * ny;
        cz = rake_z_ * nz;
    }
    const std::array<float, 5> r{x, cy, cz, half_y, half_z};
    if (r != last_rake_) {
        tracers_->set_rake(x, cy, cz, half_y, half_z);
        sim_->post([r](Tunnel& t) { t.set_dye_rake(r[0], r[1], r[2], r[3], r[4]); });
        last_rake_ = r;
    }
}

void App::record_frame(const Swapchain::Frame& f, const SimWorker::Frame& sf,
                       const TunnelStatus& st, bool shot) {
    VkCommandBuffer cmd = f.cmd;
    if (slot_written_[f.slot]) { // the slot's previous frame is complete: its GPU time
        std::uint64_t ts[2] = {};
        if (vkGetQueryPoolResults(ctx_->device(), queries_, 2 * f.slot, 2, sizeof(ts), ts,
                                  sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            render_gpu_ms_ += 0.1 * (double(ts[1] - ts[0]) * tick_ms_ - render_gpu_ms_);
    }
    vkCmdResetQueryPool(cmd, queries_, 2 * f.slot, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 2 * f.slot);

    const VkExtent2D ext = swapchain_->extent();
    bool drew_volume = false;
    if (sf.slot >= 0) {
        renderer_->resize(std::uint32_t(float(ext.width) * render_scale_),
                          std::uint32_t(float(ext.height) * render_scale_));
        // Lagrangian visuals advance by the solver steps since the snapshot
        // last drawn (0 while paused); a reset or new geometry respawns them.
        if (sf.flow_epoch != last_epoch_ || sf.geometry_version != last_geo_) {
            tracers_->request_reset();
            last_epoch_ = sf.flow_epoch;
            last_geo_ = sf.geometry_version;
            last_snap_steps_ = sf.steps;
        }
        const int steps = last_snap_steps_ < 0
                              ? 0
                              : int(std::clamp<std::int64_t>(sf.steps - last_snap_steps_, 0, 400));
        last_snap_steps_ = sf.steps;
        const float u_ref = std::max(st.u_applied, 1e-3f);
        const bool lbm = !sf.transonic; // tracers integrate in lattice time
        tracers_->set_timelines(smoke_mode_ == 1 ? pulse_steps_ : 0);
        const std::vector<std::array<float, 3>> probes(probe_pos_.begin(),
                                                       probe_pos_.begin() + n_probes_);
        const bool planes = show_planes_ && averaging_ && lbm && analysis_.wake_valid;
        tracers_->set_overlay(planes ? analysis_.x_upstream : -1, planes ? analysis_.x_survey : -1,
                              probes);
        tracers_->set_blades(lbm ? st.rotor_segments
                                 : std::vector<std::array<std::array<float, 3>, 2>>{});
        tracers_->show_wand(show_smoke_);
        if (lbm)
            tracers_->record(cmd, sf.slot, steps, u_ref, show_smoke_, show_streamlines_);

        render::Settings s = rs_;
        s.slice_axis = slice_mode_ == 1 ? 2 : slice_mode_ == 2 ? 1 : slice_mode_ == 3 ? 0 : -1;
        if (lbm) {
            s.u_ref = u_ref;
            s.rho_ref = float(st.rho_ref);
            // Cp divides by u^2: while the inlet ramps it is startup noise over
            // a tiny dynamic pressure, so the body stays plain until a third of
            // the way; the haze fades in with the speed.
            s.paint_surface = paint_mode_ > 0 && st.u_applied > 0.3f * st.u_command;
            s.stats_inv_weight = sf.stats_inv_weight;
            const float frac = std::min(1.0f, st.u_applied / std::max(st.u_command, 1e-3f));
            s.haze_gain = rs_.haze_gain * frac * frac;
            s.dye = show_dye_ && sf.dye;
        } else { // Euler units: u in Mach, p_ref = 1 / gamma, Cp = (p - p_ref) / (M^2 / 2)
            s.u_ref = std::max(st.mach_applied, 1e-3f);
            s.rho_ref = 1.0f / euler::kGamma;
            s.pscale = 1.0f;
            s.gamma = euler::kGamma;
            s.dye = false;
            s.vortex_cores = false;
            s.paint_surface = paint_mode_ > 0;
            s.stats_inv_weight = 0.0f;
            s.recirculation = false;
        }
        s.paint = static_cast<render::Paint>(std::max(paint_mode_ - 1, 0));
        std::uint32_t arrow_segments = 0;
        if (slice_arrows_ && s.slice_axis >= 0) {
            const int dims[3] = {ts_.nx, ts_.ny, ts_.nz};
            arrow_segments = tracers_->record_arrows(cmd, sf.slot, s.slice_axis,
                                                     s.slice_pos * float(dims[s.slice_axis]),
                                                     arrow_spacing_, s.u_ref);
        }
        std::vector<render::SplatDraw> splats;
        if (lbm && show_smoke_) {
            splats.push_back({smoke_id_, render::Tracers::kSmokeParticles, false, smoke_radius_,
                              0.85f, smoke_mode_ == 1 ? 1.0f : 0.0f});
        }
        if (lbm && (show_smoke_ || n_probes_ > 0 || planes || !st.rotor_segments.empty()))
            splats.push_back({marker_id_, tracers_->marker_segments(), true, 0.0f, 0.9f, 1.0f});
        if (arrow_segments > 0)
            splats.push_back({arrows_id_, arrow_segments, true, 0.0f, 0.9f, 1.0f});
        if (lbm && show_streamlines_)
            splats.push_back({lines_id_, render::Tracers::line_segments(), true, 0.0f, 0.8f});
        renderer_->record(cmd, camera_.view(fov_deg_), s, sf.slot, sf.geometry_version, splats);
        drew_volume = true;
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2 * f.slot + 1);
    slot_written_[f.slot] = true;

    if (drew_volume) { // blit the volume over the whole swapchain image
        image_barrier(cmd, f.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_BLIT_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {std::int32_t(renderer_->width()), std::int32_t(renderer_->height()),
                              1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {std::int32_t(ext.width), std::int32_t(ext.height), 1};
        vkCmdBlitImage(cmd, renderer_->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, f.image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        image_barrier(
            cmd, f.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    } else {
        image_barrier(cmd, f.image, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }
    VkRenderingAttachmentInfo att{};
    att.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    att.imageView = f.view;
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = drew_volume ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.clearValue.color = {{0.05f, 0.05f, 0.09f, 1.0f}};
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = {{0, 0}, ext};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
    VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (shot) { // copy the finished frame (with the UI) for a screenshot
        const VkDeviceSize bytes = VkDeviceSize(ext.width) * ext.height * 4;
        if (!shot_buf_ || shot_buf_->size() < bytes)
            shot_buf_ = std::make_unique<Buffer>(*ctx_, bytes, MemoryUse::Readback);
        image_barrier(cmd, f.image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {ext.width, ext.height, 1};
        vkCmdCopyImageToBuffer(cmd, f.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               shot_buf_->handle(), 1, &region);
        layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        shot_extent_ = ext;
    }
    image_barrier(cmd, f.image, layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_NONE, 0);
}

// Written once the frame that copied it has completed. `path` empty: a
// timestamped file under screenshots/ (gitignored).
void App::write_screenshot(std::string path) {
    if (!shot_buf_)
        return;
    vkQueueWaitIdle(ctx_->graphics_queue());
    const std::uint32_t w = shot_extent_.width, h = shot_extent_.height;
    std::vector<std::uint8_t> rgba(std::size_t(w) * h * 4);
    std::memcpy(rgba.data(), shot_buf_->data(), rgba.size());
    if (colour_format_ == VK_FORMAT_B8G8R8A8_UNORM || colour_format_ == VK_FORMAT_B8G8R8A8_SRGB)
        for (std::size_t i = 0; i < rgba.size(); i += 4)
            std::swap(rgba[i], rgba[i + 2]);
    for (std::size_t i = 3; i < rgba.size(); i += 4)
        rgba[i] = 255;
    if (path.empty()) {
        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);
        char name[64];
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_s(&tm, &t);
        std::strftime(name, sizeof(name), "screenshots/shot_%Y%m%d_%H%M%S.png", &tm);
        path = name;
    }
    if (write_png(path, rgba.data(), w, h)) {
        toast_ = "saved " + path;
        std::printf("screenshot: %s (%u x %u)\n", path.c_str(), w, h);
    } else {
        toast_ = "could not write " + path;
    }
    toast_t_ = Clock::now();
}

// -- keys -----------------------------------------------------------------------------------

void App::handle_keys(const TunnelStatus& st) {
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
        return;
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false))
        PostQuitMessage(0);
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
        const bool p = !st.paused;
        sim_->post([p](Tunnel& t) { t.set_paused(p); });
    }
    if (ImGui::IsKeyPressed(ImGuiKey_H, false))
        show_ui_ = !show_ui_;
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
        show_help_ = !show_help_;
    if (ImGui::IsKeyPressed(ImGuiKey_P, false))
        screenshot_pending_ = true;
    if (ImGui::IsKeyPressed(ImGuiKey_F, false))
        camera_.target = st.placed_centre;
}

// -- panels ---------------------------------------------------------------------------------

void App::ui(const TunnelStatus& st) {
    update_title(st);
    if (st.analysis_version != analysis_.version)
        analysis_ = sim_->analysis();
    if (show_plots_)
        plot_strip(st);
    legends(st);
    if (show_help_)
        help_window();
    if (!show_ui_)
        return;
    panel_tunnel(st);
    panel_model(st);
    panel_compare(st);
    panel_view(st);
    panel_analysis(st);
    reset_layout_ = want_reset_layout_; // the button, or a layout that does not fit
    want_reset_layout_ = false;
}

float App::plot_strip_height() const {
    return show_plots_ ? 0.16f * ImGui::GetMainViewport()->WorkSize.y : 0.0f;
}

// A panel: its initial layout as fractions of the viewport (ImGui remembers
// later moves and docking in windoa_imgui.ini). An undocked panel must stay
// inside the area above the plot strip: one being dragged or resized is held
// there, and a layout that does not fit (saved on a larger window, or the
// window shrunk) is put back to the defaults for the current size, since
// pulling each panel in separately piles them on top of one another. Items
// leave a fixed label column, so long labels are not clipped.
bool App::begin_panel(const char* name, float x, float y, float w, float h) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImGuiCond cond = reset_layout_ ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
    ImGui::SetNextWindowPos(
        {vp->WorkPos.x + x * vp->WorkSize.x, vp->WorkPos.y + y * vp->WorkSize.y}, cond);
    ImGui::SetNextWindowSize({w * vp->WorkSize.x, h * vp->WorkSize.y}, cond);
    if (reset_layout_)
        ImGui::SetNextWindowDockID(0, ImGuiCond_Always);
    const bool open = ImGui::Begin(name);
    if (!ImGui::IsWindowDocked()) {
        const ImVec2 lo = vp->WorkPos;
        const ImVec2 hi = {vp->WorkPos.x + vp->WorkSize.x,
                           vp->WorkPos.y + vp->WorkSize.y - plot_strip_height()};
        const ImVec2 pos = ImGui::GetWindowPos(), size = ImGui::GetWindowSize();
        const ImVec2 fit = {std::min(size.x, hi.x - lo.x), std::min(size.y, hi.y - lo.y)};
        const ImVec2 inside = {std::clamp(pos.x, lo.x, hi.x - fit.x),
                               std::clamp(pos.y, lo.y, hi.y - fit.y)};
        const bool misfit =
            fit.x != size.x || fit.y != size.y || inside.x != pos.x || inside.y != pos.y;
        if (misfit && ImGui::IsAnyMouseDown()) {
            ImGui::SetWindowSize(fit);
            ImGui::SetWindowPos(inside);
        } else if (misfit) {
            want_reset_layout_ = true;
        }
    }
    ImGui::PushItemWidth(
        -(ImGui::CalcTextSize("render quality (steps)").x + ImGui::GetStyle().ItemInnerSpacing.x));
    return open;
}

void App::end_panel() {
    ImGui::PopItemWidth();
    ImGui::End();
}

void App::panel_tunnel(const TunnelStatus& st) {
    begin_panel("Tunnel", 0.01f, 0.02f, 0.24f, 0.40f);
    int regime = st.transonic ? 1 : 0;
    static const char* kRegime[] = {"subsonic (LBM)", "TRANSONIC (Euler)"};
    if (ImGui::Combo("regime", &regime, kRegime, 2)) {
        const bool on = regime == 1;
        sim_->post([on](Tunnel& t) { t.set_transonic(on); });
        if (on && rs_.field == render::Field::Speed)
            rs_.field = render::Field::Mach;
        if (!on && rs_.field == render::Field::Mach)
            rs_.field = render::Field::Speed;
        rs_.haze_floor = -1.0f;
    }
    if (st.transonic) {
        panel_transonic(st);
        end_panel();
        return;
    }
    const bool settled = st.settled && !st.developing;
    const ImVec4 phase_col = st.paused       ? ImVec4(1.0f, 0.75f, 0.3f, 1.0f)
                             : st.developing ? ImVec4(0.55f, 0.8f, 1.0f, 1.0f)
                             : settled       ? ImVec4(0.45f, 0.95f, 0.55f, 1.0f)
                                             : ImVec4(1, 1, 1, 1);
    ImGui::PushStyleColor(ImGuiCol_Text, phase_col);
    ImGui::TextWrapped("%s", st.phase.c_str());
    ImGui::PopStyleColor();
    ImGui::Text("step %lld   %.0f MLUPS (%s)   %d steps/batch", static_cast<long long>(st.steps),
                sim_->mlups(), ts_.storage_f16 ? "f16" : "f32", sim_->last_batch_steps());
    if (!st.cache_note.empty())
        ImGui::TextDisabled("flow cache: %s", st.cache_note.c_str());
    if (!st.health_note.empty() && (st.paused || st.health_note_age < 15.0))
        ImGui::TextColored({1.0f, 0.45f, 0.4f, 1.0f}, "%s", st.health_note.c_str());
    ImGui::Text("ms: sim %.2f/step   render %.2f (GPU)   frame %.1f", sim_->ms_per_step(),
                render_gpu_ms_, frame_ms_);
    ImGui::Separator();
    ImGui::Text("Re_sim %.0f   A_ref %.0f cells^2", st.re_sim, st.a_ref);
    ImGui::Text("Cd %+.3f  Cl %+.3f  Cs %+.3f%s", st.cd, st.cl, st.cs,
                st.developing ? "   (settling)" : "");
    ImGui::Text("Cm_z %+.4f   max|u| %.3f", st.cm, st.max_speed);
    ImGui::Separator();
    // The slider reads in lattice units and, beside them, the Mach-matched
    // speed in sea-level air (airspeed.hpp; ImGui takes the label as a format).
    char speed_fmt[48];
    std::snprintf(speed_fmt, sizeof(speed_fmt), "%%.3f  = %.0f mph",
                  airspeed::mph(airspeed::metres_per_second(
                      airspeed::mach_from_lattice(double(u_command_)))));
    if (ImGui::SliderFloat("flow speed", &u_command_, 0.005f, ts_.u_max, speed_fmt)) {
        const float u = u_command_;
        sim_->post([u](Tunnel& t) { t.set_speed(u); });
    }
    ImGui::SetItemTooltip(
        "Freestream speed in lattice units (Mach = u sqrt 3, kept under ~0.19).\n"
        "mph and m/s are the speed in sea-level air at that Mach number; the\n"
        "Reynolds number stays near 10^3 whatever the speed (see Re_sim).\n"
        "Slew-limited so the lattice never shocks; a change over 10 %% re-develops.");
    ImGui::Text("  u %.3f -> %.3f   Ma %.3f", st.u_applied, st.u_command, st.airspeed_mach);
    if (ImGui::SliderFloat("inlet turbulence %", &turb_pct_, 0.0f, 2.0f, "%.1f")) {
        const float t = turb_pct_;
        sim_->post([t](Tunnel& tn) { tn.set_turbulence(t); });
    }
    ImGui::SetItemTooltip(
        "Real-tunnel freestream turbulence: a divergence-free gust field\n"
        "carried in through the inlet (0 = clean; real tunnels about 0.1 - 2 %%).");
    if (ImGui::SliderFloat("sim rate cap", &sim_rate_cap_, 0.0f, 5000.0f,
                           sim_rate_cap_ <= 0.0f ? "flat out" : "%.0f steps/s"))
        sim_->set_max_steps_per_second(sim_rate_cap_);
    ImGui::SetItemTooltip("The solver runs flat out on its own GPU queue; cap it to watch the\n"
                          "flow evolve slowly. The window stays smooth either way.");
    if (st.developing && ImGui::Button("skip developing"))
        sim_->post([](Tunnel& t) { t.skip_develop(); });
    if (st.developing)
        ImGui::SameLine();
    if (ImGui::Button(st.paused ? "resume (Space)" : "pause (Space)")) {
        const bool p = !st.paused;
        sim_->post([p](Tunnel& t) { t.set_paused(p); });
    }
    ImGui::SameLine();
    if (ImGui::Button("reset flow"))
        sim_->post([](Tunnel& t) { t.reset_flow("flow reset"); });
    ImGui::SetItemTooltip("Restart from rest with the inlet ramp (never from the flow cache).");
    ImGui::TextDisabled("keys: F1 help, H hide panels, P screenshot, F focus");
    if (!toast_.empty() && std::chrono::duration<float>(Clock::now() - toast_t_).count() < 6.0f)
        ImGui::TextColored({1.0f, 0.8f, 0.4f, 1.0f}, "%s", toast_.c_str());
    end_panel();
}

void App::panel_transonic(const TunnelStatus& st) {
    ImGui::PushStyleColor(ImGuiCol_Text, st.developing ? ImVec4(0.55f, 0.8f, 1.0f, 1.0f)
                                                       : ImVec4(0.45f, 0.95f, 0.55f, 1.0f));
    ImGui::TextWrapped("%s", st.phase.c_str());
    ImGui::PopStyleColor();
    ImGui::Text("peak local Mach %.2f   %d steps/batch   %.2f ms/step", st.peak_mach,
                sim_->last_batch_steps(), sim_->ms_per_step());
    ImGui::Text("Cd %+.3f  Cl %+.3f  Cs %+.3f", st.cd, st.cl, st.cs);
    ImGui::Text("Cm_z %+.4f", st.cm);
    ImGui::TextDisabled("  inviscid: pressure + wave drag only");
    if (!st.health_note.empty() && st.health_note_age < 15.0)
        ImGui::TextColored({1.0f, 0.45f, 0.4f, 1.0f}, "%s", st.health_note.c_str());
    char mach_fmt[48];
    std::snprintf(mach_fmt, sizeof(mach_fmt), "%%.3f  = %.0f mph",
                  airspeed::mph(airspeed::metres_per_second(double(mach_command_))));
    if (ImGui::SliderFloat("Mach", &mach_command_, ts_.mach_min, ts_.mach_max, mach_fmt)) {
        const float m = mach_command_;
        sim_->post([m](Tunnel& t) { t.set_mach(m); });
    }
    if (ImGui::Button(st.paused ? "resume (Space)" : "pause (Space)")) {
        const bool p = !st.paused;
        sim_->post([p](Tunnel& t) { t.set_paused(p); });
    }
    ImGui::SameLine();
    if (ImGui::Button("restart flow"))
        sim_->post([](Tunnel& t) { t.restart_flow(); });
    ImGui::TextDisabled("LBM paused (kept for switching back). Smoke, dye,");
    ImGui::TextDisabled("vortex cores, spin and rotors are subsonic-only.");
    ImGui::TextDisabled("Try field: schlieren for the shocks.");
}

void App::panel_model(const TunnelStatus& st) {
    begin_panel("Model", 0.01f, 0.43f, 0.24f, 0.38f);
    const MenuItem& cur = menu_[model_index_];
    if (ImGui::BeginCombo("model", cur.label.c_str(), ImGuiComboFlags_HeightLarge)) {
        std::string group;
        for (std::size_t i = 0; i < menu_.size(); ++i) {
            if (menu_[i].group != group) {
                group = menu_[i].group;
                ImGui::SeparatorText(group.c_str());
            }
            if (ImGui::Selectable(menu_[i].label.c_str(), i == model_index_) && i != model_index_)
                choose_model(i);
        }
        ImGui::EndCombo();
    }
    if (placement_dirty_)
        ImGui::TextDisabled("  [voxelising when you stop adjusting...]");
    else if (st.out_of_bounds)
        ImGui::TextColored({1.0f, 0.6f, 0.3f, 1.0f},
                           "  ! extends outside the tunnel: reduce size / turn");
    ImGui::TextDisabled("%zu tris -> %zu voxels (%.0f ms)", st.n_tris, st.n_solid, st.vox_ms);

    bool changed = false;
    static const char* kGround[] = {"aviation (free air)", "rolling road", "fixed ground"};
    int g = static_cast<int>(placement_.ground);
    if (ImGui::Combo("placement", &g, kGround, 3)) {
        placement_.ground = static_cast<catalogue::Ground>(g);
        if (placement_.ground == catalogue::Ground::Fixed)
            placement_.ride_height = 0.0f;
        else if (placement_.ground == catalogue::Ground::Road)
            placement_.ride_height = std::max(4.0f, 0.06f * placement_.length_cells);
        else
            placement_.pos_frac[1] = 0.5f;
        changed = true;
    }
    changed |=
        ImGui::SliderFloat("size [cells]", &placement_.length_cells, 12.0f, 0.8f * ts_.nx, "%.0f");
    changed |= ImGui::SliderFloat("pitch / AoA", &placement_.aoa_deg, -180.0f, 180.0f, "%.1f deg");
    changed |= ImGui::SliderFloat("yaw", &placement_.yaw_deg, -180.0f, 180.0f, "%.1f deg");
    changed |= ImGui::SliderFloat("roll", &placement_.roll_deg, -180.0f, 180.0f, "%.1f deg");
    if (ImGui::Button("reset attitude")) {
        placement_.aoa_deg = placement_.yaw_deg = placement_.roll_deg = 0.0f;
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(Ctrl+click a slider to type)");
    changed |= ImGui::SliderFloat("pos x", &placement_.pos_frac[0], 0.1f, 0.8f, "%.3f");
    if (placement_.ground != catalogue::Ground::Air) {
        changed |= ImGui::SliderFloat("ride height", &placement_.ride_height, 0.0f, 0.25f * ts_.ny,
                                      "%.1f");
        if (placement_.ride_height < 3.5f && placement_.ground == catalogue::Ground::Road)
            ImGui::TextColored({1.0f, 0.7f, 0.3f, 1.0f},
                               "  low: a thin under-body gap may destabilise");
    } else {
        changed |= ImGui::SliderFloat("pos y", &placement_.pos_frac[1], 0.15f, 0.85f, "%.3f");
    }
    if (changed) {
        placement_dirty_ = true;
        placement_edit_ = Clock::now();
    }

    static const char* kArea[] = {"frontal", "planform", "manual"};
    if (ImGui::Combo("ref area", &area_mode_, kArea, 3) ||
        (area_mode_ == 2 &&
         ImGui::SliderFloat("manual area", &a_manual_, 10.0f, 5000.0f, "%.0f"))) {
        const int m = area_mode_;
        const double a = a_manual_;
        sim_->post([m, a](Tunnel& t) { t.set_area_mode(static_cast<AreaMode>(m), a); });
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%.0f", st.a_ref);
    if (st.has_spinners) {
        bool post = ImGui::Checkbox("spinning parts", &spin_on_);
        if (spin_on_)
            post |= ImGui::SliderFloat("spin ratio (rim/U)", &spin_ratio_, 0.0f, 3.0f, "%.2f");
        if (post) {
            const bool on = spin_on_;
            const float r = spin_ratio_;
            sim_->post([on, r](Tunnel& t) { t.set_spin(on, r); });
        }
        if (spin_on_ && st.spin_scale < 0.999f)
            ImGui::TextColored({1.0f, 0.8f, 0.4f, 1.0f}, "  limited to %.2f (wall speed cap)",
                               spin_ratio_ * st.spin_scale);
    }
    if (st.has_rotors) {
        bool post = ImGui::Checkbox("rotors turning", &rotors_on_);
        ImGui::SetItemTooltip(
            "The blades are actuator lines: lines of lift and drag forces from the local flow\n"
            "(blade-element theory), drawn turning, not solid. Parked, they still take the\n"
            "wind. The tip-speed ratio is tip speed over the wind speed (a turbine runs ~7).");
        if (rotors_on_)
            post |= ImGui::SliderFloat("tip-speed ratio", &rotor_tsr_, 0.0f, 12.0f, "%.1f");
        if (post) {
            const bool on = rotors_on_;
            const float t = rotor_tsr_;
            sim_->post([on, t](Tunnel& tn) { tn.set_rotors(on, t); });
        }
        if (st.rotor_coeffs_ready)
            ImGui::TextDisabled("  rotor C_T %+.3f  C_P %+.3f%s", st.rotor_ct, st.rotor_cp,
                                st.rotor_cp > 0.0 ? " (driven)" : "");
        else
            ImGui::TextDisabled("  rotor C_T, C_P: once the wind is at speed");
        if (st.rotor_blockage > 0.05)
            ImGui::TextColored({1.0f, 0.75f, 0.35f, 1.0f},
                               "  swept area %.0f %% of the section: blocked, not free air",
                               100.0 * st.rotor_blockage);
        ImGui::SetItemTooltip("In a closed tunnel the walls hold the flow round a large rotor;\n"
                              "past ~5 %% of the section its coefficients depart from free-air\n"
                              "values (a turbine's C_P can pass even the Betz limit).\n"
                              "Reduce the size for free-air numbers.");
    }
    if (st.has_ports) {
        bool post = ImGui::Checkbox("engines (jets / intakes)", &power_on_);
        ImGui::SetItemTooltip(
            "Exhausts blow and intakes draw. Subsonic: each face moves at its speed ratio x\n"
            "throttle x U. Transonic: exhausts emit their exit state (a hot, under-expanded\n"
            "plume: shock diamonds). The balance then reads the net force, thrust included.");
        if (power_on_)
            post |= ImGui::SliderFloat("throttle", &throttle_, 0.0f, 3.0f, "%.2f");
        if (post) {
            const bool on = power_on_;
            const float t = throttle_;
            sim_->post([on, t](Tunnel& tn) { tn.set_power(on, t); });
        }
        if (power_on_ && st.jet_scale < 0.999f)
            ImGui::TextColored({1.0f, 0.8f, 0.4f, 1.0f}, "  limited to %.2f (jet speed cap)",
                               throttle_ * st.jet_scale);
    }
    end_panel();
}

void App::panel_compare(const TunnelStatus& st) {
    begin_panel("Compare", 0.75f, 0.02f, 0.24f, 0.20f);
    const Snap now{menu_[model_index_].label,
                   placement_.aoa_deg,
                   placement_.length_cells,
                   st.cd,
                   st.cl,
                   st.cm};
    if (ImGui::Button("save as A"))
        ab_[0] = now;
    ImGui::SameLine();
    if (ImGui::Button("save as B"))
        ab_[1] = now;
    for (int k = 0; k < 2; ++k) {
        if (!ab_[k])
            continue;
        const Snap& r = *ab_[k];
        ImGui::Text("%c: %.24s  AoA %+.0f  size %.0f", 'A' + k, r.model.c_str(), r.aoa, r.size);
        ImGui::Text("   Cd %+.3f  Cl %+.3f  Cm %+.4f", r.cd, r.cl, r.cm);
    }
    if (ab_[0] && ab_[1])
        ImGui::TextColored({0.6f, 0.9f, 1.0f, 1.0f}, "B-A: dCd %+.3f  dCl %+.3f",
                           ab_[1]->cd - ab_[0]->cd, ab_[1]->cl - ab_[0]->cl);
    end_panel();
}

void App::panel_view(const TunnelStatus& st) {
    begin_panel("View", 0.75f, 0.23f, 0.24f, 0.58f);
    static const char* kSurface[] = {"hidden", "voxel", "smooth"};
    int surf = static_cast<int>(rs_.surface);
    if (ImGui::Combo("surface", &surf, kSurface, 3))
        rs_.surface = static_cast<render::Surface>(surf);
    static const char* kPaint[] = {"none", "pressure (Cp)", "near-wall speed", "reversed flow",
                                   "oil flow"};
    ImGui::Combo("surface paint", &paint_mode_, kPaint, 5);
    ImGui::SetItemTooltip(
        "Cp: pressure of the air touching the model (red compression, blue suction).\n"
        "Near-wall speed: the flow one cell off the surface, which scales with the skin\n"
        "friction. Reversed flow: blue where the near-wall flow runs upstream (separated).\n"
        "Oil flow: streaks along the near-wall flow, as in a tunnel oil-film test.");
    int field = static_cast<int>(rs_.field);
    if (ImGui::Combo("field", &field, kFieldNames, IM_ARRAYSIZE(kFieldNames))) {
        rs_.field = static_cast<render::Field>(field);
        rs_.haze_floor = -1.0f;
    }
    if (field >= 6 && st.avg_samples == 0)
        ImGui::TextColored({1.0f, 0.75f, 0.35f, 1.0f}, "  needs time averaging (Analysis panel)");
    ImGui::Checkbox("flow haze (3D, translucent)", &rs_.haze);
    ImGui::SetItemTooltip("The field through the whole tunnel; undisturbed flow is transparent,\n"
                          "so only what the model changes shows. 'haze floor' sets the cut.");
    if (rs_.haze) {
        ImGui::SliderFloat("  haze strength", &rs_.haze_gain, 0.2f, 4.0f, "%.2f");
        float floor_v = rs_.haze_floor >= 0.0f ? rs_.haze_floor : render::default_floor(rs_.field);
        if (ImGui::SliderFloat("  haze floor", &floor_v, 0.0f, 0.5f, "%.3f"))
            rs_.haze_floor = floor_v;
    }
    static const char* kSlice[] = {"off", "vertical (x-y)", "horizontal (x-z)", "cross (y-z)"};
    ImGui::Combo("field slice", &slice_mode_, kSlice, 4);
    if (slice_mode_) {
        ImGui::SliderFloat("  slice pos", &rs_.slice_pos, 0.02f, 0.98f, "%.2f");
        ImGui::Checkbox("  flow texture (LIC)", &rs_.slice_lic);
        ImGui::SetItemTooltip("Line-integral convolution: noise smeared along the in-plane flow,\n"
                              "so the slice shows every streamline at once.");
        ImGui::Checkbox("  velocity arrows", &slice_arrows_);
        if (slice_arrows_)
            ImGui::SliderFloat("  arrow spacing", &arrow_spacing_, 2.0f, 12.0f, "%.0f cells");
    }
    ImGui::Separator();
    if (st.transonic) {
        ImGui::TextDisabled("(smoke, dye, streamlines, vortex cores: subsonic only)");
    } else {
        ImGui::Checkbox("smoke", &show_smoke_);
        if (show_smoke_) {
            static const char* kSmoke[] = {"streaklines (wand)", "timelines (pulsed wire)"};
            ImGui::Combo("  smoke as", &smoke_mode_, kSmoke, 2);
            ImGui::SetItemTooltip(
                "Streaklines: continuous smoke from the wand. Timelines: a line of\n"
                "particles released across the wand every few steps, as from a pulsed\n"
                "hydrogen-bubble wire; their deformation shows the velocity profile.");
            if (smoke_mode_ == 1)
                ImGui::SliderInt("  pulse every", &pulse_steps_, 20, 300, "%d steps");
        }
        ImGui::Checkbox("dye smoke (volumetric)", &show_dye_);
        ImGui::SetItemTooltip(
            "A transported concentration from nozzles on the smoke wand: fills and\n"
            "marks the wake like theatrical smoke (never leaks into the model).");
    }
    if (!st.transonic && show_dye_ != st.dye_on) {
        const bool d = show_dye_;
        sim_->post([d](Tunnel& t) { t.set_dye(d); });
    }
    if (!st.transonic && show_dye_) {
        ImGui::SliderFloat("  dye density", &rs_.dye_gain, 0.2f, 10.0f, "%.1f");
        ImGui::Checkbox("  colour by speed", &rs_.dye_by_speed);
    }
    if (!st.transonic) {
        ImGui::Checkbox("streamlines", &show_streamlines_);
        ImGui::Checkbox("vortex cores (Q)", &rs_.vortex_cores);
        ImGui::SetItemTooltip(
            "Where rotation beats strain (Q criterion on a smoothed velocity),\n"
            "thresholded at a multiple of the flow's own RMS: the dominant cores.");
        if (rs_.vortex_cores)
            ImGui::SliderFloat("  Q threshold", &rs_.q_sense, 0.1f, 10.0f, "%.2f");
        ImGui::Checkbox("mean reversed flow", &rs_.recirculation);
        ImGui::SetItemTooltip("Translucent shells where the time-averaged streamwise flow runs\n"
                              "backwards: the mean recirculation bubbles. Needs time averaging.");
        if (rs_.recirculation && st.avg_samples == 0)
            ImGui::TextColored({1.0f, 0.75f, 0.35f, 1.0f},
                               "  needs time averaging (Analysis panel)");
    }
    if (!st.transonic && (show_smoke_ || show_dye_)) {
        ImGui::Checkbox("smoke tracks model", &rake_track_);
        ImGui::Checkbox("fit smoke to model", &rake_autofit_);
        if (!rake_autofit_) {
            ImGui::SliderFloat("smoke height", &rake_h_frac_, 0.03f, 0.50f, "%.2f");
            ImGui::SliderFloat("smoke width", &rake_w_frac_, 0.03f, 0.50f, "%.2f");
        }
        if (!rake_track_) {
            ImGui::SliderFloat("smoke y", &rake_y_, 0.1f, 0.9f, "%.2f");
            ImGui::SliderFloat("smoke z", &rake_z_, 0.1f, 0.9f, "%.2f");
        }
        ImGui::SliderFloat("smoke detail", &smoke_radius_, 0.06f, 0.35f, "%.2f");
    }
    ImGui::Separator();
    ImGui::Checkbox("time plots", &show_plots_);
    if (show_plots_) {
        ImGui::SameLine();
        if (ImGui::Button("refit plots")) {
            cd_.refit_recent(0.5);
            cl_.refit_recent(0.5);
            cm_.refit_recent(0.5);
        }
    }
    ImGui::Checkbox("tunnel box", &rs_.box);
    ImGui::SameLine();
    if (ImGui::Button("reset layout"))
        want_reset_layout_ = true;
    ImGui::SetItemTooltip("Put every panel back where it starts (undocked, Analysis folded).");
    ImGui::SliderInt("render quality (steps)", &rs_.steps, 32, 256);
    ImGui::SliderFloat("render scale", &render_scale_, 0.25f, 1.0f, "%.2f");
    ImGui::SliderFloat("field of view", &fov_deg_, 20.0f, 90.0f, "%.0f deg");
    ImGui::TextDisabled("RMB orbit, MMB pan, wheel zoom, WASD/QE, F focus");
    end_panel();
}

void App::post_probes() {
    const std::vector<std::array<float, 3>> p(probe_pos_.begin(), probe_pos_.begin() + n_probes_);
    sim_->post([p](Tunnel& t) { t.set_probes(p); });
}

// Time averaging, the wake survey, probes and spectra (THEORY 12).
void App::panel_analysis(const TunnelStatus& st) {
    ImGui::SetNextWindowCollapsed(!analysis_open_,
                                  reset_layout_ ? ImGuiCond_Always : ImGuiCond_FirstUseEver);
    if (!begin_panel("Analysis", 0.505f, 0.46f, 0.24f, 0.35f)) {
        end_panel();
        return;
    }
    const TunnelAnalysis& a = analysis_;
    const ImVec4 warn{1.0f, 0.75f, 0.35f, 1.0f};
    const float plot_h = 110.0f * ImGui::GetStyle().FontScaleDpi;
    auto plot_area = [&](float h) {
        const ImVec2 p0 = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        ImGui::Dummy({w, h});
        return std::pair<ImVec2, ImVec2>{p0, {p0.x + w, p0.y + h}};
    };

    ImGui::SeparatorText("time averaging");
    if (st.transonic)
        ImGui::TextDisabled("(subsonic only)");
    if (ImGui::Checkbox("average the flow", &averaging_)) {
        const bool on = averaging_;
        sim_->post([on](Tunnel& t) { t.set_averaging(on); });
    }
    ImGui::SetItemTooltip(
        "Accumulate the mean and the variance of every cell while the flow is\n"
        "developed; a new operating point restarts the window. Feeds the mean-speed\n"
        "and turbulence fields, the reversed-flow shells and the wake survey.");
    if (averaging_) {
        ImGui::SameLine();
        if (ImGui::Button("restart"))
            sim_->post([](Tunnel& t) { t.restart_averaging(); });
        if (st.averaging_active)
            ImGui::Text("%d samples over %.2f flow-throughs", st.avg_samples, st.avg_flow_throughs);
        else
            ImGui::TextDisabled("waiting for the flow to settle");
    }

    ImGui::SeparatorText("wake survey");
    if (!averaging_) {
        ImGui::TextDisabled("needs time averaging");
    } else if (!a.wake_valid) {
        ImGui::TextDisabled("collecting the mean flow...");
    } else {
        ImGui::Text("momentum balance   Cd %.4f", a.cd_wake);
        ImGui::Text("force balance      Cd %.4f   (%+.2f %%)", a.cd_balance,
                    (a.cd_wake / std::max(std::abs(a.cd_balance), 1e-12) - 1.0) * 100.0);
        ImGui::SetItemTooltip(
            "Two measurements of one drag: the fall in the mean flow's momentum flux\n"
            "between the upstream plane and the survey plane, and the force on the\n"
            "model averaged over the same window (THEORY 12.2).");
        ImGui::TextDisabled("planes x = %d and %d;  mass flux change %.1e", a.x_upstream,
                            a.x_survey, a.mass_imbalance);
        if (a.includes_floor)
            ImGui::TextColored(warn, "ground mode: the floor's shear is inside the volume");
        std::vector<double> y(a.profile_y.begin(), a.profile_y.end()),
            uy(a.profile_uy.begin(), a.profile_uy.end()), z(a.profile_z.begin(), a.profile_z.end()),
            uz(a.profile_uz.begin(), a.profile_uz.end());
        const auto [p0, p1] = plot_area(plot_h);
        plot_xy(ImGui::GetWindowDrawList(), p0, p1,
                {{&y, &uy, IM_COL32(255, 140, 50, 255)}, {&z, &uz, IM_COL32(80, 180, 255, 255)}},
                "cells", "mean u_x / U: across y, z");
    }
    ImGui::Checkbox("show planes", &show_planes_);
    ImGui::SameLine();
    if (ImGui::Checkbox("auto plane", &wake_auto_)) {
        const int x = wake_auto_ ? -1 : (a.x_survey > 0 ? a.x_survey : ts_.nx / 2);
        wake_x_ = x < 0 ? wake_x_ : x;
        sim_->post([x](Tunnel& t) { t.set_wake_plane(x); });
    }
    if (!wake_auto_ && ImGui::SliderInt("survey plane x", &wake_x_, 1, ts_.nx - 2)) {
        const int x = wake_x_;
        sim_->post([x](Tunnel& t) { t.set_wake_plane(x); });
    }

    ImGui::SeparatorText("probes and spectra");
    int n = n_probes_;
    if (ImGui::SliderInt("probes", &n, 0, kMaxProbes)) {
        // new probes start in the wake: centreline, shear layer, further
        // downstream, and upstream as a freestream reference
        const float L = placement_.length_cells;
        const auto c = st.placed_centre;
        const std::array<std::array<float, 3>, kMaxProbes> defaults = {
            {{c[0] + 0.8f * L, c[1], c[2]},
             {c[0] + 0.8f * L, c[1] + st.ext_y_half, c[2]},
             {c[0] + 1.4f * L, c[1], c[2]},
             {std::max(4.0f, c[0] - 0.6f * L), c[1], c[2]}}};
        for (int i = n_probes_; i < n; ++i) {
            probe_pos_[i] = defaults[i];
            probe_pos_[i][0] = std::min(probe_pos_[i][0], float(ts_.nx - ts_.outlet_sponge - 2));
        }
        n_probes_ = n;
        spec_source_ = std::min(spec_source_, n_probes_);
        post_probes();
    }
    for (int i = 0; i < n_probes_; ++i) {
        ImGui::PushID(i);
        const auto& pc = render::Tracers::kProbeColours[i];
        ImGui::ColorButton("##c", {pc[0], pc[1], pc[2], 1.0f}, ImGuiColorEditFlags_NoTooltip,
                           {ImGui::GetFrameHeight(), ImGui::GetFrameHeight()});
        ImGui::SameLine();
        if (ImGui::DragFloat3("x y z", probe_pos_[i].data(), 0.25f, 0.0f, float(ts_.nx - 1),
                              "%.1f")) {
            probe_pos_[i][1] = std::clamp(probe_pos_[i][1], 0.0f, float(ts_.ny - 1));
            probe_pos_[i][2] = std::clamp(probe_pos_[i][2], 0.0f, float(ts_.nz - 1));
            post_probes();
        }
        ImGui::PopID();
    }
    std::vector<std::string> names{"lift (Cl)"};
    for (int i = 0; i < n_probes_; ++i)
        names.push_back("probe " + std::to_string(i + 1));
    std::vector<const char*> cnames;
    for (const auto& s : names)
        cnames.push_back(s.c_str());
    ImGui::Combo("signal", &spec_source_, cnames.data(), int(cnames.size()));
    if (spec_source_ > 0) {
        static const char* kComp[] = {"u_x", "u_y", "u_z", "rho (pressure)"};
        ImGui::Combo("component", &spec_comp_, kComp, 4);
    }
    const bool probe = spec_source_ > 0 && spec_source_ <= a.n_probes;
    const Spectrum& sp = probe ? a.probe_spec[spec_source_ - 1][spec_comp_] : a.lift;
    if (sp.freq.empty()) {
        ImGui::TextDisabled(st.developing ? "waiting for the flow to settle"
                                          : "collecting the developed signal...");
    } else {
        const double st_scale = a.l_ref / std::max(a.u_ref, 1e-9); // St per (cycles / step)
        const double st_peak = sp.peak_freq * st_scale;
        const double st_ac = a.f_acoustic * st_scale; // the first acoustic mode, as St
        if (!probe && a.st_shedding > 0.0)
            ImGui::Text("shedding St %.3f   (period %.0f steps)", a.st_shedding,
                        st_scale / a.st_shedding);
        if (sp.peak_amp > 0.0) {
            const double mode = st_peak / std::max(st_ac, 1e-12);
            const bool acoustic = std::abs(mode - std::round(mode)) < 0.05 && mode > 0.8;
            ImGui::Text("strongest peak St %.3f   (period %.0f steps)%s", st_peak,
                        1.0 / sp.peak_freq, acoustic ? "  = acoustic mode" : "");
        } else {
            ImGui::TextDisabled("no peak: the signal is steady");
        }
        ImGui::SetItemTooltip(
            "St = f h / U with h = %.0f cells, the body's height across the flow.\n"
            "Window: %.0f steps of the developed flow. Dashed lines: the tunnel's\n"
            "transverse acoustic modes (sound between the side walls, St %.2f apart).",
            a.l_ref, a.window_steps, st_ac);
        std::vector<double> stx, amp;
        for (std::size_t k = 1; k < sp.freq.size(); ++k) {
            const double s = sp.freq[k] * st_scale;
            if (s > std::max(2.0, 3.0 * st_peak))
                break;
            stx.push_back(s);
            amp.push_back(sp.amp[k]);
        }
        const auto [p0, p1] = plot_area(plot_h);
        std::vector<double> modes;
        for (int m = 1; m <= 8; ++m)
            modes.push_back(m * st_ac);
        plot_xy(ImGui::GetWindowDrawList(), p0, p1, {{&stx, &amp, IM_COL32(255, 220, 120, 255)}},
                "St", "amplitude spectrum", true,
                !probe && a.st_shedding > 0.0 ? a.st_shedding : st_peak, modes);
    }
    // the signal's recent history
    std::vector<double> ht = a.hist_t, hv;
    if (probe)
        hv.assign(a.hist_probe[spec_source_ - 1][spec_comp_].begin(),
                  a.hist_probe[spec_source_ - 1][spec_comp_].end());
    else
        hv = a.hist_cl;
    if (!ht.empty()) {
        const ImU32 col = probe ? ImGui::ColorConvertFloat4ToU32(
                                      {render::Tracers::kProbeColours[spec_source_ - 1][0],
                                       render::Tracers::kProbeColours[spec_source_ - 1][1],
                                       render::Tracers::kProbeColours[spec_source_ - 1][2], 1.0f})
                                : IM_COL32(69, 171, 255, 255);
        const auto [p0, p1] = plot_area(0.8f * plot_h);
        plot_xy(ImGui::GetWindowDrawList(), p0, p1, {{&ht, &hv, col}}, "step", "history");
    }
    end_panel();
}

// The window title names the model and the regime (and the phase), so a
// screenshot of the whole desktop says what it shows.
void App::update_title(const TunnelStatus& st) {
    std::string t = "wind-oa -- " + menu_[model_index_].label + " -- " +
                    (st.transonic ? "transonic (Euler)" : "subsonic (LBM)");
    if (st.paused)
        t += " -- PAUSED";
    if (t == title_)
        return;
    title_ = t;
    std::wstring w(t.begin(), t.end()); // ASCII labels: a plain widening
    SetWindowTextW(window_->hwnd(), w.c_str());
}

void App::help_window() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        {vp->WorkPos.x + 0.5f * vp->WorkSize.x, vp->WorkPos.y + 0.5f * vp->WorkSize.y},
        ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::Begin("Help (F1)", &show_help_,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking);
    ImGui::SeparatorText("Camera");
    ImGui::BulletText("Right drag: orbit.  Middle drag: pan.  Wheel or Q / E: zoom.");
    ImGui::BulletText("W A S D: pan the focus point.  F: focus on the model.");
    ImGui::SeparatorText("Run");
    ImGui::BulletText("Space: pause / resume.  H: hide the panels.  P: screenshot.  Esc: quit.");
    ImGui::BulletText("Ctrl + click a slider to type an exact value.");
    ImGui::BulletText("Panels move, fold and dock; View > reset layout puts them back.");
    ImGui::SeparatorText("Reading it");
    ImGui::BulletText("DEVELOPING: the wake is still forming -- wait for SETTLED before");
    ImGui::Text("    reading the numbers (2 - 4.5 flow-throughs: physics, not slowness).");
    ImGui::BulletText("Re_sim ~10^3, not a real car's ~10^6: trust comparisons (Compare");
    ImGui::Text("    panel) and flow topology; treat absolute Cd as qualitative.");
    ImGui::BulletText("Speed haze shows DEVIATION from the freestream: blue slower, red faster.");
    ImGui::BulletText("Transonic: inviscid Euler, Mach 0.3-1.6 -- try the schlieren field.");
    ImGui::SeparatorText("Measuring it");
    ImGui::BulletText("Analysis panel: average the flow once SETTLED, then the mean-speed and");
    ImGui::Text("    turbulence fields, the reversed-flow shells and the wake survey work.");
    ImGui::BulletText("Probes record the flow at points; the spectrum gives the shedding");
    ImGui::Text("    frequency as a Strouhal number (St = f h / U).");
    ImGui::BulletText("Surface paint: Cp, near-wall speed, reversed flow or oil-flow streaks.");
    ImGui::SeparatorText("More");
    ImGui::TextDisabled("docs/GUIDE.md (operating it), docs/MODEL.md (how it works),");
    ImGui::TextDisabled("docs/REFERENCE.md (every control and default)");
    ImGui::End();
}

// Colour legends, top centre: one per colour-coded layer that is on.
void App::legends(const TunnelStatus& st) {
    struct Entry {
        const std::vector<Stop>* stops;
        const char* title;
        const char* ends;
    };
    std::vector<Entry> list;
    const bool field_shown = rs_.haze || slice_mode_ != 0;
    const int f = static_cast<int>(rs_.field);
    if (field_shown) {
        static const char* titles[] = {"speed vs freestream",   "Cp colour scale",
                                       "|vorticity| / U",       "streamwise vorticity",
                                       "local Mach number",     "schlieren  |grad rho|",
                                       "mean speed (averaged)", "turbulence intensity"};
        static const char* ends[] = {"blue slower .. red faster",
                                     "blue suction .. red stagnation",
                                     "white weak .. red strong",
                                     "blue / orange = opposite spin",
                                     "blue subsonic, white M = 1, red super",
                                     "bright = shocks and expansions",
                                     "blue slower .. red faster (mean)",
                                     "0 .. 20 %% of U (rms of the fluctuations)"};
        const std::vector<Stop>* s = f == 3   ? &kVort
                                     : f == 5 ? &kGrey
                                     : f == 7 ? &kDye
                                              : &kCoolwarm;
        list.push_back({s, titles[f], ends[f]});
    }
    const bool surface_shown = rs_.surface != render::Surface::Hidden;
    if (surface_shown && paint_mode_ == 1 && !(field_shown && f == 1))
        list.push_back({&kCoolwarm, "surface Cp", "blue suction .. red stagnation"});
    else if (surface_shown && paint_mode_ == 2)
        list.push_back({&kDye, "surface: near-wall speed", "still .. 1.2 U (skin friction)"});
    else if (surface_shown && paint_mode_ == 3)
        list.push_back({&kCoolwarm, "surface: near-wall flow", "blue reversed .. red forward"});
    if (!st.transonic && rs_.recirculation && st.avg_samples > 0)
        list.push_back({&kRecirc, "mean reversed flow", "time-averaged u_x < 0"});
    if (!st.transonic && show_dye_ && st.dye_on && rs_.dye_by_speed)
        list.push_back({&kDye, "dye smoke: local speed", "violet still .. yellow 1.5 U"});
    if (!st.transonic && rs_.vortex_cores)
        list.push_back({&kQcore, "vortex cores (Q criterion)", "green at threshold .. amber 4x"});

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float w = 270.0f * ImGui::GetStyle().FontScaleDpi;
    float y = vp->WorkPos.y + 6.0f;
    for (std::size_t k = 0; k < list.size() && k < 4; ++k) {
        ImGui::SetNextWindowPos({vp->WorkPos.x + 0.5f * (vp->WorkSize.x - w), y});
        ImGui::SetNextWindowSize({w, 0});
        ImGui::SetNextWindowBgAlpha(0.72f);
        char id[48];
        std::snprintf(id, sizeof(id), "%s###legend%zu", list[k].title, k);
        ImGui::Begin(id, nullptr,
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing |
                         ImGuiWindowFlags_NoNav);
        ImGui::TextUnformatted(list[k].ends);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float bw = ImGui::GetContentRegionAvail().x,
                    bh = 10.0f * ImGui::GetStyle().FontScaleDpi;
        const int n = 48;
        for (int i = 0; i < n; ++i) {
            const float t0 = float(i) / n, t1 = float(i + 1) / n;
            dl->AddRectFilledMultiColor({p.x + bw * t0, p.y}, {p.x + bw * t1, p.y + bh},
                                        ramp(*list[k].stops, t0), ramp(*list[k].stops, t1),
                                        ramp(*list[k].stops, t1), ramp(*list[k].stops, t0));
        }
        ImGui::Dummy({bw, bh});
        y += ImGui::GetWindowHeight() + 4.0f;
        ImGui::End();
    }
}

// Cd / Cl / Cm along the bottom (expand-only scales; refit once settled).
void App::plot_strip(const TunnelStatus& st) {
    if (st.steps != plotted_steps_ && st.steps > 0 && !st.paused) {
        if (st.steps < plotted_steps_) { // a reset: new history
            cd_.clear();
            cl_.clear();
            cm_.clear();
        }
        cd_.append(st.cd);
        cl_.append(st.cl);
        cm_.append(st.cm);
        plotted_steps_ = st.steps;
        if (st.settled && !was_settled_) { // scaled to the transient: refocus
            cd_.refit_recent(0.25);
            cl_.refit_recent(0.25);
            cm_.refit_recent(0.25);
        }
        was_settled_ = st.settled;
    }
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float h = plot_strip_height();
    ImGui::SetNextWindowPos({vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - h});
    ImGui::SetNextWindowSize({vp->WorkSize.x, h});
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::Begin("##plots", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBringToFrontOnFocus);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    const float pw = vp->WorkSize.x / 3.0f, m = 8.0f;
    const TimeSeries* s[3] = {&cd_, &cl_, &cm_};
    for (int k = 0; k < 3; ++k)
        s[k]->draw(dl, {o.x + k * pw + m, o.y + m}, {o.x + (k + 1) * pw - m, o.y + h - m});
    ImGui::End();
}

} // namespace windoa::app
