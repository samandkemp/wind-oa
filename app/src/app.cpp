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
#include "imgui_internal.h" // settings handlers, FindWindowByName

#include <vulkan/vulkan_win32.h>

#include "png.hpp"
#include "ui_common.hpp"
#include "windoa/airspeed.hpp"
#include "windoa/voxeliser.hpp"

namespace windoa::app {

namespace {

constexpr float kRevoxSettle = 0.5f; // s after the last placement edit
// Frames at most this often, with or without vsync: every frame's rendering
// is GPU time the solver does not get, and beyond about 90 a second the view
// gains little. Uncapped (--no-vsync), the window drew ~900 frames a second
// and starved the solver's queue (3,040 -> 1,280 MLUPS); a 144 Hz display
// with vsync would draw 144.
constexpr double kMaxFps = 90.0;

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

} // namespace

// -- construction ----------------------------------------------------------------

App::App(const Options& o) : opt_(o), ts_(tunnel_preset(o.preset)) {
    ts_.storage_f16 = o.f16;
    ImGui_ImplWin32_EnableDpiAwareness();
    // Scripted runs keep a fixed size, so their pictures repeat on any screen.
    const std::array<int, 2> win = o.window       ? *o.window
                                   : o.scripted() ? std::array<int, 2>{1600, 900}
                                                  : std::array<int, 2>{0, 0};
    window_ = std::make_unique<Window>(L"wind-oa -- LBM wind tunnel", win[0], win[1]);

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
    static_assert(SimWorker::kSlots == render::kSlots, "one renderer source set per snapshot slot");
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
    // the frame cap's timer (Windows 10 1803 or later; else a plain one)
    frame_timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
    if (!frame_timer_)
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
    rs_.surface = render::Surface::Mesh;
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
    if (has("smooth"))
        rs_.surface = render::Surface::Smooth;
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
    if (has("sequential"))
        rs_.palette = 1;
    if (has("greyscale"))
        rs_.palette = 3;
    if (has("logscale"))
        rs_.log_scale = true;
    // Scripted runs are for repeatable pictures: still textures unless asked.
    animate_textures_ = opt_.max_frames <= 0 || has("animate");
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
        sim_->post("placement", [p](Tunnel& t) { t.set_placement(p); });
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
        sim_->post("mach", [m](Tunnel& t) { t.set_mach(m); });
    }
    // The smoke wand and dye nozzles go to the inlet before any stepping, so
    // a warm-up already carries smoke and dye from there.
    sim_->flush();
    update_rake(sim_->status());
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
    if (has("paused")) { // after the warm-up: every frame shows the same flow
        sim_->post([](Tunnel& t) { t.set_paused(true); });
        sim_->flush();
    }
    if (o.look > 0) // after the warm-up: a look's slice goes through the placed model
        apply_look(o.look, sim_->status());
    if (!o.record.empty()) {
        record_every_ = o.record_every;
        start_recording(o.record);
    }
    if (o.zoom > 0.0f) { // scripted close-ups
        camera_.target = sim_->status().placed_centre;
        camera_.distance *= o.zoom;
    }
    if (o.view) { // scripted viewpoint
        camera_.azimuth = (*o.view)[0] * 3.14159265f / 180.0f;
        camera_.elevation = (*o.view)[1] * 3.14159265f / 180.0f;
    }
    // An interactive run opens on the model, three-quarters from upstream,
    // gliding in from the whole tunnel ("tunnel" on the quick bar goes back);
    // a scripted one keeps the tunnel view its --zoom and --view refer to.
    if (!o.scripted() && !o.view && o.zoom <= 0.0f)
        view_preset(3, sim_->status());
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
    if (recording_)
        stop_recording();
    { // the writer drains its queue, then stops
        std::lock_guard<std::mutex> lock(png_mu_);
        png_stop_ = true;
    }
    png_cv_.notify_all();
    if (png_writer_.joinable())
        png_writer_.join();
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
    pick_buf_.reset();
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
    // The app's preferences ride in the same file, as a [windoa][prefs] entry
    // (a settings handler: Dear ImGui's hook for an application's own data).
    // Scripted runs ignore them, so their pictures do not depend on them.
    ImGuiSettingsHandler prefs;
    prefs.TypeName = "windoa";
    prefs.TypeHash = ImHashStr("windoa");
    prefs.UserData = this;
    prefs.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler* h, const char*) -> void* {
        return h->UserData;
    };
    prefs.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler*, void* app, const char* line) {
        static_cast<App*>(app)->read_pref(line);
    };
    prefs.WriteAllFn = [](ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* out) {
        static_cast<const App*>(h->UserData)->write_prefs(*out);
    };
    ImGui::AddSettingsHandler(&prefs);
    ImGui::LoadIniSettingsFromDisk(io.IniFilename);
    if (!opt_.scripted()) {
        ui_scale_ = saved_prefs_.ui_scale;
        ui_font_ = saved_prefs_.font;
        speed_unit_ = saved_prefs_.speed_unit;
        render_scale_ = saved_prefs_.render_scale;
    }
    if (opt_.ui_scale > 0.0f)
        ui_scale_ = std::clamp(opt_.ui_scale, kUiScaleMin, kUiScaleMax);

    // Fonts: the system's UI font (Segoe UI, part of Windows: nothing to
    // download) and Dear ImGui's built-in scalable font, the fallback. Both
    // are rasterised at whatever size is asked for (ImGui 1.92 dynamic
    // fonts), so a scale change stays sharp.
    fonts_[1] = io.Fonts->AddFontDefaultVector();
    wchar_t windir[MAX_PATH] = {};
    const UINT n = GetWindowsDirectoryW(windir, MAX_PATH);
    const std::filesystem::path segoe =
        std::filesystem::path(std::wstring(windir, n)) / "Fonts" / "segoeui.ttf";
    std::error_code ec;
    if (n > 0 && std::filesystem::is_regular_file(segoe, ec)) {
        const std::u8string path = segoe.u8string();
        fonts_[0] = io.Fonts->AddFontFromFileTTF(reinterpret_cast<const char*>(path.c_str()));
    }
    apply_style();
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

// The style for the monitor's DPI x the user's scale, rebuilt from the base
// (ScaleAllSizes multiplies, so scaling a scaled style would compound).
void App::apply_style() {
    const float dpi = window_ ? ImGui_ImplWin32_GetDpiScaleForHwnd(window_->hwnd()) : 1.0f;
    if (dpi == style_dpi_ && ui_scale_ == style_scale_ && ui_font_ == style_font_)
        return;
    const float ratio = style_dpi_ > 0.0f ? dpi * ui_scale_ / (style_dpi_ * style_scale_) : 1.0f;
    ImGuiStyle st;
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 4.0f;
    st.FrameRounding = 3.0f;
    st.Colors[ImGuiCol_WindowBg].w = 0.86f;
    st.ScaleAllSizes(dpi * ui_scale_);
    const bool system_font = ui_font_ == 0 && fonts_[0];
    st.FontSizeBase = system_font ? 17.0f : 15.0f; // ~10 pt Segoe UI at 100 %
    st.FontScaleMain = ui_scale_;
    st.FontScaleDpi = dpi;
    ImGui::GetStyle() = st;
    ImGui::GetIO().FontDefault = system_font ? fonts_[0] : fonts_[1];
    // Undocked panels keep their place and grow or shrink with the text, so
    // a larger scale does not leave their contents clipped.
    if (ratio != 1.0f)
        for (const char* name : kPanelNames)
            if (const ImGuiWindow* w = ImGui::FindWindowByName(name); w && !w->DockIsActive)
                ImGui::SetWindowSize(name, {w->SizeFull.x * ratio, w->SizeFull.y * ratio});
    style_dpi_ = dpi;
    style_scale_ = ui_scale_;
    style_font_ = ui_font_;
}

void App::set_ui_scale(float s) {
    const float v = std::clamp(std::round(s * 20.0f) / 20.0f, kUiScaleMin, kUiScaleMax);
    if (v == ui_scale_)
        return;
    ui_scale_ = v;
    ImGui::MarkIniSettingsDirty();
    char msg[48];
    std::snprintf(msg, sizeof(msg), "UI scale %.0f %%", 100.0f * v);
    toast_ = msg;
    toast_t_ = Clock::now();
}

// Parsed into saved_prefs_ whatever the run; applied by init_imgui() unless
// scripted, and written back unchanged by a scripted run.
void App::read_pref(const char* line) {
    float f = 0.0f;
    int i = 0;
    if (sscanf_s(line, "UiScale=%f", &f) == 1)
        saved_prefs_.ui_scale = std::clamp(f, kUiScaleMin, kUiScaleMax);
    else if (sscanf_s(line, "Font=%d", &i) == 1)
        saved_prefs_.font = std::clamp(i, 0, 1);
    else if (sscanf_s(line, "SpeedUnit=%d", &i) == 1)
        saved_prefs_.speed_unit = std::clamp(i, 0, 2);
    else if (sscanf_s(line, "RenderScale=%f", &f) == 1)
        saved_prefs_.render_scale = std::clamp(f, 0.25f, 1.0f);
}

void App::write_prefs(ImGuiTextBuffer& out) const {
    const Prefs p =
        opt_.scripted() ? saved_prefs_ : Prefs{ui_scale_, ui_font_, speed_unit_, render_scale_};
    out.appendf("[windoa][prefs]\nUiScale=%.3f\nFont=%d\nSpeedUnit=%d\nRenderScale=%.3f\n\n",
                p.ui_scale, p.font, p.speed_unit, p.render_scale);
}

// -- model -------------------------------------------------------------------------------

void App::choose_model(std::size_t index) {
    const MenuItem& it = menu_[index];
    Model m;
    try {
        m = it.entry ? model_from_catalogue(*it.entry) : model_from_stl(it.stl);
    } catch (const std::exception& e) {
        toast_ = std::string("could not load: ") + e.what();
        toast_t_ = Clock::now();
        return;
    }
    if (m.mesh.triangles() > Voxeliser::kMaxTriangles) {
        toast_ = "could not load: " + std::to_string(m.mesh.triangles()) +
                 " triangles (the voxeliser takes up to 1,048,576)";
        toast_t_ = Clock::now();
        return;
    }
    model_index_ = index; // loaded: the menu shows it
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
        sim_->post("rotors", [on, tsr](Tunnel& t) { t.set_rotors(on, tsr); });
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
            sim_->post("placement", [p](Tunnel& t) { t.set_placement(p); });
            placement_dirty_ = false;
        }
        update_rake(st);

        // -- UI --
        apply_style(); // a new monitor DPI or UI scale
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
        if (opt_.pick && frames == 10 && renderer_->width() > 0) { // a scripted Ctrl + click
            pick_request_ = std::array<std::uint32_t, 2>{
                std::uint32_t((*opt_.pick)[0] * float(renderer_->width())),
                std::uint32_t((*opt_.pick)[1] * float(renderer_->height()))};
            pick_mode_ = click_mode_;
        }
        const bool record_due = recording_ && record_tick_++ % record_every_ == 0;
        const bool shot = screenshot_pending_ || (!opt_.shot.empty() && last_frame) || record_due;
        record_frame(f, sf, st, shot);
        std::vector<Swapchain::Timeline> waits, signals;
        if (sf.slot >= 0)
            waits.push_back(
                {sim_->sim_timeline(), sf.value, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
        signals.push_back({sim_->render_timeline(), rv, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
        if (!swapchain_->end(f, waits, signals))
            swapchain_->rebuild(w, h);
        if (pick_recorded_)
            resolve_pick(st);
        if (record_due) { // hand the frame to the writer; drop it if the writer is behind
            PendingPng p;
            p.rgba = capture_rgba(p.w, p.h);
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%05d.png", record_count_);
            p.path = (std::filesystem::path(record_dir_) / name).string();
            std::lock_guard<std::mutex> lock(png_mu_);
            if (png_queue_.size() < 6) {
                png_queue_.push_back(std::move(p));
                ++record_count_;
                png_cv_.notify_one();
            } else {
                ++record_dropped_;
            }
        }
        if (screenshot_pending_ || (!opt_.shot.empty() && last_frame)) {
            write_screenshot(screenshot_pending_ ? std::string() : opt_.shot);
            screenshot_pending_ = false;
        }

        { // the frame cap: wait out the frame, then yield up to the deadline
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
    const render::VolumeRenderer::PassTimes& pt = renderer_->pass_times();
    std::printf("render passes, ms: field %.2f, cores %.2f, copies %.2f, shape %.2f, march %.2f, "
                "splats %.2f; tracers etc. %.2f\n",
                pt.prepare, pt.q, pt.copies, pt.mesh, pt.march, pt.splats,
                std::max(0.0f, float(render_gpu_ms_) - pt.total));
    return 0;
}

void App::update_rake(const TunnelStatus& st) {
    // The smoke wand (and the dye nozzles on it) sits at the inlet, so the
    // smoke runs the whole tunnel: centred on the model's y, z while it
    // follows the model, else where the y / z controls (or a click) put it.
    const float ny = float(ts_.ny), nz = float(ts_.nz);
    float half_y = rake_autofit_ ? st.ext_y_half * 1.1f : rake_h_frac_ * ny;
    float half_z = rake_autofit_ ? st.ext_z_half * 1.1f : rake_w_frac_ * nz;
    const float x = ts_.rake_x;
    float cy, cz;
    if (rake_track_) {
        cy = st.placed_centre[1];
        cz = st.placed_centre[2];
        rake_y_ = cy / ny;
        rake_z_ = cz / nz;
    } else {
        cy = rake_y_ * ny;
        cz = rake_z_ * nz;
    }
    const std::array<float, 5> r{x, cy, cz, half_y, half_z};
    if (r != last_rake_) {
        tracers_->set_rake(x, cy, cz, half_y, half_z);
        sim_->post("rake", [r](Tunnel& t) { t.set_dye_rake(r[0], r[1], r[2], r[3], r[4]); });
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

        if (sf.geometry_version != mesh_seen_) { // the placed triangles (true shape)
            const SimWorker::PlacedMesh pm = sim_->placed_mesh();
            if (pm.mesh && pm.version == sf.geometry_version) {
                renderer_->set_mesh(*pm.mesh);
                mesh_seen_ = pm.version;
            }
        }
        render::Settings s = rs_;
        s.slice_axis = slice_mode_ == 1 ? 2 : slice_mode_ == 2 ? 1 : slice_mode_ == 3 ? 0 : -1;
        s.lic_phase = animate_textures_ ? lic_phase_ : -1.0f;
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
        if (pick_request_) { // read the depth under the cursor once the frame is done
            if (!pick_buf_)
                pick_buf_ = std::make_unique<Buffer>(*ctx_, 16, MemoryUse::Readback);
            pick_px_ = *pick_request_;
            pick_view_ = camera_.view(fov_deg_);
            renderer_->record_pick(cmd, pick_px_[0], pick_px_[1], *pick_buf_);
            pick_recorded_ = true;
            pick_request_.reset();
        }
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
// The frame copied for a screenshot, as RGBA (it waits for the frame).
std::vector<std::uint8_t> App::capture_rgba(std::uint32_t& w, std::uint32_t& h) {
    w = h = 0;
    if (!shot_buf_)
        return {};
    vkQueueWaitIdle(ctx_->graphics_queue());
    w = shot_extent_.width;
    h = shot_extent_.height;
    std::vector<std::uint8_t> rgba(std::size_t(w) * h * 4);
    std::memcpy(rgba.data(), shot_buf_->data(), rgba.size());
    if (colour_format_ == VK_FORMAT_B8G8R8A8_UNORM || colour_format_ == VK_FORMAT_B8G8R8A8_SRGB)
        for (std::size_t i = 0; i < rgba.size(); i += 4)
            std::swap(rgba[i], rgba[i + 2]);
    for (std::size_t i = 3; i < rgba.size(); i += 4)
        rgba[i] = 255;
    return rgba;
}

void App::write_screenshot(std::string path) {
    std::uint32_t w = 0, h = 0;
    const std::vector<std::uint8_t> rgba = capture_rgba(w, h);
    if (rgba.empty())
        return;
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
    // Esc closes the help; otherwise it asks once and quits on a second press
    // within 2 s (one stray key no longer ends a long run).
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        if (show_help_) {
            show_help_ = false;
        } else if (Clock::now() - quit_armed_ < std::chrono::seconds(2)) {
            PostQuitMessage(0);
        } else {
            quit_armed_ = Clock::now();
            toast_ = "press Esc again to quit";
            toast_t_ = quit_armed_;
        }
    }
    if (io.KeyCtrl) { // the UI scale: Ctrl + = / - / 0, as in a browser
        if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd))
            set_ui_scale(ui_scale_ * 1.1f);
        if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract))
            set_ui_scale(ui_scale_ / 1.1f);
        if (ImGui::IsKeyPressed(ImGuiKey_0, false) || ImGui::IsKeyPressed(ImGuiKey_Keypad0, false))
            set_ui_scale(1.0f);
        if (!io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            request_pick(click_mode_); // click to place
        return;                        // Ctrl + a key is never one of the plain keys below
    }
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
        camera_.fly_to(st.placed_centre, camera_.distance, camera_.azimuth, camera_.elevation);
    for (int k = 0; k < kLooks; ++k) // 1 - 9: the looks
        if (ImGui::IsKeyPressed(static_cast<ImGuiKey>(ImGuiKey_1 + k), false))
            apply_look(k + 1, st);
    if (ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false))
        step_model(-1);
    if (ImGui::IsKeyPressed(ImGuiKey_RightBracket, false))
        step_model(+1);
    if (ImGui::IsKeyPressed(ImGuiKey_O, false))
        orbit_ = !orbit_;
    if (!io.WantCaptureMouse && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        request_pick(0); // a double click focuses the camera there
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        if (recording_)
            stop_recording();
        else
            start_recording("");
    }
}

// The render pixel under the cursor, read back after the next frame
// (resolve_pick); `mode` as click_mode_.
void App::request_pick(int mode) {
    if (renderer_->width() == 0)
        return;
    const ImGuiIO& io = ImGui::GetIO();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float fx = (io.MousePos.x - vp->Pos.x) / std::max(vp->Size.x, 1.0f);
    const float fy = (io.MousePos.y - vp->Pos.y) / std::max(vp->Size.y, 1.0f);
    if (fx < 0.0f || fx >= 1.0f || fy < 0.0f || fy >= 1.0f)
        return;
    pick_request_ = std::array<std::uint32_t, 2>{std::uint32_t(fx * float(renderer_->width())),
                                                 std::uint32_t(fy * float(renderer_->height()))};
    pick_mode_ = mode;
}

// -- panels ---------------------------------------------------------------------------------

void App::ui(const TunnelStatus& st) {
    update_title(st);
    if (st.analysis_version != analysis_.version)
        analysis_ = sim_->analysis();
    // Kept here, not in the View panel, so a look or a key works with the
    // panels hidden.
    if (!st.transonic && show_dye_ != st.dye_on) {
        const bool d = show_dye_;
        sim_->post("dye", [d](Tunnel& t) { t.set_dye(d); });
    }
    if (animate_textures_) // about one ripple every two seconds
        lic_phase_ = std::fmod(lic_phase_ + 0.5f * ImGui::GetIO().DeltaTime, 1.0f);
    camera_.orbit_rate = orbit_ ? 0.25f : 0.0f; // a turn in 25 s
    if (sweep_) {                               // eased to and fro between the two speeds
        sweep_t_ += ImGui::GetIO().DeltaTime;
        const float ph =
            0.5f - 0.5f * std::cos(6.2831853f * sweep_t_ / std::max(sweep_period_, 2.0f));
        if (st.transonic) {
            mach_command_ = std::clamp(sweep_lo_mach_ + (sweep_hi_mach_ - sweep_lo_mach_) * ph,
                                       ts_.mach_min, ts_.mach_max);
            const float m = mach_command_;
            sim_->post("mach", [m](Tunnel& t) { t.set_mach(m); });
        } else {
            const double v = sweep_lo_mps_ + (sweep_hi_mps_ - sweep_lo_mps_) * ph;
            u_command_ = std::clamp(float(mps_to_lattice(v)), 0.005f, ts_.u_max);
            const float u = u_command_;
            sim_->post("speed", [u](Tunnel& t) { t.set_speed(u); });
        }
    }
    if (show_plots_)
        plot_strip(st);
    quick_bar_height_ = 0.0f;
    if (show_ui_)
        quick_bar(st);
    legends(st);
    if (show_help_)
        help_window();
    toast_overlay();
    if (!show_ui_)
        return;
    const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
    relayout_from_ = layout_size_.x > 0.0f && (size.x != layout_size_.x || size.y != layout_size_.y)
                         ? layout_size_
                         : ImVec2(0.0f, 0.0f);
    panel_tunnel(st);
    panel_model(st);
    panel_compare(st);
    panel_view(st);
    panel_analysis(st);
    layout_size_ = size;
    reset_layout_ = want_reset_layout_; // the button, or a layout that does not fit
    want_reset_layout_ = false;
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

void App::step_model(int delta) {
    const int n = int(menu_.size());
    if (n == 0)
        return;
    choose_model(std::size_t((int(model_index_) + delta % n + n) % n));
}

// -- recording (F8) ------------------------------------------------------------------

// Every frame (or every N-th) as a numbered PNG in `dir` (default a fresh
// screenshots/rec_<time> folder). A background thread compresses and writes
// them, so the window keeps its pace; frames it cannot keep up with are
// dropped and counted, never queued without bound.
void App::start_recording(std::string dir) {
    if (dir.empty()) {
        char name[64];
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        localtime_s(&tm, &t);
        std::strftime(name, sizeof(name), "screenshots/rec_%Y%m%d_%H%M%S", &tm);
        dir = name;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        toast_ = "could not create " + dir;
        toast_t_ = Clock::now();
        return;
    }
    if (!png_writer_.joinable())
        png_writer_ = std::thread([this] { png_writer_loop(); });
    record_dir_ = dir;
    record_tick_ = record_count_ = record_dropped_ = 0;
    recording_ = true;
    toast_ = "recording to " + dir;
    toast_t_ = Clock::now();
}

void App::stop_recording() {
    if (!recording_)
        return;
    recording_ = false;
    char msg[160];
    std::snprintf(msg, sizeof(msg), "recorded %d frames to %s%s", record_count_,
                  record_dir_.c_str(), record_dropped_ > 0 ? " (some dropped)" : "");
    toast_ = msg;
    toast_t_ = Clock::now();
    std::printf("%s\n", msg);
}

void App::png_writer_loop() {
    for (;;) {
        PendingPng p;
        {
            std::unique_lock<std::mutex> lock(png_mu_);
            png_cv_.wait(lock, [&] { return png_stop_ || !png_queue_.empty(); });
            if (png_queue_.empty())
                return; // stopping, and nothing left to write
            p = std::move(png_queue_.front());
            png_queue_.pop_front();
        }
        if (!write_png(p.path, p.rgba.data(), p.w, p.h))
            std::fprintf(stderr, "could not write %s\n", p.path.c_str());
    }
}

// -- click to place (F10) --------------------------------------------------------------

// The point under a Ctrl + click (the model's surface or the slice): the
// camera's focus glides there, a probe is placed there (the four in turn),
// or the smoke wand moves there.
void App::resolve_pick(const TunnelStatus& st) {
    pick_recorded_ = false;
    vkQueueWaitIdle(ctx_->graphics_queue()); // the frame that copied the depth
    float t = 1e30f;
    std::memcpy(&t, pick_buf_->data(), sizeof(t));
    toast_t_ = Clock::now();
    if (!(t < 1e29f)) {
        toast_ = "nothing under the cursor to place on (click the model or a slice)";
        std::printf("pick: nothing under the cursor\n");
        return;
    }
    const std::array<float, 3> p = renderer_->point_at(pick_view_, pick_px_[0], pick_px_[1], t);
    char msg[96];
    if (pick_mode_ == 0) {
        camera_.fly_to(p, camera_.distance, camera_.azimuth, camera_.elevation);
        std::snprintf(msg, sizeof(msg), "focus at (%.0f, %.0f, %.0f)", p[0], p[1], p[2]);
    } else if (pick_mode_ == 1) {
        const int i = next_probe_;
        probe_pos_[std::size_t(i)] = {std::clamp(p[0], 0.0f, float(ts_.nx - 1)),
                                      std::clamp(p[1], 0.0f, float(ts_.ny - 1)),
                                      std::clamp(p[2], 0.0f, float(ts_.nz - 1))};
        n_probes_ = std::max(n_probes_, i + 1);
        next_probe_ = (i + 1) % kMaxProbes;
        post_probes();
        std::snprintf(msg, sizeof(msg), "probe %d at (%.0f, %.0f, %.0f)", i + 1, p[0], p[1], p[2]);
    } else { // the wand stays at the inlet, on the point's y, z
        show_smoke_ = true;
        rake_track_ = false;
        rake_y_ = std::clamp(p[1] / float(ts_.ny), 0.02f, 0.98f);
        rake_z_ = std::clamp(p[2] / float(ts_.nz), 0.02f, 0.98f);
        std::snprintf(msg, sizeof(msg), "smoke from the inlet at y %.0f, z %.0f", p[1], p[2]);
    }
    (void)st;
    toast_ = msg;
    std::printf("pick: %s (depth %.1f)\n", msg, t);
}

} // namespace windoa::app
