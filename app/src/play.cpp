// Play (docs/REVIEW.md F1 - F4): the quick bar, the looks and the camera
// views.
#include "app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "imgui.h"
#include "imgui_internal.h" // BringWindowToDisplayFront
#include "ui_common.hpp"

namespace windoa::app {

// The quick bar, top centre: the controls a player reaches for -- the model,
// the wind in familiar units with presets, the looks and the camera views --
// in one strip (docs/REVIEW.md F1 - F4). The panels keep the full set.
void App::quick_bar(const TunnelStatus& st) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float w = std::min(std::max(0.48f * vp->WorkSize.x, em(40.0f)), vp->WorkSize.x);
    ImGui::SetNextWindowPos({vp->WorkPos.x + 0.5f * (vp->WorkSize.x - w), vp->WorkPos.y + 6.0f});
    ImGui::SetNextWindowSize({w, 0.0f});
    ImGui::SetNextWindowBgAlpha(0.80f);
    ImGui::Begin("##quick", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_AlwaysAutoResize);
    // Drawn over the panels: a panel left (or saved) across the top centre
    // must not cover the wind.
    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    const float sp = ImGui::GetStyle().ItemSpacing.x;

    // Row 1: the model and the wind.
    if (ImGui::ArrowButton("##prev", ImGuiDir_Left))
        step_model(-1);
    ImGui::SetItemTooltip("Previous model ([)");
    ImGui::SameLine(0.0f, 2.0f);
    const float name_w = 13.0f * ImGui::GetFontSize();
    const std::string& name = menu_[model_index_].label;
    ImGui::Button(name.c_str(), {name_w, 0.0f});
    ImGui::SetItemTooltip("%s", name.c_str());
    ImGui::SameLine(0.0f, 2.0f);
    if (ImGui::ArrowButton("##next", ImGuiDir_Right))
        step_model(+1);
    ImGui::SetItemTooltip("Next model (])");
    ImGui::SameLine(0.0f, 2.0f * sp);

    const double k = unit_per_mps(speed_unit_);
    const char* unit = kUnitNames[speed_unit_];
    char fmt[32];
    std::snprintf(fmt, sizeof(fmt), speed_unit_ == 2 ? "wind %%.1f %s" : "wind %%.0f %s", unit);
    const float presets_w = 3.0f * ImGui::CalcTextSize("M 1.50").x + 6.0f * sp +
                            ImGui::CalcTextSize("sweep").x + ImGui::GetFrameHeight() + 2.0f * sp;
    const float unit_w = ImGui::CalcTextSize("km/h").x + 2.0f * ImGui::GetFrameHeight();
    ImGui::SetNextItemWidth(
        std::max(ImGui::GetContentRegionAvail().x - unit_w - presets_w - 2.0f * sp, 80.0f));
    if (!st.transonic) {
        float v = float(lattice_to_mps(u_command_) * k);
        const float lo = float(lattice_to_mps(0.005) * k),
                    hi = float(lattice_to_mps(ts_.u_max) * k);
        if (ImGui::SliderFloat("##wind", &v, lo, hi, fmt)) {
            u_command_ = std::clamp(float(mps_to_lattice(v / k)), 0.005f, ts_.u_max);
            const float u = u_command_;
            sim_->post("speed", [u](Tunnel& t) { t.set_speed(u); });
        }
        ImGui::SetItemTooltip("The speed of sea-level air at the lattice's Mach number. The\n"
                              "Reynolds number stays near 10^3 whatever the speed (THEORY 1.2).");
    } else {
        float v = float(airspeed::metres_per_second(mach_command_) * k);
        const float lo = float(airspeed::metres_per_second(ts_.mach_min) * k),
                    hi = float(airspeed::metres_per_second(ts_.mach_max) * k);
        if (ImGui::SliderFloat("##wind", &v, lo, hi, fmt)) {
            mach_command_ = std::clamp(float(v / k / airspeed::kSeaLevelSoundSpeed), ts_.mach_min,
                                       ts_.mach_max);
            const float m = mach_command_;
            sim_->post("mach", [m](Tunnel& t) { t.set_mach(m); });
        }
        ImGui::SetItemTooltip("Mach %.2f in sea-level air.", mach_command_);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(unit_w);
    ImGui::Combo("##unit", &speed_unit_, kUnitNames, 3);
    // Presets: familiar road speeds, and the top of the range.
    const double road[3][2] = {{30.0, 70.0}, {50.0, 110.0}, {15.0, 30.0}};
    for (int i = 0; i < 3; ++i) {
        ImGui::SameLine();
        char label[32];
        ImGui::PushID(i);
        if (st.transonic) {
            const float m = i == 0 ? 0.8f : i == 1 ? 1.2f : ts_.mach_max;
            std::snprintf(label, sizeof(label), "M %.2g", m);
            if (ImGui::SmallButton(label)) {
                mach_command_ = m;
                sim_->post("mach", [m](Tunnel& t) { t.set_mach(m); });
            }
        } else {
            const float u = i < 2 ? std::clamp(float(mps_to_lattice(road[speed_unit_][i] / k)),
                                               0.005f, ts_.u_max)
                                  : ts_.u_max;
            if (i < 2)
                std::snprintf(label, sizeof(label), "%.0f", road[speed_unit_][i]);
            else
                std::snprintf(label, sizeof(label), "max");
            if (ImGui::SmallButton(label)) {
                u_command_ = u;
                sim_->post("speed", [u](Tunnel& t) { t.set_speed(u); });
            }
            if (i == 2)
                ImGui::SetItemTooltip("The fastest the lattice is trusted with: Mach %.2f.",
                                      airspeed::mach_from_lattice(ts_.u_max));
        }
        ImGui::PopID();
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("sweep", &sweep_))
        sweep_t_ = 0.0f;
    ImGui::SetItemTooltip("The wind eased to and fro between two speeds; right-click to set\n"
                          "them and the period.");
    if (ImGui::BeginPopupContextItem("##sweep")) {
        if (!st.transonic) {
            char ufmt[24];
            std::snprintf(ufmt, sizeof(ufmt), speed_unit_ == 2 ? "%%.1f %s" : "%%.0f %s", unit);
            const float a = float(lattice_to_mps(0.005) * k),
                        b = float(lattice_to_mps(ts_.u_max) * k);
            float lo = float(sweep_lo_mps_ * k), hi = float(sweep_hi_mps_ * k);
            if (ImGui::SliderFloat("from", &lo, a, b, ufmt))
                sweep_lo_mps_ = float(lo / k);
            if (ImGui::SliderFloat("to", &hi, a, b, ufmt))
                sweep_hi_mps_ = float(hi / k);
        } else {
            ImGui::SliderFloat("from Mach", &sweep_lo_mach_, ts_.mach_min, ts_.mach_max, "%.2f");
            ImGui::SliderFloat("to Mach", &sweep_hi_mach_, ts_.mach_min, ts_.mach_max, "%.2f");
        }
        ImGui::SliderFloat("period", &sweep_period_, 4.0f, 120.0f, "%.0f s");
        ImGui::EndPopup();
    }

    // Row 2: the looks.
    for (int i = 0; i < kLooks; ++i) {
        if (i > 0)
            ImGui::SameLine(0.0f, 3.0f);
        char label[32];
        std::snprintf(label, sizeof(label), "%d %s", i + 1, kLookNames[i]);
        const bool active = look_ == i + 1;
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        if (ImGui::SmallButton(label))
            apply_look(i + 1, st);
        if (active)
            ImGui::PopStyleColor();
        ImGui::SetItemTooltip("%s", kLookHints[i]);
    }

    // Row 3: the camera, and the flow's own controls.
    for (int i = 0; i < kViews; ++i) {
        if (i > 0)
            ImGui::SameLine(0.0f, 3.0f);
        if (ImGui::SmallButton(kViewNames[i]))
            view_preset(i, st);
    }
    ImGui::SameLine(0.0f, 2.0f * sp);
    static const char* kClick[] = {"focus", "probe", "smoke"};
    ImGui::SetNextItemWidth(ImGui::CalcTextSize("probe").x + 2.0f * ImGui::GetFrameHeight());
    ImGui::Combo("##click", &click_mode_, kClick, 3);
    ImGui::SetItemTooltip("Ctrl + click on the model or a slice: focus the camera there, place a\n"
                          "probe (the four in turn) or aim the smoke there (the wand stays at the\n"
                          "inlet and moves to that height and span).");
    ImGui::SameLine();
    ImGui::Checkbox("orbit (O)", &orbit_);
    ImGui::SameLine();
    ImGui::Checkbox("moving textures", &animate_textures_);
    ImGui::SetItemTooltip("Flow textures (slice LIC, oil flow) travel with the flow.");
    ImGui::SameLine();
    bool rec = recording_;
    if (ImGui::Checkbox("record (R)", &rec)) {
        if (rec)
            start_recording("");
        else
            stop_recording();
    }
    ImGui::SetItemTooltip("Every frame as a numbered PNG in screenshots/rec_<time>; H hides the\n"
                          "panels for clean frames.");
    if (recording_) {
        ImGui::SameLine();
        ImGui::TextColored({1.0f, 0.35f, 0.3f, 1.0f}, "REC %d", record_count_);
    }
    ImGui::SameLine(0.0f, 2.0f * sp);
    if (ImGui::SmallButton(st.paused ? "resume" : "pause")) {
        const bool p = !st.paused;
        sim_->post([p](Tunnel& t) { t.set_paused(p); });
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("reset flow")) {
        if (st.transonic)
            sim_->post([](Tunnel& t) { t.restart_flow(); });
        else
            sim_->post([](Tunnel& t) { t.reset_flow("flow reset"); });
    }
    quick_bar_height_ = ImGui::GetWindowHeight();
    ImGui::End();
}

// A look sets what is drawn, as a whole: it switches off what it does not use.
void App::apply_look(int look, const TunnelStatus& st) {
    if (look < 1 || look > kLooks)
        return;
    look_ = look;
    rs_.haze = false;
    rs_.haze_gain = 1.0f;
    rs_.haze_floor = -1.0f;
    rs_.surface = render::Surface::Mesh;
    rs_.slice_lic = false;
    rs_.vortex_cores = false;
    rs_.recirculation = false;
    rs_.field = render::Field::Speed;
    paint_mode_ = 1;
    slice_mode_ = 0;
    slice_arrows_ = false;
    show_smoke_ = false;
    show_dye_ = false;
    show_streamlines_ = false;
    // a slice through the model: vertical (x-y at its z) or horizontal (x-z at its y)
    auto slice_through = [&](int mode) {
        slice_mode_ = mode;
        rs_.slice_pos = mode == 1 ? std::clamp(st.placed_centre[2] / float(ts_.nz), 0.02f, 0.98f)
                                  : std::clamp(st.placed_centre[1] / float(ts_.ny), 0.02f, 0.98f);
    };
    switch (look) {
    case 1: // tunnel: the start look
        show_smoke_ = true;
        smoke_mode_ = 0;
        rs_.haze = true;
        break;
    case 2: // smoke
        show_smoke_ = true;
        smoke_mode_ = 0;
        paint_mode_ = 0;
        break;
    case 3: // pressure
        rs_.field = render::Field::Pressure;
        rs_.haze = true;
        slice_through(1);
        break;
    case 4: // vortices
        rs_.vortex_cores = true;
        rs_.field = render::Field::VortX;
        rs_.haze = true;
        paint_mode_ = 0;
        break;
    case 5: // texture
        slice_through(1);
        rs_.slice_lic = true;
        break;
    case 6: // oil
        paint_mode_ = 4;
        break;
    case 7: // dye
        show_dye_ = true;
        rs_.dye_by_speed = true;
        break;
    case 8: // wake
        rs_.field = render::Field::MeanSpeed;
        slice_through(2);
        rs_.slice_lic = true;
        rs_.recirculation = true;
        if (!averaging_) {
            averaging_ = true;
            sim_->post([](Tunnel& t) { t.set_averaging(true); });
        }
        break;
    case 9: // schlieren
        rs_.field = render::Field::Schlieren;
        slice_through(1);
        break;
    }
}

// The camera views: the model framed from the front (upstream), the side,
// above, three-quarters upstream and behind, or the whole tunnel (the
// scripted runs' start view), gliding there.
void App::view_preset(int view, const TunnelStatus& st) {
    if (view < 0 || view >= kViews)
        return;
    orbit_ = false;
    if (view == 5) {
        const float nx = float(ts_.nx);
        camera_.fly_to({0.42f * nx, 0.17f * nx, 0.19f * nx}, 1.15f * nx,
                       OrbitCamera::radians(35.0f), OrbitCamera::radians(18.0f));
        return;
    }
    const float L = std::max(placement_.length_cells, 16.0f);
    const float d = 0.95f * L / std::tan(OrbitCamera::radians(0.5f * fov_deg_));
    const float az[] = {180.0f, 90.0f, 90.0f, 135.0f, 30.0f};
    const float el[] = {6.0f, 4.0f, 85.0f, 22.0f, 16.0f};
    camera_.fly_to(st.placed_centre, d, OrbitCamera::radians(az[view]),
                   OrbitCamera::radians(el[view]));
}

} // namespace windoa::app
