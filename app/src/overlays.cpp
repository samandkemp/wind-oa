// What is drawn over the view: the colour legends, the force plots along
// the bottom and the help window.
#include "app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "imgui.h"
#include "ui_common.hpp"

namespace windoa::app {

void App::help_window() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        {vp->WorkPos.x + 0.5f * vp->WorkSize.x, vp->WorkPos.y + 0.5f * vp->WorkSize.y},
        ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::Begin("Help (F1)", &show_help_,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDocking);
    ImGui::SeparatorText("Camera");
    ImGui::BulletText("Left or right drag: orbit.  Middle or Shift + drag: pan.  Wheel or Q / E:");
    ImGui::Text("    zoom.  Double-click the model or a slice: focus the camera there.");
    ImGui::BulletText("W A S D: pan the focus point.  F: focus on the model.");
    ImGui::SeparatorText("Play");
    ImGui::BulletText("The quick bar (top centre): the model, the wind in mph / km/h / m/s,");
    ImGui::Text("    the looks and the camera views.");
    ImGui::BulletText("1 - 9: the looks.  [ / ]: previous / next model.  O: orbit the camera.");
    ImGui::BulletText("R: record the frames as PNGs.  The sweep box eases the wind to and fro.");
    ImGui::BulletText("Ctrl + click the model or a slice: focus, a probe or the smoke there");
    ImGui::Text("    (the choice beside the camera views).");
    ImGui::SeparatorText("Run");
    ImGui::BulletText("Space: pause / resume.  H: hide the panels.  P: screenshot.");
    ImGui::BulletText("Esc twice: quit.  Ctrl + = / - / 0: the UI scale (View panel too).");
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

// The latest message (a screenshot saved, recording, the UI scale, Esc to
// quit), bottom centre above the plots, for six seconds; the quit prompt for
// the two its second press is waited for. With the panels hidden (clean
// frames) only the quit prompt shows.
void App::toast_overlay() {
    if (toast_.empty())
        return;
    const float age = std::chrono::duration<float>(Clock::now() - toast_t_).count();
    const bool quit_prompt = toast_t_ == quit_armed_;
    if (age > (quit_prompt ? 2.0f : 6.0f) || (!show_ui_ && !quit_prompt))
        return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos({vp->WorkPos.x + 0.5f * vp->WorkSize.x,
                             vp->WorkPos.y + vp->WorkSize.y - plot_strip_height() - em(0.8f)},
                            ImGuiCond_Always, {0.5f, 1.0f});
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::Begin("##toast", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextColored({1.0f, 0.8f, 0.4f, 1.0f}, "%s", toast_.c_str());
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
    std::string field_ends; // the field's legend text with its range note, while list lives
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
                                     "0 .. 20 % of U (rms of the fluctuations)"};
        // without colour words, for the View panel's other maps
        static const char* neutral[] = {"slower .. faster",
                                        "suction .. stagnation",
                                        "weak .. strong",
                                        "one sense of spin .. the other",
                                        "subsonic .. M = 1 .. supersonic",
                                        "weak .. strong gradients",
                                        "slower .. faster (mean)",
                                        "0 .. 20 % of U (rms of the fluctuations)"};
        const std::vector<Stop>* s = f == 3   ? &kVort
                                     : f == 5 ? &kGrey
                                     : f == 7 ? &kDye
                                              : &kCoolwarm;
        const char* e = ends[f];
        if (rs_.palette > 0) {
            s = rs_.palette == 1 ? &kSeq : rs_.palette == 2 ? &kCoolwarm : &kGrey;
            e = neutral[f];
        }
        if (rs_.range != 1.0f || rs_.log_scale) {
            char note[48];
            if (rs_.range == 1.0f)
                std::snprintf(note, sizeof(note), "  (log)");
            else
                std::snprintf(note, sizeof(note), "  (range x%.2g%s)", rs_.range,
                              rs_.log_scale ? ", log" : "");
            field_ends = std::string(e) + note;
            e = field_ends.c_str();
        }
        list.push_back({s, titles[f], e});
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
    const float w = em(21.0f);
    float y = vp->WorkPos.y + 6.0f + (quick_bar_height_ > 0.0f ? quick_bar_height_ + 4.0f : 0.0f);
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
        const float bw = ImGui::GetContentRegionAvail().x, bh = em(0.75f);
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
