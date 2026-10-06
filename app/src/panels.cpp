// The app's panels: Tunnel (and its transonic form), Model, Compare, View
// and Analysis, with the placement logic that keeps them on screen.
#include "app.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

#include "imgui.h"
#include "ui_common.hpp"

namespace windoa::app {

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
    // 26 font sizes wide (labels and sliders fit at any UI scale, and a
    // large window does not stretch them), at most 0.3 of the window; a
    // right-hand panel keeps its right edge at x + w.
    const float width = std::min(em(26.0f), 0.3f * vp->WorkSize.x);
    const float left = x > 0.5f ? vp->WorkPos.x + (x + w) * vp->WorkSize.x - width
                                : vp->WorkPos.x + x * vp->WorkSize.x;
    ImGui::SetNextWindowPos({left, vp->WorkPos.y + y * vp->WorkSize.y}, cond);
    ImGui::SetNextWindowSize({width, h * vp->WorkSize.y}, cond);
    if (reset_layout_)
        ImGui::SetNextWindowDockID(0, ImGuiCond_Always);
    const bool open = ImGui::Begin(name);
    if (relayout_from_.x > 0.0f && !reset_layout_ && !ImGui::IsWindowDocked()) {
        // The window was resized: a panel in the right (lower) half keeps its
        // distance to the right (bottom) edge.
        ImVec2 pos = {ImGui::GetWindowPos().x - vp->WorkPos.x,
                      ImGui::GetWindowPos().y - vp->WorkPos.y};
        const ImVec2 size = ImGui::GetWindowSize();
        if (pos.x + 0.5f * size.x > 0.5f * relayout_from_.x)
            pos.x += vp->WorkSize.x - relayout_from_.x;
        if (pos.y + 0.5f * size.y > 0.5f * relayout_from_.y)
            pos.y += vp->WorkSize.y - relayout_from_.y;
        // ... kept inside the free area when it fits (the plot strip scales
        // with the window, so a bottom panel can land on it)
        const float room_x = vp->WorkSize.x - size.x;
        const float room_y = vp->WorkSize.y - plot_strip_height() - size.y;
        if (room_x >= 0.0f)
            pos.x = std::clamp(pos.x, 0.0f, room_x);
        if (room_y >= 0.0f)
            pos.y = std::clamp(pos.y, 0.0f, room_y);
        ImGui::SetWindowPos({vp->WorkPos.x + pos.x, vp->WorkPos.y + pos.y});
    }
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
    worker_failure();
    ImGui::Text("ms: sim %.2f/step   render %.2f (GPU)   frame %.1f", sim_->ms_per_step(),
                render_gpu_ms_, frame_ms_);
    {
        const render::VolumeRenderer::PassTimes& pt = renderer_->pass_times();
        ImGui::SetItemTooltip(
            "GPU time per pass, ms: field %.2f, vortex cores %.2f, texture copies "
            "%.2f,\ntrue shape %.2f, ray march %.2f, splats %.2f; tracers and "
            "the rest %.2f.",
            pt.prepare, pt.q, pt.copies, pt.mesh, pt.march, pt.splats,
            std::max(0.0f, float(render_gpu_ms_) - pt.total));
    }
    ImGui::Separator();
    ImGui::Text("Re_sim %.0f   A_ref %.0f cells^2", st.re_sim, st.a_ref);
    if (st.full_scale_length_m > 0.0) {
        ImGui::TextDisabled("  full size (%.3g m) at this speed: Re %s", st.full_scale_length_m,
                            readable(st.full_scale_re).c_str());
        ImGui::SetItemTooltip(
            "The Reynolds number the real object would have at the speed shown, in\n"
            "sea-level air. The tunnel runs %.0f times lower, so absolute coefficients\n"
            "are qualitative and comparisons are what hold (THEORY 1.2, 4.5).",
            st.full_scale_re / std::max(st.re_sim, 1.0));
    }
    ImGui::Text("Cd %+.3f  Cl %+.3f  Cs %+.3f%s", st.cd, st.cl, st.cs,
                st.developing ? "   (settling)" : "");
    ImGui::Text("Cm_z %+.4f   max|u| %.3f", st.cm, st.max_speed);
    ImGui::Separator();
    // The slider reads in lattice units and, beside them, the Mach-matched
    // speed in sea-level air (airspeed.hpp; ImGui takes the label as a format).
    char speed_fmt[48];
    std::snprintf(speed_fmt, sizeof(speed_fmt), "%%.3f  = %.0f %s",
                  lattice_to_mps(double(u_command_)) * unit_per_mps(speed_unit_),
                  kUnitNames[speed_unit_]);
    if (ImGui::SliderFloat("flow speed", &u_command_, 0.005f, ts_.u_max, speed_fmt)) {
        const float u = u_command_;
        sim_->post("speed", [u](Tunnel& t) { t.set_speed(u); });
    }
    ImGui::SetItemTooltip(
        "Freestream speed in lattice units (Mach = u sqrt 3, kept under ~0.19).\n"
        "mph and m/s are the speed in sea-level air at that Mach number; the\n"
        "Reynolds number stays near 10^3 whatever the speed (see Re_sim).\n"
        "Slew-limited so the lattice never shocks; a change over 10 %% re-develops.");
    ImGui::Text("  u %.3f -> %.3f   Ma %.3f", st.u_applied, st.u_command, st.airspeed_mach);
    if (ImGui::SliderFloat("inlet turbulence %", &turb_pct_, 0.0f, 2.0f, "%.1f")) {
        const float t = turb_pct_;
        sim_->post("turbulence", [t](Tunnel& tn) { tn.set_turbulence(t); });
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
    end_panel();
}

// A failure on the worker thread, until it is old news; a stopped solver
// keeps its note until resumed.
void App::worker_failure() {
    const SimWorker::Failure f = sim_->failure();
    if (f.what.empty())
        return;
    if (!f.stopped && std::chrono::duration<float>(Clock::now() - f.when).count() > 20.0f)
        return;
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.4f, 1.0f));
    ImGui::TextWrapped("%s", f.what.c_str());
    ImGui::PopStyleColor();
    if (f.stopped && ImGui::Button("resume the solver"))
        sim_->resume();
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
    worker_failure();
    char mach_fmt[48];
    std::snprintf(mach_fmt, sizeof(mach_fmt), "%%.3f  = %.0f %s",
                  airspeed::metres_per_second(double(mach_command_)) * unit_per_mps(speed_unit_),
                  kUnitNames[speed_unit_]);
    if (ImGui::SliderFloat("Mach", &mach_command_, ts_.mach_min, ts_.mach_max, mach_fmt)) {
        const float m = mach_command_;
        sim_->post("mach", [m](Tunnel& t) { t.set_mach(m); });
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
        sim_->post("area", [m, a](Tunnel& t) { t.set_area_mode(static_cast<AreaMode>(m), a); });
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
            sim_->post("spin", [on, r](Tunnel& t) { t.set_spin(on, r); });
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
            sim_->post("rotors", [on, t](Tunnel& tn) { tn.set_rotors(on, t); });
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
            sim_->post("power", [on, t](Tunnel& tn) { tn.set_power(on, t); });
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
    if (ImGui::Button("append to results.csv"))
        append_result(st);
    ImGui::SetItemTooltip("The model, its placement, the wind and the coefficients now, as a row\n"
                          "of results/results.csv: a table to compare models, sizes and angles\n"
                          "in a spreadsheet. Wait for SETTLED first (GUIDE 7).");
    end_panel();
}

// One row per press; a header when the file is new. The same columns as
// tunnel_run --csv where they overlap, so the two tables can be joined.
void App::append_result(const TunnelStatus& st) {
    std::error_code ec;
    std::filesystem::create_directories("results", ec);
    const std::filesystem::path path = std::filesystem::path("results") / "results.csv";
    const bool fresh =
        !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
    std::ofstream out(path, std::ios::app);
    toast_t_ = Clock::now();
    if (!out) {
        toast_ = "could not write results/results.csv";
        return;
    }
    if (fresh)
        out << "time,model,preset,size_cells,aoa_deg,yaw_deg,roll_deg,ground,regime,speed_mps,"
               "speed_mph,re_sim,re_full_size,steps,settled,flow_throughs,cd,cl,cs,cm,"
               "a_ref_cells\n";
    char when[32];
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tm);
    static const char* kGround[] = {"air", "road", "fixed"};
    const double mps = st.transonic ? airspeed::metres_per_second(double(st.mach_applied))
                                    : airspeed::metres_per_second(st.airspeed_mach);
    char row[640];
    std::snprintf(row, sizeof(row),
                  "%s,\"%s\",%s,%.0f,%.2f,%.2f,%.2f,%s,%s,%.2f,%.1f,%.0f,%.4g,%lld,%d,%.2f,%.5f,"
                  "%.5f,%.5f,%.5f,%.1f\n",
                  when, menu_[model_index_].label.c_str(), opt_.preset.c_str(),
                  placement_.length_cells, placement_.aoa_deg, placement_.yaw_deg,
                  placement_.roll_deg, kGround[std::clamp(int(placement_.ground), 0, 2)],
                  st.transonic ? "transonic" : "subsonic", mps,
                  mps / airspeed::kMetresPerSecondPerMph, st.re_sim, st.full_scale_re,
                  static_cast<long long>(st.steps), st.settled ? 1 : 0, st.flow_throughs, st.cd,
                  st.cl, st.cs, st.cm, st.a_ref);
    out << row;
    toast_ =
        std::string("appended to results/results.csv") + (st.settled ? "" : " (not yet settled)");
}

void App::panel_view(const TunnelStatus& st) {
    begin_panel("View", 0.75f, 0.23f, 0.24f, 0.58f);
    static const char* kSurface[] = {"hidden", "voxels (the solver's cells)", "smoothed cells",
                                     "true shape (mesh)"};
    int surf = static_cast<int>(rs_.surface);
    if (ImGui::Combo("surface", &surf, kSurface, 4))
        rs_.surface = static_cast<render::Surface>(surf);
    ImGui::SetItemTooltip(
        "True shape: the model's own triangles, sharp at any zoom. Voxels: the cells the\n"
        "solver treats as solid -- what the flow actually meets (its walls then sit at\n"
        "their sub-cell positions). Smoothed: those cells, blurred. The paints read the\n"
        "same flow in every case (THEORY 10.10).");
    static const char* kPaint[] = {"none", "pressure (Cp)", "near-wall speed", "reversed flow",
                                   "oil flow"};
    ImGui::Combo("surface paint", &paint_mode_, kPaint, 5);
    if (paint_mode_ == 4)
        ImGui::Checkbox("  animate the streaks", &animate_textures_);
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
    ImGui::Combo("colour map", &rs_.palette, kPaletteNames, 4);
    ImGui::SliderFloat("colour range", &rs_.range, 0.1f, 5.0f, "x %.2f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("The field value that fills the colour scale, as a multiple of the\n"
                          "default: below 1 more contrast, above 1 a wider span. It moves the\n"
                          "haze's opacity with the colour.");
    ImGui::Checkbox("log scale", &rs_.log_scale);
    ImGui::SetItemTooltip("Opens up the weak end of the scale (vorticity, schlieren).");
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
        if (rs_.slice_lic)
            ImGui::Checkbox("  animate the texture", &animate_textures_);
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
        ImGui::Checkbox("smoke follows the model", &rake_track_);
        ImGui::SetItemTooltip(
            "The smoke wand and the dye nozzles sit at the inlet, so the smoke runs the\n"
            "whole tunnel. On: centred on the model's height and span. Off: set where on\n"
            "the inlet with smoke y / z below, or Ctrl + click a point (click mode smoke).");
        ImGui::Checkbox("fit smoke to model", &rake_autofit_);
        if (!rake_autofit_) {
            ImGui::SliderFloat("smoke height", &rake_h_frac_, 0.03f, 0.50f, "%.2f");
            ImGui::SliderFloat("smoke width", &rake_w_frac_, 0.03f, 0.50f, "%.2f");
        }
        if (!rake_track_) {
            ImGui::SliderFloat("smoke y (inlet)", &rake_y_, 0.02f, 0.98f, "%.2f");
            ImGui::SetItemTooltip("Height of the wand's centre on the inlet, as a fraction.");
            ImGui::SliderFloat("smoke z (inlet)", &rake_z_, 0.02f, 0.98f, "%.2f");
            ImGui::SetItemTooltip("Span-wise position of the wand's centre, as a fraction.");
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
    // The UI scale by buttons, not a slider: a slider would resize itself
    // under the cursor as it was dragged.
    ImGui::Text("UI scale %.0f %%", 100.0f * ui_scale_);
    ImGui::SameLine();
    if (ImGui::SmallButton(" - "))
        set_ui_scale(ui_scale_ / 1.1f);
    ImGui::SameLine();
    if (ImGui::SmallButton(" + "))
        set_ui_scale(ui_scale_ * 1.1f);
    ImGui::SameLine();
    if (ImGui::SmallButton("100 %"))
        set_ui_scale(1.0f);
    ImGui::SetItemTooltip("Ctrl + = / - / 0. Text and panels scale together, on top of the\n"
                          "monitor's own scale; remembered between runs.");
    if (fonts_[0]) {
        static const char* kFont[] = {"Segoe UI (system)", "built-in"};
        ImGui::Combo("UI font", &ui_font_, kFont, 2);
    }
    ImGui::TextDisabled("drag orbit, Shift+drag pan, wheel zoom, double-click focus");
    end_panel();
}

void App::post_probes() {
    const std::vector<std::array<float, 3>> p(probe_pos_.begin(), probe_pos_.begin() + n_probes_);
    sim_->post("probes", [p](Tunnel& t) { t.set_probes(p); });
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
    const float plot_h = em(8.5f);
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
        sim_->post("wake", [x](Tunnel& t) { t.set_wake_plane(x); });
    }
    if (!wake_auto_ && ImGui::SliderInt("survey plane x", &wake_x_, 1, ts_.nx - 2)) {
        const int x = wake_x_;
        sim_->post("wake", [x](Tunnel& t) { t.set_wake_plane(x); });
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

} // namespace windoa::app
