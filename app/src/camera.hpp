// Turntable camera about a target point, in lattice cells, driven by Dear
// ImGui's input state:
//   LMB or RMB drag          orbit (azimuth / elevation)
//   MMB drag, Shift + drag   pan the target in the view plane (Shift for a
//                            touchpad or a mouse without a middle button)
//   wheel                    zoom
// (Ctrl + click is the app's: click to place.)
//   W A S D      pan;  Q / E  zoom out / in  (F, refocus, is the app's)
// Input is ignored while ImGui wants the mouse / keyboard (over a panel).
// fly_to() glides to a view (the app's view presets); orbit_rate turns the
// camera about the vertical through the target (a turntable). Any manual
// move ends a glide.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "imgui.h"

#include "windoa/volume.hpp"

namespace windoa::app {

class OrbitCamera {
  public:
    // Frame a box of nx x ny x nz cells from upstream, above and to the side.
    void frame_box(int nx, int ny, int nz) {
        box_ = {float(nx), float(ny), float(nz)};
        target = {0.5f * nx, 0.5f * ny, 0.5f * nz};
        distance = 1.25f * float(std::max({nx, ny, nz}));
        azimuth = radians(125.0f); // upstream, +z side: flow runs left to right
        elevation = radians(22.0f);
    }

    // Glide to a view over `seconds` (eased at both ends); the azimuth takes
    // the shorter way round.
    void fly_to(const std::array<float, 3>& to_target, float to_distance, float to_azimuth,
                float to_elevation, float seconds = 0.45f) {
        from_ = {target[0], target[1], target[2], distance, azimuth, elevation};
        float daz = std::remainder(to_azimuth - azimuth, 2.0f * kPi);
        to_ = {to_target[0],  to_target[1],
               to_target[2],  to_distance,
               azimuth + daz, std::clamp(to_elevation, radians(-85.0f), radians(85.0f))};
        fly_t_ = 0.0f;
        fly_time_ = std::max(seconds, 1e-3f);
        flying_ = true;
    }
    bool flying() const { return flying_; }
    float orbit_rate = 0.0f; // radians per second about the vertical; 0 = still

    void update(const ImGuiIO& io) {
        const float w = std::max(io.DisplaySize.x, 1.0f);
        // A drag: LMB without Ctrl (a click to place) or RMB / MMB, moving.
        const bool lmb = ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyCtrl;
        const bool rmb = ImGui::IsMouseDown(ImGuiMouseButton_Right);
        const bool mmb = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
        const bool moved = io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f;
        const bool drag = !io.WantCaptureMouse && moved && (lmb || rmb || mmb);
        const bool panning = mmb || (io.KeyShift && (lmb || rmb));
        const bool manual = drag || (!io.WantCaptureMouse && io.MouseWheel != 0.0f) ||
                            (!io.WantCaptureKeyboard &&
                             (ImGui::IsKeyDown(ImGuiKey_W) || ImGui::IsKeyDown(ImGuiKey_A) ||
                              ImGui::IsKeyDown(ImGuiKey_S) || ImGui::IsKeyDown(ImGuiKey_D) ||
                              ImGui::IsKeyDown(ImGuiKey_Q) || ImGui::IsKeyDown(ImGuiKey_E)));
        if (manual)
            flying_ = false;
        if (flying_) {
            fly_t_ += io.DeltaTime;
            const float x = std::min(fly_t_ / fly_time_, 1.0f);
            const float e = x * x * (3.0f - 2.0f * x); // smoothstep
            auto mix = [&](int k) { return from_[k] + (to_[k] - from_[k]) * e; };
            target = {mix(0), mix(1), mix(2)};
            distance = mix(3);
            azimuth = mix(4);
            elevation = mix(5);
            if (x >= 1.0f)
                flying_ = false;
        } else if (orbit_rate != 0.0f) {
            azimuth += orbit_rate * io.DeltaTime;
        }
        if (!io.WantCaptureMouse) {
            if (drag && panning) {
                const float k = distance / w; // ~1 pixel of drag = 1 pixel of motion
                pan(-io.MouseDelta.x * k, io.MouseDelta.y * k);
            } else if (drag) {
                azimuth += io.MouseDelta.x / w * kOrbitRate;
                elevation = std::clamp(elevation + io.MouseDelta.y / w * kOrbitRate,
                                       radians(-85.0f), radians(85.0f));
            }
            if (io.MouseWheel != 0.0f)
                distance *= std::pow(0.9f, io.MouseWheel);
        }
        if (!io.WantCaptureKeyboard) {
            const float step = 0.8f * distance * io.DeltaTime;
            if (ImGui::IsKeyDown(ImGuiKey_A))
                pan(-step, 0.0f);
            if (ImGui::IsKeyDown(ImGuiKey_D))
                pan(step, 0.0f);
            if (ImGui::IsKeyDown(ImGuiKey_W))
                pan(0.0f, step);
            if (ImGui::IsKeyDown(ImGuiKey_S))
                pan(0.0f, -step);
            if (ImGui::IsKeyDown(ImGuiKey_Q))
                distance *= 1.0f + 1.5f * io.DeltaTime;
            if (ImGui::IsKeyDown(ImGuiKey_E))
                distance /= 1.0f + 1.5f * io.DeltaTime;
        }
        distance = std::max(distance, 2.0f);
    }

    // The start view: target (0.42, 0.17, 0.19) nx and distance 1.15 nx, seen
    // from downstream, above, +z.
    void frame_start(int nx, int ny, int nz) {
        box_ = {float(nx), float(ny), float(nz)};
        target = {0.42f * nx, 0.17f * nx, 0.19f * nx};
        distance = 1.15f * nx;
        azimuth = radians(35.0f);
        elevation = radians(18.0f);
    }

    render::View view(float fov_deg) const {
        render::View v;
        const std::array<float, 3> dir = direction();
        for (int k = 0; k < 3; ++k)
            v.eye[k] = target[k] + dir[k] * distance;
        v.target = target;
        v.fov_deg = fov_deg;
        return v;
    }

    std::array<float, 3> target{};
    float distance = 100.0f;
    float azimuth = 0.0f, elevation = 0.0f; // radians

    static float radians(float deg) { return deg * kPi / 180.0f; }

  private:
    static constexpr float kPi = 3.14159265f;
    static constexpr float kOrbitRate = 3.5f; // radians per full-window drag

    std::array<float, 3> direction() const { // target -> eye, unit
        const float ce = std::cos(elevation);
        return {ce * std::cos(azimuth), std::sin(elevation), ce * std::sin(azimuth)};
    }

    // Move the target by (dx right, dy up) in the view plane.
    void pan(float dx, float dy) {
        const std::array<float, 3> d = direction();
        const std::array<float, 3> fwd{-d[0], -d[1], -d[2]};
        // right = fwd x world_up (0, 1, 0); up = right x fwd
        std::array<float, 3> right{-fwd[2], 0.0f, fwd[0]};
        const float rn = std::max(std::sqrt(right[0] * right[0] + right[2] * right[2]), 1e-9f);
        right = {right[0] / rn, 0.0f, right[2] / rn};
        const std::array<float, 3> up{right[1] * fwd[2] - right[2] * fwd[1],
                                      right[2] * fwd[0] - right[0] * fwd[2],
                                      right[0] * fwd[1] - right[1] * fwd[0]};
        for (int k = 0; k < 3; ++k)
            target[k] += right[k] * dx + up[k] * dy;
    }

    std::array<float, 3> box_{64.0f, 64.0f, 64.0f};
    // a glide: target xyz, distance, azimuth, elevation at its ends
    std::array<float, 6> from_{}, to_{};
    float fly_t_ = 0.0f, fly_time_ = 1.0f;
    bool flying_ = false;
};

} // namespace windoa::app
