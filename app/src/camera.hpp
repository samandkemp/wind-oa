// Turntable camera about a target point, in lattice cells, driven by Dear
// ImGui's input state:
//   RMB drag     orbit (azimuth / elevation)
//   MMB drag     pan the target in the view plane
//   wheel        zoom
//   W A S D      pan;  Q / E  zoom out / in  (F, refocus, is the app's)
// Input is ignored while ImGui wants the mouse / keyboard (over a panel).
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

    void update(const ImGuiIO& io) {
        const float w = std::max(io.DisplaySize.x, 1.0f);
        if (!io.WantCaptureMouse) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                azimuth += io.MouseDelta.x / w * kOrbitRate;
                elevation = std::clamp(elevation + io.MouseDelta.y / w * kOrbitRate,
                                       radians(-85.0f), radians(85.0f));
            }
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
                const float k = distance / w; // ~1 pixel of drag = 1 pixel of motion
                pan(-io.MouseDelta.x * k, io.MouseDelta.y * k);
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

  private:
    static constexpr float kOrbitRate = 3.5f; // radians per full-window drag
    static float radians(float deg) { return deg * 3.14159265f / 180.0f; }

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
};

} // namespace windoa::app
