# The interactive wind tunnel: Win32 window + Vulkan swapchain + Dear ImGui
# on top of the headless engine. The only target that sees a window.
add_executable(windoa_app
    src/main.cpp
    src/app.cpp
    src/overlays.cpp
    src/panels.cpp
    src/play.cpp
    src/sim.cpp
    src/window.cpp
    src/swapchain.cpp
)
target_link_libraries(windoa_app PRIVATE windoa_render windoa_engine imgui)
# WIN32_LEAN_AND_MEAN / NOMINMAX: keep <windows.h> from defining min / max
# macros (they break std::min) and dragging in winsock et al. UNICODE: the
# Win32 macros (IDC_ARROW, ...) resolve to the wide-char API the app calls.
target_compile_definitions(windoa_app PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE
                           VK_USE_PLATFORM_WIN32_KHR)
