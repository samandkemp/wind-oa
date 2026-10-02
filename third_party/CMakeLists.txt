# Vendored third-party code (see each directory's VERSION.md). Built with
# the upstream sources unmodified and this project's warnings off: /W4
# findings in vendored code are noise that would never be fixed here.
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS /permissive- /utf-8)

# Dear ImGui (docking) + the Win32 and Vulkan backends. Used by app/ only --
# engine/ must never link it.
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_win32.cpp
    imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui SYSTEM PUBLIC imgui imgui/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
target_compile_definitions(imgui PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
