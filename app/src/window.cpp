#include "window.hpp"

#include <algorithm>
#include <stdexcept>

#include "imgui_impl_win32.h"

// Declared by the backend header only behind #if 0 (so it does not force
// <windows.h> on its users); it is declared here instead, as the backend's
// documentation advises.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
                                                             LPARAM lParam);

namespace windoa::app {

static const wchar_t* kClassName = L"windoa_window";

Window::Window(const wchar_t* title, int width, int height) {
    hinst_ = GetModuleHandleW(nullptr);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = &Window::proc;
    wc.hInstance = hinst_;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = kClassName;
    RegisterClassExW(&wc);

    // No size given: 80 % of the primary screen's work area, centred on it
    // (in pixels: the process is DPI-aware, so a fixed size would be small on
    // a large screen and large on a small one).
    RECT work{0, 0, 1920, 1080};
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0);
    const int work_w = work.right - work.left, work_h = work.bottom - work.top;
    if (width <= 0 || height <= 0) {
        width = std::max(640, int(0.8 * work_w));
        height = std::max(360, int(0.8 * work_h));
    }
    // Size the client area (what Vulkan draws into), not the outer frame.
    RECT r{0, 0, width, height};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    const int outer_w = r.right - r.left, outer_h = r.bottom - r.top;
    const int x = work.left + std::max(0, (work_w - outer_w) / 2);
    const int y = work.top + std::max(0, (work_h - outer_h) / 2);
    // `this` rides along in lpParam so proc() can find the object.
    hwnd_ = CreateWindowExW(0, kClassName, title, WS_OVERLAPPEDWINDOW, x, y, outer_w, outer_h,
                            nullptr, nullptr, hinst_, this);
    if (!hwnd_)
        throw std::runtime_error("CreateWindowExW failed");
    ShowWindow(hwnd_, SW_SHOWDEFAULT);
    UpdateWindow(hwnd_);
}

Window::~Window() {
    if (hwnd_)
        DestroyWindow(hwnd_);
    UnregisterClassW(kClassName, hinst_);
}

bool Window::pump() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        if (msg.message == WM_QUIT)
            closed_ = true;
    }
    return !closed_;
}

void Window::client_size(std::uint32_t& w, std::uint32_t& h) const {
    RECT r{};
    GetClientRect(hwnd_, &r);
    w = static_cast<std::uint32_t>(r.right - r.left);
    h = static_cast<std::uint32_t>(r.bottom - r.top);
}

bool Window::take_resized() {
    const bool r = resized_;
    resized_ = false;
    return r;
}

LRESULT CALLBACK Window::proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
        return 1;

    // Win32 idiom: stash the object pointer in the window's user data on
    // WM_NCCREATE (the first message), then fetch it for every later one.
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    }
    auto* self = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg) {
    case WM_SIZE:
        if (self)
            self->resized_ = true;
        return 0;
    case WM_DPICHANGED: { // moved to a monitor of another scale: take the size it suggests
        const RECT* r = reinterpret_cast<const RECT*>(lp);
        SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_SYSCOMMAND:
        if ((wp & 0xfff0) == SC_KEYMENU)
            return 0; // Alt would freeze the loop in a menu
        break;
    case WM_CLOSE:
        if (self)
            self->closed_ = true;
        return 0;
    case WM_DESTROY:
        if (self)
            self->hwnd_ = nullptr;
        return 0;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

} // namespace windoa::app
