// A plain Win32 window: create, pump messages, report size / close. The
// message procedure forwards input to Dear ImGui first.
#pragma once

#include <cstdint>

#include <windows.h>

namespace windoa::app {

class Window {
  public:
    Window(const wchar_t* title, int width, int height);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    HWND hwnd() const { return hwnd_; }
    HINSTANCE hinstance() const { return hinst_; }

    // Drain the message queue; false once the window has been closed.
    bool pump();
    // Client-area size in pixels (0 x 0 while minimised).
    void client_size(std::uint32_t& w, std::uint32_t& h) const;
    // True once after each resize (the swapchain must be rebuilt).
    bool take_resized();

  private:
    static LRESULT CALLBACK proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

    HINSTANCE hinst_ = nullptr;
    HWND hwnd_ = nullptr;
    bool closed_ = false;
    bool resized_ = false;
};

} // namespace windoa::app
