// The simulation worker: runs the Tunnel on its own thread, on the engine's
// (async compute) queue, so the UI never waits for the solver -- the P3d
// goal ("the UI never blocks on GPU work").
//
// Hand-off to the renderer, without stalls either way:
//   - after each batch the worker copies the live fields (macro, flags when
//     the geometry changed, dye) into one of two snapshot slots and signals
//     its timeline semaphore; the slot it writes is never the one most
//     recently published;
//   - a frame takes the latest published slot, makes its submit wait on the
//     worker's timeline value (GPU-side), and signals the render timeline;
//   - before overwriting a slot the worker waits (host-side, on its own
//     thread) until every frame that read that slot has completed.
// Commands from the UI are closures run on the worker between batches, so
// the Tunnel is only ever touched by one thread.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "windoa/context.hpp"
#include "windoa/tunnel.hpp"

namespace windoa::app {

class SimWorker {
  public:
    static constexpr int kSlots = 2;

    struct Snapshot {
        std::unique_ptr<Buffer> macro; // vec4 per cell (u, rho)
        std::unique_ptr<Buffer> flags; // uint per cell
        std::unique_ptr<Buffer> dye;   // float per cell (zeros when dye is off)
        std::unique_ptr<Buffer> aux;   // float per cell: Euler rho (transonic)
        std::unique_ptr<Buffer> mean;  // vec4 per cell: time-averaged u, rho - 1
        std::unique_ptr<Buffer> m2;    // vec4 per cell: summed squared deviations
        std::uint64_t geometry_version = UINT64_MAX;
        int stats_samples = -1; // the averaging sample count the copies hold
    };

    // What a frame renders: a published slot and the value to wait for.
    struct Frame {
        int slot = -1;
        std::uint64_t value = 0;            // worker timeline value to wait on
        std::uint64_t geometry_version = 0; // flags version in the slot
        std::int64_t steps = 0;             // solver steps at the snapshot
        std::uint64_t flow_epoch = 0;
        bool dye = false;
        bool transonic = false;        // macro = (u, p), aux = rho
        float stats_inv_weight = 0.0f; // 1 / averaging weight (0: no averages yet)
    };

    SimWorker(Context& ctx, const TunnelSettings& s, const std::filesystem::path& cache_dir);
    ~SimWorker();
    SimWorker(const SimWorker&) = delete;
    SimWorker& operator=(const SimWorker&) = delete;

    // Run `cmd` on the worker thread before its next batch.
    void post(std::function<void(Tunnel&)> cmd);
    // Blocks until every posted command has run (UI start-up only).
    void flush();

    void set_running(bool r) { running_ = r; }
    bool running() const { return running_; }
    // Solver time per batch (the publish cadence), and an optional cap on
    // the simulation rate (0 = flat out).
    void set_batch_ms(float ms) { batch_ms_ = ms; }
    void set_max_steps_per_second(float s) { max_sps_ = s; }

    TunnelStatus status() const;
    // The latest measurements (wake survey, spectra), refreshed by the
    // tunnel every analysis_every batches.
    TunnelAnalysis analysis() const;
    double mlups() const { return mlups_.load(); }
    double ms_per_step() const { return ms_per_step_.load(); }
    int last_batch_steps() const { return last_batch_.load(); }

    const Snapshot& snapshot(int slot) const { return snaps_[slot]; }
    VkSemaphore sim_timeline() const { return sim_tl_; }
    VkSemaphore render_timeline() const { return render_tl_; }

    // The latest published snapshot for a frame that will signal the render
    // timeline to `render_value` when done. slot < 0: nothing published yet.
    Frame acquire(std::uint64_t render_value);

  private:
    void loop();
    void publish();

    Context& ctx_;
    std::unique_ptr<Tunnel> tunnel_;
    std::array<Snapshot, kSlots> snaps_;
    VkSemaphore sim_tl_ = VK_NULL_HANDLE;
    VkSemaphore render_tl_ = VK_NULL_HANDLE;
    std::uint64_t sim_value_ = 0;

    mutable std::mutex mu_; // guards everything below
    std::condition_variable cv_;
    std::deque<std::function<void(Tunnel&)>> commands_;
    std::uint64_t posted_ = 0, done_ = 0;
    TunnelStatus status_;
    TunnelAnalysis analysis_;
    Frame published_;
    std::array<std::uint64_t, kSlots> last_read_{}; // render value of the last frame per slot
    std::array<Frame, kSlots> slot_meta_{};

    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{true};
    std::atomic<float> batch_ms_{16.0f};
    std::atomic<float> max_sps_{0.0f};
    std::atomic<double> mlups_{0.0}, ms_per_step_{1.0};
    std::atomic<int> last_batch_{0};
    std::thread thread_;
};

} // namespace windoa::app
