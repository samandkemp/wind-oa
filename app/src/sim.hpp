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
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "windoa/context.hpp"
#include "windoa/tunnel.hpp"

namespace windoa::app {

class SimWorker {
  public:
    // one drawn, one written by the batch in flight, one free (render::kSlots)
    static constexpr int kSlots = 3;

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
    // The same, superseding a pending command with this key: a slider posts
    // every frame it moves, and only its latest value matters. The newer
    // command joins the back of the queue, so its order relative to other
    // commands is that of the latest post.
    void post(const char* key, std::function<void(Tunnel&)> cmd);
    // Blocks until every posted command has run (UI start-up only).
    void flush();

    void set_running(bool r) { running_ = r; }
    bool running() const { return running_; }

    // The last failure on the worker thread: a command that threw (the change
    // was not applied) or a batch that threw (stepping stops until resume()).
    struct Failure {
        std::string what;
        bool stopped = false;
        std::chrono::steady_clock::time_point when{};
    };
    Failure failure() const;
    void resume(); // clears the failure and steps again
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
    // The placed model's triangles (lattice cells), with the render geometry
    // version they belong to (Frame::geometry_version); refreshed by the
    // worker when the geometry changes, for the renderer's true-shape view.
    struct PlacedMesh {
        std::shared_ptr<const geometry::Mesh> mesh;
        std::uint64_t version = UINT64_MAX;
    };
    PlacedMesh placed_mesh() const;
    VkSemaphore sim_timeline() const { return sim_tl_; }
    VkSemaphore render_timeline() const { return render_tl_; }

    // The latest published snapshot for a frame that will signal the render
    // timeline to `render_value` when done. slot < 0: nothing published yet.
    Frame acquire(std::uint64_t render_value);

  private:
    void loop();
    // A snapshot for the next batch to carry: a slot neither drawn nor
    // pending, its copies recorded as the batch's tail, made the frames'
    // latest (committed) once that batch completes -- so a frame never waits
    // on a batch. publish_now(): the same at once, for a change made between
    // batches (nothing in flight).
    struct Publish {
        bool on = false;
        int slot = -1;
        std::uint64_t value = 0;
        bool dye = false, transonic = false;
        float stats_inv_weight = 0.0f;
        std::int64_t steps = -1; // solver steps at the batch's end (-1: from the status)
        int batch_steps = 0;
    };
    Publish prepare_publish(lbm::Solver::Tail& tail);
    void commit_publish(const Publish& p);
    void publish_now();
    bool wait_for_readers(int slot); // frames that read it are done (false: stopping)
    std::deque<Publish> flight_;     // one per tunnel batch in flight, oldest first
    std::uint64_t published_epoch_ = UINT64_MAX, published_geo_ = UINT64_MAX;
    std::int64_t published_steps_ = -1;

    Context& ctx_;
    std::unique_ptr<Tunnel> tunnel_;
    std::array<Snapshot, kSlots> snaps_;
    VkSemaphore sim_tl_ = VK_NULL_HANDLE;
    VkSemaphore render_tl_ = VK_NULL_HANDLE;
    std::uint64_t sim_value_ = 0;
    std::uint64_t mesh_built_for_ = UINT64_MAX; // worker thread only

    mutable std::mutex mu_; // guards everything below
    std::condition_variable cv_;
    struct Command {
        const char* key = nullptr; // nullptr: never superseded
        std::function<void(Tunnel&)> run;
    };
    std::deque<Command> commands_;
    std::uint64_t posted_ = 0, done_ = 0;
    TunnelStatus status_;
    TunnelAnalysis analysis_;
    Failure failure_;
    Frame published_;
    std::array<std::uint64_t, kSlots> last_read_{}; // render value of the last frame per slot
    PlacedMesh mesh_;
    std::array<Frame, kSlots> slot_meta_{};

    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{true};
    std::atomic<bool> consumed_{true}; // a frame has taken the latest snapshot
    std::atomic<float> batch_ms_{16.0f};
    std::atomic<float> max_sps_{0.0f};
    std::atomic<double> mlups_{0.0}, ms_per_step_{1.0};
    std::atomic<int> last_batch_{0};
    std::thread thread_;
};

} // namespace windoa::app
