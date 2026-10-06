#include "sim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace windoa::app {

namespace {
using Clock = std::chrono::steady_clock;

void copy(VkCommandBuffer cmd, const Buffer& src, const Buffer& dst) {
    VkBufferCopy region{0, 0, std::min(src.size(), dst.size())};
    vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
}
} // namespace

SimWorker::SimWorker(Context& ctx, const TunnelSettings& s, const std::filesystem::path& cache_dir)
    : ctx_(ctx) {
    tunnel_ = std::make_unique<Tunnel>(ctx, s, cache_dir);
    const std::size_t cells = tunnel_->solver().cells();
    for (Snapshot& sn : snaps_) {
        sn.macro = std::make_unique<Buffer>(ctx, cells * 16);
        sn.flags = std::make_unique<Buffer>(ctx, cells * 4);
        sn.dye = std::make_unique<Buffer>(ctx, cells * 4);
        sn.aux = std::make_unique<Buffer>(ctx, cells * 4);
        sn.mean = std::make_unique<Buffer>(ctx, cells * 16);
        sn.m2 = std::make_unique<Buffer>(ctx, cells * 16);
        ctx.fill_zero(*sn.dye);
        ctx.fill_zero(*sn.aux);
        ctx.fill_zero(*sn.mean);
        ctx.fill_zero(*sn.m2);
    }
    sim_tl_ = ctx.create_timeline(0);
    render_tl_ = ctx.create_timeline(0);
    status_ = tunnel_->status();
    thread_ = std::thread([this] { loop(); });
}

SimWorker::~SimWorker() {
    stop_ = true;
    cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
    // A batch and a snapshot copy may still be on the GPU (pipelined, not
    // waited for); with the worker gone nothing else submits to its queue.
    vkQueueWaitIdle(ctx_.queue());
    tunnel_.reset(); // before the buffers it may still be using finish
    vkDestroySemaphore(ctx_.device(), sim_tl_, nullptr);
    vkDestroySemaphore(ctx_.device(), render_tl_, nullptr);
}

void SimWorker::post(std::function<void(Tunnel&)> cmd) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        commands_.push_back({nullptr, std::move(cmd)});
        ++posted_;
    }
    cv_.notify_all();
}

void SimWorker::post(const char* key, std::function<void(Tunnel&)> cmd) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto old = std::find_if(commands_.begin(), commands_.end(), [&](const Command& c) {
            return c.key && std::strcmp(c.key, key) == 0;
        });
        if (old != commands_.end())
            commands_.erase(old); // superseded: it will not run, so not counted again
        else
            ++posted_;
        commands_.push_back({key, std::move(cmd)});
    }
    cv_.notify_all();
}

void SimWorker::flush() {
    std::unique_lock<std::mutex> lock(mu_);
    const std::uint64_t target = posted_;
    cv_.wait(lock, [&] { return done_ >= target || stop_; });
}

SimWorker::Failure SimWorker::failure() const {
    std::lock_guard<std::mutex> lock(mu_);
    return failure_;
}

void SimWorker::resume() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        failure_ = {};
    }
    running_ = true;
    cv_.notify_all();
}

TunnelStatus SimWorker::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

TunnelAnalysis SimWorker::analysis() const {
    std::lock_guard<std::mutex> lock(mu_);
    return analysis_;
}

SimWorker::PlacedMesh SimWorker::placed_mesh() const {
    std::lock_guard<std::mutex> lock(mu_);
    return mesh_;
}

SimWorker::Frame SimWorker::acquire(std::uint64_t render_value) {
    std::lock_guard<std::mutex> lock(mu_);
    Frame f = published_;
    if (f.slot >= 0) {
        last_read_[f.slot] = render_value;
        consumed_ = true;
    }
    return f;
}

bool SimWorker::wait_for_readers(int j) {
    std::uint64_t must_finish = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        must_finish = last_read_[j];
    }
    // Host wait on this thread only, with a timeout so shutdown never hangs.
    for (;;) {
        std::uint64_t done = 0;
        vkGetSemaphoreCounterValue(ctx_.device(), render_tl_, &done);
        if (done >= must_finish)
            return true;
        if (stop_)
            return false;
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &render_tl_;
        wi.pValues = &must_finish;
        vkWaitSemaphores(ctx_.device(), &wi, 50'000'000); // 50 ms
    }
}

SimWorker::Publish SimWorker::prepare_publish(lbm::Solver::Tail& tail) {
    Publish p;
    int drawn = -1, pending = -1;
    {
        std::lock_guard<std::mutex> lock(mu_);
        drawn = published_.slot;
    }
    for (const Publish& f : flight_)
        if (f.on)
            pending = f.slot;
    int j = -1;
    for (int k = 0; k < kSlots && j < 0; ++k)
        if (k != drawn && k != pending)
            j = k;
    if (j < 0 || !wait_for_readers(j)) // every frame that read slot j must be done
        return p;

    lbm::Solver& s = tunnel_->solver();
    euler::Solver* e = tunnel_->transonic() ? tunnel_->euler_solver() : nullptr;
    Snapshot& sn = snaps_[j];
    const TunnelStatus st = tunnel_->status();
    const bool geo = sn.geometry_version != st.render_geometry;
    Dye* dye = tunnel_->dye();
    const bool dye_on = st.dye_on && dye && !e;
    // the time averages, when they have changed since this slot copied them
    const FlowStats* stats = tunnel_->stats();
    const int samples = stats && !e ? stats->samples() : 0;
    const bool stats_copy = samples > 0 && samples != sn.stats_samples;
    p.on = true;
    p.slot = j;
    p.value = ++sim_value_;
    p.dye = dye_on;
    p.transonic = e != nullptr;
    p.stats_inv_weight = samples > 0 ? float(1.0 / stats->weight()) : 0.0f;
    tail.record = [=, &s, &sn](VkCommandBuffer cmd, int live) {
        if (e) {
            copy(cmd, e->macro_buffer(), *sn.macro);
            copy(cmd, e->rho_buffer(), *sn.aux);
            if (geo)
                copy(cmd, e->flag_buffer(), *sn.flags);
        } else {
            copy(cmd, s.macro_buffer(live), *sn.macro);
            if (geo)
                copy(cmd, s.flag_buffer(), *sn.flags);
            if (dye_on)
                copy(cmd, dye->concentration_buffer(), *sn.dye);
            if (stats_copy) {
                copy(cmd, stats->mean(), *sn.mean);
                copy(cmd, stats->m2(), *sn.m2);
            }
        }
    };
    tail.signal = sim_tl_;
    tail.value = p.value;
    sn.geometry_version = st.render_geometry;
    if (stats_copy || samples == 0)
        sn.stats_samples = samples;
    return p;
}

void SimWorker::commit_publish(const Publish& p) {
    const TunnelStatus st = tunnel_->status();
    std::shared_ptr<const geometry::Mesh> mesh; // new geometry: its triangles, as placed
    if (st.render_geometry != mesh_built_for_) {
        mesh = std::make_shared<const geometry::Mesh>(tunnel_->placed_mesh());
        mesh_built_for_ = st.render_geometry;
    }
    Frame f;
    f.slot = p.slot;
    f.value = p.value;
    f.geometry_version = snaps_[p.slot].geometry_version;
    f.steps = p.steps >= 0 ? p.steps : st.steps;
    f.flow_epoch = st.flow_epoch;
    f.dye = p.dye;
    f.transonic = p.transonic;
    f.stats_inv_weight = p.stats_inv_weight;
    published_epoch_ = st.flow_epoch;
    published_geo_ = st.render_geometry;
    published_steps_ = f.steps;

    std::lock_guard<std::mutex> lock(mu_);
    if (mesh)
        mesh_ = {std::move(mesh), st.render_geometry};
    slot_meta_[p.slot] = f;
    published_ = f;
    consumed_ = false;
    status_ = st;
    if (tunnel_->analysis().version != analysis_.version)
        analysis_ = tunnel_->analysis();
}

void SimWorker::publish_now() {
    lbm::Solver::Tail tail;
    const Publish p = prepare_publish(tail);
    if (!p.on)
        return;
    const int live = tunnel_->solver().live_index();
    ctx_.submit_and_wait([&](VkCommandBuffer cmd) { tail.record(cmd, live); }, tail.signal,
                         tail.value);
    commit_publish(p);
}

void SimWorker::loop() {
    auto last_publish = Clock::now();
    auto last_complete = Clock::now();
    auto stop_solver = [&](const std::exception& e) {
        std::fprintf(stderr, "[sim] advance failed: %s\n", e.what());
        running_ = false;
        std::lock_guard<std::mutex> lock(mu_);
        failure_ = {std::string("the solver stopped: ") + e.what(), true, Clock::now()};
    };
    // The oldest batch in flight, and its snapshot when it carried one.
    auto complete_one = [&]() {
        const bool used = tunnel_->advance_complete();
        const Publish p = flight_.front();
        flight_.pop_front();
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - last_complete).count();
        if (p.batch_steps > 0 && ms < 500.0) { // between completions: the batch's GPU time
            ms_per_step_ = ms_per_step_ + 0.2 * (ms / p.batch_steps - ms_per_step_);
            mlups_ = double(tunnel_->solver().cells()) / (ms_per_step_ * 1e3);
        }
        last_complete = now;
        if (p.on && used) {
            commit_publish(p);
            last_publish = now;
        }
    };
    auto finish_all = [&]() {
        while (tunnel_->batches_in_flight() > 0)
            complete_one();
        flight_.clear();
    };

    while (!stop_) {
        // 1. commands from the UI: they change the tunnel, so what is in
        // flight finishes first
        std::deque<Command> cmds;
        {
            std::lock_guard<std::mutex> lock(mu_);
            cmds.swap(commands_);
        }
        if (!cmds.empty()) {
            try {
                finish_all();
            } catch (const std::exception& e) {
                stop_solver(e);
            }
        }
        for (auto& c : cmds) {
            try {
                c.run(*tunnel_);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[sim] command failed: %s\n", e.what());
                std::lock_guard<std::mutex> lock(mu_);
                failure_ = {std::string("a change could not be applied: ") + e.what(), false,
                            Clock::now()};
            }
        }
        if (!cmds.empty()) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                done_ += cmds.size();
                status_ = tunnel_->status();
            }
            cv_.notify_all();
        }

        // 2. a batch: the next one goes out, then the one before it completes
        // while the GPU runs the new one, so the GPU is not left idle while
        // the host takes a batch's readings. A batch carries a snapshot of its
        // end state when a frame has taken the last one and ~12 ms have passed
        // (at most about one a frame: on the fine grid a snapshot per batch
        // cost ~20 % of the solver's throughput; a minimised window takes none).
        bool stepped = false;
        const TunnelStatus before = tunnel_->status();
        if (running_ && !before.paused && before.n_solid > 0) {
            const int steps =
                std::clamp(int(batch_ms_ / std::max(ms_per_step_.load(), 1e-3)), 1, 2000);
            const auto t0 = Clock::now();
            try {
                lbm::Solver::Tail tail;
                bool pending_snapshot = false;
                for (const Publish& f : flight_)
                    pending_snapshot = pending_snapshot || f.on;
                const bool want = consumed_ && !pending_snapshot &&
                                  Clock::now() - last_publish >= std::chrono::milliseconds(12);
                Publish p = want ? prepare_publish(tail) : Publish{};
                p.batch_steps = steps;
                if (tunnel_->advance_submit(steps, tail)) {
                    if (tunnel_->batches_in_flight() == 0) { // the blocking (compressible) batch
                        const double ms =
                            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                        ms_per_step_ = ms_per_step_ + 0.2 * (ms / steps - ms_per_step_);
                        mlups_ = double(tunnel_->solver().cells()) / (ms_per_step_ * 1e3);
                        if (p.on) {
                            commit_publish(p);
                            last_publish = Clock::now();
                        }
                    } else {
                        p.steps = tunnel_->solver().steps_taken();
                        flight_.push_back(p);
                        if (tunnel_->batches_in_flight() > 1)
                            complete_one();
                    }
                    stepped = true;
                }
            } catch (const std::exception& e) {
                stop_solver(e);
            }
            last_batch_ = steps;
            const float cap = max_sps_;
            if (cap > 0.0f) { // optional sim-rate cap
                const double ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                const double want_ms = 1000.0 * steps / cap;
                if (want_ms > ms)
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(long(1000 * (want_ms - ms))));
            }
        }
        if (!stepped && tunnel_->batches_in_flight() > 0) { // paused or stopped: finish
            try {
                finish_all();
            } catch (const std::exception& e) {
                stop_solver(e);
            }
        }

        // 3. snapshots outside batches: a change made between them (commands,
        // a reset, new geometry), or the last state once stepping stops
        const TunnelStatus st = tunnel_->status();
        if (!cmds.empty() || st.flow_epoch != published_epoch_ ||
            st.render_geometry != published_geo_ ||
            (!stepped && st.steps != published_steps_ && consumed_)) {
            try {
                finish_all();
                publish_now();
                last_publish = Clock::now();
            } catch (const std::exception& e) {
                stop_solver(e);
            }
        } else {
            std::lock_guard<std::mutex> lock(mu_);
            status_ = st;
        }
        if (!stepped) { // idle: wait for a command (or poll the pause state)
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(20),
                         [&] { return !commands_.empty() || stop_; });
        }
    }
    try { // nothing left on the GPU for a tunnel about to go
        finish_all();
    } catch (const std::exception&) {
    }
}

} // namespace windoa::app
