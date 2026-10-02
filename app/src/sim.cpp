#include "sim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

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
        ctx.fill_zero(*sn.dye);
        ctx.fill_zero(*sn.aux);
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
    tunnel_.reset(); // before the buffers it may still be using finish
    vkDestroySemaphore(ctx_.device(), sim_tl_, nullptr);
    vkDestroySemaphore(ctx_.device(), render_tl_, nullptr);
}

void SimWorker::post(std::function<void(Tunnel&)> cmd) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        commands_.push_back(std::move(cmd));
        ++posted_;
    }
    cv_.notify_all();
}

void SimWorker::flush() {
    std::unique_lock<std::mutex> lock(mu_);
    const std::uint64_t target = posted_;
    cv_.wait(lock, [&] { return done_ >= target || stop_; });
}

TunnelStatus SimWorker::status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

SimWorker::Frame SimWorker::acquire(std::uint64_t render_value) {
    std::lock_guard<std::mutex> lock(mu_);
    Frame f = published_;
    if (f.slot >= 0)
        last_read_[f.slot] = render_value;
    return f;
}

void SimWorker::publish() {
    std::uint64_t must_finish = 0;
    int j = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        j = published_.slot < 0 ? 0 : 1 - published_.slot;
        must_finish = last_read_[j];
    }
    // Every frame that read slot j must be done before it is overwritten.
    // Host wait on this thread only, with a timeout so shutdown never hangs.
    for (;;) {
        std::uint64_t done = 0;
        vkGetSemaphoreCounterValue(ctx_.device(), render_tl_, &done);
        if (done >= must_finish || stop_)
            break;
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &render_tl_;
        wi.pValues = &must_finish;
        vkWaitSemaphores(ctx_.device(), &wi, 50'000'000); // 50 ms
    }
    if (stop_)
        return;

    lbm::Solver& s = tunnel_->solver();
    euler::Solver* e = tunnel_->transonic() ? tunnel_->euler_solver() : nullptr;
    Snapshot& sn = snaps_[j];
    const TunnelStatus st = tunnel_->status();
    const bool geo = sn.geometry_version != st.render_geometry;
    Dye* dye = tunnel_->dye();
    const bool dye_on = st.dye_on && dye && !e;
    const std::uint64_t v = ++sim_value_;
    ctx_.submit_and_wait(
        [&](VkCommandBuffer cmd) {
            if (e) {
                copy(cmd, e->macro_buffer(), *sn.macro);
                copy(cmd, e->rho_buffer(), *sn.aux);
                if (geo)
                    copy(cmd, e->flag_buffer(), *sn.flags);
            } else {
                copy(cmd, s.macro_buffer(s.live_index()), *sn.macro);
                if (geo)
                    copy(cmd, s.flag_buffer(), *sn.flags);
                if (dye_on)
                    copy(cmd, dye->concentration_buffer(), *sn.dye);
            }
        },
        sim_tl_, v);
    sn.geometry_version = st.render_geometry;

    std::lock_guard<std::mutex> lock(mu_);
    Frame f;
    f.slot = j;
    f.value = v;
    f.geometry_version = sn.geometry_version;
    f.steps = st.steps;
    f.flow_epoch = st.flow_epoch;
    f.dye = dye_on;
    f.transonic = e != nullptr;
    slot_meta_[j] = f;
    published_ = f;
    status_ = st;
}

void SimWorker::loop() {
    std::uint64_t published_epoch = UINT64_MAX, published_geo = UINT64_MAX;
    bool unpublished = false;
    auto last_publish = Clock::now();
    while (!stop_) {
        // 1. commands from the UI
        std::deque<std::function<void(Tunnel&)>> cmds;
        {
            std::lock_guard<std::mutex> lock(mu_);
            cmds.swap(commands_);
        }
        for (auto& c : cmds) {
            try {
                c(*tunnel_);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[sim] command failed: %s\n", e.what());
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

        // 2. a batch of steps
        bool stepped = false;
        const TunnelStatus before = tunnel_->status();
        if (running_ && !before.paused && before.n_solid > 0) {
            const int steps =
                std::clamp(int(batch_ms_ / std::max(ms_per_step_.load(), 1e-3)), 1, 2000);
            const auto t0 = Clock::now();
            try {
                tunnel_->advance(steps);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[sim] advance failed: %s\n", e.what());
                running_ = false;
            }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            ms_per_step_ = ms_per_step_ + 0.2 * (ms / steps - ms_per_step_);
            mlups_ = double(tunnel_->solver().cells()) / (ms_per_step_ * 1e3);
            last_batch_ = steps;
            stepped = true;
            const float cap = max_sps_;
            if (cap > 0.0f) { // optional sim-rate cap
                const double want_ms = 1000.0 * steps / cap;
                if (want_ms > ms)
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(long(1000 * (want_ms - ms))));
            }
        }

        // 3. publish what changed -- at most about once a frame: on the fine
        // grid a batch can be a single step, and a snapshot per step (a
        // 160 MB copy + a hand-off) cost ~20 % of the solver's throughput.
        const TunnelStatus st = tunnel_->status();
        unpublished |= stepped;
        const bool due = Clock::now() - last_publish >= std::chrono::milliseconds(12);
        if ((unpublished && (due || !stepped)) || !cmds.empty() ||
            st.flow_epoch != published_epoch || st.render_geometry != published_geo) {
            publish();
            published_epoch = st.flow_epoch;
            published_geo = st.render_geometry;
            unpublished = false;
            last_publish = Clock::now();
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
}

} // namespace windoa::app
