// When is the flow "developed"? The detector that decides when the
// dashboard numbers can be trusted after a change.
//
// The coefficient history is cut into consecutive windows of `window_ft`
// flow-throughs (simulated time: steps x u / nx). Developed when
//   - at least ft_min flow-throughs have passed (guards the early
//     overshoot, where two windows straddling the peak look alike), and
//   - for every coefficient the last two window means differ by less than
//     max(tol |mean|, abs_tol), and on a monotone approach the geometric
//     (Aitken) tail estimate of the remaining change is inside it too.
// Released anyway at ft_max ("not settled") so it never develops forever.
// Defaults tuned on real Ahmed Cd histories (V22 checks them).
// Specification: THEORY 9.5.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace windoa {

class ConvergenceMonitor {
  public:
    static constexpr int kCoeffs = 2; // Cd, Cl
    using Coeffs = std::array<double, kCoeffs>;

    explicit ConvergenceMonitor(double window_ft = 0.5, double tol = 0.01, double abs_tol = 0.01,
                                double ft_min = 1.5, double ft_max = 6.0);

    // A change happened: forget the history and develop again.
    void restart(const std::string& reason = "");
    void finish() { developing_ = false; } // the user's "skip developing"
    void mark_restored();                  // loaded settled from the flow cache

    // Coefficients averaged over n_steps solver steps at inlet speed u in a
    // tunnel nx cells long.
    void add(const Coeffs& c, int n_steps, double u, int nx);

    bool developing() const { return developing_; }
    bool settled() const { return settled_; }
    bool gave_up() const { return gave_up_; }
    bool restored() const { return restored_; }
    double flow_throughs() const { return ft_; }
    const std::string& reason() const { return reason_; }
    std::string status() const; // one line for the dashboard
    // The last completed window's means (what a settled flow reports), or
    // nothing before the first window closes.
    std::optional<Coeffs> last_window_means() const {
        if (windows_.empty())
            return std::nullopt;
        return windows_.back();
    }

  private:
    bool coefficient_settled(double w1, double w2, double w3) const;
    void check();

    double window_ft_, tol_, abs_tol_, ft_min_, ft_max_;
    std::string reason_;
    double ft_ = 0.0;
    double verdict_ft_ = 0.0; // flow-throughs when settled / gave up (ft_ keeps counting)
    long long steps_ = 0;
    bool developing_ = true, settled_ = false, gave_up_ = false, restored_ = false;
    std::vector<Coeffs> windows_;
    Coeffs acc_{};
    double acc_ft_ = 0.0;
};

} // namespace windoa
