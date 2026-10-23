// Native model with an optional double-precision device head.
#ifndef JOPE_CUDA_MODEL_HPP
#define JOPE_CUDA_MODEL_HPP
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace jope_cuda {
using Rows = std::vector<std::vector<double>>;
using StringRows = std::vector<std::vector<std::string>>;
class Error : public std::runtime_error {
public:
    const int status;
    Error(int code, const std::string& message) : std::runtime_error(message), status(code) {}
};

// Each instance owns one native handle. Separate instances may run concurrently.
// Moving, refitting, or destroying an instance requires exclusive caller access.
// Prediction retains the fitted state for its entire native call.
class Model {
public:
    Model() noexcept = default;
    ~Model() noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;

    // Fits atomically: a failed refit leaves the previous model usable.
    Model& fit(const Rows& rows, const std::vector<double>& targets);
    // Opens the embedded raw-string predictor. Training recipes refuse load().
    static Model load();
    std::vector<double> predict(const Rows& rows) const;
    double score(const Rows& rows, const std::vector<double>& targets) const;
    std::vector<double> predict_strings(const StringRows& rows) const;
    std::vector<double> predict_csv(const std::string& csv, std::size_t expected_rows) const;

    // Explicit GPU execution; no automatic fallback. Fitted feature processing
    // stays on the host. Only admitted linear heads run on the device.
    // stream is a borrowed cudaStream_t on the current device, or nullptr.
    // This operation synchronizes that stream before returning copied results.
    std::vector<double> predict_cuda(const Rows& rows, void* stream = nullptr) const;
    bool supports_cuda() const;
    static bool cuda_available() noexcept;
    void close() noexcept;

private:
    std::uint64_t handle_ = 0;
    std::size_t width_ = 0;
    void check() const;
};
} // namespace jope_cuda
#endif
