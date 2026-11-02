use super::*;

pub(super) const IO_BLOCK: &str = r#"// ---- caller-owned C ABI ----------------------------------------------------

extern "C" {

struct model_handle {
    Model model;
    std::size_t width;

    model_handle(Model&& fitted, std::size_t fitted_width)
        : model(std::move(fitted)), width(fitted_width) {}
};

std::uint32_t model_abi_version(void) noexcept {
    return 1U;
}

std::size_t model_workspace_bytes(const model_handle*, std::size_t) noexcept {
    return 0U;
}

int model_fit_numeric(
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    const double* targets,
    std::size_t target_len,
    model_handle** out
) noexcept {
    if (out == nullptr) return 1;
    *out = nullptr;
    if (values == nullptr || targets == nullptr || rows == 0U || columns == 0U) return 1;
    if (target_len != rows) return 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values_len < rows * columns) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        std::vector<double> y;
        y.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            if (!std::isfinite(targets[row])) return 1;
            x.push_back(std::move(cells));
            y.push_back(targets[row]);
        }
        Model fitted = Model::fit(x, y);
        *out = new model_handle(std::move(fitted), columns);
        return 0;
    } catch (...) {
        return 2;
    }
}

int model_predict_numeric_batch(
    const model_handle* handle,
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    void* workspace,
    std::size_t workspace_bytes,
    double* out,
    std::size_t out_len
) noexcept {
    (void)workspace;
    (void)workspace_bytes;
    if (handle == nullptr || columns == 0U || columns != handle->width) return 1;
    if (rows == 0U) return values_len == 0U ? 0 : 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values == nullptr || values_len < rows * columns || out == nullptr || out_len < rows) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            x.push_back(std::move(cells));
        }
        std::vector<double> predictions = handle->model.predict(x);
        if (predictions.size() != rows) return 2;
        for (std::size_t row = 0; row < rows; ++row) out[row] = predictions[row];
        return 0;
    } catch (...) {
        return 2;
    }
}

void model_unload(model_handle* handle) noexcept {
    delete handle;
}

}  // extern "C"
"#;
