// deploy/ci/emitters/driver.cpp — appended after an emitted model.cpp by deploy/ci/emitters.sh.
// Usage: program train.csv test_features.csv predictions   (headerless CSV; the last train column
// is the target). Fits and predicts through the emitted C ABI and writes one prediction per line
// with 17 significant digits (round-trips the double exactly).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool read_csv(const char* path, std::vector<double>& cells, std::size_t& rows, std::size_t& columns) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    rows = 0;
    columns = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream fields(line);
        std::string cell;
        std::size_t count = 0;
        while (std::getline(fields, cell, ',')) {
            cells.push_back(std::strtod(cell.c_str(), nullptr));
            ++count;
        }
        if (columns == 0) columns = count;
        else if (count != columns) return false;
        ++rows;
    }
    return rows > 0 && columns > 0;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s train.csv test_features.csv predictions\n", argv[0]);
        return 2;
    }
    std::vector<double> train_cells, test_cells;
    std::size_t train_rows = 0, train_cols = 0, test_rows = 0, test_cols = 0;
    if (!read_csv(argv[1], train_cells, train_rows, train_cols)) return 3;
    if (!read_csv(argv[2], test_cells, test_rows, test_cols)) return 3;
    if (train_cols < 2 || test_cols != train_cols - 1) return 4;
    const std::size_t columns = train_cols - 1;
    std::vector<double> train_x(train_rows * columns), targets(train_rows);
    for (std::size_t r = 0; r < train_rows; ++r) {
        for (std::size_t c = 0; c < columns; ++c) train_x[r * columns + c] = train_cells[r * train_cols + c];
        targets[r] = train_cells[r * train_cols + columns];
    }
    model_handle* model = nullptr;
    if (model_fit_numeric(train_x.data(), train_x.size(), train_rows, columns, targets.data(), targets.size(), &model) != 0) return 10;
    std::vector<double> out(test_rows);
    if (model_predict_numeric_batch(model, test_cells.data(), test_cells.size(), test_rows, columns, nullptr, 0, out.data(), out.size()) != 0) return 11;
    model_unload(model);
    std::FILE* file = std::fopen(argv[3], "w");
    if (file == nullptr) return 5;
    for (double value : out) std::fprintf(file, "%.17g\n", value);
    std::fclose(file);
    return 0;
}
