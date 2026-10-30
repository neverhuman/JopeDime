#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <limits>
#include <new>
#include <string>

namespace {

constexpr std::size_t kHeaderLength = 56;
constexpr std::size_t kChecksumLength = 32;
constexpr std::uint16_t kCodecVersion = 2;
constexpr unsigned char kMagic[8] = {'J', 'D', 'F', 'C', 'P', 'P', '3', 0};

struct Error {
  std::int32_t code;
  char stage[32];
  char message[256];
};

using Predict = int (*)(const double *, std::size_t, const std::uint8_t *,
                        std::size_t, std::size_t, std::size_t, double *,
                        std::size_t);
using Columns = std::size_t (*)();

struct Model {
  void *library;
  Predict predict;
  Columns columns;
};

void set_error(Error *error, std::int32_t code, const char *stage,
               const std::string &message) noexcept {
  if (error == nullptr) return;
  error->code = code;
  std::memset(error->stage, 0, sizeof(error->stage));
  std::memset(error->message, 0, sizeof(error->message));
  std::strncpy(error->stage, stage, sizeof(error->stage) - 1);
  std::strncpy(error->message, message.c_str(), sizeof(error->message) - 1);
}

std::uint16_t read_u16(const std::uint8_t *bytes) noexcept {
  return static_cast<std::uint16_t>(bytes[0]) |
         (static_cast<std::uint16_t>(bytes[1]) << 8);
}

std::uint32_t read_u32(const std::uint8_t *bytes) noexcept {
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    value |= static_cast<std::uint32_t>(bytes[shift / 8]) << shift;
  }
  return value;
}

std::uint64_t read_u64(const std::uint8_t *bytes) noexcept {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(bytes[shift / 8]) << shift;
  }
  return value;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t *result) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) return false;
  *result = left + right;
  return true;
}

bool library_section(const std::uint8_t *codec, std::size_t codec_length,
                     const std::uint8_t **library,
                     std::size_t *library_length) noexcept {
  if (codec == nullptr || codec_length < kHeaderLength + kChecksumLength ||
      std::memcmp(codec, kMagic, sizeof(kMagic)) != 0 ||
      read_u16(codec + 8) != kCodecVersion ||
      read_u16(codec + 10) != kHeaderLength || read_u32(codec + 12) != 0 ||
      read_u64(codec + 16) != codec_length) {
    return false;
  }
  const std::uint64_t raw_lengths[4] = {
      read_u64(codec + 24), read_u64(codec + 32), read_u64(codec + 40),
      read_u64(codec + 48)};
  std::size_t offset = kHeaderLength;
  for (std::uint64_t raw : raw_lengths) {
    if (raw > std::numeric_limits<std::size_t>::max()) return false;
    std::size_t next = 0;
    if (!checked_add(offset, static_cast<std::size_t>(raw), &next) ||
        next > codec_length - kChecksumLength) {
      return false;
    }
    offset = next;
  }
  if (offset != codec_length - kChecksumLength || raw_lengths[3] == 0) {
    return false;
  }
  const std::size_t size = static_cast<std::size_t>(raw_lengths[3]);
  *library = codec + offset - size;
  *library_length = size;
  return true;
}

std::string loader_error() {
  const char *message = dlerror();
  return message == nullptr ? "dynamic loader returned no detail" : message;
}

}  // namespace

extern "C" int jope_runtime_model_load(const std::uint8_t *codec,
                                        std::size_t codec_length,
                                        const char *library_path,
                                        void **output,
                                        Error *error) noexcept {
  if (output != nullptr) *output = nullptr;
  if (library_path == nullptr || output == nullptr) {
    set_error(error, 1, "load", "null output or library path");
    return 1;
  }
  const std::uint8_t *library_bytes = nullptr;
  std::size_t library_length = 0;
  if (!library_section(codec, codec_length, &library_bytes, &library_length)) {
    set_error(error, 2, "codec", "malformed fitted-model codec");
    return 2;
  }
  try {
    std::ofstream stream(library_path, std::ios::binary | std::ios::trunc);
    if (!stream ||
        !stream.write(reinterpret_cast<const char *>(library_bytes),
                      static_cast<std::streamsize>(library_length)) ||
        !stream.flush()) {
      set_error(error, 3, "load", "write fitted model image");
      return 3;
    }
    dlerror();
    void *library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      set_error(error, 4, "load", loader_error());
      return 4;
    }
    dlerror();
    void *predict = dlsym(library, "jope_fitted_predict");
    const char *predict_error = dlerror();
    void *columns = dlsym(library, "jope_fitted_columns");
    const char *columns_error = dlerror();
    if (predict_error != nullptr || columns_error != nullptr ||
        predict == nullptr || columns == nullptr) {
      std::string message = predict_error != nullptr
                                ? predict_error
                                : (columns_error != nullptr ? columns_error
                                                            : "missing model symbol");
      dlclose(library);
      set_error(error, 5, "load", message);
      return 5;
    }
    Model *model = new (std::nothrow) Model{
        library, reinterpret_cast<Predict>(predict),
        reinterpret_cast<Columns>(columns)};
    if (model == nullptr) {
      dlclose(library);
      set_error(error, 6, "load", "allocate fitted model handle");
      return 6;
    }
    *output = model;
    set_error(error, 0, "load", "ok");
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, 7, "load", exception.what());
    return 7;
  } catch (...) {
    set_error(error, 7, "load", "unknown C++ exception");
    return 7;
  }
}

extern "C" int jope_runtime_model_predict(
    void *opaque, const double *values, std::size_t values_length,
    const std::uint8_t *missing, std::size_t missing_length, std::size_t rows,
    std::size_t columns, double *output, std::size_t output_length,
    Error *error) noexcept {
  if (opaque == nullptr) {
    set_error(error, 8, "predict", "null fitted model handle");
    return 8;
  }
  Model *model = static_cast<Model *>(opaque);
  if (columns != model->columns()) {
    set_error(error, 9, "predict", "typed-table column count mismatch");
    return 9;
  }
  const int status = model->predict(values, values_length, missing,
                                    missing_length, rows, columns, output,
                                    output_length);
  if (status != 0) {
    set_error(error, 100 + status, "predict",
              "fitted model prediction rejected typed table");
    return 100 + status;
  }
  set_error(error, 0, "predict", "ok");
  return 0;
}

extern "C" void jope_runtime_model_free(void *opaque) noexcept {
  Model *model = static_cast<Model *>(opaque);
  if (model == nullptr) return;
  dlclose(model->library);
  delete model;
}
