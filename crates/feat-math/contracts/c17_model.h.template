/* Native model API, version 1. Double-precision row-major inputs. */
#ifndef JOPE_MODEL_C17_H
#define JOPE_MODEL_C17_H
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#ifdef __cplusplus
static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
              "64-bit IEEE double required");
extern "C" {
#else
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "64-bit IEEE double required");
#endif

enum jope_model_status {
    JOPE_MODEL_OK = 0,
    JOPE_MODEL_INVALID_ARGUMENT = 1,
    JOPE_MODEL_INVALID_HANDLE = 2,
    JOPE_MODEL_PANIC = 3,
    JOPE_MODEL_INTERNAL_ERROR = 4
};

/* Build the companion Rust sources once as a static or shared native library.
 * Buffers must be valid, aligned, and accessible for the declared lengths.
 * Lengths are element counts, not bytes. Inputs are copied before outputs change.
 * All floating-point input bits are preserved; no CSV cleaning is applied.
 * On failure, output buffers and output handles are unchanged.
 * Handles are owned by this loaded library. Release each successful fit once.
 * Predict may run concurrently; free does not cancel a prediction in progress.
 * Empty prediction batches accept NULL input/output with zero lengths.
 * Allocation exhaustion and process-aborting builds cannot be recovered.
 */
uint32_t jope_model_abi_version(void);
int32_t jope_model_fit(const double *x, size_t x_len,
                       const double *y, size_t y_len,
                       size_t rows, size_t cols, uint64_t *out_handle);
int32_t jope_model_predict(uint64_t handle, const double *x, size_t x_len,
                           size_t rows, size_t cols,
                           double *out, size_t out_len);
/* The recipe's R-squared score on nonempty labeled rows. */
int32_t jope_model_score(uint64_t handle, const double *x, size_t x_len,
                         const double *y, size_t y_len, size_t rows, size_t cols,
                         double *out_score);
int32_t jope_model_free(uint64_t handle);

#ifdef __cplusplus
}
#endif
#endif
