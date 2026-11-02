//! Native C boundary. Arithmetic stays in the safe companion module.
#![deny(unsafe_op_in_unsafe_fn)]
#[path = "model_core.rs"]
mod model_core;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};

const INVALID_ARGUMENT: i32 = 1;
const INVALID_HANDLE: i32 = 2;
const PANIC: i32 = 3;
const INTERNAL_ERROR: i32 = 4;
struct Handles {
    next: u64,
    models: BTreeMap<u64, Arc<model_core::C17Model>>,
}
fn handles() -> &'static Mutex<Handles> {
    static TABLE: LazyLock<Mutex<Handles>> = LazyLock::new(|| {
        Mutex::new(Handles {
            next: 1,
            models: BTreeMap::new(),
        })
    });
    &TABLE
}
fn boundary(f: impl FnOnce() -> Result<(), i32>) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(status)) => status,
        Err(_) => PANIC,
    }
}
fn buffer<T>(ptr: *const T, len: usize) -> bool {
    len <= isize::MAX as usize / std::mem::size_of::<T>()
        && (len == 0 || (!ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0))
}
fn shape(rows: usize, cols: usize, len: usize) -> bool {
    cols != 0 && rows.checked_mul(cols) == Some(len)
}
unsafe fn rows_from(ptr: *const f64, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    if rows == 0 {
        return Vec::new();
    }
    // SAFETY: callers validate the multiplication and pointer bounds before
    // calling. The C allocation remains owned by its caller for this borrow.
    unsafe { values_from(ptr, rows * cols) }
        .chunks_exact(cols)
        .map(|row| row.to_vec())
        .collect()
}

/// Copy a checked C-owned float buffer without taking ownership of its allocation.
unsafe fn values_from(ptr: *const f64, len: usize) -> Vec<f64> {
    // SAFETY: callers validate `len`, non-nullness, alignment and checked
    // dimensions; the C contract guarantees one live allocation readable for
    // `len` initialized f64 values. This shared borrow is copied, not owned.
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}
#[no_mangle]
pub extern "C" fn jope_model_abi_version() -> u32 {
    1
}

/// Pointer validity is the caller's C contract; malformed dimensions are refused.
#[no_mangle]
pub unsafe extern "C" fn jope_model_fit(
    x: *const f64,
    x_len: usize,
    y: *const f64,
    y_len: usize,
    rows: usize,
    cols: usize,
    out_handle: *mut u64,
) -> i32 {
    boundary(|| {
        if rows == 0
            || !shape(rows, cols, x_len)
            || y_len != rows
            || !buffer(x, x_len)
            || !buffer(y, y_len)
            || !buffer(out_handle, 1)
        {
            return Err(INVALID_ARGUMENT);
        }
        // Copy foreign memory before constructing state or touching caller output.
        // SAFETY: `shape` checked the product, `buffer` checked alignment and
        // non-nullness, and the C caller supplies the live initialized array.
        // `rows_from` copies values; allocation ownership remains with C.
        let x = unsafe { rows_from(x, rows, cols) };
        // SAFETY: `y_len == rows > 0`; `buffer` checked alignment and
        // non-nullness. The C caller keeps this initialized allocation live;
        // `values_from` copies it without taking ownership.
        let y = unsafe { values_from(y, y_len) };
        let model = Arc::new(model_core::C17Model::fit(&x, &y, cols));
        let mut table = handles().lock().map_err(|_| INTERNAL_ERROR)?;
        let id = table.next;
        table.next = id.checked_add(1).ok_or(INTERNAL_ERROR)?;
        table.models.insert(id, model);
        // SAFETY: `buffer(out_handle, 1)` checked alignment/non-nullness; the
        // C caller guarantees one writable `u64` at this address.
        unsafe {
            out_handle.write(id);
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn jope_model_predict(
    handle: u64,
    x: *const f64,
    x_len: usize,
    rows: usize,
    cols: usize,
    out: *mut f64,
    out_len: usize,
) -> i32 {
    boundary(|| {
        if !shape(rows, cols, x_len) || out_len < rows || !buffer(x, x_len) || !buffer(out, out_len)
        {
            return Err(INVALID_ARGUMENT);
        }
        // Clone the shared owner, then release the table lock before numerical work.
        let model = handles()
            .lock()
            .map_err(|_| INTERNAL_ERROR)?
            .models
            .get(&handle)
            .cloned()
            .ok_or(INVALID_HANDLE)?;
        if cols != model.width {
            return Err(INVALID_ARGUMENT);
        }
        if rows == 0 {
            return Ok(());
        }
        // SAFETY: `shape` checked the product, `buffer` checked alignment and
        // non-nullness, and the caller's C contract guarantees that `x` retains
        // provenance from one live, contiguous, readable C allocation. The
        // helper only borrows/copies it; allocation ownership stays with C.
        let x = unsafe { rows_from(x, rows, cols) };
        let predictions = model.predict(&x);
        if predictions.len() != rows {
            return Err(INTERNAL_ERROR);
        }
        // Commit output only after the complete prediction succeeds. The temporary
        // result owns disjoint memory, including when caller input aliases output.
        // SAFETY: `buffer(out, out_len)` guarantees alignment/non-nullness,
        // `rows <= out_len`, and the C caller guarantees writable output memory.
        // `predictions` owns a separate allocation, so the regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(predictions.as_ptr(), out, rows);
        }
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn jope_model_score(
    handle: u64,
    x: *const f64,
    x_len: usize,
    y: *const f64,
    y_len: usize,
    rows: usize,
    cols: usize,
    out_score: *mut f64,
) -> i32 {
    boundary(|| {
        if rows == 0
            || !shape(rows, cols, x_len)
            || y_len != rows
            || !buffer(x, x_len)
            || !buffer(y, y_len)
            || !buffer(out_score, 1)
        {
            return Err(INVALID_ARGUMENT);
        }
        let model = handles()
            .lock()
            .map_err(|_| INTERNAL_ERROR)?
            .models
            .get(&handle)
            .cloned()
            .ok_or(INVALID_HANDLE)?;
        if cols != model.width {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: `shape` checked the product, `buffer` checked alignment and
        // non-nullness, and the C caller supplies the live initialized array.
        // `rows_from` copies values; allocation ownership remains with C.
        let x = unsafe { rows_from(x, rows, cols) };
        // SAFETY: `y_len == rows > 0`; `buffer` checked alignment and
        // non-nullness. The C caller keeps this initialized allocation live;
        // `values_from` copies it without taking ownership.
        let y = unsafe { values_from(y, y_len) };
        let score = model.score(&x, &y);
        // SAFETY: `buffer(out_score, 1)` checked alignment/non-nullness; the
        // C caller guarantees one writable `f64` at this address.
        unsafe {
            out_score.write(score);
        }
        Ok(())
    })
}
#[no_mangle]
pub extern "C" fn jope_model_free(handle: u64) -> i32 {
    boundary(|| {
        let model = handles()
            .lock()
            .map_err(|_| INTERNAL_ERROR)?
            .models
            .remove(&handle)
            .ok_or(INVALID_HANDLE)?;
        drop(model);
        Ok(())
    })
}
