//! The `jope` binary's counting global allocator: the system allocator, plus a report of every
//! allocation and release to [`jope_runtime::meter`], so resource-aware objectives can read a
//! candidate's peak heap growth (`fit_peak_heap_bytes`, `predict_peak_heap_bytes`).
//!
//! The only `unsafe` here is the `GlobalAlloc` contract itself; every method forwards unchanged
//! to [`System`] and reports sizes through safe atomics that never allocate.
use std::alloc::{GlobalAlloc, Layout, System};

pub struct Counting;

// SAFETY: every method delegates to `System`, which upholds the `GlobalAlloc` contract; the
// added bookkeeping only reads `layout.size()` and updates atomics, and never allocates.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's `layout` obligations are passed through unchanged.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            jope_runtime::meter::note_alloc(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `alloc`.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            jope_runtime::meter::note_alloc(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator (hence `System`) with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        jope_runtime::meter::note_dealloc(layout.size());
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr`/`layout` come from this allocator and `new_size` meets the caller's
        // contract; both are forwarded unchanged.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            jope_runtime::meter::note_dealloc(layout.size());
            jope_runtime::meter::note_alloc(new_size);
        }
        moved
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;
