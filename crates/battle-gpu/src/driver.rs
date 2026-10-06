//! Minimal raw CUDA *driver API* FFI: `dlopen("libcuda.so.1")` + `dlsym` of
//! exactly the symbols the ridge screen needs. No CUDA crates, no build-time
//! CUDA dependency — hosts without a driver get a clean `Err`/`None`, never a
//! panic or abort, and the caller's seam falls back to the exact CPU path.
//!
//! Unsafe surface policy: every raw call is wrapped in a checked helper that
//! converts a nonzero `CUresult` into `Err(String)` (via `cuGetErrorName`);
//! device memory and the context/module/stream live in RAII wrappers or in
//! the process-lifetime backend singleton. No pointer arithmetic outside this
//! module.

use libc::{c_char, c_int, c_uint, c_void, dlclose, dlerror, dlopen, dlsym, RTLD_LAZY, RTLD_LOCAL};
use std::ffi::CStr;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT — the only attribute we query.
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;

macro_rules! api {
    ($($name:ident: fn($($arg:ty),*) -> CUresult;)*) => {
        /// Resolved driver entry points. One `dlsym` per field at load time.
        /// Field names deliberately match the CUDA symbol names.
        #[allow(non_snake_case)]
        pub struct Api {
            $(pub $name: unsafe extern "C" fn($($arg),*) -> CUresult,)*
        }

        impl Api {
            #[cfg(test)]
            fn test_api() -> Api {
                Api {
                    $($name: {
                        unsafe extern "C" fn reject_call($(_: $arg),*) -> CUresult { 1 }
                        reject_call
                    },)*
                }
            }

            fn load(handle: *mut c_void) -> Result<Api, String> {
                Ok(Api {
                    $($name: {
                        let symbol = concat!(stringify!($name), "\0");
                        let ptr = unsafe { dlsym(handle, symbol.as_ptr() as *const c_char) };
                        if ptr.is_null() {
                            return Err(format!("libcuda: missing symbol {}", stringify!($name)));
                        }
                        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> CUresult>(ptr) }
                    },)*
                })
            }
        }
    };
}

api! {
    cuInit: fn(c_uint) -> CUresult;
    cuDriverGetVersion: fn(*mut c_int) -> CUresult;
    cuDeviceGet: fn(*mut CUdevice, c_int) -> CUresult;
    cuDeviceGetName: fn(*mut c_char, c_int, CUdevice) -> CUresult;
    cuDeviceGetAttribute: fn(*mut c_int, c_int, CUdevice) -> CUresult;
    cuDeviceTotalMem_v2: fn(*mut usize, CUdevice) -> CUresult;
    cuCtxCreate_v2: fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
    cuCtxSetCurrent: fn(CUcontext) -> CUresult;
    cuCtxDestroy_v2: fn(CUcontext) -> CUresult;
    cuModuleLoadData: fn(*mut CUmodule, *const c_void) -> CUresult;
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize) -> CUresult;
    cuMemFree_v2: fn(CUdeviceptr) -> CUresult;
    cuMemcpyHtoD_v2: fn(CUdeviceptr, *const c_void, usize) -> CUresult;
    cuMemcpyDtoH_v2: fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
    cuMemGetInfo_v2: fn(*mut usize, *mut usize) -> CUresult;
    cuLaunchKernel: fn(
        CUfunction,
        c_uint, c_uint, c_uint,
        c_uint, c_uint, c_uint,
        c_uint, CUstream,
        *mut *mut c_void,
        *mut *mut c_void
    ) -> CUresult;
    cuStreamCreate: fn(*mut CUstream, c_uint) -> CUresult;
    cuStreamSynchronize: fn(CUstream) -> CUresult;
    cuStreamDestroy: fn(CUstream) -> CUresult;
    cuGetErrorName: fn(CUresult, *mut *const c_char) -> CUresult;
}

/// Owned `dlopen` handle; closed on drop (only on backend-init failure — the
/// process-lifetime singleton never drops it).
pub struct Library {
    handle: *mut c_void,
}

// The raw handle is only ever used behind the backend Mutex.
unsafe impl Send for Library {}

impl Library {
    pub fn open(path: &str) -> Result<Library, String> {
        let name = format!("{path}\0");
        unsafe { dlerror() }; // clear any stale error
        let handle = unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_LAZY | RTLD_LOCAL) };
        if handle.is_null() {
            let detail = unsafe {
                let err = dlerror();
                if err.is_null() {
                    "unknown dlopen failure".to_string()
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                }
            };
            return Err(format!("dlopen({path}) failed: {detail}"));
        }
        Ok(Library { handle })
    }

    pub fn api(&self) -> Result<Api, String> {
        Api::load(self.handle)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { dlclose(self.handle) };
    }
}

impl Api {
    /// Human-readable name for a nonzero CUresult.
    pub fn error_name(&self, code: CUresult) -> String {
        let mut name: *const c_char = std::ptr::null();
        let status = unsafe { (self.cuGetErrorName)(code, &mut name) };
        if status == CUDA_SUCCESS && !name.is_null() {
            unsafe { CStr::from_ptr(name).to_string_lossy().into_owned() }
        } else {
            format!("CUDA_ERROR_{code}")
        }
    }

    pub fn check(&self, code: CUresult, what: &str) -> Result<(), String> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: {}", self.error_name(code)))
        }
    }
}

/// Device allocation that frees itself; kept alive for the duration of one
/// wave. Frees are best-effort (an error during free cannot be surfaced from
/// Drop and is intentionally ignored — the context outlives the process).
pub struct DeviceBuffer<'api> {
    api: &'api Api,
    ptr: CUdeviceptr,
    // Only allocation construction can set the capacity used by transfer checks.
    bytes: usize,
}

mod scalar {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for u64 {}
}

/// Padding-free scalar types for which every bit pattern is a valid value.
/// Sealing prevents a device download from constructing invalid Rust values.
pub trait DeviceScalar: Copy + scalar::Sealed {}
impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}
impl DeviceScalar for i32 {}
impl DeviceScalar for u64 {}

impl<'api> DeviceBuffer<'api> {
    pub fn alloc(api: &'api Api, bytes: usize) -> Result<DeviceBuffer<'api>, String> {
        let mut ptr: CUdeviceptr = 0;
        // Zero-sized allocs are illegal in the driver API; round up to 8.
        let bytes = bytes.max(8);
        api.check(
            unsafe { (api.cuMemAlloc_v2)(&mut ptr, bytes) },
            "cuMemAlloc",
        )?;
        Ok(DeviceBuffer { api, ptr, bytes })
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn upload<T: DeviceScalar>(&self, host: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        if bytes == 0 {
            return Ok(());
        }
        if bytes > self.bytes {
            return Err(format!(
                "device transfer of {bytes} bytes exceeds allocation of {} bytes",
                self.bytes
            ));
        }
        // SAFETY: these initialized scalars have no padding, and the checked byte count
        // fits the live device allocation. The driver borrows the host slice for this call.
        self.api.check(
            unsafe { (self.api.cuMemcpyHtoD_v2)(self.ptr, host.as_ptr() as *const c_void, bytes) },
            "cuMemcpyHtoD",
        )
    }

    pub fn download<T: DeviceScalar>(&self, host: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        if bytes == 0 {
            return Ok(());
        }
        if bytes > self.bytes {
            return Err(format!(
                "device transfer of {bytes} bytes exceeds allocation of {} bytes",
                self.bytes
            ));
        }
        // SAFETY: the checked count fits the device allocation and the live mutable
        // host slice. DeviceScalar permits every downloaded bit pattern.
        self.api.check(
            unsafe {
                (self.api.cuMemcpyDtoH_v2)(host.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            },
            "cuMemcpyDtoH",
        )
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.cuMemFree_v2)(self.ptr) };
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());
    static COPY_CALLS: AtomicUsize = AtomicUsize::new(0);
    static COPY_BYTES: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn free(_: CUdeviceptr) -> CUresult {
        CUDA_SUCCESS
    }

    unsafe extern "C" fn upload(_: CUdeviceptr, _: *const c_void, bytes: usize) -> CUresult {
        COPY_CALLS.fetch_add(1, Ordering::SeqCst);
        COPY_BYTES.store(bytes, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    unsafe extern "C" fn download(host: *mut c_void, _: CUdeviceptr, bytes: usize) -> CUresult {
        COPY_CALLS.fetch_add(1, Ordering::SeqCst);
        COPY_BYTES.store(bytes, Ordering::SeqCst);
        // SAFETY: these tests pass live mutable numeric slices covering the copy size.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0xff, bytes) };
        CUDA_SUCCESS
    }

    fn api() -> Api {
        COPY_CALLS.store(0, Ordering::SeqCst);
        COPY_BYTES.store(0, Ordering::SeqCst);
        let mut api = Api::test_api();
        api.cuMemFree_v2 = free;
        api.cuMemcpyHtoD_v2 = upload;
        api.cuMemcpyDtoH_v2 = download;
        api
    }

    #[test]
    fn oversized_upload_is_refused_before_the_driver_call() {
        let _lock = LOCK.lock().unwrap();
        let api = api();
        let buffer = DeviceBuffer {
            api: &api,
            ptr: 42,
            bytes: 8,
        };
        let error = buffer.upload(&[1u64, 2]).unwrap_err();
        assert!(error.contains("16 bytes exceeds allocation of 8 bytes"));
        assert_eq!(COPY_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn oversized_download_is_refused_without_modifying_the_host() {
        let _lock = LOCK.lock().unwrap();
        let api = api();
        let buffer = DeviceBuffer {
            api: &api,
            ptr: 42,
            bytes: 8,
        };
        let mut host = [3u64, 4];
        let error = buffer.download(&mut host).unwrap_err();
        assert!(error.contains("16 bytes exceeds allocation of 8 bytes"));
        assert_eq!(COPY_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(host, [3, 4]);
    }

    #[test]
    fn exact_size_and_smaller_transfers_execute_the_checked_driver_calls() {
        let _lock = LOCK.lock().unwrap();
        let api = api();
        let buffer = DeviceBuffer {
            api: &api,
            ptr: 42,
            bytes: 8,
        };
        buffer.upload(&[1f64]).unwrap();
        assert_eq!(COPY_BYTES.load(Ordering::SeqCst), 8);
        buffer.upload(&[1f32]).unwrap();
        assert_eq!(COPY_BYTES.load(Ordering::SeqCst), 4);
        buffer.upload(&[1i32]).unwrap();
        let mut host = [0f64];
        buffer.download(&mut host).unwrap();
        assert_eq!(host[0].to_bits(), u64::MAX);
        assert_eq!(COPY_BYTES.load(Ordering::SeqCst), 8);
        assert_eq!(COPY_CALLS.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn empty_transfers_do_not_call_the_driver() {
        let _lock = LOCK.lock().unwrap();
        let api = api();
        let buffer = DeviceBuffer {
            api: &api,
            ptr: 42,
            bytes: 0,
        };
        buffer.upload(&[] as &[f32]).unwrap();
        buffer.download(&mut [] as &mut [f64]).unwrap();
        assert_eq!(COPY_CALLS.load(Ordering::SeqCst), 0);
    }
}
