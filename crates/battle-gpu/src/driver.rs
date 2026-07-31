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
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

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

    pub fn upload<T: Copy>(&self, host: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        if bytes == 0 {
            return Ok(());
        }
        debug_assert!(bytes <= self.bytes);
        self.api.check(
            unsafe { (self.api.cuMemcpyHtoD_v2)(self.ptr, host.as_ptr() as *const c_void, bytes) },
            "cuMemcpyHtoD",
        )
    }

    pub fn download<T: Copy>(&self, host: &mut [T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(host);
        if bytes == 0 {
            return Ok(());
        }
        debug_assert!(bytes <= self.bytes);
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
