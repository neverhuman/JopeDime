pub(crate) struct RuntimeThreads {
    pub(crate) threads: usize,
}

pub(crate) fn init_runtime_threads(_name: &str, requested: usize) -> RuntimeThreads {
    RuntimeThreads { threads: requested }
}
