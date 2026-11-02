use super::*;

/// An injected remote fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The request never reaches the store.
    Disconnect,
    /// The store applies the request, then the response is lost.
    ResponseLost,
}

/// A [`Remote`] that injects faults by call number and records every request, so tests can
/// assert order and count requests (failed ones included).
pub struct Faulty<'a> {
    pub inner: &'a dyn Remote,
    /// `put_immutable` call index (0-based) to fault.
    pub put_faults: RefCell<BTreeMap<usize, Fault>>,
    /// `put_pointer` call index (0-based) to fault.
    pub pointer_faults: RefCell<BTreeMap<usize, Fault>>,
    /// `(operation, key, outcome)` for every request.
    pub log: RefCell<Vec<(String, String, String)>>,
    puts: RefCell<usize>,
    pointers: RefCell<usize>,
}

impl<'a> Faulty<'a> {
    pub fn new(inner: &'a dyn Remote) -> Faulty<'a> {
        Faulty {
            inner,
            put_faults: RefCell::default(),
            pointer_faults: RefCell::default(),
            log: RefCell::default(),
            puts: RefCell::new(0),
            pointers: RefCell::new(0),
        }
    }

    pub fn fault_put(self, call: usize, fault: Fault) -> Self {
        self.put_faults.borrow_mut().insert(call, fault);
        self
    }

    pub fn fault_pointer(self, call: usize, fault: Fault) -> Self {
        self.pointer_faults.borrow_mut().insert(call, fault);
        self
    }

    /// Keys of successful or failed requests of `op`, in order.
    pub fn keys(&self, op: &str) -> Vec<String> {
        self.log
            .borrow()
            .iter()
            .filter(|(o, _, _)| o == op)
            .map(|(_, k, _)| k.clone())
            .collect()
    }

    fn note(&self, op: &str, key: &str, outcome: &str) {
        self.log
            .borrow_mut()
            .push((op.into(), key.into(), outcome.into()));
    }

    fn faulted<T>(
        &self,
        op: &str,
        key: &str,
        fault: Option<Fault>,
        call: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match fault {
            Some(Fault::Disconnect) => {
                self.note(op, key, "disconnect");
                bail!("injected disconnect: {op} {key}")
            }
            Some(Fault::ResponseLost) => {
                let applied = call();
                self.note(
                    op,
                    key,
                    if applied.is_ok() {
                        "applied-response-lost"
                    } else {
                        "failed-response-lost"
                    },
                );
                bail!("injected response loss: {op} {key}")
            }
            None => {
                let result = call();
                self.note(op, key, if result.is_ok() { "ok" } else { "error" });
                result
            }
        }
    }
}

impl Remote for Faulty<'_> {
    fn url(&self) -> &str {
        self.inner.url()
    }

    fn put_immutable(&self, key: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let n = {
            let mut n = self.puts.borrow_mut();
            *n += 1;
            *n - 1
        };
        let fault = self.put_faults.borrow_mut().remove(&n);
        self.faulted("put", key, fault, || self.inner.put_immutable(key, bytes))
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let result = self.inner.get(key);
        self.note("get", key, if result.is_ok() { "ok" } else { "error" });
        result
    }

    fn get_observed(&self, key: &str) -> Result<Option<ObservedObject>> {
        let result = self.inner.get_observed(key);
        self.note("get", key, if result.is_ok() { "ok" } else { "error" });
        result
    }

    fn put_pointer(
        &self,
        key: &str,
        bytes: &[u8],
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        let n = {
            let mut n = self.pointers.borrow_mut();
            *n += 1;
            *n - 1
        };
        let fault = self.pointer_faults.borrow_mut().remove(&n);
        self.faulted("pointer", key, fault, || {
            self.inner.put_pointer(key, bytes, observed)
        })
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let result = self.inner.list(prefix);
        self.note("list", prefix, if result.is_ok() { "ok" } else { "error" });
        result
    }
}
