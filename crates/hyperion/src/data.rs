//! Column-major float matrix — the 12-line `Matrix` the preserved host build
//! (`parallel/native/hyperion-host/src/lib.rs`) defined in place of feat-core's data layer.
//! Column `j` occupies `data[j*n .. j*n + n]`.
use std::sync::Arc;

#[derive(Clone)]
pub struct Matrix {
    pub n: usize,
    pub c: usize,
    pub data: Arc<Vec<f32>>,
}

impl Matrix {
    pub fn new(data: Vec<f32>, n: usize, c: usize) -> Self {
        assert_eq!(data.len(), n * c);
        Self {
            n,
            c,
            data: Arc::new(data),
        }
    }
    pub fn col(&self, j: usize) -> &[f32] {
        &self.data[j * self.n..(j + 1) * self.n]
    }
    pub fn at(&self, row: usize, col: usize) -> f32 {
        if row < self.n && col < self.c {
            self.data[col * self.n + row]
        } else {
            f32::NAN
        }
    }
}
