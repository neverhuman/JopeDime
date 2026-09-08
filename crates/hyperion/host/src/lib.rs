//! Host build of the preserved encoder. Only unrelated dataset/GP imports are
//! omitted; the encoder, sweep RNG, numerical kernels and CLI source are reused.
include!(concat!(env!("OUT_DIR"),"/modules.rs"));
#[path="../../../assets/source/hyperion-jain/crates/feat-core/src/progress.rs"]
pub mod progress;
#[path="../../../assets/source/hyperion-jain/crates/feat-core/src/rng.rs"]
pub mod rng;
#[path="../../../assets/source/hyperion-jain/crates/feat-core/src/hashing.rs"]
pub mod hashing;
pub mod data {
    #[derive(Clone)]
    pub struct Matrix {pub n:usize,pub c:usize,pub data:std::sync::Arc<Vec<f32>>}
    impl Matrix {
        pub fn new(data:Vec<f32>,n:usize,c:usize)->Self {assert_eq!(data.len(),n*c);Self{n,c,data:std::sync::Arc::new(data)}}
        pub fn col(&self,j:usize)->&[f32]{&self.data[j*self.n..(j+1)*self.n]}
        pub fn at(&self,row:usize,col:usize)->f32{if row<self.n&&col<self.c{self.data[col*self.n+row]}else{f32::NAN}}
    }
}
