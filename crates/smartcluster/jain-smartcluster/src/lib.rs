#![deny(unsafe_code)]

mod client;
mod framing;
pub mod jope;
mod stream;

pub use client::{
    job_subscription, CancelReceipt, Client, ClientConfig, DispatchError, JobHandle, JobMonitor,
    JobStatus, WorkloadDispatcher, DEFAULT_SOCKET_PATH,
};

pub use framing::{FrameCodec, FrameError, ABSOLUTE_MAX_FRAME_BYTES, DEFAULT_MAX_FRAME_BYTES};
pub use jain_smartcluster_core::*;
pub use stream::{AttachSender, Attachment, EventStream};
