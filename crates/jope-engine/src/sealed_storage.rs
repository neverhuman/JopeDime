//! Content-verified sealed storage shared by resident ranks over private IPC.
use anyhow::{ensure,Context,Result};
use std::{fs,io::{BufRead,BufReader,Read,Write},os::{fd::{AsRawFd,FromRawFd,OwnedFd},unix::net::UnixStream},path::Path};

pub(crate) fn send_fd(stream:&UnixStream,fd:i32)->Result<()> {
    let mut marker=[b'L'];let mut iov=libc::iovec{iov_base:marker.as_mut_ptr().cast(),iov_len:1};
    let mut control=[0usize;8];let mut msg:libc::msghdr=unsafe{std::mem::zeroed()};
    msg.msg_iov=&mut iov;msg.msg_iovlen=1;msg.msg_control=control.as_mut_ptr().cast();
    msg.msg_controllen=unsafe{libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32)} as usize;
    unsafe {
        let header=libc::CMSG_FIRSTHDR(&msg);(*header).cmsg_level=libc::SOL_SOCKET;(*header).cmsg_type=libc::SCM_RIGHTS;
        (*header).cmsg_len=libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize;
        std::ptr::write_unaligned(libc::CMSG_DATA(header).cast::<i32>(),fd);
        ensure!(libc::sendmsg(stream.as_raw_fd(),&msg,libc::MSG_NOSIGNAL)==1,"GPU lease transfer failed");
    }
    Ok(())
}
pub(crate) fn receive_fd(stream:&UnixStream)->Result<OwnedFd> {
    let mut marker=[0u8];let mut iov=libc::iovec{iov_base:marker.as_mut_ptr().cast(),iov_len:1};
    let mut control=[0usize;8];let mut msg:libc::msghdr=unsafe{std::mem::zeroed()};
    msg.msg_iov=&mut iov;msg.msg_iovlen=1;msg.msg_control=control.as_mut_ptr().cast();msg.msg_controllen=std::mem::size_of_val(&control);
    unsafe {
        ensure!(libc::recvmsg(stream.as_raw_fd(),&mut msg,libc::MSG_CMSG_CLOEXEC)==1 && marker[0]==b'L'
            && msg.msg_flags & libc::MSG_CTRUNC == 0,"GPU lease transfer missing/truncated");
        let header=libc::CMSG_FIRSTHDR(&msg);
        ensure!(!header.is_null() && (*header).cmsg_level==libc::SOL_SOCKET && (*header).cmsg_type==libc::SCM_RIGHTS
            && (*header).cmsg_len==libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize,"invalid GPU lease descriptor");
        Ok(OwnedFd::from_raw_fd(std::ptr::read_unaligned(libc::CMSG_DATA(header).cast::<i32>())))
    }
}

/// The private cache is the verifier. Ranks accept only its fully sealed fd,
/// exact content identity and dimensions; mutable source paths are never mmaped.
pub fn shared(path:&Path, expected:&str, length:usize)->Result<Option<fs::File>> {
    let Some(socket)=std::env::var_os("JOPE_TEACHING_CACHE") else {return Ok(None);};
    let mut stream=UnixStream::connect(socket)?;
    let request=serde_json::json!({"path":path.canonicalize()?,"sha256":expected,"bytes":length});
    stream.write_all(&serde_json::to_vec(&request)?)?;stream.write_all(b"\n")?;
    let file=fs::File::from(receive_fd(&stream)?);
    let mut response=String::new();BufReader::new(stream).take(65536).read_line(&mut response)?;
    let response:serde_json::Value=serde_json::from_str(&response)?;
    ensure!(response["sha256"]==expected && response["bytes"]==length
        && response["epoch"].as_str().context("cache restart identity absent")?.len()==32,
        "shared teaching identity changed");
    let seals=libc::F_SEAL_WRITE|libc::F_SEAL_GROW|libc::F_SEAL_SHRINK|libc::F_SEAL_SEAL;
    ensure!(file.metadata()?.len()==length as u64
        && unsafe{libc::fcntl(file.as_raw_fd(),libc::F_GET_SEALS)} & seals == seals,
        "shared teaching storage is not fully sealed");
    Ok(Some(file))
}
