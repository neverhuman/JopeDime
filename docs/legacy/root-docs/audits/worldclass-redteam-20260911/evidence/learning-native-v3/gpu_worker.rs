//! Resident CUDA ranks. Task clients transfer their existing GPU lease by fd;
//! disconnecting a client terminates only its worker, releasing that lease.
use crate::manifest::*;
use crate::sealed_storage::{send_fd,receive_fd};
use anyhow::{ensure,Context,Result};
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::{fs,io::{BufRead,BufReader,Read,Write},os::{fd::{AsRawFd},
    unix::{fs::{MetadataExt,PermissionsExt},net::{UnixListener,UnixStream}}},path::{Path,PathBuf},
    sync::{Arc,atomic::{AtomicBool,Ordering}}};

#[derive(Clone,Serialize,Deserialize)]
#[serde(tag="kind")]
pub enum Task {
    Train {corpus:PathBuf,run:PathBuf,launch:PathBuf,world:usize,micro:usize,steps:usize,
        checkpoint_steps:usize,resume:bool,initialize:Option<PathBuf>},
    Proposal {corpus:PathBuf,checkpoint:PathBuf,output:PathBuf,world:usize,cuts:usize,cut_start:usize,shuffled:bool},
    Monitor {corpus:PathBuf,checkpoint:PathBuf,panel:PathBuf,output:PathBuf},
}
impl Task {
    fn receipt(&self,rank:usize)->PathBuf {match self {
        Self::Train{launch,..}=>launch.join(format!("complete-{rank}.json")),
        Self::Proposal{output,..}=>output.join(format!("complete-{rank}.json")),
        Self::Monitor{output,..}=>output.clone(),
    }}
    fn run(&self,root:&Path,rank:usize)->Result<()> {match self {
        Self::Train{corpus,run,launch,world,micro,steps,checkpoint_steps,resume,initialize}=>
            crate::training::rank(root,corpus,run,launch,rank,*world,*micro,*steps,*checkpoint_steps,*resume,initialize.as_deref()),
        Self::Proposal{corpus,checkpoint,output,world,cuts,cut_start,shuffled}=>
            crate::proposals::rank(root,corpus,checkpoint,output,rank,*world,*cuts,*cut_start,*shuffled),
        Self::Monitor{corpus,checkpoint,panel,output}=>crate::monitoring::evaluate(root,corpus,checkpoint,panel,output),
    }}
}

pub fn enabled()->bool {
    std::env::var_os("JOPE_GPU_POOL").is_some() && std::env::var_os("JOPE_GPU_LEASE_FD").is_some()
        && std::env::var_os("JOPE_GPU_WORKER_ACTIVE").is_none()
}

pub fn inherit_lease(command:&mut std::process::Command,lease:&fs::File)->Result<()> {
    if std::env::var_os("JOPE_GPU_POOL").is_some() {
        ensure!(unsafe {libc::fcntl(lease.as_raw_fd(),libc::F_SETFD,0)}==0,"cannot inherit GPU lease");
        command.env("JOPE_GPU_LEASE_FD",lease.as_raw_fd().to_string());
    }
    Ok(())
}


fn wait_for_rank(pool:&Path,rank:usize)->Result<()> {
    let started=std::time::Instant::now();
    loop {
        let pool_state=read_json::<Value>(&pool.join("pool.json")).unwrap_or_default();
        let state=read_json::<Value>(&pool.join(format!("rank-{rank}.json"))).unwrap_or_default();
        let pid=state["pid"].as_i64().unwrap_or(0) as i32;
        if pid>1 && state["status"]=="idle" && pool_state["pids"][rank.to_string()]==pid
            && unsafe{libc::kill(pid,0)}==0 {return Ok(());}
        ensure!(started.elapsed().as_secs()<30,"resident rank {rank} did not recover within thirty seconds");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

pub fn submit(root:&Path,rank:usize,task:Task)->Result<()> {
    let pool=PathBuf::from(std::env::var_os("JOPE_GPU_POOL").context("resident pool absent")?);
    let lease=std::env::var("JOPE_GPU_LEASE_FD")?.parse::<i32>()?;
    let mut env=serde_json::Map::new();
    for key in ["JOPE_CHAMPION_SNAPSHOT","JOPE_GP_EXECUTABLE","JOPE_EXPERIMENT_DIR"] {
        if let Ok(value)=std::env::var(key) {env.insert(key.into(),json!(value));}
    }
    let request=json!({"root":root,"rank":rank,"task":task,"env":env,
        "executable_sha256":file_hash(&std::env::current_exe()?)?});
    wait_for_rank(&pool,rank)?;
    let mut stream=UnixStream::connect(pool.join(format!("rank-{rank}.sock")))?;
    send_fd(&stream,lease)?;
    stream.write_all(&serde_json::to_vec(&request)?)?;stream.write_all(b"\n")?;
    let mut response=String::new();BufReader::new(stream).take(1024*1024).read_line(&mut response)?;
    let result:Value=serde_json::from_str(&response)?;
    ensure!(result["request_sha256"]==hash(&serde_json::to_vec(&request)?),"resident task response changed");
    ensure!(result["success"]==true,"resident GPU task failed: {}",result["error"]);
    ensure!(file_hash(&task.receipt(rank))?==result["receipt_sha256"],"resident task receipt changed");
    Ok(())
}

pub fn serve(root:&Path,pool:&Path,rank:usize)->Result<()> {
    ensure!(rank<8 && pool.starts_with(root.join("runs")),"invalid resident worker namespace");
    fs::create_dir_all(pool)?;fs::set_permissions(pool,fs::Permissions::from_mode(0o700))?;
    let guard=fs::OpenOptions::new().create(true).truncate(false).write(true).open(pool.join(format!("rank-{rank}.lock")))?;
    ensure!(unsafe{libc::flock(guard.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"resident rank already owned");
    let socket=pool.join(format!("rank-{rank}.sock"));
    if socket.exists(){fs::remove_file(&socket)?;}
    let listener=UnixListener::bind(&socket)?;
    let executable_sha=file_hash(&std::env::current_exe()?)?;
    std::env::set_var("JOPE_GPU_WORKER_ACTIVE","1");
    crate::neural::configure_runtime();
    // Initialize CUDA lazily under the first transferred task lease. A
    // replacement rank must be able to accept its parent's still-held lease;
    // acquiring a new one here would deadlock communicator recovery.
    write_json(&pool.join(format!("rank-{rank}.json")),&json!({"rank":rank,"status":"idle",
        "pid":std::process::id(),"timestamp":crate::telemetry::now()}))?;
    for incoming in listener.incoming() {
        write_json(&pool.join(format!("rank-{rank}.json")),&json!({"rank":rank,"status":"idle",
            "pid":std::process::id(),"timestamp":crate::telemetry::now()}))?;
        let mut stream=incoming?;
        let lease=receive_fd(&stream)?;
        let file=fs::File::from(lease);let expected=root.join("runs/encoding-service.lock").metadata()?;
        let actual=file.metadata()?;
        ensure!(expected.dev()==actual.dev() && expected.ino()==actual.ino()
            && unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)}==0,"task did not transfer the exclusive GPU lease");
        let mut line=String::new();BufReader::new(stream.try_clone()?).take(1024*1024).read_line(&mut line)?;
        let request:Value=serde_json::from_str(&line)?;
        ensure!(request["root"]==json!(root) && request["rank"]==rank
            && request["executable_sha256"]==executable_sha,"resident task root/rank/executable identity changed");
        let task:Task=serde_json::from_value(request["task"].clone())?;
        let id=hash(&serde_json::to_vec(&request)?);let directory=pool.join("tasks").join(&id);
        fs::create_dir_all(&directory)?;
        let intent=directory.join("request.json");
        if intent.exists(){ensure!(read_json::<Value>(&intent)?==request,"resident task intent changed");}
        else {write_json(&intent,&request)?;}
        let receipt=directory.join("complete.json");
        if receipt.is_file() {
            let result:Value=read_json(&receipt)?;
            ensure!(file_hash(&task.receipt(rank))?==result["receipt_sha256"],"committed resident task changed");
            stream.write_all(&serde_json::to_vec(&result)?)?;stream.write_all(b"\n")?;continue;
        }
        for key in ["JOPE_CHAMPION_SNAPSHOT","JOPE_GP_EXECUTABLE","JOPE_EXPERIMENT_DIR"] {
            if let Some(value)=request["env"][key].as_str(){std::env::set_var(key,value);}else{std::env::remove_var(key);}
        }
        std::env::set_var("JOPE_GPU_WORKER_LEASE_FD",file.as_raw_fd().to_string());
        write_json(&pool.join(format!("rank-{rank}.json")),&json!({"rank":rank,"status":"busy",
            "pid":std::process::id(),"request_sha256":id,"task":task,"timestamp":crate::telemetry::now()}))?;
        let done=Arc::new(AtomicBool::new(false));let observed=Arc::clone(&done);let watching=stream.try_clone()?;
        let watcher=std::thread::spawn(move||{
            while !observed.load(Ordering::Acquire){
                let mut poll=libc::pollfd{fd:watching.as_raw_fd(),events:libc::POLLRDHUP,revents:0};
                unsafe{libc::poll(&mut poll,1,100);}
                if poll.revents & (libc::POLLRDHUP|libc::POLLHUP|libc::POLLERR)!=0 && !observed.load(Ordering::Acquire){
                    // No live task client remains to supervise collectives. The
                    // pool supervisor restarts this rank from durable receipts.
                    unsafe{libc::_exit(75);}
                }
            }
        });
        let result=task.run(root,rank);
        done.store(true,Ordering::Release);let _=watcher.join();
        std::env::remove_var("JOPE_GPU_WORKER_LEASE_FD");
        let response=json!({"request_sha256":id,"success":result.is_ok(),"error":result.err().map(|e|format!("{e:#}")),
            "receipt_sha256":if task.receipt(rank).is_file(){Some(file_hash(&task.receipt(rank))?)}else{None}});
        if response["success"]==true{write_json(&receipt,&response)?;}
        else{write_json(&directory.join(format!("failure-{}.json",crate::telemetry::now())),&response)?;}
        stream.write_all(&serde_json::to_vec(&response)?)?;stream.write_all(b"\n")?;
        write_json(&pool.join(format!("rank-{rank}.json")),&json!({"rank":rank,"status":"idle",
            "pid":std::process::id(),"timestamp":crate::telemetry::now()}))?;
        ensure!(response["success"]==true,"failed task requires a fresh CUDA worker");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transferred_lease_survives_sender_and_releases_with_worker() {
        let d=tempfile::tempdir().unwrap();let p=d.path().join("lease");
        let lease=fs::OpenOptions::new().create(true).truncate(false).write(true).open(&p).unwrap();
        assert_eq!(unsafe{libc::flock(lease.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)},0);
        let (sender,receiver)=UnixStream::pair().unwrap();send_fd(&sender,lease.as_raw_fd()).unwrap();
        let received=receive_fd(&receiver).unwrap();drop(lease);
        let competing=fs::OpenOptions::new().write(true).open(p).unwrap();
        assert_ne!(unsafe{libc::flock(competing.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)},0);
        drop(received);
        assert_eq!(unsafe{libc::flock(competing.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)},0);
    }
}
