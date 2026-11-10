//! Fixed-size native build filesystem and persistent per-checkout build target selector.
//! Native Rust replacement for `native_build_pool.py` and `build_pool_target.py`.

use std::fs;
use std::os::unix::fs::{symlink, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cache_namespaces::is_symlink;
use crate::json::Json;
use crate::json_obj;
use crate::sha256::sha256_str;
use crate::sys::{
    atomic_write_str, base_dir, current_gid, current_uid, disk_usage, home_dir, hostname,
};

pub const CAPACITY: u64 = 40 * 1024 * 1024 * 1024;
pub const MOUNT_PATH: &str = "/mnt/jope-build-native";

fn pool_config() -> (PathBuf, &'static str, &'static str) {
    let host = hostname();
    if host == "xbabe3" {
        (
            PathBuf::from("/mnt/xbabe3-external/disk-steward-build-cache-v1.ext4"),
            "exfat",
            "/mnt/xbabe3-external",
        )
    } else {
        (
            PathBuf::from("/mnt/fast-scratch/disk-steward/disk-steward-build-cache-v1.ext4"),
            "ext4",
            "/mnt/fast-scratch",
        )
    }
}

fn run_cmd(args: &[&str]) -> Result<String, String> {
    let out = Command::new(args[0])
        .args(&args[1..])
        .output()
        .map_err(|e| format!("exec {}: {e}", args[0]))?;
    if !out.status.success() {
        return Err(format!(
            "{} failed: {}",
            args[0],
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn find_mount(path: &Path) -> Option<Json> {
    let out = Command::new("findmnt")
        .args([
            "--json",
            "--mountpoint",
            &path.to_string_lossy(),
            "-o",
            "TARGET,SOURCE,FSTYPE,UUID",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let parsed = Json::parse(&String::from_utf8_lossy(&out.stdout)).ok()?;
    parsed
        .get("filesystems")
        .and_then(|v| v.as_arr())
        .and_then(|a| a.first().cloned())
}

fn find_provider(path: &Path) -> Option<Json> {
    let out = Command::new("findmnt")
        .args([
            "--json",
            "--target",
            &path.to_string_lossy(),
            "-o",
            "TARGET,SOURCE,FSTYPE,UUID",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let parsed = Json::parse(&String::from_utf8_lossy(&out.stdout)).ok()?;
    parsed
        .get("filesystems")
        .and_then(|v| v.as_arr())
        .and_then(|a| a.first().cloned())
}

pub fn check_image(record: &Json) -> Result<(), String> {
    let (image, provider_kind, provider_mount) = pool_config();
    if record.get_str("status") != Some("formatted")
        || record.get_str("image") != Some(&image.to_string_lossy())
        || record.get_u64("capacity_bytes") != Some(CAPACITY)
    {
        return Err(
            "No completed initialization record; existing images are never reformatted automatically"
                .to_string(),
        );
    }
    let parent = image.parent().unwrap_or_else(|| Path::new("/"));
    let Some(provider) = find_provider(parent) else {
        return Err("External provider/image does not match the initialized pool".to_string());
    };
    let parent_dev = fs::metadata(parent).map(|m| m.dev()).unwrap_or(0);
    let root_dev = fs::metadata("/").map(|m| m.dev()).unwrap_or(0);
    let img_size = fs::metadata(&image).map(|m| m.len()).unwrap_or(0);
    if provider.get_str("fstype") != Some(provider_kind)
        || provider.get_str("target") != Some(provider_mount)
        || parent_dev == root_dev
        || is_symlink(&image)
        || img_size != CAPACITY
    {
        return Err("External provider/image does not match the initialized pool".to_string());
    }
    let uuid = run_cmd(&[
        "sudo",
        "-n",
        "blkid",
        "-p",
        "-s",
        "UUID",
        "-o",
        "value",
        &image.to_string_lossy(),
    ])?;
    if Some(uuid.as_str()) != record.get_str("uuid") {
        return Err("Image UUID changed".to_string());
    }
    Ok(())
}

pub fn check_mounted(record: &Json) -> Result<Json, String> {
    let (image, _, _) = pool_config();
    let mount = Path::new(MOUNT_PATH);
    let Some(mounted) = find_mount(mount) else {
        return Err("Native pool is not mounted with its recorded filesystem UUID".to_string());
    };
    if mounted.get_str("fstype") != Some("ext4") {
        return Err("Native pool is not mounted with its recorded filesystem UUID".to_string());
    }
    let src_dev = mounted.get_str("source").unwrap_or("");
    let mounted_uuid = run_cmd(&[
        "sudo", "-n", "blkid", "-p", "-s", "UUID", "-o", "value", src_dev,
    ])?;
    if Some(mounted_uuid.as_str()) != record.get_str("uuid") {
        return Err("Mounted filesystem UUID changed".to_string());
    }
    let backing = run_cmd(&[
        "sudo",
        "-n",
        "losetup",
        "--noheadings",
        "--output",
        "BACK-FILE",
        src_dev,
    ])?;
    if backing != image.to_string_lossy() {
        return Err("Loop device does not refer to the recorded image".to_string());
    }
    let stats = disk_usage(mount).map_err(|e| format!("statvfs pool: {e}"))?;
    let mount_dev = fs::metadata(mount).map(|m| m.dev()).unwrap_or(0);
    let parent_dev = image
        .parent()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.dev())
        .unwrap_or(0);
    if stats.total_capacity_bytes > CAPACITY || mount_dev == parent_dev {
        return Err("Pool capacity or filesystem isolation check failed".to_string());
    }
    Ok(json_obj! {
        "mount" => MOUNT_PATH,
        "uuid" => record.get_str("uuid").unwrap_or(""),
        "image_capacity_bytes" => CAPACITY,
        "filesystem_bytes" => stats.total_capacity_bytes,
        "available_bytes" => stats.available_bytes,
    })
}

pub fn run_native_pool(do_mount: bool) -> i32 {
    let state_path = base_dir().join("state/native-build-pool.json");
    let record = match Json::load_file(&state_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("disk-steward: cannot read {}: {e}", state_path.display());
            return 1;
        }
    };
    if let Err(e) = check_image(&record) {
        eprintln!("disk-steward: {e}");
        return 1;
    }
    let mount = Path::new(MOUNT_PATH);
    if do_mount && find_mount(mount).is_none() {
        let is_empty_dir = mount.is_dir()
            && !is_symlink(mount)
            && fs::read_dir(mount)
                .map(|mut rd| rd.next().is_none())
                .unwrap_or(false);
        if !is_empty_dir {
            eprintln!("disk-steward: Mountpoint must be an empty real directory");
            return 1;
        }
        let (image, _, _) = pool_config();
        let owner = format!("{}:{}", current_uid(), current_gid());
        if let Err(e) = run_cmd(&[
            "sudo",
            "-n",
            "mount",
            "-t",
            "ext4",
            "-o",
            "loop,nodev,nosuid",
            &image.to_string_lossy(),
            MOUNT_PATH,
        ])
        .and_then(|_| run_cmd(&["sudo", "-n", "chown", &owner, MOUNT_PATH]))
        .and_then(|_| run_cmd(&["chmod", "750", MOUNT_PATH]))
        {
            eprintln!("disk-steward: mount failed: {e}");
            return 1;
        }
    }
    match check_mounted(&record) {
        Ok(res) => {
            println!("{}", res.to_compact());
            0
        }
        Err(e) => {
            eprintln!("disk-steward: {e}");
            1
        }
    }
}

pub fn run_build_pool_target(source_arg: Option<&Path>, profile: &str) -> i32 {
    let state_path = base_dir().join("state/native-build-pool.json");
    let host = hostname();
    // On xbabe1 /mnt/jope-build-native is a dedicated physical NVMe partition (/dev/nvme1n1p1)
    if host != "xbabe1" || state_path.exists() {
        if state_path.exists() {
            let Ok(record) = Json::load_file(&state_path) else {
                eprintln!("disk-steward: unreadable native-build-pool.json");
                return 1;
            };
            if let Err(e) = check_image(&record).and_then(|_| check_mounted(&record)) {
                eprintln!("disk-steward: {e}");
                return 1;
            }
        }
    }
    let cwd = source_arg
        .map(|p| p.to_path_buf())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let Ok(toplevel) = run_cmd(&[
        "git",
        "-C",
        &cwd.to_string_lossy(),
        "rev-parse",
        "--show-toplevel",
    ]) else {
        eprintln!("disk-steward: git rev-parse --show-toplevel failed");
        return 1;
    };
    let Ok(source) = fs::canonicalize(&toplevel) else {
        eprintln!("disk-steward: canonicalize source failed");
        return 1;
    };
    let Ok(commit) = run_cmd(&[
        "git",
        "-C",
        &source.to_string_lossy(),
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]) else {
        eprintln!("disk-steward: git rev-parse HEAD^{{commit}} failed");
        return 1;
    };
    let cargo_rustc = home_dir().join(".cargo/bin/rustc");
    let rustc_bin = if cargo_rustc.is_file() {
        cargo_rustc.to_string_lossy().to_string()
    } else {
        "rustc".to_string()
    };
    let Ok(compiler_out) = Command::new(&rustc_bin)
        .arg("-vV")
        .current_dir(&source)
        .output()
    else {
        eprintln!("disk-steward: rustc -vV failed");
        return 1;
    };
    let compiler = String::from_utf8_lossy(&compiler_out.stdout).to_string();
    let toolchain = &sha256_str(&compiler)[..16];
    let lane = &sha256_str(&source.to_string_lossy())[..24];
    let mount = Path::new(MOUNT_PATH);
    let parent = mount.join(toolchain).join(profile).join(lane);
    if let Err(e) = fs::create_dir_all(&parent) {
        eprintln!("disk-steward: mkdir {}: {e}", parent.display());
        return 1;
    }
    let pointer = parent.join("source");
    if is_symlink(&pointer) {
        if fs::canonicalize(&pointer).ok().as_ref() != Some(&source) {
            eprintln!("disk-steward: Source pointer collision");
            return 1;
        }
    } else if pointer.exists() {
        eprintln!("disk-steward: Unexpected source pointer path");
        return 1;
    } else if let Err(e) = symlink(&source, &pointer) {
        eprintln!("disk-steward: symlink source pointer: {e}");
        return 1;
    }
    let target = parent.join("target");
    let _ = fs::create_dir_all(&target);
    if is_symlink(&target) || fs::canonicalize(&target).ok().as_ref() != Some(&target) {
        eprintln!("disk-steward: Target must remain on the mounted native filesystem");
        return 1;
    }
    let meta_obj = json_obj! {
        "source" => source.to_string_lossy().to_string(),
        "commit" => commit,
        "compiler" => compiler,
        "profile" => profile,
    };
    let _ = atomic_write_str(
        &parent.join("source-commit.json"),
        &format!("{}\n", meta_obj.to_pretty()),
    );
    println!("{}", target.display());
    0
}
