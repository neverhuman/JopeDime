#!/usr/bin/env python3
"""Flock-serialized reader/writer for `/home/ubuntu/JopeDime/shared_CHAT.md` (`Wave H458`).

Guarantees:
  1. Byte-offset cursor tracking (`shared_chat_cursor_agy.json`) so `agy_apex_w10` reads 100% of peer posts
     without truncation before every write or milestone decision.
  2. Exclusive `fcntl.flock` on `/home/ubuntu/JopeDime/shared_CHAT.md.lock`, re-reading any concurrent bytes
     inside the lock before appending and calling `os.fsync`.
  3. Hard Git-staging guard helper (`assert_shared_chat_untracked`) so `shared_CHAT.md*` is never committed to a PR.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

DEFAULT_CHAT = Path("/home/ubuntu/JopeDime/shared_CHAT.md")
DEFAULT_LOCK = Path("/home/ubuntu/JopeDime/shared_CHAT.md.lock")
DEFAULT_CURSOR = Path("/home/ubuntu/jope-data/h458-flywheel/logs/shared_chat_cursor_agy.json")


def utc_hm() -> str:
    return time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_since_cursor(
    chat_path: Path = DEFAULT_CHAT,
    cursor_path: Path = DEFAULT_CURSOR,
    advance: bool = True,
) -> Dict[str, object]:
    raw = chat_path.read_bytes() if chat_path.is_file() else b""
    prev_off = 0
    if cursor_path.is_file():
        try:
            prev_off = int(json.loads(cursor_path.read_text(encoding="utf-8")).get("offset", 0))
        except Exception:
            prev_off = 0
    if prev_off < 0 or prev_off > len(raw):
        prev_off = 0
    new_bytes = raw[prev_off:]
    new_text = new_bytes.decode("utf-8", errors="replace")
    if advance:
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = cursor_path.with_suffix(".tmp.json")
        tmp_p.write_text(
            json.dumps({"offset": len(raw), "updated_utc": utc_iso()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_p, cursor_path)
    return {
        "chat_path": str(chat_path),
        "prev_offset": prev_off,
        "new_offset": len(raw),
        "delta_bytes": len(new_bytes),
        "new_text": new_text,
    }


def append_message_locked(
    sender: str,
    recipients: str,
    body: str,
    chat_path: Path = DEFAULT_CHAT,
    lock_path: Path = DEFAULT_LOCK,
    cursor_path: Path = DEFAULT_CURSOR,
    timeout_s: float = 30.0,
) -> Dict[str, object]:
    body_clean = body.strip()
    if not body_clean:
        raise ValueError("Refusing to post empty shared_CHAT message")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    t0 = time.monotonic()
    with open(lock_path, "r+", encoding="utf-8") as lf:
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if (time.monotonic() - t0) >= timeout_s:
                    raise TimeoutError(f"Timed out waiting {timeout_s}s for flock on {lock_path}")
                time.sleep(0.1)
        try:
            unread = read_since_cursor(chat_path=chat_path, cursor_path=cursor_path, advance=False)
            raw_cur = chat_path.read_bytes() if chat_path.is_file() else b""
            prefix_nl = "" if (not raw_cur or raw_cur.endswith(b"\n\n")) else ("\n" if raw_cur.endswith(b"\n") else "\n\n")
            stamp = utc_hm()
            formatted = f"{prefix_nl}[{stamp}] {sender} -> {recipients}: {body_clean}\n"
            with open(chat_path, "a", encoding="utf-8") as cf:
                cf.write(formatted)
                cf.flush()
                os.fsync(cf.fileno())
            after = read_since_cursor(chat_path=chat_path, cursor_path=cursor_path, advance=True)
            return {
                "posted_utc": stamp,
                "unread_delta_bytes_before_post": unread["delta_bytes"],
                "unread_before_post": unread["new_text"],
                "new_offset": after["new_offset"],
            }
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def assert_shared_chat_untracked(repo_dir: Path = Path("/home/ubuntu/JopeDime"), index_file: str | None = None) -> None:
    env = os.environ.copy()
    if index_file:
        env["GIT_INDEX_FILE"] = index_file
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-files", "--", "shared_CHAT.md", "shared_CHAT.md.lock"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if r.stdout.strip():
        raise RuntimeError(f"REFUSING: shared_CHAT file is tracked in index: {r.stdout.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat", default=str(DEFAULT_CHAT))
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    ap.add_argument("--cursor", default=str(DEFAULT_CURSOR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read")
    p_read.add_argument("--peek", action="store_true", help="Do not advance stored cursor")

    p_post = sub.add_parser("post")
    p_post.add_argument("--sender", default="agy_apex_w10")
    p_post.add_argument("--to", default="cl_h458, cl_fwriter, cl_mtc (cc Jepson)")
    p_post.add_argument("--msg-file", default="")
    p_post.add_argument("--msg", default="")

    a = ap.parse_args()
    chat_p, lock_p, cur_p = Path(a.chat), Path(a.lock), Path(a.cursor)

    if a.cmd == "read":
        res = read_since_cursor(chat_p, cur_p, advance=not a.peek)
        sys.stdout.write(
            json.dumps(
                {k: v for k, v in res.items() if k != "new_text"},
                sort_keys=True,
            )
            + "\n"
        )
        if res["new_text"]:
            sys.stdout.write(str(res["new_text"]))
            if not str(res["new_text"]).endswith("\n"):
                sys.stdout.write("\n")
    elif a.cmd == "post":
        text = Path(a.msg_file).read_text(encoding="utf-8") if a.msg_file else a.msg
        res = append_message_locked(a.sender, a.to, text, chat_path=chat_p, lock_path=lock_p, cursor_path=cur_p)
        if res["unread_delta_bytes_before_post"]:
            sys.stderr.write(f"[UNREAD BEFORE POST ({res['unread_delta_bytes_before_post']}B)]:\n{res['unread_before_post']}\n")
        print(json.dumps({k: v for k, v in res.items() if k != "unread_before_post"}, sort_keys=True))


if __name__ == "__main__":
    main()
