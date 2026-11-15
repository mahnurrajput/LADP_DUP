"""
bench_disk_read.py — LADP
=========================
Sequential read throughput of the feature-cache drive, measured with
**unbuffered** Windows I/O (FILE_FLAG_NO_BUFFERING), so the OS page cache
(up to ~18 GiB on this machine) cannot inflate the result.  Files are read
whole, one after another, in the order training would read a shard.

Read-only: nothing is written except the JSON report.

  python scripts/bench_disk_read.py --gb 20
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x1000, 0x2000, 0x8000, 0x04
INVALID = ctypes.c_void_p(-1).value

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
k32.VirtualAlloc.restype = wintypes.LPVOID
k32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
k32.CloseHandle.argtypes = [wintypes.HANDLE]


def read_unbuffered(path: Path, buf, buf_size: int) -> int:
    h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    if h == INVALID or h is None:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed: {path}")
    total, got = 0, wintypes.DWORD(0)
    try:
        while True:
            if not k32.ReadFile(h, buf, buf_size, ctypes.byref(got), None):
                raise OSError(ctypes.get_last_error(), f"ReadFile failed: {path}")
            if got.value == 0:
                break
            total += got.value
            if got.value < buf_size:
                break
    finally:
        k32.CloseHandle(h)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default=str(ROOT / "data" / "cache" / "coco_train"))
    ap.add_argument("--gb", type=float, default=20.0, help="how much to read (decimal GB)")
    ap.add_argument("--start", type=int, default=0, help="first file index")
    ap.add_argument("--out", default=str(ROOT / "results" / "throughput" / "bench_disk_read.json"))
    args = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("Windows only (uses FILE_FLAG_NO_BUFFERING).")

    cache = Path(args.cache_dir)
    files = sorted(cache.glob("features_*.pt"))[args.start:]
    buf_size = 8 * 1024 * 1024                       # 8 MiB, sector-aligned (VirtualAlloc is page-aligned)
    buf = k32.VirtualAlloc(None, buf_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    target = args.gb * 1e9
    n_bytes, n_files, t0, marks = 0, 0, time.perf_counter(), []
    try:
        for f in files:
            n_bytes += read_unbuffered(f, buf, buf_size)
            n_files += 1
            if n_files % 500 == 0:
                el = time.perf_counter() - t0
                marks.append({"files": n_files, "GB": n_bytes / 1e9, "MB_per_s": n_bytes / 1e6 / el})
                print(f"  {n_files} files, {n_bytes / 1e9:.1f} GB, {n_bytes / 1e6 / el:.1f} MB/s", flush=True)
            if n_bytes >= target:
                break
    finally:
        k32.VirtualFree(buf, 0, MEM_RELEASE)
    el = time.perf_counter() - t0
    rate = n_bytes / el
    full_bytes = 118_287 * (files[0].stat().st_size if files else 5_900_261)
    rep = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "cache_dir": str(cache),
           "method": "CreateFileW FILE_FLAG_NO_BUFFERING|SEQUENTIAL_SCAN, 8 MiB reads, whole files in sorted order",
           "files_read": n_files, "first_index": args.start, "bytes_read": n_bytes, "seconds": el,
           "MB_per_s": rate / 1e6, "files_per_s": n_files / el,
           "progress": marks,
           "full_cache_bytes": full_bytes,
           "est_full_epoch_read_min": full_bytes / rate / 60,
           "free_bytes_now": shutil.disk_usage(cache).free}
    print(json.dumps({k: v for k, v in rep.items() if k != "progress"}, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print("Report written to", args.out)


if __name__ == "__main__":
    main()
