#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time


def read_status(pid):
    out = {"rss_kb": 0, "hwm_kb": 0}
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    out["hwm_kb"] = int(line.split()[1])
    except FileNotFoundError:
        pass
    return out


def children(pid):
    path = f"/proc/{pid}/task/{pid}/children"
    try:
        direct = [int(x) for x in open(path, "r", encoding="utf-8").read().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    out = []
    for child in direct:
        out.append(child)
        out.extend(children(child))
    return out


def sample_tree(pid):
    pids = [pid] + children(pid)
    rss = hwm = 0
    live = []
    for p in pids:
        st = read_status(p)
        if st["rss_kb"] or st["hwm_kb"]:
            live.append(p)
            rss += st["rss_kb"]
            hwm += st["hwm_kb"]
    return live, rss, hwm


def main():
    ap = argparse.ArgumentParser(description="Run a command and sample process-tree RSS.")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--csv", default="rss_trace.csv")
    ap.add_argument("--summary", default="rss_summary.json")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        ap.error("missing command after --")

    start = time.time()
    proc = subprocess.Popen(args.command)
    peak_rss = peak_hwm = 0
    samples = 0

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_ms", "pid", "pids", "rss_kb", "hwm_kb"])
        while proc.poll() is None:
            live, rss, hwm = sample_tree(proc.pid)
            peak_rss = max(peak_rss, rss)
            peak_hwm = max(peak_hwm, hwm)
            samples += 1
            w.writerow([int((time.time() - start) * 1000), proc.pid, " ".join(map(str, live)), rss, hwm])
            f.flush()
            time.sleep(max(args.interval, 0.05))
        live, rss, hwm = sample_tree(proc.pid)
        peak_rss = max(peak_rss, rss)
        peak_hwm = max(peak_hwm, hwm)
        w.writerow([int((time.time() - start) * 1000), proc.pid, " ".join(map(str, live)), rss, hwm])

    summary = {
        "command": args.command,
        "returncode": proc.returncode,
        "elapsed_ms": int((time.time() - start) * 1000),
        "samples": samples,
        "peak_rss_kb": peak_rss,
        "peak_hwm_kb": peak_hwm,
        "csv": args.csv,
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
