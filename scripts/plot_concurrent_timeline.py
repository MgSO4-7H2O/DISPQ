#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(
        description="Plot concurrent workload query, insert, merge, and rebuild timelines."
    )
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-max", action="store_true")
    args = parser.parse_args()

    summary_path = args.summary_json
    stem = summary_path.stem
    timeseries_path = summary_path.with_name(f"{stem}_timeseries.csv")
    insert_path = summary_path.with_name(f"{stem}_insert.csv")
    events_path = summary_path.with_name(f"{stem}_events.csv")
    for path in (summary_path, timeseries_path, insert_path, events_path):
        if not path.is_file():
            parser.error(f"required input file not found: {path}")

    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    timeseries = pd.read_csv(timeseries_path)
    inserts = pd.read_csv(insert_path)
    events = pd.read_csv(events_path)

    query_workers = summary.get("query_workers", "?")
    query_threads = summary.get("query_threads_per_request", "?")
    insert_threads = summary.get("insert_threads", "?")
    maintenance_threads = summary.get("maintenance_threads", "?")
    target_vecps = summary.get("target_insert_vecps", "?")
    title = (f"Q{query_workers}x{query_threads} / I{insert_threads} / "
             f"M{maintenance_threads} / target={target_vecps}")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    qps_x = (timeseries["window_start_ms"] + timeseries["window_end_ms"]) / 2000.0
    axes[0].plot(qps_x, timeseries["query_qps"], color="tab:blue", label="Query QPS")
    axes[0].set_ylabel("Query QPS")

    latency_x = qps_x
    axes[1].plot(latency_x, timeseries["query_latency_p50_ms"], label="p50")
    axes[1].plot(latency_x, timeseries["query_latency_p95_ms"], label="p95")
    axes[1].plot(latency_x, timeseries["query_latency_p99_ms"], label="p99")
    if args.show_max:
        axes[1].scatter(latency_x, timeseries["query_latency_max_ms"],
                         s=12, alpha=0.55, label="max")
    axes[1].set_ylabel("Query latency (ms)")

    if not inserts.empty:
        insert_x = inserts["end_ms"] / 1000.0
        axes[2].plot(insert_x, inserts["batch_ms"], marker="o", label="batch_ms")
        axes[2].plot(insert_x, inserts["insert_ms"], marker=".", linestyle="--",
                     alpha=0.75, label="insert_ms")
    axes[2].set_ylabel("Insert latency (ms)")
    axes[2].set_xlabel("Elapsed time (s)")

    event_colors = {"merge": "tab:green", "rebuild": "tab:red"}
    has_rebuild_wait = False
    has_merge_commit = False
    has_commit_wait = False
    if not events.empty:
        for event in events.itertuples(index=False):
            event_type = str(event.type)
            color = event_colors.get(event_type, "gray")
            start_s = float(event.start_ms) / 1000.0
            end_s = float(event.end_ms) / 1000.0
            for axis in axes:
                if end_s > start_s:
                    axis.axvspan(start_s, end_s, color=color,
                                 alpha=0.12 if event_type == "merge" else 0.16)
                if (event_type == "merge" and
                        event.commit_done_ms > event.commit_lock_acquired_ms):
                    axis.axvspan(float(event.commit_lock_acquired_ms) / 1000.0,
                                 float(event.commit_done_ms) / 1000.0,
                                 color="darkgreen", alpha=0.30)
                    has_merge_commit = True
                if (event_type == "merge" and
                        event.commit_wait_start_ms > event.start_ms):
                    axis.axvline(float(event.commit_wait_start_ms) / 1000.0,
                                 color="darkgreen", linestyle="--", alpha=0.7)
                    has_commit_wait = True
                if event_type == "rebuild" and event.request_ms < event.start_ms:
                    axis.axvline(float(event.request_ms) / 1000.0,
                                 color=color, linestyle=":", alpha=0.65)
                    has_rebuild_wait = True

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")

    event_handles = [
        Patch(facecolor=event_colors["merge"], alpha=0.25, label="Merge"),
        Patch(facecolor=event_colors["rebuild"], alpha=0.25, label="Global Rebuild"),
    ]
    if has_merge_commit:
        event_handles.append(Patch(facecolor="darkgreen", alpha=0.4,
                                   label="Merge commit critical section"))
    if has_commit_wait:
        event_handles.append(Line2D([0], [0], color="darkgreen", linestyle="--",
                                    label="Commit lock wait start"))
    if has_rebuild_wait:
        event_handles.append(Line2D([0], [0], color=event_colors["rebuild"],
                                    linestyle=":", label="Rebuild requested"))
    fig.legend(handles=event_handles, loc="upper right", frameon=True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = args.output or summary_path.with_name(f"{stem}_timeline.png")
    fig.savefig(output_path, dpi=160)
    print(output_path)


if __name__ == "__main__":
    main()
