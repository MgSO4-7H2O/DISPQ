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
    progress_path = summary_path.with_name(f"{stem}_progress.csv")
    insert_path = summary_path.with_name(f"{stem}_insert.csv")
    events_path = summary_path.with_name(f"{stem}_events.csv")
    for path in (summary_path, progress_path, insert_path, events_path):
        if not path.is_file():
            parser.error(f"required input file not found: {path}")

    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    progress = pd.read_csv(progress_path)
    inserts = pd.read_csv(insert_path)
    events = pd.read_csv(events_path)

    query_workers = summary.get("query_workers", "?")
    query_threads = summary.get("query_threads_per_request", "?")
    insert_threads = summary.get("insert_threads", "?")
    maintenance_threads = summary.get("maintenance_threads", "?")
    target_vecps = summary.get("target_insert_vecps", "?")
    title = (f"Q{query_workers}x{query_threads} / I{insert_threads} / "
             f"M{maintenance_threads} / target={target_vecps}")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 12))
    progress_x = progress["insert_progress_pct"]
    axes[0].plot(progress_x, progress["recall_at_k"], color="tab:purple",
                 marker="o", label="Recall@k")
    axes[0].set_ylabel("Recall@k")

    axes[1].plot(progress_x, progress["query_qps"], color="tab:blue", marker="o",
                 label="Query QPS")
    axes[1].set_ylabel("Query QPS")

    axes[2].plot(progress_x, progress["query_latency_p50_ms"], label="p50")
    axes[2].plot(progress_x, progress["query_latency_p95_ms"], label="p95")
    axes[2].plot(progress_x, progress["query_latency_p99_ms"], label="p99")
    if args.show_max:
        axes[2].scatter(progress_x, progress["query_latency_max_ms"],
                        s=12, alpha=0.55, label="max")
    axes[2].set_ylabel("Query latency (ms)")

    if not inserts.empty:
        axes[3].plot(progress_x, progress["batch_ms"], marker="o", label="batch_ms")
        axes[3].plot(progress_x, progress["insert_ms"], marker=".", linestyle="--",
                     alpha=0.75, label="insert_ms")
    axes[3].set_ylabel("Insert latency (ms)")
    axes[3].set_xlabel("Dynamic Insert Progress (%)")

    insert_rows = list(inserts.itertuples(index=False))
    insert_progress = progress["insert_progress_pct"].tolist()

    def time_to_progress(timestamp_ms):
        if not insert_rows:
            return 0.0
        timestamp_ms = float(timestamp_ms)
        first_start = float(insert_rows[0].start_ms)
        if timestamp_ms <= first_start:
            return float(insert_progress[0])
        for index, batch in enumerate(insert_rows):
            start_ms = float(batch.start_ms)
            end_ms = float(batch.end_ms)
            current_progress = float(insert_progress[index])
            previous_progress = (float(insert_progress[index - 1])
                                 if index > 0 else current_progress)
            if timestamp_ms <= end_ms:
                if timestamp_ms <= start_ms or end_ms <= start_ms:
                    return previous_progress
                fraction = min(1.0, max(0.0,
                    (timestamp_ms - start_ms) / (end_ms - start_ms)))
                return previous_progress + fraction * (current_progress - previous_progress)
            if index + 1 < len(insert_rows) and \
                    timestamp_ms < float(insert_rows[index + 1].start_ms):
                return current_progress
        return float(insert_progress[-1])

    event_colors = {"merge": "tab:green", "rebuild": "tab:red"}
    has_rebuild_wait = False
    has_merge_commit = False
    has_commit_wait = False
    if not events.empty:
        for event in events.itertuples(index=False):
            event_type = str(event.type)
            color = event_colors.get(event_type, "gray")
            request_progress = time_to_progress(event.request_ms)
            start_progress = time_to_progress(event.start_ms)
            commit_lock_progress = time_to_progress(event.commit_lock_acquired_ms)
            commit_done_progress = time_to_progress(event.commit_done_ms)
            end_progress = time_to_progress(event.end_ms)
            for axis in axes:
                if end_progress > start_progress:
                    axis.axvspan(start_progress, end_progress, color=color,
                                 alpha=0.12 if event_type == "merge" else 0.16)
                elif float(event.end_ms) > float(event.start_ms):
                    axis.axvline(start_progress, color=color, alpha=0.35, linewidth=2)
                if (event_type == "merge" and
                        event.commit_done_ms > event.commit_lock_acquired_ms):
                    if commit_done_progress > commit_lock_progress:
                        axis.axvspan(commit_lock_progress, commit_done_progress,
                                     color="darkgreen", alpha=0.30)
                    else:
                        axis.axvline(commit_lock_progress, color="darkgreen",
                                     alpha=0.75, linewidth=3)
                    has_merge_commit = True
                if (event_type == "merge" and
                        event.commit_wait_start_ms > event.start_ms):
                    axis.axvline(time_to_progress(event.commit_wait_start_ms),
                                 color="darkgreen", linestyle="--", alpha=0.7)
                    has_commit_wait = True
                if event_type == "rebuild" and event.request_ms < event.start_ms:
                    axis.axvline(request_progress,
                                 color=color, linestyle=":", alpha=0.65)
                    has_rebuild_wait = True

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")
        axis.set_xlim(0, 100)

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
