#!/usr/bin/env python3
"""Summarize the four SIFT1M PQ-layout runs into JSON, CSV, and Markdown."""

import csv
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("RESULT_ROOT", ROOT / "result" / "sift1M" / "drifted"))
OUT = Path(__file__).resolve().parent
RUNS = [
    ("sift1m_layout_aos_record", "AoS codebook", "record-major codes", False, False),
    ("sift1m_layout_dim_record", "dimension-major codebook", "record-major codes", True, False),
    ("sift1m_layout_aos_subq", "AoS codebook", "subquantizer-major codes", False, True),
    ("sift1m_layout_dim_subq", "dimension-major codebook", "subquantizer-major codes", True, True),
]


def mib(value):
    return value / (1024.0**2)


def gib(value):
    return value / (1024.0**3)


def parse_run(run_name, codebook_layout, code_layout, codebook_dim, codes_subq):
    run_dir = RESULTS / run_name
    metrics_doc = json.loads((run_dir / "online_eval.json").read_text())
    memory_doc = json.loads((run_dir / "memory_trace.json").read_text())
    metrics = metrics_doc["metrics"]
    summary = metrics_doc["summary"]
    final_event = next(e for e in reversed(memory_doc["events"]) if e["stage"] == "final")
    components = final_event["components"]
    index_bytes = sum(v for k, v in components.items() if k.endswith(".ivf.total"))
    codebook_soa_bytes = sum(
        v for k, v in components.items() if k.endswith(".pq_codebooks_soa")
    )
    pq_codes_aos_bytes = sum(
        v for k, v in components.items() if k.endswith(".compact_pq_codes")
    )
    pq_codes_soa_bytes = sum(
        v for k, v in components.items() if k.endswith(".compact_pq_codes_soa")
    )

    log = (run_dir / "run.log").read_text(errors="replace")
    merge_profiles = re.findall(r"\[MERGE_PROFILE\].*?pq_code_assignment_us=([0-9.eE+-]+)", log)
    global_rebuild_encodes = re.findall(
        r"\[GLOBAL_REBUILD_ADD_PROFILE\].*?encode_us=([0-9.eE+-]+)", log
    )
    merge_encode_us = sum(float(x) for x in merge_profiles)
    global_rebuild_encode_us = sum(float(x) for x in global_rebuild_encodes)
    total_pq_assignment_seconds = (
        metrics.get("initial_main_pq_encode_us", 0.0)
        + metrics.get("initial_delta_seed_pq_encode_us", 0.0)
        + 1000.0 * metrics.get("update_insert_encode_ms", 0.0)
        + merge_encode_us
        + global_rebuild_encode_us
    ) / 1e6

    return {
        "run": run_name,
        "codebook_layout": codebook_layout,
        "pq_code_layout": code_layout,
        "pq_codebook_dimension_major": codebook_dim,
        "pq_codes_subquantizer_major": codes_subq,
        "recall_at_10": metrics.get("recall@10", 0.0),
        "latency_ms": metrics.get("latency_ms", 0.0),
        "query_qps": metrics.get("query_qps", 0.0),
        "pq_lut_build_us_per_query": metrics.get("avg_pq_lut_build_us", 0.0),
        "pq_adc_scan_us_per_query": metrics.get("avg_pq_adc_scan_us", 0.0),
        "initial_main_pq_encode_us": metrics.get("initial_main_pq_encode_us", 0.0),
        "initial_delta_seed_pq_encode_us": metrics.get("initial_delta_seed_pq_encode_us", 0.0),
        "stream_insert_pq_encode_ms": metrics.get("update_insert_encode_ms", 0.0),
        "merge_pq_assignment_us": merge_encode_us,
        "global_rebuild_pq_encode_us": global_rebuild_encode_us,
        "pq_assignment_total_s": total_pq_assignment_seconds,
        "merge_count": summary.get("merge_count", 0),
        "global_rebuild_count": summary.get("global_rebuild_count", 0),
        "stream_update_throughput_vecps": metrics.get("throughput", 0.0),
        "maintenance_total_ms": metrics.get("total_maintenance_ms", 0.0),
        "index_total_mib": mib(index_bytes),
        "codebook_dimension_major_cache_mib": mib(codebook_soa_bytes),
        "pq_codes_record_major_mib": mib(pq_codes_aos_bytes),
        "pq_codes_subquantizer_major_cache_mib": mib(pq_codes_soa_bytes),
        "peak_accounted_mib": mib(memory_doc["summary"]["max_accounted_bytes"]),
        "peak_rss_gib": gib(memory_doc["summary"]["max_process_peak_rss_bytes"]),
        "final_process_rss_gib": gib(summary.get("process_rss_bytes", 0.0)),
        "omp_threads": metrics_doc["params"].get("omp_max_threads", 0),
        "snapshot_span": metrics_doc["params"].get("snapshot_span", 0),
    }


def main_effect(rows, field, factor):
    low = [r[field] for r in rows if not r[factor]]
    high = [r[field] for r in rows if r[factor]]
    low_avg = sum(low) / len(low)
    high_avg = sum(high) / len(high)
    return {"low_mean": low_avg, "high_mean": high_avg, "difference": high_avg - low_avg}


def write_outputs(rows):
    json_path = OUT / "summary.json"
    csv_path = OUT / "summary.csv"
    md_path = OUT / "SUMMARY.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    effects = {
        "dimension_major_on_lut_us": main_effect(rows, "pq_lut_build_us_per_query", "pq_codebook_dimension_major"),
        "subquantizer_major_on_adc_us": main_effect(rows, "pq_adc_scan_us_per_query", "pq_codes_subquantizer_major"),
        "dimension_major_on_assignment_s": main_effect(rows, "pq_assignment_total_s", "pq_codebook_dimension_major"),
        "subquantizer_major_on_qps": main_effect(rows, "query_qps", "pq_codes_subquantizer_major"),
    }
    baseline = next(r for r in rows if not r["pq_codebook_dimension_major"] and
                    not r["pq_codes_subquantizer_major"])
    aligned = next(r for r in rows if r["pq_codebook_dimension_major"] and
                   r["pq_codes_subquantizer_major"])
    combined_qps_change = 100.0 * (aligned["query_qps"] / baseline["query_qps"] - 1.0)
    combined_latency_change = 100.0 * (aligned["latency_ms"] / baseline["latency_ms"] - 1.0)
    combined_assignment_speedup = baseline["pq_assignment_total_s"] / aligned["pq_assignment_total_s"]
    combined_rss_change = 100.0 * (aligned["peak_rss_gib"] / baseline["peak_rss_gib"] - 1.0)
    codebook_lut_reduction = 100.0 * (
        1.0 - effects["dimension_major_on_lut_us"]["high_mean"] /
        effects["dimension_major_on_lut_us"]["low_mean"]
    )
    codebook_assignment_reduction = 100.0 * (
        1.0 - effects["dimension_major_on_assignment_s"]["high_mean"] /
        effects["dimension_major_on_assignment_s"]["low_mean"]
    )
    subq_adc_reduction = 100.0 * (
        1.0 - effects["subquantizer_major_on_adc_us"]["high_mean"] /
        effects["subquantizer_major_on_adc_us"]["low_mean"]
    )
    subq_qps_change = 100.0 * (
        effects["subquantizer_major_on_qps"]["high_mean"] /
        effects["subquantizer_major_on_qps"]["low_mean"] - 1.0
    )

    metadata_path = RESULTS / RUNS[0][0] / "run_meta.txt"
    run_metadata = {}
    if metadata_path.is_file():
        for line in metadata_path.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                run_metadata[key] = value
    if {"omp_num_threads", "numa_node", "cpu_list"}.issubset(run_metadata):
        run_conditions = (
            f"Each setting used the same seed, {run_metadata['omp_num_threads']} OpenMP threads "
            f"pinned to NUMA node {run_metadata['numa_node']} CPUs `{run_metadata['cpu_list']}`, "
            f"with memory bound to NUMA node {run_metadata['numa_node']}."
        )
    else:
        run_conditions = (
            "All four settings used the same seed and thread/NUMA binding; see each run's "
            "`run_meta.txt` for machine-specific details."
        )

    lines = [
        "# SIFT1M PQ layout ablation",
        "",
        f"This is a 2×2 layout ablation over the drifted SIFT1M stream. {run_conditions} "
        "Each setting used 200k initial Main rows, a 10k Delta training window, 10k insertion batches, "
        "the configured online maintenance settings, and 10k queries. The snapshot interval is 1M, "
        "so the stream is evaluated before ingestion and at the final 1M-row state.",
        "",
        "| Codebook layout | PQ-code layout | Recall@10 | QPS | App avg latency (ms) | LUT µs/query | ADC µs/query | PQ assignment (s) | Index MiB | Peak RSS GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['codebook_layout']} | {r['pq_code_layout']} | {r['recall_at_10']:.6f} | {r['query_qps']:.0f} | {r['latency_ms']:.6f} | {r['pq_lut_build_us_per_query']:.2f} | {r['pq_adc_scan_us_per_query']:.2f} | {r['pq_assignment_total_s']:.4f} | {r['index_total_mib']:.1f} | {r['peak_rss_gib']:.3f} |"
        )
    lines += [
        "",
        "## Layout memory",
        "",
        "| Variant | Dimension-major codebook cache (MiB) | Subquantizer-major PQ-code cache (MiB) |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['codebook_dimension_major_cache_mib']:.2f} | {r['pq_codes_subquantizer_major_cache_mib']:.2f} |"
        )
    lines += [
        "",
        "## Factor means",
        "",
        f"- Dimension-major codebook mean LUT time: {effects['dimension_major_on_lut_us']['low_mean']:.2f} → {effects['dimension_major_on_lut_us']['high_mean']:.2f} µs/query ({codebook_lut_reduction:.1f}% lower).",
        f"- Dimension-major codebook mean measured PQ-assignment time: {effects['dimension_major_on_assignment_s']['low_mean']:.4f} → {effects['dimension_major_on_assignment_s']['high_mean']:.4f} s ({codebook_assignment_reduction:.1f}% lower).",
        f"- Subquantizer-major code mean ADC scan time: {effects['subquantizer_major_on_adc_us']['low_mean']:.2f} → {effects['subquantizer_major_on_adc_us']['high_mean']:.2f} µs/query ({subq_adc_reduction:.1f}% lower).",
        f"- Subquantizer-major code mean QPS: {effects['subquantizer_major_on_qps']['low_mean']:.0f} → {effects['subquantizer_major_on_qps']['high_mean']:.0f} ({subq_qps_change:+.1f}%).",
        f"- Both layouts vs. neither: QPS {baseline['query_qps']:.0f} → {aligned['query_qps']:.0f} ({combined_qps_change:+.1f}%), app average latency {baseline['latency_ms']:.6f} → {aligned['latency_ms']:.6f} ms ({combined_latency_change:+.1f}%), measured PQ-assignment speedup {combined_assignment_speedup:.2f}×, peak RSS {combined_rss_change:+.1f}%; Recall@10 remains {aligned['recall_at_10']:.6f} in all four runs.",
        "",
        "## Reading the measurements",
        "",
        "All four runs completed 790k streamed insertions, 38 merges, and 2 Global Rebuilds. `LUT µs/query` and `ADC µs/query` are averages of per-query stage timers summed over the searched routes. The ADC timer covers candidate code access, distance accumulation, and top-k heap updates; it excludes routing, LUT construction, whitening, and exact reranking. `App avg latency` and QPS use the evaluator's parallel batch wall time. The code does not record an independent per-request P99, so its snapshot-series P99 is not reported as query P99. `PQ assignment` sums the instrumented initial Main and Delta-seed encoding, stream insertion encoding, merge re-encoding, and Global Rebuild encoding timers. Peak RSS includes the evaluator, transformed data, and transient merge state; the two cache columns isolate the additional layout storage reported by the index.",
        "",
        "Each configuration was run once. Treat timing differences as preliminary; repeat each variant at least three times before presenting confidence intervals or making paper-level claims. The paper's Table 4(a) reports both layouts together on SIFT100M drift-fast, so these SIFT1M results are a separate 2×2 decomposition and are not directly comparable in scale or workload.",
        "",
        f"See [run.sh](run.sh), [configs](configs), [summary.csv](summary.csv), and the per-run JSON/logs under `{RESULTS}`. The paper's layout design is Section 5.1 (PDF p. 8); combined layout results are Table 4(a) (PDF p. 11) in [PVLDB27_DiSPQ.pdf](../../PVLDB27_DiSPQ.pdf).",
        "",
    ]
    md_path.write_text("\n".join(lines))


def main():
    if len(sys.argv) > 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} [RESULT_ROOT]")
    global RESULTS
    if len(sys.argv) == 2:
        RESULTS = Path(sys.argv[1]).resolve()
    elif "RESULT_ROOT" in os.environ:
        RESULTS = Path(os.environ["RESULT_ROOT"]).resolve()

    rows = [parse_run(*run) for run in RUNS]
    write_outputs(rows)
    print(f"Wrote {OUT / 'SUMMARY.md'}")
    print(f"Wrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
