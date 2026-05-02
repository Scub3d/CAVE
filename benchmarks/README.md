# Benchmarks

PowerShell scripts under `scripts/` run search-mode performance sweeps.
Reference results for the project's primary hardware (dual Quadro RTX 8000)
live in `reference/`.

The original snapshot was taken 2026-05-01 right before `/build` was
gitignored, then carried over from the old repo's `bench_archive/`.

## Layout

```
benchmarks/
├── scripts/
│   ├── debug/      # bench_*.ps1 to run against build/Debug/Cave.exe
│   └── release/    # bench_*.ps1 to run against build/Release/Cave.exe
├── reference/
│   ├── debug/      # historical CSV results from Debug runs
│   └── release/    # historical CSV results from Release runs
└── README.md
```

CSV schema: one row per `(workgroup, partition)` config; columns
`WallSeconds, Viable, Possible, Unviable, Total, TimedOut`.

## Running a sweep

Build Release first. Then from the repo root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/scripts/release/<sweep>.ps1
```

Per-chunk output appears under `data/search/<run>/` (gitignored). Each
script writes a results CSV plus a progress log; commit only the CSV (and
only if it's a new reference worth archiving).

## Adding a new bench

1. Copy an existing script as a starting point.
2. Adjust the parameter sweep (workgroup sizes, partition counts, etc.).
3. Output paths should write to `data/search/<descriptive_name>/` so they
   land in the gitignored area.
4. Commit your script to `scripts/<flavor>/` and the small `_results.csv`
   to `reference/<flavor>/`. Do NOT commit the per-rule `*.jsonl` files —
   they're large and regenerable.

## Notable historical runs

| File | What it covers |
|------|----------------|
| `reference/debug/bench_results.csv` | First chunks-per-config sweep (Debug, p=1,2,4,8,16,32 at 25³ and 49³ FE K=2). |
| `reference/debug/bench_extended_results.csv` | Extension to p=16 and p=32 of the original sweep. |
| `reference/debug/bench_grand_results_DEBUG.csv` | Grand cross-product (workgroup × partitions, Debug). Truncated by descriptor-pool OOM at p=128 — fixed in `deviceContext.cpp` later. |
| `reference/release/bench_grand_results.csv` | Release-mode follow-up; only 1 row (canceled to investigate shared-mem reduction). |

Per-run progress logs and the huge per-chunk `*.jsonl` outputs were
intentionally omitted — they're regenerable and aggregate-redundant with
the CSVs.

## When to re-archive

If a perf-tuning session produces multiple new result CSVs or distinct
methodology scripts, drop the artifacts into a dated subfolder here
(e.g. `reference/2026-06-15/`) so prior sessions stay readable.
