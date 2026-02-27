# RMCTS in lc0

Maintained by:
- Benjamin Howard (bhoward73@gmail.com, bjhowa3@idaccr.org)
- Keith Frankston (k.frankston@fastmail.com, k.frankston@idaccr.org)

The core RMCTS implementation in this tree is in `src/rmcts/src/c/RMCTS.cc`.
This branch integrates RMCTS as an experimental lc0 search mode so it can be
tested directly against classic search under comparable conditions.

---

## What this README covers

- How to run lc0 in RMCTS mode through UCI.
- How to run and use the `play` utility (`src/play.cc` target).
- How to compare RMCTS vs classic using the helper scripts.
- How to keep script/command arguments in one shared parameters file.

Reference docs:

- Original RMCTS paper (v1 context):
	https://github.com/bhoward73/rmcts/blob/main/paper/rmcts.pdf
- This branch's implementation-difference summary:
	`src/rmcts/RMCTS.v2.txt`

The shared parameters file for these workflows is:

- `src/rmcts/params.toml`

---

## Quick start

From repo root:

```bash
# Build (uses the safe CUDA/NVCC script if needed on your machine)
./build_cuda_safe.sh

# Optional: build play utility explicitly
ninja -C build/release play -j4

# Show UCI rmcts options
./build/release/lc0 rmcts --help

# Show play options
./build/release/play --help
```

If your build environment is sensitive to compiler/CUDA versions, use
`build_cuda_safe.sh` as documented in the top-level README.

---

## Shared parameters file (`src/rmcts/params.toml`)

All RMCTS helper scripts now accept a `--config` option and default to:

- `src/rmcts/params.toml`

Scripts using this config:

- `scripts/bot_match.py`
- `scripts/single_position_compare.py`
- `scripts/rmcts_plot.py`
- `scripts/rmcts_run.py`

### Structure

The file is organized into sections:

- `[common]` - common defaults (engine path and shared backend args)
- `[bot_match]` - defaults for paired bot matches
- `[single_position_compare]` - defaults for per-ply trace analysis
- `[rmcts_plot]` - plotting defaults
- `[commands]` - defaults for launcher commands (`rmcts_run.py`)

In each script, CLI flags still override config values.

Portability note:

- `params.toml` intentionally avoids machine-specific hardcoded network paths.
- For ONNX backends, set `--weights=...` in `shared_args` (or pass on CLI)
	before running comparison/play commands.

### GPU selection (single GPU vs multi-GPU)

Defaults in `src/rmcts/params.toml` are intentionally GPU-agnostic (no explicit
`gpu=<id>` in backend options), so they work out of the box on a machine with
only one NVIDIA GPU.

If you have multiple GPUs and want to pin workloads, set explicit GPU ids in
`single_position_compare` args, for example:

```toml
rmcts_args = "rmcts --rmcts-chunk-sims=128 --backend-opts=gpu=1,batch=16,steps=1"
classic_args = "classic --minibatch-size=136 --max-prefetch=136 --backend-opts=gpu=0,batch=136,steps=1"
policy_args = "policyhead --backend-opts=gpu=0,batch=32,steps=1"
```

---

## Running RMCTS via UCI (`lc0`)

### Direct usage

```bash
./build/release/lc0 rmcts --backend=onnx-trt --weights=weights/<your-net>.pb.gz
```

### Classic baseline

```bash
./build/release/lc0 classic --backend=onnx-trt --weights=weights/<your-net>.pb.gz
```

### Config-driven usage (recommended)

Use launcher:

```bash
python3 scripts/rmcts_run.py uci-rmcts
python3 scripts/rmcts_run.py uci-classic

# Validate config before running (checks ONNX + weights requirements)
python3 scripts/rmcts_run.py check-config
```

Both commands read `src/rmcts/params.toml` and expand configured args.
The launcher also performs an early check and exits with a clear error if an
ONNX backend is selected without `--weights`.

You can preview command expansion without execution:

```bash
python3 scripts/rmcts_run.py --dry-run uci-rmcts
```

---

## Using `play` (`src/play.cc`)

`play` is a console utility for interactive/human-play style sessions.
Built binary path is typically:

- `build/release/play`

Basic usage:

```bash
./build/release/play --weights=weights/<your-net>.pb.gz --backend=onnx-trt
```

Useful options:

- `--movetime-ms=N`
- `--move-overhead-ms=N`
- `--threads=N`
- `--minibatch-size=N`
- `--max-prefetch=N`
- `--max-half-moves=N`
- `--human`
- `--no-pause`

In interactive mode:

- Enter moves in UCI format (examples: `e2e4`, `e7e5`).
- Use `show policy` to inspect classic vs RMCTS root policy snapshot.
- Use `quit` / `q` / `exit` to leave.

Config-driven launcher invocation:

```bash
python3 scripts/rmcts_run.py play
```

---

## RMCTS vs classic comparison workflows

### 1) Paired bot matches (`scripts/bot_match.py`)

Purpose:

- Random start positions.
- Two games per start (color-swapped).
- Reports W/D/L and node totals for A vs B.

Default run (from params file):

```bash
python3 scripts/bot_match.py
```

Explicit config file:

```bash
python3 scripts/bot_match.py --config src/rmcts/params.toml
```

Equivalent launcher path:

```bash
python3 scripts/rmcts_run.py bot-match
```

### 2) Single-line per-ply comparison (`scripts/single_position_compare.py`)

Purpose:

- For each ply of one driven line, query RMCTS, classic, and policyhead.
- Save per-ply CSV with moves, values, and disagreement indicators.
- Save a plot for value trajectories and disagreement markers.

Analyze mode:

```bash
python3 scripts/single_position_compare.py --mode analyze
```

Overlay mode (compare RMCTS-driven vs classic-driven lines):

```bash
python3 scripts/single_position_compare.py --mode overlay \
	--overlay-rmcts-csv /tmp/rmcts_line.csv \
	--overlay-classic-csv /tmp/classic_line.csv
```

Launcher path:

```bash
python3 scripts/rmcts_run.py single-compare
```

### 3) RMCTS chunk-trace plotting (`scripts/rmcts_plot.py`)

Purpose:

- Plot posterior/Q/visit trajectories from RMCTS trace CSV output.

Usage with explicit CSV:

```bash
python3 scripts/rmcts_plot.py /path/to/rmcts_trace.csv --metric all
```

Or set `rmcts_plot.csv` in `params.toml` and run:

```bash
python3 scripts/rmcts_plot.py
```

Launcher path:

```bash
python3 scripts/rmcts_run.py plot
```

---

## Typical end-to-end experiment loop

1. Build:

```bash
./build_cuda_safe.sh
```

2. Edit experiment parameters in `src/rmcts/params.toml`.
	 - In particular, set a valid `--weights=...` path in `shared_args` for ONNX
		 backends.

3. Run paired match:

```bash
python3 scripts/rmcts_run.py bot-match
```

4. Run detailed single-line analysis:

```bash
python3 scripts/rmcts_run.py single-compare
```

5. Plot RMCTS trace CSV (if generated):

```bash
python3 scripts/rmcts_run.py plot -- /path/to/rmcts_trace.csv --metric all
```

---

## Notes on reproducibility

- Keep `seed` fixed in script sections when comparing revisions.
- Keep backend, weights, and movetime synchronized across RMCTS/classic.
- Prefer paired/color-swapped tests to reduce opening/color bias.
- For fair throughput comparisons, track node totals and wall-clock settings.

---

## Quick glossary (v1 paper -> lc0 RMCTS)

For the full comparison, see `src/rmcts/RMCTS.v2.txt`.

- **RMCTS prior policy (paper)** -> root prior from lc0 backend eval at current root.
- **RMCTS posterior policy (paper)** -> chunk/epoch-updated root policy used for move choice/logging.
- **Simulation budget N (paper)** -> operationally split by `rmcts-num-sims`, `rmcts-epochs`, and `rmcts-chunk-sims`.
- **Fixed-budget run (paper framing)** -> usually chunked execution with checkpointing and optional time-deadline stop.
- **Experiment timing loop (paper scripts)** -> lc0 UCI clock/movetime handling.
- **Standalone comparison harness (paper repo)** -> branch tools `bot_match.py`, `single_position_compare.py`, `rmcts_plot.py` with shared `params.toml`.

---

## Copyright

Copyright (c) 2025, Institute for Defense Analyses,
730 Glebe Rd, Alexandria, VA 22305-3086; 703-845-2500.

This material may be reproduced by or for the U.S. Government pursuant to all
applicable FAR and DFARS clauses.
