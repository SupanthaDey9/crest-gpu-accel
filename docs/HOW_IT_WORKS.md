# How it works

The patches change four things in CREST main at commit cfdc301f. Everything else is upstream code.

## 1. Thread layout and host code (always on)

- **Thread environment.** `scripts/crest-env.sh` sets the following:
  - a static OpenMP schedule;
  - threads bound to cores;
  - a 256 MB thread stack;
  - the glibc thread-stack cache;
  - one OpenBLAS thread.

  The default schedule (dynamic, chunk 1) had hidden almost all thread scaling of the SCC. On one EPYC 7502 node, MTD became 2.18x faster per energy and gradient with the same binary.
- **Rewritten loops.** 43 places in tblite, mctc-lib, multicharge, dftd4 and simple-dftd3 built 3x3 outer products with Fortran `spread`. That allocates temporaries in inner loops, so they are now explicit loops with the same operand order. A single-thread energy and gradient became 11% faster, and stays bitwise identical.

## 2. One GPU for the trajectories (`tblite_solver = "cuda"`, `CREST_ATM_GPU=1`)

- **SCC eigensolve.** During MTD and MD, each SCC eigensolve goes through a small C++ bridge (`src/calculator/cuda/crest_cuda_bridge.cpp`) to cuSOLVER `dsygvd`; cuBLAS builds the density matrix. Every OpenMP worker has its own CUDA stream and buffers, so the six MTD solves overlap on one device.
- **Three-body term.** The D4 Axilrod-Teller-Muto dispersion term runs in a deterministic FP64 kernel (`src/cuda_atm/`). It has a fixed reduction order and no atomics, and is compiled with `--fmad=false`. It matches the CPU routine to 4.4e-18 Eh/bohr.
- **CPU fallback.** Any CUDA error switches the process back to the CPU routine.
- **Optimisation passes stay on the CPU.** `CREST_OPT_SOLVER=cpu` keeps the CPU solver, and the ATM phase switch in `crest_oloop` keeps the CPU dispersion. The GPU was slower there, and CPU results can be checked bit for bit.

## 3. Distributed optimisation passes (`CREST_DIST_OLOOP=<config>`)

For every optimisation pass of at least `min_structures` structures, the CREST process acts as a driver:

1. **Write the pass.** It writes the exact binary coordinates into a pass folder. The folder is named by a hash of the settings, the binary's sha256 and the input, so a requeued driver finds and reuses finished chunks.
2. **Submit workers.** It submits chunks of `chunk` structures as one Slurm array (`slurm/dist_worker.sbatch`, `--export=NONE`). Each worker runs the same binary, refuses mismatched settings, and writes an atomic result file with a checksum, its binary hash and its CPU model.
3. **Check and merge.** It recomputes `verify` structures from workers on its own CPU model, which must agree bitwise, and `verify_other` from other CPU models, which must agree to 1e-7 Eh and 1e-4 Å. It then merges the results in input order through the in-process code path. A same-model mismatch sends the whole pass back in-process; an other-model failure recomputes those chunks.

| `dist_oloop.conf` key | meaning | default |
|---|---|---|
| `workdir` | absolute folder for pass data (required) | |
| `worker_script` | path of `dist_worker.sbatch` (required if `submit` is not 0) | |
| `worker_env` | file sourced by each worker (modules, threads) | none |
| `sbatch_args` | extra sbatch options for worker arrays | none |
| `worker_threads` | `-T` of a worker | 32 |
| `chunk` | structures per array task | 128 |
| `throttle` | maximum concurrent tasks (`%N`) | 40 |
| `poll` | seconds between `squeue` checks (minimum 30) | 60 |
| `min_structures` | smallest pass that is distributed | 1000 |
| `verify` / `verify_other` | structures re-checked by the driver | 32 / same as `verify` |
| `local` | 1: the driver also computes chunks while waiting | 1 |
| `lost_wait` | seconds before a vanished chunk is resubmitted (minimum 60) | 90 |
| `max_rounds` | submissions per chunk before the driver computes it itself | 10 |
| `submit` | 0: never submit workers (the driver computes everything; for tests) | 1 |
| `same_cpu` | 1: recompute every result that came from another CPU model | 0 |
| `local_after` | with `local = 0`: seconds without a running worker or a new result before the driver starts computing too (0: never) | 1800 |

The production searches used `worker_threads = 16`, `chunk = 64`, `throttle = 60`, `min_structures = 128` and `verify = verify_other = 32`.

## 4. Checkpoints (`CREST_CKPT=N`, `CREST_RESUME_POST=fallback`)

- **What is saved:**
  - every N steps, each trajectory writes an exact binary checkpoint (coordinates, velocities, counters, bias list, SHAKE list, SCC charges);
  - every optimised structure is journalled;
  - `crest.restart` records the position with an atomic write.
- **Resume.** Set `restart = true` in the TOML input and keep the same `CREST_CKPT` value.
- **After the MTD loop.** The stage after the loop restarts from its beginning when `CREST_RESUME_POST=fallback`; with the default `refuse`, it stops with exit code 3.
- **Reproducibility.** With nested threads, a resumed trajectory is statistically equivalent to an uninterrupted one, not bitwise identical. Two uninterrupted multithreaded runs also differ.
- **Known defect.** A resume late in the MTD loop (multimd occurrence 4 or later) can stop with "File crest_rotamers_3.xyz is empty".
