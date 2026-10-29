# How it works

FastCREST's patches change four things in CREST commit cfdc301f (master branch). Everything else is upstream code.

## 1. Thread layout and rewritten loops

- **Thread environment (set by `scripts/crest-env.sh`).** It sets:
  - a static OpenMP schedule;
  - threads bound to cores;
  - a 256 MB thread stack;
  - the glibc thread-stack cache;
  - one OpenBLAS thread.

  The default schedule (dynamic, chunk 1) hid almost all thread scaling of the SCC. With the same binary on one EPYC 7502 node, these settings made MTD 2.18x faster per energy and gradient.
- **Rewritten loops (always on).** 43 statements in tblite, mctc-lib, multicharge, dftd4 and simple-dftd3 built 3x3 outer products with Fortran `spread`. That allocates temporaries in inner loops, so they are now explicit loops with the same operand order. A single-thread energy and gradient is 11% faster and still bitwise identical.

## 2. One GPU for the trajectories (`tblite_solver = "cuda"`, `CREST_ATM_GPU=1`)

- **SCC eigensolve.** During MTD and MD, each SCC eigensolve goes through a small C++ bridge (added by `patches/0001-crest-cfdc301f.patch` as `src/calculator/cuda/crest_cuda_bridge.cpp`) to cuSOLVER `dsygvd`; cuBLAS builds the density matrix. Each concurrent trajectory has its own CUDA stream and buffers, so the eigensolves of concurrent MTDs (six for the test system) overlap on one GPU.
- **Three-body term.** The D4 Axilrod-Teller-Muto dispersion term runs in a deterministic FP64 kernel (`src/cuda_atm/`, also added by that patch). It has a fixed reduction order and no atomics, and is compiled with `--fmad=false`. It matches the CPU routine to 4.4e-18 Eh/bohr.
- **CPU fallback.** A CUDA error in the three-body kernel moves that term to the CPU routine for the rest of the run. A CUDA error in the eigensolve stops the calculation, for example on a GPU with an uncorrectable ECC error.
- **Optimization passes stay on the CPU.** `CREST_OPT_SOLVER=cpu` keeps the eigensolve on the CPU, and the three-body term always uses the CPU routine during optimization passes. The GPU was slower there, and CPU results can be checked bit for bit.

## 3. Distributed optimization passes (`CREST_DIST_OLOOP=<config>`)

For every optimization pass of at least `min_structures` structures, the CREST process acts as a driver:

1. **Write the pass.** It writes the exact binary coordinates into a pass folder. The folder is named by a hash of the settings, the binary's sha256 and the input, so a requeued driver finds and reuses finished chunks.
2. **Submit workers.** It submits chunks of `chunk` structures as one Slurm array (`slurm/dist_worker.sbatch`, `--export=NONE`). Each worker runs the same binary, refuses mismatched settings, and writes an atomic result file with a checksum, its binary hash and its CPU model.
3. **Check and merge.** It recomputes `verify` structures that came from workers with its own CPU model; these must agree bitwise. It also recomputes `verify_other` structures from workers with other CPU models; these must agree to 1e-7 Eh in energy and 1e-4 Å in coordinates. It then merges all results in input order through the usual code path. A mismatch on its own CPU model makes the driver redo the whole pass itself; a mismatch on another CPU model makes it recompute the affected chunks.

The defaults in the table are built into the code. `slurm/crest_search.sbatch` writes `worker_threads = 16`, `chunk = 64`, `throttle = 60`, `min_structures = 128` and `verify = verify_other = 32`, the settings of the benchmark searches (the first pass of the FastCREST search in the paper used 32-thread workers with 128 structures per chunk).

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

## 4. Checkpoints (`CREST_CKPT=N`, `CREST_RESUME_POST=fallback`)

- **What is saved:**
  - every N steps, each trajectory writes an exact binary checkpoint (coordinates, velocities, counters, bias list, SHAKE list, SCC charges);
  - every optimized structure is journaled;
  - CREST's restart file `crest.restart` records how far the search has progressed, written atomically.
- **Resume.** Set `restart = true` in the TOML input and keep the same `CREST_CKPT` value.
- **After the MTD loop.** The stage after the loop restarts from its beginning when `CREST_RESUME_POST=fallback`; with the default `refuse`, it stops with exit code 3.
- **Reproducibility.** With nested threads, a resumed trajectory is not bitwise identical to an uninterrupted one, just as two uninterrupted multithreaded runs differ. We expect them to be statistically equivalent but did not test this.
- **Known defect.** A resume late in the MTD loop (from the fourth MTD batch on) can stop with "File crest_rotamers_3.xyz is empty".
