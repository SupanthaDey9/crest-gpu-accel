# How it works

FastCREST's patches change four things in CREST commit cfdc301f (master branch). Everything else is upstream code. Settings marked opt-in or optional are off unless you switch them on.

## 1. Thread layout and rewritten loops

- **Thread environment (set by `scripts/crest-env.sh`).** It sets:
  - a static OpenMP schedule;
  - threads bound to cores;
  - a 256 MB thread stack;
  - the glibc thread-stack cache;
  - one OpenBLAS thread.

  The default schedule (dynamic, chunk 1) hid almost all thread scaling of the SCC. With the same binary on one EPYC 7502 node, these settings made MTD 2.18x faster per energy and gradient.
- **Rewritten loops (always on).** 43 statements in tblite, mctc-lib, multicharge, dftd4 and simple-dftd3 built 3x3 outer products with Fortran `spread`. That allocates temporaries in inner loops, so they are now explicit loops with the same operand order. A single-thread energy and gradient is 11% faster and still bitwise identical.
- **Optional: LLVM's OpenMP runtime (`CREST_LIBOMP_DIR`).** CREST runs the parallel loops of each trajectory in a nested OpenMP team. The GNU runtime (libgomp) starts the threads of such a team again at every parallel region; LLVM's runtime (libomp) keeps them between regions. LLVM installs libomp together with a `libgomp.so.1` alias, so the same binary runs with it when that directory comes first in `LD_LIBRARY_PATH`. `scripts/crest-env.sh` does this, and sets `KMP_HOT_TEAMS_MAX_LEVEL=2` and `KMP_HOT_TEAMS_MODE=1`, when `CREST_LIBOMP_DIR` names that directory. Check with `ldd bin/crest | grep gomp`: a build whose RPATH names the GNU library ignores `LD_LIBRARY_PATH`. In single points with 8 threads (CPU solver, GPU solver and Jacobi option), the energy was the same with both runtimes and the gradient differed by at most 2e-15 Eh/bohr; two runs with the same runtime differed by up to 1.7e-13 Eh.
- **Optional: one CPU socket per trajectory (`scripts/crest-places.sh`).** With `OMP_PLACES=cores`, the block of cores that a trajectory gets can span both sockets of a node when the job's cores are split unevenly between them. That trajectory ran 5 to 16% slower than the others, and each MTD iteration waits for the slowest trajectory. `eval "$(scripts/crest-places.sh 6 5)"` lists six groups of five cores, each on one socket, and sets `CREST_THREADS=30`; run CREST with `-T $CREST_THREADS`. Any 34 cores on two sockets contain six such groups.

## 2. One GPU for the trajectories (`tblite_solver = "cuda"`, `CREST_ATM_GPU=1`)

- **SCC eigensolve.** During MTD and MD, each SCC eigensolve goes through a small C++ bridge (added by `patches/0001-crest-cfdc301f.patch` as `src/calculator/cuda/crest_cuda_bridge.cpp`) to cuSOLVER `dsygvd`; cuBLAS builds the density matrix. Each concurrent trajectory has its own CUDA stream and buffers, so the eigensolves of concurrent MTDs (six for the test system) overlap on one GPU.
- **Three-body term.** The D4 Axilrod-Teller-Muto dispersion term runs in a deterministic FP64 kernel (`src/cuda_atm/`, also added by that patch). It has a fixed reduction order and no atomics, and is compiled with `--fmad=false`. It matches the CPU routine to 4.4e-18 Eh/bohr.
- **CPU fallback.** A CUDA error in the three-body kernel moves that term to the CPU routine for the rest of the run. A CUDA error in the eigensolve stops the calculation, for example on a GPU with an uncorrectable ECC error.
- **Eigensolve that does not converge.** A positive `info` from cuSOLVER `dsygvd` (the eigenproblem did not converge, or the overlap is not positive definite) is a numerical result, as LAPACK's `info`, not a CUDA error. Since 1.1.1 that one solve is repeated on the CPU with tblite's LAPACK `dsygvd` solver, the eigensolver of conventional CREST, on the same matrices; the GPU stays in use for the next solves. If LAPACK fails too, the calculation fails as it would in conventional CREST: an MD then ends early and the search goes on. `crest.err` lists the first 20 repeats (`CREST_CUDA_CPU_RETRY`) and the totals at exit (`CREST_CUDA_CPU_RETRY_STATS`). Up to 1.1.0 such a solve stopped the whole search. With `CREST_CUDA_DEVSCC=1` the repeat rebuilds that step's Hamiltonian on the GPU, copies it and the overlap to the CPU, and uploads the CPU eigenvectors for the density. `CREST_CUDA_TEST_NOCONV=k` (test only) makes the k-th GPU eigensolve report that it did not converge, and `k,cpu` makes its CPU repeat fail too; `tests/run_tests.sh` uses both.
- **Optimization passes stay on the CPU.** `CREST_OPT_SOLVER=cpu` keeps the eigensolve on the CPU, and the three-body term always uses the CPU routine during optimization passes. The GPU was slower there, and CPU results can be checked bit for bit. The Hessians of the thermochemistry (entropy mode, `--rrhoav`) always use the CPU route too (since 1.1.2).
- **Opt-in: warm-started Jacobi eigensolver (`CREST_CUDA_WARMJACOBI=1`).** The overlap matrix does not change during the SCC iterations of one energy-and-gradient evaluation. The bridge therefore keeps its Cholesky factor after the first iteration, and solves the transformed problem with cuSOLVER's Jacobi solver (`syevj`), started from the previous iteration's eigenvectors (2.8 sweeps on average for the test system). A solve that does not converge falls back to `dsygvd` within the same call. Energies and gradients agree with the standard solver to about 1e-13 Eh, but not bit for bit.
- **Opt-in: GPU-resident SCC (`CREST_CUDA_DEVSCC=1`).** In the standard route, each SCC iteration sends the Hamiltonian to the GPU and brings the eigenvectors and the density matrix back, and the host computes the charges and multipole moments from the density. With this setting, the zeroth-order Hamiltonian, the overlap and the dipole and quadrupole integrals go to the GPU once per evaluation. Each iteration then sends only the potential shifts (per orbital and per atom). The GPU builds the Hamiltonian and solves it, and returns the eigenvalues; the host sets the occupations (Fermi smearing) and sends them back. The GPU then forms the density matrix and returns 11 numbers per orbital: its Mulliken population, dipole and quadrupole contributions and band energy. Charge mixing stays on the host. The eigenvectors and the density cross the bus once, after the last iteration, for the gradient. For the 274-atom test system (671 basis functions) this cut the data moved per evaluation from about 154 MB to 48 MB. The kernels (`src/calculator/cuda/crest_cuda_scc.cu`) follow tblite's formulas with the same operation order within one matrix element; only the sums over orbitals run in another order. Energies and gradients therefore agree with the standard GPU route to about 2e-13 Eh and 2e-15 Eh/bohr, but not bit for bit. The setting works with both eigensolvers. Open-shell calculations (two spin channels) keep the standard route, and on a GPU for which the build has no kernel code the setting declines and the SCC stays on the host.

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
- **Runs that end with failures.** With checkpoints on, a run that ends with failures keeps its files, as a killed run does, so it can be resumed (since 1.1.2; earlier versions deleted the per-iteration ensembles `.cre_*` in CREST's final cleanup). Before the ensembles of an MTD cycle are merged, a resume checks that every iteration of the cycle left its ensemble; if one is missing it stops with exit code 3 (`CREST_RESUME_FORCE=1` continues without it). `CREST_CUDA_TEST_NOCONV=k,fault` (test only) makes the k-th GPU eigensolve report a CUDA error, which ends a run with failures; `tests/run_tests.sh` uses it and checks that the resume merges one ensemble per MTD iteration. `CREST_MD_SEED` (test only) fixes the initial MD velocities of the test searches.
- **Resume in a later cycle.** A resume replays the completed steps without merging the ensembles after an MTD loop, so leftover files shift the numbering of `crest_rotamers_*` by one in the replayed stage. Up to 1.1.2 a resume in the first MTD batch of a later cycle therefore stopped with "File crest_rotamers_3.xyz is empty" if the last replayed step was a sort with a widened energy window; since 1.2.0 it goes on. `CREST_TEST_NEWCYCLE=1` (test only) forces a new cycle right after such a sort, whatever the energies, because the small test molecule never starts one; `tests/run_tests.sh` uses it.
