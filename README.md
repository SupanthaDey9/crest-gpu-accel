# crest-gpu-accel

Faster [CREST](https://github.com/crest-lab/crest) conformer searches (iMTD-GC, GFN2-xTB) with one GPU and many CPU jobs, computing the same thing as CREST.

A CREST search spends its time on two different kinds of work:

- **Trajectories.** A few long metadynamics and MD trajectories, each waiting on one energy-and-gradient call per step.
- **Optimisations.** About 15,000 independent geometry optimisations per iteration.

This build gives each kind its own hardware:

- **Trajectories:** one GPU (the SCC eigensolve and the D4 three-body dispersion term), plus a tuned OpenMP layout.
- **Optimisation passes:** Slurm array jobs on ordinary CPU nodes. The driver re-checks a sample of the results before merging them.
- **Checkpoints:** written during trajectories and passes, so a search survives preemption and time limits.

The build is CREST main at commit cfdc301f plus the patches in `patches/`. With every switch below unset, it gives the same results as an unmodified build of that commit: single-thread energies and gradients are bitwise identical, because the rewritten loops in the patches keep the operand order.

| 274-atom Ga peptide, `--quick`, ALPB water | wall time | throughput (E/G calls per hour) | CPU-hours | GPU-hours |
|---|---|---|---|---|
| CREST 3.0.2 release, 32 threads | 321.5 h | 1.0x | 10,288 | 0 |
| this build, 1 A100 + CPU workers | 54.25 h | 8.1x | 8,936 | 54 |

The two searches did different amounts of work (adaptive restarts), so compare the throughput column. Each speed number in this README is a single measurement (n = 1) on ISU Nova, documented in the paper [reference to be added]; `docs/VALIDATION.md` covers correctness.

## What you need

| Component | Tested with | Notes |
|---|---|---|
| Linux x86-64 | RHEL 9 | |
| gfortran, gcc, g++ | 11.5.0 | gfortran >= 11 |
| CMake, Ninja | 3.30.2, 1.12.1 | CMake >= 3.17 |
| OpenBLAS (OpenMP build) | 0.3.27 | |
| CUDA toolkit with cuSOLVER and cuBLAS | 12.4.1 | needed at build and run time |
| GPU with good FP64 | A100, H200 | see "Which GPU" |
| bash >= 4.4, git, patch, python3, binutils | | |
| Slurm | | only for distributed passes |

## Build and test

```bash
module load gcc cmake ninja openblas cuda     # or your site's equivalents
git clone https://github.com/SupanthaDey9/crest-gpu-accel.git && cd crest-gpu-accel
./build.sh --jobs 8                           # fetches CREST cfdc301f and pinned dependencies, patches, builds bin/crest
tests/run_tests.sh bin/crest                  # on a GPU node: single points (CPU, GPU) and a short GPU search
```

- **Build time.** On ISU Nova a clean build took 6 minutes on 8 cores.
- **No internet on the build node:** run `./build.sh --fetch-only SRC` on a login node, then `./build.sh --sources SRC`.
- **Specific libraries:** choose installations with `--cuda DIR` and `--openblas /path/libopenblas.so`.
- **Other test modes:** `tests/run_tests.sh bin/crest --no-gpu` runs only the CPU single point (for example on a login node).
  `--dist "<sbatch options>"` also runs the short search with real Slurm workers; try it once before a real search.
- **Scope of the tests.** `run_tests.sh` uses a 12-atom molecule. It checks that the build works on your machine (GPU single point equal to the CPU single point to 1e-8 Eh, the three-body kernel really ran on the GPU, searches finish). It does not repeat the validation of `docs/VALIDATION.md`.

## Run on one machine with a GPU (no Slurm)

All optimisations run in-process; the trajectories use the GPU. Run in a folder of its own, because CREST writes many files.

```bash
R=/path/to/crest-gpu-accel
mkdir run && cd run && cp /path/to/your_molecule.xyz molecule.xyz
source $R/scripts/crest-env.sh gpu                         # thread layout + GPU switches
export OMP_NUM_THREADS=32 CREST_CKPT=40 CREST_RESUME_POST=fallback   # use the number of physical cores
cat > search.toml <<'EOF'
input = "molecule.xyz"
runtype = "imtd-gc"
threads = 32

[[calculation.level]]
method = "gfn2"
charge = 0
uhf = 0
alpb = "water"
tblite_solver = "cuda"
EOF
$R/bin/crest --input search.toml --newversion --gfn2 --chrg 0 --uhf 0 --alpb water --quick -T 32 > crest.out 2> crest.err
```

To resume after an interruption, add `restart = true` as the first line of `search.toml` and start the same command again with the same `CREST_CKPT`. `crest.err` ends with a line `CREST_ATM_GPU summary gpu_calls N ...`; N > 0 means the three-body term ran on the GPU.

## Run on a Slurm cluster with distributed optimisation passes

This is the mode behind the numbers above.

1. Edit `slurm/worker_env.sh` so that workers load the same OpenBLAS and CUDA modules as the build. `slurm/site/nova_worker_env.sh` is a complete example; to use a site file instead, `export WORKER_ENV=/absolute/path/to/site_env.sh` before `sbatch`. Workers are started with `--export=NONE`.
2. Load the same modules in your shell, then submit the driver:

```bash
export WORKER_SBATCH_ARGS="--partition=<cpu partition> --account=<account> --mem=8G --time=02:00:00"
sbatch --partition=<gpu partition> --account=<account> slurm/crest_search.sbatch molecule.xyz 0 0 --quick
```

**What the driver does.** The driver job takes one GPU and 32 cores. For every optimisation pass of 128 or more structures it submits worker arrays of 16-core tasks, with 64 structures each and at most 60 at once. The settings are in `OUT/dist_oloop.conf`; the keys are explained in `docs/HOW_IT_WORKS.md`. Quote any `|` inside `WORKER_SBATCH_ARGS`.

**Resuming.** Resubmitting with `OUT=<folder>` resumes a search. `search.toml` and `dist_oloop.conf` are written only the first time; edit them in the folder to change settings.

**Time limits.** Give the driver the time the whole search needs. The stage after the MTD loop can only restart from its beginning, so short segments may never finish it.

**Scratch space.** Workers use `$TMPDIR` (or `/tmp`) and do not delete it; most sites clean it after each job.

## Switches

| Setting | Where | Effect |
|---|---|---|
| `tblite_solver = "cuda"` | `[[calculation.level]]` in the TOML input | SCC eigensolve on the GPU (cuSOLVER `dsygvd`) |
| `CREST_OPT_SOLVER=cpu` | environment | keep every optimisation pass on the CPU solver. Always set it together with the line above: on one GPU, 32 concurrent optimisations ran slower than on the CPU, and only CPU passes can be distributed and checked bit for bit |
| `CREST_ATM_GPU=1` | environment | D4 three-body dispersion term on the GPU, in every calculation except optimisation passes |
| `CREST_CKPT=40` | environment | exact checkpoint of every trajectory every 40 steps, and a journal of every optimised structure |
| `CREST_RESUME_POST=fallback` | environment | a resume after the MTD loop rewinds to the start of that stage instead of stopping with exit code 3 |
| `CREST_DIST_OLOOP=<file>` | environment | distribute optimisation passes as Slurm arrays (file format in `docs/HOW_IT_WORKS.md`) |
| `CREST_CUDA_MULTI_DEVICE=1` | environment | spread GPU sessions over several visible GPUs. Not recommended: two GPUs gave only 1.19x for one search, so run two searches instead |
| `OMP_*`, `OPENBLAS_NUM_THREADS=1`, ... | `scripts/crest-env.sh` | thread layout. Keep OpenBLAS at one thread; otherwise it floods the output with "Detect OpenMP Loop" warnings |

`scripts/crest-env.sh gpu` sets `CREST_ATM_GPU=1`, `CREST_OPT_SOLVER=cpu` and the thread layout.

**Option names.** The CMake options `WITH_TBLITE_GPU_DENSITY_EXPERIMENT` and `WITH_TBLITE_GPU_PERSISTENCE_EXPERIMENT` that `build.sh` sets carry historical names. They are required, because the validated binary was compiled with them.

## Which GPU

The MTD step is still mostly host work: about two thirds of each step ran on the CPU in our profile. A fast host CPU therefore matters as much as the GPU. MTD time per step for the 274-atom test system:

| Hardware | time per step | three-body term |
|---|---|---|
| original release, 32 threads | 2.8 s | CPU |
| CPU only, zen5, 64 threads | 0.74 s | CPU |
| A100 + EPYC 7543 | 0.46 s | GPU |
| H200 | 0.39 s | GPU |
| RTX PRO 6000 + EPYC 9645 | 0.68 s | falls back to CPU (no sm_120 code) |
| V100 + older Xeon, 8 threads | 1.86 s | falls back to CPU (no sm_70 code) |

**Architectures.** The three-body kernel is compiled for sm_80 and sm_90. L40S (sm_89) runs the sm_80 code; we saw it run on one L40S job. To add targets, pass the full list, defaults included, for example `./build.sh --gencode "-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_120,code=sm_120"`. sm_120 needs CUDA 12.8 or newer; the tested CUDA 12.4.1 cannot compile it. Such builds have not been validated, and the value is kept in the CMake cache until you delete `build/`.

**GPUs per search.** Use one GPU per search, and put several searches on a multi-GPU node.

## How we checked that nothing changed

See `docs/VALIDATION.md`. In short:

- **Startup:** the decisions CREST makes before the first trajectory are identical across the 3.0.2 release, an unmodified build of cfdc301f, and this build.
- **Single points:** energies and gradients on 306 structures agree with the release binary to 1.01e-10 Eh. The CPU route of this build is bitwise identical to the unmodified build; the GPU route agrees with it to 4e-13 Eh.
- **Controlled MD:** from identical starts and velocities, the fraction of topology changes is equivalent within a preset margin of ±0.15 (two one-sided tests, 90% intervals).
- **Replicate searches:** 16 GPU searches and 17 searches with unmodified numerics, all from the same input. The shift in the lowest energy after the first MTD round was +4.09 kcal/mol, with a 95% interval of −4.94 to +11.75 kcal/mol, so no shift was detected. Equivalence within 2 kcal/mol could not be shown, because CREST itself varies far more than that between runs on this system (interquartile range about 13 kcal/mol).
- **The release build:** built from GitHub with `build.sh`, it gives a single-thread single point byte-identical to the binary validated above.

## Known limitations

- **One test system:** all measurements come from one system (a 274-atom, +4 Ga peptide) with `--quick`.
- **Late-resume failure:** a resume late in the MTD loop (later cycles) can fail with "File crest_rotamers_3.xyz is empty". Give jobs enough time that late preemptions are rare.
- **Open-shell metals:** for open-shell Cu and Ni peptides at these settings, most MTDs stopped early. The same happened for Fe with the release build.
- **Untested combinations:** the full benchmark searches used the Slurm mode. The single-machine mode was tested on small molecules only.

## License and citation

- **CREST, tblite, dftd4, simple-dftd3:** these projects are LGPL-3.0-or-later, and the patches to them, the new CUDA, C++ and Fortran sources and the scripts are distributed under the same license (`COPYING`, `COPYING.LESSER`).
- **mctc-lib, multicharge:** these projects are Apache-2.0, and the patches to them (`patches/deps/mctc-lib.patch`, `patches/deps/multicharge.patch`) are offered under Apache-2.0 (`LICENSE-Apache-2.0`).
- **Citation:** please cite the CREST and GFN2-xTB papers and our paper [reference to be added].
