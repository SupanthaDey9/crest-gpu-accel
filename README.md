# FastCREST

Faster [CREST](https://github.com/crest-lab/crest) conformer searches (iMTD-GC, GFN2-xTB) with one GPU and many CPU jobs, with unchanged energies and gradients. FastCREST was first released as crest-gpu-accel.

A CREST search spends its time on two different kinds of work:

- **Trajectories.** A few long metadynamics and MD trajectories, each waiting on one energy-and-gradient call per step.
- **Optimizations.** About 15,000 independent geometry optimizations per iteration.

FastCREST gives each kind its own hardware:

- **Trajectories:** one GPU (the SCC eigensolve and the D4 three-body dispersion term), plus a tuned OpenMP layout.
- **Optimization passes:** Slurm array jobs on ordinary CPU nodes. The main CREST job (the driver) re-checks a sample of the results before merging them.

Checkpoints written during trajectories and passes let a search survive preemption and time limits.

FastCREST is CREST commit cfdc301f (master branch, 80 commits after v3.0.2) plus the patches in `patches/`. With every switch below unset, it gives the same results as an unmodified build of that commit: single-thread energies and gradients are bitwise identical, because the rewritten loops in the patches keep the operand order.

| 274-atom Ga peptide, `--quick`, ALPB water | wall time | throughput (E/G calls per hour, relative) | CPU-hours | GPU-hours |
|---|---|---|---|---|
| conventional search: CREST 3.0.2 release, 32 threads | 321.5 h | 1.0x | 10,288 (32 threads x wall) | 0 |
| FastCREST, 1 A100 + CPU workers | 54.3 h | 8.1x | 8,936 | 54.3 |

The two searches did different amounts of work, because CREST repeats iterations until it stops finding a lower conformer; compare the throughput column. Each speed number in this README is a single measurement (n = 1) on the Nova cluster at Iowa State University, documented in our paper (in preparation); `docs/VALIDATION.md` covers correctness.

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
git clone https://github.com/chowdhurylab/FastCREST.git && cd FastCREST
./build.sh --jobs 8                           # fetches CREST cfdc301f and pinned dependencies, patches, builds bin/crest
tests/run_tests.sh bin/crest                  # on a GPU node: single points (CPU, GPU) and a short GPU search
```

- **Build time.** On Nova a clean build took 5 to 7 minutes on 8 cores.
- **No internet on the build node:** run `./build.sh --fetch-only SRC` on a login node, then `./build.sh --sources SRC`.
- **Specific libraries:** choose installations with `--cuda DIR` and `--openblas /path/libopenblas.so`.
- **Other test modes:** `tests/run_tests.sh bin/crest --no-gpu` runs only the CPU single point (for example on a login node).
  `--dist "<sbatch options>"` also runs the short search with Slurm worker jobs; run it once before your first full search.
- **Scope of the tests.** `run_tests.sh` uses a 12-atom test molecule. It checks that the build works on your machine (GPU single point equal to the CPU single point to 1e-8 Eh, the three-body kernel ran on the GPU without falling back to the CPU, searches finish). It does not repeat the validation of `docs/VALIDATION.md`.

## Run on one machine with a GPU (no Slurm)

All optimizations run in-process; the trajectories use the GPU. Run in a folder of its own, because CREST writes many files.

```bash
R=/path/to/FastCREST
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

## Run on a Slurm cluster with distributed optimization passes

This is the mode behind the numbers above.

1. Uncomment and edit the module line in `slurm/worker_env.sh` so that workers load the same compiler runtime, OpenBLAS and CUDA modules as the build. `slurm/site/nova_worker_env.sh` is a complete example; to use a site file instead, `export WORKER_ENV=/absolute/path/to/site_env.sh` before `sbatch`. Workers are started with `--export=NONE`.
2. Make sure that the output folder, this repository and `bin/crest` are on a filesystem that the CPU worker nodes also mount, and that jobs may run `sbatch`, `squeue` and `scancel`. Load the same modules in your shell, then submit the driver:

```bash
export WORKER_SBATCH_ARGS="--partition=<cpu partition> --account=<account> --mem=8G --time=02:00:00"
sbatch --partition=<gpu partition> --account=<account> slurm/crest_search.sbatch molecule.xyz 0 0 --quick
```

**What the driver does.** The driver job takes one GPU and 32 cores. For every optimization pass of 128 or more structures it submits worker arrays of 16-core tasks, with 64 structures each and at most 60 at once. The settings are written to `dist_oloop.conf` in the search folder; the keys are explained in `docs/HOW_IT_WORKS.md`. The driver passes `WORKER_SBATCH_ARGS` to `/bin/sh`, so quote values that contain `|` or other shell characters, for example `--constraint="epyc-7502|epyc-7413"`.

**Resuming.** Resubmitting with `OUT=<folder>` resumes a search. `search.toml` and `dist_oloop.conf` are written only the first time, and an existing `dist_oloop.conf` is used again; edit them in the folder to change settings. To run every pass inside the driver, rename `dist_oloop.conf` and resubmit without `WORKER_SBATCH_ARGS`.

**Finding the repository.** The job finds this repository from its own command line. If that fails it stops and asks for `REPO=/path/to/FastCREST`. `CREST_BIN`, `SOLVENT` and `EXTRA_TOML` are described at the top of `slurm/crest_search.sbatch`.

**Time limits.** Give the driver the time the whole search needs. The stage after the MTD loop can only restart from its beginning, so a search run as a chain of short requeued jobs may never finish it.

**Scratch space.** Workers use `$TMPDIR` (or `/tmp`) and do not delete it; most sites clean it after each job.

## Switches

| Setting | Where | Effect |
|---|---|---|
| `tblite_solver = "cuda"` | `[[calculation.level]]` in the TOML input | SCC eigensolve on the GPU (cuSOLVER `dsygvd`) |
| `CREST_OPT_SOLVER=cpu` | environment | keep every optimization pass on the CPU solver. Always set it together with the line above: on one GPU, 32 concurrent optimizations ran slower than on the CPU, and only CPU passes can be distributed and checked bit for bit |
| `CREST_ATM_GPU=1` | environment | D4 three-body dispersion term on the GPU, in every calculation except optimization passes |
| `CREST_CKPT=40` | environment | exact checkpoint of every trajectory every 40 steps, and a journal of every optimized structure |
| `CREST_RESUME_POST=fallback` | environment | a resume after the MTD loop restarts the following stage from its beginning instead of stopping with exit code 3 |
| `CREST_DIST_OLOOP=<file>` | environment | distribute optimization passes as Slurm arrays (file format in `docs/HOW_IT_WORKS.md`) |
| `CREST_CUDA_MULTI_DEVICE=1` | environment | spread GPU sessions over several visible GPUs. Not recommended: two GPUs gave only 1.19x for one search, so run two searches instead |
| `OMP_*`, `OPENBLAS_NUM_THREADS=1`, ... | `scripts/crest-env.sh` | thread layout. Keep OpenBLAS at one thread; otherwise it floods the output with "Detect OpenMP Loop" warnings |

`scripts/crest-env.sh gpu` sets `CREST_ATM_GPU=1`, `CREST_OPT_SOLVER=cpu` and the thread layout.

**Option names.** `build.sh` sets the CMake options `WITH_TBLITE_GPU_DENSITY_EXPERIMENT` (density matrix built on the GPU) and `WITH_TBLITE_GPU_PERSISTENCE_EXPERIMENT` (per-thread GPU buffers kept between calls). Despite their names, both are required: the validated binary was built with them.

## Which GPU

The MTD step is still mostly host work: in a profile of an earlier build with only the eigensolve on the GPU, 63.5% of each step was CPU time. A fast host CPU therefore matters as much as the GPU. MTD time per step for the 274-atom test system (GPU rows at 32 threads unless stated):

| Hardware | time per step | three-body term |
|---|---|---|
| conventional search (CREST 3.0.2 release), 32 threads | 2.8 s | CPU |
| rewritten loops, CPU only, Zen 5, 64 threads | 0.74 s | CPU |
| A100 + EPYC 7543 | 0.49 s | GPU |
| H200 + Zen 5 | 0.39 s | GPU |
| RTX PRO 6000 + EPYC 9645 | 0.68 s | falls back to CPU (no sm_120 code) |
| V100 + older Xeon, 8 threads | 1.86 s (one partial trajectory) | falls back to CPU (no sm_70 code) |

The two Zen 5 rows show the GPU's own effect on the same host CPU type: about 1.9x per step.

**Architectures.** The three-body kernel is compiled for sm_80 and sm_90. L40S (sm_89) runs the sm_80 code: in one job on an L40S the kernel ran without falling back to the CPU. The code-level tests did not cover L40S GPUs. To add targets, pass the full list, defaults included, for example `./build.sh --gencode "-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_120,code=sm_120"`. sm_120 needs CUDA 12.8 or newer; the tested CUDA 12.4.1 cannot compile it. Such builds have not been validated, and the value is kept in the CMake cache until you delete `build/`.

**GPUs per search.** Use one GPU per search, and put several searches on a multi-GPU node.

## How we checked the results

See `docs/VALIDATION.md`. In short:

- **Startup:** the decisions CREST makes before the first trajectory are identical across the 3.0.2 release, an unmodified build of cfdc301f, and FastCREST.
- **Single points:** on 306 structures, every configuration agrees with the release binary within 1.01e-10 Eh in energy and 8.16e-11 Eh/bohr in gradient, two orders of magnitude inside the preset 1e-8 Eh criterion. FastCREST on the CPU is bitwise identical to the unmodified build; FastCREST on the GPU agrees with it to 4e-13 Eh.
- **Upstream change:** one commit between the 3.0.2 release and cfdc301f stops unconverged ANCOPT optimizations after 200 cycles. It left 2 to 3 more structures per 100 unconverged than the release, more than our preset limit of 1 per 100; we kept it (details in `docs/VALIDATION.md`).
- **Controlled MD:** from identical starts and velocities, FastCREST on CPU and GPU, the unmodified build and the release change bonding topology in equivalent fractions of frames in all paired comparisons, within a preset margin of ±0.15 (two one-sided tests, 90% intervals).
- **Replicate searches:** 16 FastCREST searches and 17 stock searches (unmodified numerics), all from the same input. The lowest energy after the first MTD iteration was typically 4.09 kcal/mol higher in FastCREST searches than in stock searches (Hodges-Lehmann shift; 95% interval −4.94 to +11.75 kcal/mol). The interval includes zero, so no shift was detected. The test is inconclusive: equivalence within 2 kcal/mol could not be shown, because CREST itself varies far more than that between runs on this system (interquartile range about 13 kcal/mol).
- **Structures:** all 10 finished FastCREST searches and 13 of the 15 finished stock searches ended in compact conformers; the other two are explained under Known limitations. Median final energies were 10.5 kcal/mol (FastCREST) and 15.3 kcal/mol (stock) above the conventional search. Outside the Ga site, which CREST's topology check holds fixed, both differ from the conventional search mainly at the peptide termini. The lowest FastCREST conformer lies 0.7 kcal/mol above the conventional one.
- **The released code:** built from GitHub with `build.sh`, it gives a single-thread single point byte-identical to the binary validated above. `tests/site/nova/` holds the Nova jobs used for this check.

## Known limitations

- **One test system:** all measurements come from one system (a 274-atom, +4 Ga peptide) with `--quick`.
- **CREST fallback to the input geometry:** when CREST's topology check rejects every structure of one CREGEN call, CREGEN keeps the reference geometry labeled with the lowest energy seen, and the search continues from it. This is upstream CREST code (also in the 3.0.2 release). It sent 2 of our 15 finished stock searches (and none of the 10 finished FastCREST searches) back to the helical input. If `crest_best.xyz` is almost identical to your input, check `crest.out` for CREGEN calls where all structures were removed by the topology check.
- **Late-resume failure:** a resume late in the MTD loop (from the fourth MTD batch on) can fail with "File crest_rotamers_3.xyz is empty". Give jobs enough time that late preemptions are rare.
- **Open-shell metals:** in single searches of open-shell Cu and Ni peptides at these settings, most MTDs stopped early. Fe searches with the release build did the same, so we cannot tell whether FastCREST plays a part.
- **Untested combinations:** the full benchmark searches used the Slurm mode. The single-machine mode was tested on small molecules only.
- **CPU-only use:** the build always includes the GPU code, so a CUDA toolkit is needed to build and run it. For a CPU-only search leave out `tblite_solver` and source `scripts/crest-env.sh` without `gpu`; `slurm/crest_search.sbatch` always asks for a GPU.
- **Compiler flags:** do not add `-march=native` or other flags that allow FMA contraction. The bitwise identity with the unmodified build relies on the default x86-64 target that `build.sh` uses.
- **Test input:** the repository ships only a 12-atom test molecule (`examples/smoke/struc.xyz`). The Ga peptide input and the raw outputs are available from the authors on request.

## License and citation

- **CREST, tblite, dftd4, simple-dftd3:** these projects are LGPL-3.0-or-later, and the patches to them, the new CUDA, C++ and Fortran sources and the scripts are distributed under the same license (`COPYING`, `COPYING.LESSER`).
- **mctc-lib, multicharge:** these projects are Apache-2.0, and the patches to them (`patches/deps/mctc-lib.patch`, `patches/deps/multicharge.patch`) are offered under Apache-2.0 (`LICENSE-Apache-2.0`).
- **Citation:** GitHub's "Cite this repository" uses `CITATION.cff`. Please also cite the papers that CREST asks for: P. Pracht, F. Bohle, S. Grimme, *Phys. Chem. Chem. Phys.* **22**, 7169 (2020); S. Grimme, *J. Chem. Theory Comput.* **15**, 2847 (2019); and P. Pracht et al., *J. Chem. Phys.* **160**, 114110 (2024). Also cite GFN2-xTB (C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652 (2019)) and our paper (in preparation).
