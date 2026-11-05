# Validation record

All tests ran on the Nova cluster at Iowa State University. The code-level tests and the replicate searches used the 274-atom, +4 Ga peptide in ALPB water; `tests/run_tests.sh` uses the 12-atom test molecule in `examples/smoke/`. The pass criteria of the code-level tests (startup, single points, controlled MD) were fixed before the runs. The decision rule for the replicate searches was fixed after some individual results had been seen, but before any group statistic was computed.

Names: **conventional CREST** means CREST without our GPU code; the **reference search** is the original 321.5 h conventional CREST search, made with the 3.0.2 release. Five configurations were compared, and the two opt-in settings as three more:

| Name | What it is |
|---|---|
| release | CREST 3.0.2 release (conda-forge), used by the reference search |
| unmodified | our build of CREST commit cfdc301f without these patches, run from the command line |
| unmodified, TOML | the same build, run from TOML input |
| FastCREST, CPU | FastCREST with every GPU switch off |
| FastCREST, GPU | FastCREST with `tblite_solver = "cuda"` and `CREST_ATM_GPU=1` |
| FastCREST, Jacobi | FastCREST, GPU with `CREST_CUDA_WARMJACOBI=1` (opt-in) |
| FastCREST, GPU-resident SCC | FastCREST, GPU with `CREST_CUDA_DEVSCC=1` (opt-in) |
| FastCREST, both | FastCREST, GPU with both opt-in settings |

## Startup decisions

Every configuration, the opt-in settings included, made the same decisions before the first trajectory:

- the same initial-optimization energies to 1e-10 Eh (first frame -415.2625315087 Eh on the release, -415.2625315088 Eh on the others);
- the same flexibility measure (0.391);
- six MTDs of 250 ps;
- 274 bonds with bond order > 0.5 and 272 SHAKE constraints (counts compared).

## Single points: 306 structures

The structures are the reference search's 135 final conformers, 50 of its MD frames, 4 input and pre-optimized geometries, and 117 structures from a search with an earlier build that ran only the eigensolve on the GPU.

| Comparison | max abs dE (Eh) | max abs dG (Eh/bohr) |
|---|---|---|
| unmodified vs release | 1.01e-10 | 8.16e-11 |
| unmodified: TOML vs command line | 0 (bitwise) | 0 |
| FastCREST, CPU vs unmodified | 0 (bitwise) | 0 |
| FastCREST: GPU vs CPU | 3.98e-13 | 4.0e-15 |
| FastCREST, GPU vs release | 1.01e-10 | 8.16e-11 |
| FastCREST: Jacobi vs GPU | 3.41e-13 | 3.4e-14 |
| FastCREST: GPU-resident SCC vs GPU | 3.41e-13 | 5.0e-15 |
| FastCREST: both vs Jacobi | 2.84e-13 | 3.4e-14 |
| FastCREST: Jacobi, GPU-resident SCC, both vs CPU | 3.41e-13, 4.55e-13, 4.55e-13 | 3.3e-14, 4.0e-15, 6.0e-15 |
| FastCREST: Jacobi, GPU-resident SCC, both vs release | 1.00e-10 each | 8.16e-11 each |

Criterion, fixed before the runs: max abs dE <= 1e-8 Eh and max abs dG <= 1e-6 Eh/bohr. Every comparison passes; the largest differences, to the release, are two orders of magnitude inside it.

## Upstream change since the release

Of the 80 upstream commits between the 3.0.2 release and cfdc301f, one can change results: unconverged ANCOPT optimizations now stop after 200 cycles (the release ran up to 239 cycles in our test). On 200 MD frames this left 2 to 3 more unconverged structures per 100 than the release, above our preset limit of 1 per 100, so this test failed. We kept the commit because every structure that converged in both builds got the same topology-check result, and the controlled MD below, whose post-MD optimizations are capped only in our builds, passed.

## Controlled MD

Each comparison used 20 ps runs from four starts, at 400 and 500 K, with identical initial velocities on both sides. The score is the fraction of frames whose bonding topology changed. A pair counts as equivalent if the 90% confidence interval of the mean paired difference lies within ±0.15 (two one-sided tests).

| Comparison | raw frames | after optimization |
|---|---|---|
| unmodified, same binary twice | -0.024 to +0.017 | -0.084 to +0.041 |
| unmodified vs release | -0.015 to +0.017 | -0.039 to +0.030 |
| unmodified: TOML vs command line | -0.005 to +0.034 | -0.030 to +0.110 |
| FastCREST, CPU vs unmodified (TOML) | -0.043 to +0.005 | -0.137 to -0.007 |
| FastCREST: GPU vs CPU | 0.000 to +0.050 | +0.008 to +0.134 |
| FastCREST, GPU vs unmodified (command line) | -0.001 to +0.030 | +0.005 to +0.072 |
| FastCREST, GPU vs release | -0.003 to +0.033 | -0.014 to +0.082 |
| FastCREST: Jacobi vs GPU | -0.033 to +0.010 | -0.076 to +0.025 |
| FastCREST, Jacobi vs release | -0.020 to +0.026 | -0.048 to +0.064 |
| FastCREST: GPU-resident SCC vs GPU | -0.024 to +0.016 | -0.065 to +0.040 |
| FastCREST: both vs Jacobi | -0.021 to +0.021 | -0.059 to +0.040 |
| FastCREST, GPU-resident SCC vs release | -0.005 to +0.027 | -0.029 to +0.072 |
| FastCREST, both vs release | -0.013 to +0.020 | -0.044 to +0.042 |

All 13 comparisons are equivalent. Three intervals after optimization exclude zero while staying inside the margin; one of them compares two CPU configurations whose single-thread forces are bitwise identical, so we read them as sampling noise.

## Replicate searches

**Endpoint.** The lowest crude-level energy after the first MTD iteration.

**Groups.** 16 FastCREST searches against 17 conventional CREST searches: CPU only, with the unmodified build or the FastCREST binary with its GPU code off (bitwise the same single-thread numbers), all with the tuned thread settings and most with distributed passes, which change speed but not the computation.

**Conditions.** The groups ran under different conditions. An uncorrectable memory error on one GPU stopped the resumed jobs of 17 of 40 FastCREST replicates at startup; after seeing interim results we did not resubmit them, and 7 more were canceled early or still queued. Thirteen of the 16 FastCREST searches but 7 of the 17 conventional CREST ones resumed from checkpoints before the endpoint. Twelve of the 16 FastCREST searches finished their first MTD iteration on L40S or V100 GPUs, which the code-level tests did not cover, and 6 of the 16 computed the three-body term on the CPU in at least part of the run.

**Result.**

- **Shift:** FastCREST endpoints were typically 4.09 kcal/mol higher than conventional CREST ones (Hodges-Lehmann shift), with an exact 95% interval of -4.94 to +11.75 kcal/mol.
- **Verdict:** inconclusive; no detectable shift.
- **Limit:** equivalence within 2 kcal/mol could not be shown. Conventional CREST searches alone spread over about 13 kcal/mol (interquartile range) on this system.
- **Run-to-run spread:** thirteen of the 15 finished conventional CREST searches ended between -1.1 and +24.1 kcal/mol relative to the reference search. The other two ended more than 110 kcal/mol higher because of an upstream CREST fallback. When the topology check rejects every structure of a CREGEN call, CREGEN keeps the reference geometry (`coord`, here the helical input) labeled with the lowest energy seen, and the MD stage then starts from it. The fallback is in `cregen.f90`, unchanged by our patches and identical in the 3.0.2 release.

## Structures

Final ensembles of the reference search and of the 25 finished replicate searches (10 FastCREST, 15 conventional CREST), superposed on the Ga site:

- CREST's topology check holds the input's five Ga-O bonds, so every final conformer keeps them.
- All 10 FastCREST and 13 of the 15 conventional CREST searches ended in compact conformers (radius of gyration 6.2 to 6.7 Å, against 7.56 Å for the helical input); the other two are the fallback cases above.
- The lowest FastCREST conformer lies 0.7 kcal/mol above the reference search's lowest, with the same six Ga-O contacts (distances within 0.08 Å). Its Cα atoms coincide with the reference ones within 2.1 Å from Arg4 to Lys15; Met1 to Thr3 point elsewhere.
- Median final energies: FastCREST +10.5 and conventional CREST +15.3 kcal/mol relative to the reference search (post hoc Hodges-Lehmann shift -2.7 kcal/mol, 95% interval -11.3 to +6.5).

## Check of the released build

`build.sh` builds from upstream plus `patches/`. Two checks are run on the result:

- **Same numbers as the benchmarked binary:** a single-thread single point of the Ga peptide must give a `crest.engrad` byte-identical to the one from the FastCREST binary used for the results above.
- **Repeatable checks:** `tests/run_tests.sh` repeats the single-point checks (CPU against a reference binary, GPU against CPU) and runs a short GPU search, optionally with distributed passes. Since 1.1.1 it also forces a GPU eigensolve that does not converge (`CREST_CUDA_TEST_NOCONV`): repeated on the CPU, the single point must match the CPU single point; if the CPU repeat fails too, the single point must fail as an ordinary SCC failure and a search must still end normally. Since 1.1.2 it also runs the Hessian thermochemistry with the GPU route selected, which must equal the CPU solver's, and resumes a search that a forced CUDA error ended with failures: the resume must end normally and merge one ensemble per MTD iteration, and the same folder without those ensembles must be refused.

### Results on Nova (2026-10-28, fresh clone of this repository from GitHub, commit d6c51ab)

| Item | Value |
|---|---|
| Patches tested (sha256 prefix) | `0001-crest` 3d0d8ca6, `0002-tblite` a4ae8643, dftd4 08ac3376, mctc-lib dedf3bb8, multicharge 5ea75735, s-dftd3 b3387586 |
| Build | `build.sh` with sources fetched from GitHub, gfortran 11.5.0, CMake 3.30.2, CUDA 12.4.1, OpenBLAS 0.3.27; 414 s on 8 cores of a Xeon Platinum 8260 node (earlier builds of the same patches took 291 and 322 s) |
| Binary | sha256 0d9a8311... A fresh build is not expected to be byte-identical to the benchmarked binary (different build paths and link order), so the single point below compares results instead |
| Single point vs benchmarked binary | Ga peptide, `-T 1`, CPU solver: `crest.engrad` byte-identical to the benchmarked binary (sha256 of the file 39765069...; E = -415.262561312859191 Eh) |
| `run_tests.sh`, A100 80GB PCIe node | Test 1 (CPU single point): pass. Test 2 (against the benchmarked binary): bitwise identical. Test 3 (GPU single point vs CPU): max abs dE 8.9e-15 Eh, max abs dG 1.0e-15 Eh/bohr, three-body kernel ran on the GPU (1 call, 0 fallbacks). Test 4 (short GPU search): normal termination |
| Distributed passes (`--dist`, GPU driver, CPU workers on a CPU partition) | a short search in which 635 structures were optimized by Slurm worker jobs on other nodes, checked by the driver and merged; normal termination in 252 s |

The same checks passed on 2026-10-26 and 2026-10-27 for earlier revisions of the patches, which differed only in source comments and two CMake message strings.

### Version 1.1.1 (2026-11-03): eigensolve that does not converge, gfortran 13

| Item | Value |
|---|---|
| Nova build | `build.sh`, gfortran 11.5.0, CUDA 12.4.1, OpenBLAS 0.3.27: rc 0; Ga peptide CPU single point (`-T 1`) byte-identical to the 1.1.0 binary (E = -415.262561312859191 Eh) |
| `run_tests.sh`, RTX PRO 5000 Blackwell (sm_120 build, conda-forge gfortran 13.4, CUDA 13.3) | 7 of 7 passed. Tests 3 and 3b: max abs dE 8.9e-15 and 7.1e-15 Eh against the CPU. Test 3e (forced non-converged GPU solve, repeated on the CPU): max abs dE 8.9e-15 Eh, one CPU repeat. Test 3f (the CPU repeat fails too): ordinary failure, no GPU failure. Tests 4 and 4b: normal termination; in 4b one MD ended early. With 1.1.0 code, tests 4 and 4b fail with this compiler ("solvent: '' is not parametrized") |
| `run_tests.sh`, Nova A100 80GB PCIe (the Nova build above) | 9 of 9 passed: 1; 2 (bitwise identical to the 1.1.0 binary); 3, 3b; 3e (max abs dE 1.4e-14 Eh, one CPU repeat); 3f; 4, 4b; 5 (distributed passes: 601 structures optimized by Slurm workers and merged, normal termination in 497 s) |
| GPU single points vs 1.1.0, same A100 | Ga peptide, `-T 1`: `crest.engrad` byte-identical to the 1.1.0 binary on the GPU route and with `CREST_CUDA_WARMJACOBI=1` |

### Version 1.1.2 (2026-11-04): entropy mode with the GPU route, resume after a run that ended with failures

| Item | Value |
|---|---|
| Nova build | `build.sh`, gfortran 11.5.0, CUDA 12.4.1, OpenBLAS 0.3.27: rc 0 in 348 s; Ga peptide CPU single point (`-T 1`) byte-identical to the 1.1.1 binary (E = -415.262561312859191 Eh) |
| `run_tests.sh`, RTX PRO 5000 Blackwell (sm_120 build, conda-forge gfortran 13.4, CUDA 13.3) | 10 of 10 passed: 1; 3, 3b; 3e, 3f; 3h (Hessian thermochemistry with the GPU route selected: the 298.15 K result equals the CPU solver's, 70.800194); 4, 4b; 4c (a CUDA error at GPU eigensolve 25,493 of 50,986 ended the run with failures, and the run kept the ensemble of its finished MTD iteration; the resume ended normally and merged 2 ensembles for the 2 MTD iterations of the cycle); 4d (the same folder without that ensemble: resume refused with exit code 3) |
| `run_tests.sh`, Nova H200 NVL (the Nova build above) | 1; 2 (bitwise identical to the 1.1.1 binary); 3, 3b, 3e, 3f, 3h; 4, 4b, 4c, 4d; 5 (distributed passes: 599 structures optimized by Slurm workers and merged, normal termination in 250 s) passed. This run used a draft of the script in which 4c also required the resumed conformer file to equal test 4's byte for byte. That failed because two seeded test searches are not bitwise identical, so the condition was dropped; 4c passes on the same output |
| GPU single points vs 1.1.1, same H200 | Ga peptide, `-T 1`: `crest.engrad` byte-identical to the 1.1.1 binary on the GPU route and with `CREST_CUDA_WARMJACOBI=1` |
| Failures at other points of a search, RTX PRO 5000 Blackwell | The search of test 4 with a CUDA error at 3, 10, 25, 40, 60, 75, 85, 92 and 98% of its 50,986 GPU eigensolves, then a resume. All nine failed runs kept their files. All nine resumes ended normally with the ensembles of both MTD iterations: merged after the loop for failures during the MTD iterations, restored from the post-loop snapshot for failures in the MDs after the loop. This used the 12-atom test molecule; a failure in a later cycle was not tested |

### Version 1.2.0 (2026-11-05): GPU-resident SCC (opt-in), fastest mode, resume in a later cycle

| Item | Value |
|---|---|
| GPU-resident SCC (`CREST_CUDA_DEVSCC=1`) | Single points of 306 structures and controlled MD, alone and with the Jacobi solver: see the sections above |
| Whole searches in the fastest mode | Two searches of the Ga peptide with both opt-in settings, LLVM's OpenMP runtime and one socket per trajectory (one A100 per main job, optimizations on CPU worker jobs): 35.2 h and 42.7 h, 12.4x and 12.7x the throughput of the reference search |
| The four searches that stopped | Replayed on copies of their folders. With the previous build all four stop with "File crest_rotamers_3.xyz is empty" (exit code 155), as they did. With the fix all four reach their target MTD batch, the identity check passes and all six trajectories resume from their checkpoints; each run was stopped after 240 s |
| `run_tests.sh`, Nova A100 80GB PCIe | 16 of 16 passed, with new test 4e (a resume in the first MTD batch of a forced second cycle) and test 2 (bitwise identical to the previous build) |
| `run_tests.sh`, RTX PRO 5000 Blackwell | 14 of 14 passed. A build without the fix fails only test 4e, with exit code 155 |
| Every resume point | The search of test 4, forced into a second cycle, killed with SIGKILL at each of its 33 tracked steps, then resumed: all 33 resumes ended normally. CUDA errors at 8 points of the second cycle: all 8 resumes ended normally. This used the 12-atom test molecule, with threads, so it checks normal termination and the merged ensembles, not bitwise equality |
| Numbers | Ga peptide, `-T 1`: CPU single point and GPU single points (GPU route, `CREST_CUDA_DEVSCC=1`, both opt-in settings) byte-identical to the previous build |
| Clean clone of this release, Nova | Commit c41896b (the tested code plus documentation), cloned and built by `tests/site/nova/build_and_identity.sbatch` with sources fetched from GitHub, gfortran 11.5.0, CUDA 12.4.1, OpenBLAS 0.3.27: rc 0 in 270 s. Ga peptide CPU single point (`-T 1`) byte-identical to the tested build and to the benchmarked binary (E = -415.262561312859191 Eh). `run_tests.sh` on an A100 80GB PCIe: 16 of 16 passed (test 2 bitwise identical to the tested build; test 5: 619 structures optimized by Slurm workers and merged). Ga peptide GPU single points byte-identical to the tested build on the GPU route, with `CREST_CUDA_DEVSCC=1` and with both opt-in settings |
| Clean clone of this release, RTX PRO 5000 Blackwell | `build.sh` with sources fetched from GitHub, sm_120, conda-forge gfortran 13.4, CUDA 13.3: rc 0 in 223 s. `run_tests.sh`: 15 of 15 passed (test 2 bitwise identical to the tested build on the same machine) |
