# Validation record

All tests ran on the ISU Nova cluster. T0-T2, the replicate searches and the identity gate used the 274-atom, +4 Ga peptide in ALPB water; `tests/run_tests.sh` uses the 12-atom smoke molecule. The pass criteria were written down before the runs. The routes compared:

| Route | Build |
|---|---|
| a | CREST 3.0.2 release (conda-forge), the reference |
| b | our unmodified build of CREST main at cfdc301f, from the command line |
| c | the same build, from TOML input |
| f | this build on the CPU |
| d | this build on the GPU |

## Startup (T0)

All routes made the same decisions before the first trajectory:

- the same initial-optimisation energies to 1e-10 Eh (first frame -415.2625315087 Eh on the release, -415.2625315088 Eh on the other routes);
- the same flexibility measure (0.391);
- six MTDs of 250 ps;
- 274 bonds with bond order > 0.5 and 272 SHAKE constraints on every route (counts compared).

## Single points (T1): 306 structures

The structures are the reference's 135 final conformers, 50 of its MD frames and 121 others.

| Comparison | max abs dE (Eh) | max abs dG (Eh/bohr) |
|---|---|---|
| b vs a (our build vs release) | 1.01e-10 | 8.16e-11 |
| c vs b (TOML vs command line) | 0 (bitwise) | 0 |
| f vs c (this build on the CPU vs unmodified) | 0 (bitwise) | 0 |
| d vs f (GPU vs CPU route) | 3.98e-13 | 4.0e-15 |
| d vs a (GPU route vs release) | 1.01e-10 | 8.16e-11 |

## Controlled MD (T2)

Each comparison used 20 ps runs from four starts, at 400 and 500 K, with identical initial velocities on both routes. The score is the fraction of frames whose bonding topology changed. A pair is equivalent if the 90% confidence interval of the mean paired difference lies within ±0.15 (two one-sided tests).

| Comparison | raw frames | after optimisation |
|---|---|---|
| b vs b (same binary twice) | -0.024 to +0.017 | -0.084 to +0.041 |
| b vs a | -0.015 to +0.017 | -0.039 to +0.030 |
| d vs f (GPU vs CPU route) | -0.000 to +0.050 | +0.008 to +0.134 |
| d vs a (GPU route vs release) | -0.003 to +0.033 | -0.014 to +0.082 |

Every comparison is equivalent.

## Replicate searches

**Endpoint.** The lowest crude-level energy after the first MTD round.

**Arms.** 16 GPU searches against 17 stock-numerics searches.

**Result.**

- **Shift:** the Hodges-Lehmann shift was +4.09 kcal/mol, with an exact 95% interval of -4.94 to +11.75.
- **Verdict:** no detectable shift.
- **Limit:** equivalence within 2 kcal/mol could not be shown. Stock CREST alone spreads over about 13 kcal/mol (interquartile range) on this system, and its final energies over 125 kcal/mol between runs from the same input.

## Release build check

`build.sh` builds from upstream plus `patches/`. The tests for the released build are:

- **Byte-identity gate:** a single-thread single point of the Ga peptide must give a `crest.engrad` byte-identical to the binary validated above (sha256 ec9ec385).
- **Repeatable checks:** `tests/run_tests.sh` repeats the single-point checks (CPU vs reference, GPU vs CPU) and runs a short GPU search, optionally with distributed passes.

### Results on ISU Nova (2026-10-26)

| Item | Value |
|---|---|
| Patches tested (sha256 prefix) | `0001-crest` b8c2eabb, `0002-tblite` de427923, dftd4 a0d1c23f, mctc-lib dedf3bb8, multicharge 5ea75735, s-dftd3 b3387586 |
| Build | `build.sh` of this repository with sources fetched from GitHub, gfortran 11.5.0, CMake 3.30.2, CUDA 12.4.1, OpenBLAS 0.3.27; 322 s on 8 cores (an earlier test of the same patches took 334 s) |
| Binary | sha256 46fce78b... (the earlier test build: e6a6231d...). A fresh build is not expected to match ec9ec385 byte for byte (different build paths and link order, not checked in detail); the identity gate below compares results |
| Identity gate | Ga peptide, `-T 1`, CPU solver: `crest.engrad` byte-identical to ec9ec385 (sha256 39765069...; E = -415.262561312859191 Eh) |
| `run_tests.sh`, A100 node | 1 CPU single point: pass. 2 vs ec9ec385: bitwise identical. 3 GPU single point vs CPU: max abs dE 8.9e-15 Eh, max abs dG 1.0e-15 Eh/bohr, three-body kernel ran on the GPU. 4 short GPU search: normal termination |
| Distributed passes (`--dist`, A100 driver, CPU workers on a CPU partition) | a short search in which 651 structures were optimised by Slurm worker jobs on other nodes, checked by the driver and merged; normal termination in 251 s (626 structures in the earlier test) |

