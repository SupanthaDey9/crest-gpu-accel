# Validation record

All tests ran on the ISU Nova cluster. The code-level tests and the replicate searches used the 274-atom, +4 Ga peptide in ALPB water; `tests/run_tests.sh` uses the 12-atom smoke molecule. The pass criteria of the code-level tests were fixed before the runs; the rule for the replicate searches was fixed before any group summary, after some single-run values had been seen. Five configurations were compared:

| Name | What it is |
|---|---|
| release | CREST 3.0.2 release (conda-forge), used by the conventional search |
| unmodified | our build of CREST commit cfdc301f without these patches, run from the command line |
| unmodified, TOML | the same build, run from TOML input |
| FastCREST, CPU | FastCREST with every GPU switch off |
| FastCREST, GPU | FastCREST with `tblite_solver = "cuda"` and `CREST_ATM_GPU=1` |

## Startup decisions

Every configuration made the same decisions before the first trajectory:

- the same initial-optimization energies to 1e-10 Eh (first frame -415.2625315087 Eh on the release, -415.2625315088 Eh on the others);
- the same flexibility measure (0.391);
- six MTDs of 250 ps;
- 274 bonds with bond order > 0.5 and 272 SHAKE constraints (counts compared).

## Single points: 306 structures

The structures are the conventional search's 135 final conformers, 50 of its MD frames and 121 others.

| Comparison | max abs dE (Eh) | max abs dG (Eh/bohr) |
|---|---|---|
| unmodified vs release | 1.01e-10 | 8.16e-11 |
| unmodified: TOML vs command line | 0 (bitwise) | 0 |
| FastCREST, CPU vs unmodified | 0 (bitwise) | 0 |
| FastCREST: GPU vs CPU | 3.98e-13 | 4.0e-15 |
| FastCREST, GPU vs release | 1.01e-10 | 8.16e-11 |

## Controlled MD

Each comparison used 20 ps runs from four starts, at 400 and 500 K, with identical initial velocities on both sides. The score is the fraction of frames whose bonding topology changed. A pair counts as equivalent if the 90% confidence interval of the mean paired difference lies within ±0.15 (two one-sided tests).

| Comparison | raw frames | after optimization |
|---|---|---|
| unmodified, same binary twice | -0.024 to +0.017 | -0.084 to +0.041 |
| unmodified vs release | -0.015 to +0.017 | -0.039 to +0.030 |
| unmodified: TOML vs command line | -0.005 to +0.034 | -0.030 to +0.110 |
| FastCREST, CPU vs unmodified (TOML) | -0.043 to +0.005 | -0.137 to -0.007 |
| FastCREST: GPU vs CPU | -0.000 to +0.050 | +0.008 to +0.134 |
| FastCREST, GPU vs unmodified (command line) | -0.001 to +0.030 | +0.005 to +0.072 |
| FastCREST, GPU vs release | -0.003 to +0.033 | -0.014 to +0.082 |

Every comparison is equivalent. Three intervals after optimization exclude zero while staying inside the margin; one of them compares two CPU configurations whose single-thread forces are bitwise identical, so we read them as sampling noise.

## Replicate searches

**Endpoint.** The lowest crude-level energy after the first MTD round.

**Groups.** 16 FastCREST searches against 17 stock searches (unmodified numerics, including the FastCREST binary with its GPU code off).

**Result.**

- **Shift:** the Hodges-Lehmann shift was +4.09 kcal/mol, with an exact 95% interval of -4.94 to +11.75.
- **Verdict:** inconclusive; no detectable shift.
- **Limit:** equivalence within 2 kcal/mol could not be shown. Stock searches alone spread over about 13 kcal/mol (interquartile range) on this system. Thirteen of the 15 finished stock searches ended between -1.1 and +24.1 kcal/mol relative to the conventional search; the other two ended more than 110 kcal/mol higher because of an upstream CREST fallback: when the topology check rejects every structure of a CREGEN call, CREGEN keeps the reference geometry (`coord`, here the helical input) labeled with the lowest energy seen, and the MD stage then starts from it. The fallback is in `cregen.f90`, unchanged by our patches and identical in the 3.0.2 release.

## Check of the released build

`build.sh` builds from upstream plus `patches/`. Two checks are run on the result:

- **Same numbers as the benchmarked binary:** a single-thread single point of the Ga peptide must give a `crest.engrad` byte-identical to the one from the binary that produced every result above.
- **Repeatable checks:** `tests/run_tests.sh` repeats the single-point checks (CPU against a reference binary, GPU against CPU) and runs a short GPU search, optionally with distributed passes.

### Results on ISU Nova (2026-10-28, fresh clone of this repository from GitHub, commit d6c51ab)

| Item | Value |
|---|---|
| Patches tested (sha256 prefix) | `0001-crest` 3d0d8ca6, `0002-tblite` a4ae8643, dftd4 08ac3376, mctc-lib dedf3bb8, multicharge 5ea75735, s-dftd3 b3387586 |
| Build | `build.sh` with sources fetched from GitHub, gfortran 11.5.0, CMake 3.30.2, CUDA 12.4.1, OpenBLAS 0.3.27; 414 s on 8 cores of a Xeon Platinum 8260 node (earlier builds of the same patches took 291 and 322 s) |
| Binary | sha256 0d9a8311... A fresh build is not expected to be byte-identical to the benchmarked binary (different build paths and link order), so the single point below compares results instead |
| Single point vs benchmarked binary | Ga peptide, `-T 1`, CPU solver: `crest.engrad` byte-identical to the benchmarked binary (sha256 of the file 39765069...; E = -415.262561312859191 Eh) |
| `run_tests.sh`, A100 80GB PCIe node | 1 CPU single point: pass. 2 against the benchmarked binary: bitwise identical. 3 GPU single point vs CPU: max abs dE 8.9e-15 Eh, max abs dG 1.0e-15 Eh/bohr, three-body kernel ran on the GPU (1 call, 0 fallbacks). 4 short GPU search: normal termination |
| Distributed passes (`--dist`, GPU driver, CPU workers on a CPU partition) | a short search in which 635 structures were optimized by Slurm worker jobs on other nodes, checked by the driver and merged; normal termination in 252 s |

The same checks passed on 2026-10-26 and 2026-10-27 for earlier revisions of the patches, which differed only in source comments and two CMake message strings.
