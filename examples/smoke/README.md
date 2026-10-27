# Smoke test molecule

`struc.xyz` is the 12-atom example from the CREST repository (`examples/expl-0`). `tests/run_tests.sh` uses it to check that a build works: CPU and GPU single points and a short GPU search.

A molecule this small does not benefit from the GPU. The trajectory step only pays off when the eigensolve is large: our test system had 274 atoms and 671 basis functions, giving a 6x faster MTD step on an A100. For a speed check, run your own system of 100 atoms or more.
