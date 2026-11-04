#!/usr/bin/env bash
# run_tests.sh: check a FastCREST build.
#
#   tests/run_tests.sh bin/crest                         # CPU and GPU single points + a short GPU search
#   tests/run_tests.sh bin/crest --reference /path/crest # also compare the CPU single point with a reference build
#   tests/run_tests.sh bin/crest --dist "--partition=X --account=Y --time=00:30:00"
#                                                        # also run the short search with distributed passes (Slurm)
#   tests/run_tests.sh bin/crest --no-gpu                # CPU-only checks (e.g. on a login node)
# Options: --molecule FILE --charge Q --uhf U (default: examples/smoke/struc.xyz, 0, 0), --workdir DIR, -T N,
#          --worker-env FILE (worker environment for --dist; default slurm/worker_env.sh).
#
# Tests
#   1 sp_cpu    single point, CPU solver, 1 thread, all CREST_* switches unset (the stock code path)
#   2 sp_ref    the same with --reference: must agree to 1e-8 Eh, 1e-6 Eh/bohr (bitwise for the same CREST commit)
#   3 sp_gpu    single point with tblite_solver = "cuda" and CREST_ATM_GPU=1: must match test 1 to 1e-8 Eh, 1e-6 Eh/bohr
#   3b sp_jacobi  the same with CREST_CUDA_WARMJACOBI=1 (opt-in): must match test 1 likewise, with warm Jacobi solves
#               and no fallback to the standard solver
#   3e sp_retry   test 3 with CREST_CUDA_TEST_NOCONV=2: the second GPU eigensolve reports that it did not converge
#               and is repeated on the CPU (LAPACK); must match test 1 likewise, with one CPU repeat and no GPU failure
#   3f sp_retry_fail  the same with CREST_CUDA_TEST_NOCONV=2,cpu (the CPU repeat fails too): the single point must
#               fail as an ordinary SCC failure, as in conventional CREST, not as a GPU failure
#   3h thermo     Hessian thermochemistry (--rrhoav, as in the entropy mode) with the GPU route selected: its
#               Hessians use the CPU route, so the result must be finite and byte-identical to the CPU solver's
#   4 search    --quick iMTD-GC search on one machine with the GPU route and checkpoints: must end normally
#   4b search_retry_fail  test 4 with CREST_CUDA_TEST_NOCONV=5000,cpu: one GPU eigensolve fails on the GPU and the
#               CPU, so that MD ends early as in conventional CREST; the search must still end normally
#   4c resume_after_failure  test 4 with a CUDA error injected halfway (CREST_CUDA_TEST_NOCONV=k,fault): the run ends
#               with failures but keeps its files; the resume must end normally and merge the ensembles of all MTD
#               iterations of its cycle (its conformer file is compared with test 4's for information only)
#   4d resume_guard  that failed folder without its ensemble backups (.cre_*, as runs before 1.1.2 left it): the
#               resume must be refused with exit code 3
#   5 dist      the same search with CREST_DIST_OLOOP (optimization passes as Slurm array jobs)
set -uo pipefail

[[ ${1:-} == -h || ${1:-} == --help ]] && { sed -n '2,31p' "$0"; exit 0; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CREST=$(realpath "${1:?usage: run_tests.sh CREST [options]}") || exit 64; shift
REF=''; DIST=''; WENV=$HERE/../slurm/worker_env.sh; GPU=1; MOL=$HERE/../examples/smoke/struc.xyz; CHRG=0; UHF=0; WORK=$PWD/crest-tests; T=4
while [[ $# -gt 0 ]]; do
  case $1 in
    --reference) REF=$(realpath "${2:?--reference needs a path}") || exit 64; shift 2 ;;
    --dist) DIST=$2; shift 2 ;;
    --worker-env) WENV=$2; shift 2 ;;
    --no-gpu) GPU=0; shift ;;
    --molecule) MOL=$2; shift 2 ;;
    --charge) CHRG=$2; shift 2 ;;
    --uhf) UHF=$2; shift 2 ;;
    --workdir) WORK=$2; shift 2 ;;
    -T) T=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done
if [[ -n $DIST && $GPU == 0 ]]; then echo "--dist runs only together with the GPU tests; drop --no-gpu" >&2; exit 64; fi
MOL=$(realpath "$MOL")
mkdir -p "$WORK"; WORK=$(realpath "$WORK")
unset $(env | sed -n 's/^\(CREST_[A-Z_]*\)=.*/\1/p') 2>/dev/null || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PASS=0; FAIL=0
ok() { echo "PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL  $1"; FAIL=$((FAIL + 1)); }

toml() {  # toml <runtype> <solver> <threads>
  cat <<EOF
input = "mol.xyz"
runtype = "$1"
threads = $3
omp_nested = false

[[calculation.level]]
method = "gfn2"
charge = $CHRG
uhf = $UHF
alpb = "water"
etemp = 300.0
accuracy = 1.0
maxscc = 500
EOF
  # the GPU solver keys are written only when needed, so that an unpatched CREST can serve as --reference
  if [[ $2 == cuda ]]; then printf 'tblite_solver = "cuda"\ntblite_gpu_audit = false\n'; fi
}

sp() {  # sp <dir> <binary> <solver> [env assignments...]
  local d=$WORK/$1 bin=$2 solver=$3; shift 3
  rm -rf "$d"; mkdir -p "$d"; cp "$MOL" "$d/mol.xyz"; toml singlepoint "$solver" 1 > "$d/sp.toml"
  ( cd "$d" && env "$@" "$bin" --input sp.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" --alpb water -T 1 \
      > crest.out 2> crest.err )
  [[ -s $d/crest.engrad ]]
}

echo "== CREST under test: $CREST"
if sp sp_cpu "$CREST" cpu; then ok "1 sp_cpu: $(sed -n 8p "$WORK/sp_cpu/crest.engrad" | tr -d ' ') Eh"; else bad "1 sp_cpu (see $WORK/sp_cpu/crest.err)"; fi

if [[ -n $REF ]]; then
  # same CPU model and OpenBLAS: an unmodified build of the same CREST commit gives byte-identical files; the 3.0.2
  # release differs at the 1e-10 Eh level, inside the 1e-8 Eh tolerance
  if sp sp_ref "$REF" cpu; then
    r=$(python3 "$HERE/compare_engrad.py" "$WORK/sp_cpu/crest.engrad" "$WORK/sp_ref/crest.engrad" 1e-8 1e-6); rc=$?
    if [[ $rc == 0 ]]; then ok "2 sp_ref: $r"; else bad "2 sp_ref: $r"; fi
  else
    bad "2 sp_ref: reference build produced no result (see $WORK/sp_ref/crest.err)"
  fi
fi

if [[ $GPU == 1 ]]; then
  # the three-body kernel must really run: a summary with gpu_calls 0 means the CPU fallback was used
  if sp sp_gpu "$CREST" cuda CREST_ATM_GPU=1 && grep -q 'CREST_ATM_GPU summary gpu_calls [1-9]' "$WORK/sp_gpu/crest.err" "$WORK/sp_gpu/crest.out"; then
    r=$(python3 "$HERE/compare_engrad.py" "$WORK/sp_gpu/crest.engrad" "$WORK/sp_cpu/crest.engrad" 1e-8 1e-6); rc=$?
    summ=$(grep -h 'CREST_ATM_GPU summary' "$WORK/sp_gpu/crest.err" "$WORK/sp_gpu/crest.out" | tail -1)
    if [[ $rc == 0 ]]; then ok "3 sp_gpu: $r; $summ"; else bad "3 sp_gpu: $r"; fi
  else
    bad "3 sp_gpu: no GPU result, or the three-body kernel fell back to the CPU (see $WORK/sp_gpu/crest.err)"
  fi

  # opt-in warm-started Jacobi eigensolver: after the first SCC iteration the overlap factor is reused and syevj starts
  # from the previous eigenvectors; STATS must show warm solves and no fallback to the standard solver
  if sp sp_jacobi "$CREST" cuda CREST_ATM_GPU=1 CREST_CUDA_WARMJACOBI=1; then
    stats=$(grep -h 'CREST_CUDA_WARMJACOBI_STATS' "$WORK/sp_jacobi/crest.err" | tail -1)
    r=$(python3 "$HERE/compare_engrad.py" "$WORK/sp_jacobi/crest.engrad" "$WORK/sp_cpu/crest.engrad" 1e-8 1e-6); rc=$?
    if [[ $rc == 0 && $stats =~ \ warm\ [1-9] && $stats =~ \ stock\ 0\ failed\ 0\  ]]; then
      ok "3b sp_jacobi: $r; ${stats#CREST_CUDA_WARMJACOBI_STATS }"
    else
      bad "3b sp_jacobi: $r; ${stats:-no STATS line}"
    fi
  else
    bad "3b sp_jacobi: no result (see $WORK/sp_jacobi/crest.err)"
  fi

  # A GPU eigensolve that does not converge (sygvd devInfo > 0) is repeated on the CPU with LAPACK, as conventional
  # CREST computes it. CREST_CUDA_TEST_NOCONV=k makes the k-th GPU eigensolve report that without running.
  retry_stats() { grep -h 'CREST_CUDA_CPU_RETRY_STATS' "$WORK/$1/crest.err" | tail -1; }
  gpu_failed() { grep -q -E 'CREST CUDA [a-z ]+ failed|GPU failure' "$WORK/$1/crest.err" "$WORK/$1/crest.out"; }
  if sp sp_retry "$CREST" cuda CREST_ATM_GPU=1 CREST_CUDA_TEST_NOCONV=2; then
    stats=$(retry_stats sp_retry)
    r=$(python3 "$HERE/compare_engrad.py" "$WORK/sp_retry/crest.engrad" "$WORK/sp_cpu/crest.engrad" 1e-8 1e-6); rc=$?
    if [[ $rc == 0 && $stats == *' solves 1 cpu_solved 1 cpu_failed 0'* ]] && ! gpu_failed sp_retry; then
      ok "3e sp_retry: $r; ${stats#CREST_CUDA_CPU_RETRY_STATS }"
    else
      bad "3e sp_retry: $r; ${stats:-no STATS line}"
    fi
  else
    bad "3e sp_retry: no result (see $WORK/sp_retry/crest.err)"
  fi

  # the CPU repeat fails too: the single point fails as an ordinary SCC failure (no crest.engrad, no GPU failure)
  sp sp_retry_fail "$CREST" cuda CREST_ATM_GPU=1 CREST_CUDA_TEST_NOCONV=2,cpu
  stats=$(retry_stats sp_retry_fail)
  if [[ ! -s $WORK/sp_retry_fail/crest.engrad && $stats == *' solves 1 cpu_solved 0 cpu_failed 1'* ]] && ! gpu_failed sp_retry_fail; then
    ok "3f sp_retry_fail: ${stats#CREST_CUDA_CPU_RETRY_STATS }; ordinary failure, no GPU failure"
  else
    bad "3f sp_retry_fail: ${stats:-no STATS line} (see $WORK/sp_retry_fail)"
  fi

  # Hessian thermochemistry with the GPU route selected (the entropy mode's last step, here via --rrhoav on the test
  # molecule): no GPU scope is open there, so its Hessians use the CPU route; the 298.15 K results must be finite and
  # byte-identical to those with the CPU solver
  thermo() {  # thermo <dir> <solver>
    local d=$WORK/$1
    rm -rf "$d"; mkdir -p "$d"; cp "$MOL" "$d/mol.xyz"; toml singlepoint "$2" 4 > "$d/thermo.toml"
    ( cd "$d" && env CREST_ATM_GPU=1 "$CREST" mol.xyz --input thermo.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" \
        --alpb water -T 4 --rrhoav mol.xyz > crest.out 2> crest.err )
    grep -E '^ +298\.15 ' "$d/crest.out"
  }
  tc=$(thermo thermo_cpu cpu); tg=$(thermo thermo_gpu cuda)
  if [[ -n $tc && $tc == "$tg" ]] && ! grep -q -i -E 'nan|infinity' "$WORK/thermo_gpu/crest.out"; then
    ok "3h thermo: GPU route selected = CPU solver at 298.15 K ($(head -1 <<< "$tc" | tr -s ' '))"
  else
    bad "3h thermo: CPU [$(head -1 <<< "$tc" | tr -s ' ')] GPU route [$(head -1 <<< "$tg" | tr -s ' ')] (see $WORK/thermo_gpu)"
  fi

  search() {  # search <dir> [env...]
    local d=$WORK/$1; shift
    rm -rf "$d"; mkdir -p "$d"; cp "$MOL" "$d/mol.xyz"; toml imtd-gc cuda "$T" > "$d/search.toml"
    ( cd "$d" && source "$HERE/../scripts/crest-env.sh" gpu && export OMP_NUM_THREADS=$T CREST_CKPT=40 CREST_RESUME_POST=fallback "$@" && \
      "$CREST" --input search.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" --alpb water --quick --noreftopo -T "$T" \
        > crest.out 2> crest.err )
    grep -q 'CREST terminated normally' "$d/crest.out"
  }
  t0=$(date +%s)
  if search search CREST_CUDA_TEST_NOCONV=0 CREST_MD_SEED=20261104; then
    ok "4 search: normal termination in $(( $(date +%s) - t0 )) s, $(grep -c '' "$WORK/search/crest_conformers.xyz" 2>/dev/null) conformer-file lines"
  else
    bad "4 search (see $WORK/search/crest.out)"
  fi

  # A CUDA error halfway through test 4's GPU eigensolves stops the run with failures; with checkpoints the run keeps
  # its files, and the resume must end normally and merge one ensemble per MTD iteration of its cycle (before 1.1.2
  # the failed run deleted them and the resume merged only the iterations it computed itself). CREST_MD_SEED (test
  # only) fixes the initial MD velocities, but two identical test searches are still not bitwise identical, so the
  # conformer file is compared with test 4's for information only. The same failed folder without its per-iteration
  # ensemble backups must be refused (exit 3) instead of silently dropping those conformers.
  resume() {  # resume <dir>: rerun the search in place with restart = true
    local d=$WORK/$1
    sed -i '1i restart = true' "$d/search.toml"
    ( cd "$d" && source "$HERE/../scripts/crest-env.sh" gpu && export OMP_NUM_THREADS=$T CREST_CKPT=40 CREST_RESUME_POST=fallback CREST_MD_SEED=20261104 && \
      "$CREST" --input search.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" --alpb water --quick --noreftopo -T "$T" \
        >> crest.out 2>> crest.err )
  }
  n4=$(sed -n 's/^CREST_CUDA_TEST_NOCONV solves \([0-9]*\)$/\1/p' "$WORK/search/crest.err" | tail -1)
  if [[ -n $n4 && $n4 -gt 1 ]]; then
    search search_fault CREST_CUDA_TEST_NOCONV=$(( n4 / 2 )),fault CREST_MD_SEED=20261104
    f=$WORK/search_fault
    ncre=$(ls -a "$f" | grep -c '^\.cre_')
    if grep -q 'CREST terminated with failures' "$f/crest.out" && grep -q 'its files are kept for a resume' "$f/crest.out"; then
      rm -rf "$WORK/search_guard"; cp -a "$f" "$WORK/search_guard"
      nl=$(grep -c '' "$f/crest.out")
      resume search_fault; rc=$?
      part=$(tail -n +$(( nl + 1 )) "$f/crest.out")
      iters=$(sed -n '1,/Collecting ensmbles/p' <<< "$part" | grep -c 'Meta-Dynamics Iteration')
      merged=$(grep -m1 -o 'post-loop entry snapshot: [0-9]*' <<< "$part" | awk '{print $NF}')
      same=$(cmp -s "$f/crest_conformers.xyz" "$WORK/search/crest_conformers.xyz" && echo "identical to" || echo "differs from")
      if [[ $rc == 0 ]] && grep -q 'CREST terminated normally' <<< "$part" && [[ -n $merged && $merged == "$iters" ]]; then
        ok "4c resume_after_failure: fault at solve $(( n4 / 2 )) of $n4 ($ncre ensemble backups kept); resume merged $merged ensembles for $iters MTD iterations; conformer file $same test 4"
      else
        bad "4c resume_after_failure: resume rc $rc; merged ${merged:-none} ensembles for $iters MTD iterations (see $f)"
      fi
      if [[ $ncre -gt 0 ]]; then
        g=$WORK/search_guard
        mkdir -p "$g/moved"; mv "$g"/.cre_* "$g/moved/"
        resume search_guard; rc=$?
        if [[ $rc == 3 ]] && grep -q 'this resume would drop the others' "$g/crest.err"; then
          ok "4d resume_guard: refused with exit code 3 ($(grep -m1 -o 'only [0-9]* of the [0-9]* MTD iterations' "$g/crest.err"))"
        else
          bad "4d resume_guard: resume rc $rc, no refusal (see $g)"
        fi
      else
        echo "NOTE  4d resume_guard not run: the fault came before an MTD iteration finished (no ensemble backup yet)"
      fi
    else
      bad "4c resume_after_failure: the injected fault did not end the run with failures and kept files (see $f)"
    fi
  else
    bad "4c resume_after_failure: no GPU eigensolve count from test 4 (see $WORK/search/crest.err)"
  fi

  # one GPU eigensolve inside the search fails on the GPU and on the CPU: that MD ends early, as in conventional CREST,
  # and the search must still end normally
  t0=$(date +%s)
  if search search_retry_fail CREST_CUDA_TEST_NOCONV=5000,cpu && \
     [[ $(retry_stats search_retry_fail) == *' cpu_failed 1'* ]] && ! gpu_failed search_retry_fail; then
    ok "4b search_retry_fail: normal termination in $(( $(date +%s) - t0 )) s; $(grep -c 'terminated EARLY' "$WORK/search_retry_fail/crest.out") MD(s) ended early"
  else
    bad "4b search_retry_fail: $(retry_stats search_retry_fail) (see $WORK/search_retry_fail/crest.out)"
  fi

  if [[ -n $DIST ]]; then
    mkdir -p "$WORK/dist"
    cat > "$WORK/dist_oloop.conf" <<EOF
workdir = $WORK/dist/passes
worker_script = $(realpath "$HERE/../slurm/dist_worker.sbatch")
worker_env = $(realpath "$WENV")
sbatch_args = $DIST
worker_threads = 4
chunk = 8
throttle = 20
poll = 30
min_structures = 16
verify = 4
verify_other = 4
same_cpu = 0
local = 0
lost_wait = 90
submit = 1
max_rounds = 10
EOF
    # local = 0: the driver waits for the workers instead of computing chunks itself, so the worker path is tested
    t0=$(date +%s)
    if search dist "CREST_DIST_OLOOP=$WORK/dist_oloop.conf" && grep -q 'CREST_DIST>' "$WORK/dist/crest.out" "$WORK/dist/crest.err"; then
      nw=$(grep -h 'all .* chunks done' "$WORK/dist/crest.out" | sed -n 's/.*same CPU model) \([0-9]*\), workers (other CPU model) \([0-9]*\).*/\1 \2/p' | awk '{s+=$1+$2} END {print s+0}')
      if [[ $nw -gt 0 ]]; then
        ok "5 dist: normal termination in $(( $(date +%s) - t0 )) s; $nw structures optimized by Slurm workers and merged"
      else
        bad "5 dist: the search finished but no structure came from a worker"
      fi
    else
      bad "5 dist (see $WORK/dist/crest.out)"
    fi
  fi
fi
echo "== $PASS passed, $FAIL failed (work folder $WORK)"
[[ $FAIL == 0 ]]
