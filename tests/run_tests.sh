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
#   4 search    --quick iMTD-GC search on one machine with the GPU route and checkpoints: must end normally
#   5 dist      the same search with CREST_DIST_OLOOP (optimization passes as Slurm array jobs)
set -uo pipefail

[[ ${1:-} == -h || ${1:-} == --help ]] && { sed -n '2,17p' "$0"; exit 0; }
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

  search() {  # search <dir> [env...]
    local d=$WORK/$1; shift
    rm -rf "$d"; mkdir -p "$d"; cp "$MOL" "$d/mol.xyz"; toml imtd-gc cuda "$T" > "$d/search.toml"
    ( cd "$d" && source "$HERE/../scripts/crest-env.sh" gpu && export OMP_NUM_THREADS=$T CREST_CKPT=40 CREST_RESUME_POST=fallback "$@" && \
      "$CREST" --input search.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" --alpb water --quick --noreftopo -T "$T" \
        > crest.out 2> crest.err )
    grep -q 'CREST terminated normally' "$d/crest.out"
  }
  t0=$(date +%s)
  if search search; then
    ok "4 search: normal termination in $(( $(date +%s) - t0 )) s, $(grep -c '' "$WORK/search/crest_conformers.xyz" 2>/dev/null) conformer-file lines"
  else
    bad "4 search (see $WORK/search/crest.out)"
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
