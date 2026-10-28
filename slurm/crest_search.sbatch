#!/bin/bash -l
# crest_search.sbatch: one FastCREST iMTD-GC search as a Slurm job (driver with one GPU + distributed CPU workers).
#
#   sbatch [your partition/account options] slurm/crest_search.sbatch molecule.xyz CHARGE UHF [extra crest options]
#
# Environment (export before sbatch, or edit the defaults below):
#   CREST_BIN            FastCREST crest binary (default: <repo>/bin/crest)
#   OUT                  search folder (default: $PWD/crest-<jobid>); resubmit with the same OUT to resume a folder
#   WORKER_SBATCH_ARGS   sbatch options for worker jobs, e.g. "--partition=cpu --account=me --mem=8G --time=02:00:00".
#                        Empty: no distribution, every optimization pass runs inside this job. Quote any '|' inside.
#   WORKER_ENV           worker environment file (default: <repo>/slurm/worker_env.sh; edit it for your site)
#   SOLVENT              ALPB solvent (default: water; "none" for gas phase)
#   EXTRA_TOML           extra lines for the [[calculation.level]] block
#   REPO                 path of this repository (default: found from the submitted script path)
#
# search.toml and dist_oloop.conf are written on the first run only; edit them in OUT to change a running search.
# Load the compiler runtime, OpenBLAS and CUDA modules the binary was built with before calling sbatch (or below).
#
# Time limit: give the job the time the whole search needs. The stage after the MTD loop (MD stage + its optimization
# pass) can only restart from its beginning, so short self-requeuing segments may never finish it.
#SBATCH --job-name=crest-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
set -Euo pipefail
# The repository is found from the submitted script path (first word of the job command), or set REPO explicitly.
if [[ -z ${REPO:-} ]]; then
  cmd=$(scontrol show job "${SLURM_JOB_ID:?}" | sed -n 's/^ *Command=//p' | head -1)
  REPO=$(cd "$(dirname "${cmd%% *}")/.." 2>/dev/null && pwd) || REPO=''
fi
[[ -f ${REPO:-/nonexistent}/scripts/crest-env.sh ]] || { echo "cannot find the repository (set REPO=/path/to/FastCREST)" >&2; exit 2; }
MOL=$(realpath "${1:?usage: crest_search.sbatch molecule.xyz CHARGE UHF [crest options]}"); CHRG=${2:?charge}; UHF=${3:?uhf}; shift 3
REPO=$(cd "$REPO" && pwd)
CREST_BIN=$(realpath -m "${CREST_BIN:-$REPO/bin/crest}")
OUT=$(realpath -m "${OUT:-$PWD/crest-$SLURM_JOB_ID}")
WORKER_ENV=$(realpath -m "${WORKER_ENV:-$REPO/slurm/worker_env.sh}")
SOLVENT=${SOLVENT:-water}
T=${SLURM_CPUS_PER_TASK:-32}
mkdir -p "$OUT" && cd "$OUT" || exit 2
SEG=$(( $(ls segment-*.txt 2>/dev/null | wc -l) + 1 ))
echo "segment $SEG start $(date -u +%FT%TZ) node $(hostname -s) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" | tee "segment-$SEG.txt"

if [[ ! -f search.toml ]]; then
  cp "$MOL" input.xyz || exit 2
  solv=''; [[ $SOLVENT != none ]] && solv="alpb = \"$SOLVENT\""
  cat > search.toml <<EOF
input = "input.xyz"
runtype = "imtd-gc"
threads = $T

[[calculation.level]]
method = "gfn2"
charge = $CHRG
uhf = $UHF
$solv
tblite_solver = "cuda"
${EXTRA_TOML:-}
EOF
fi
if grep -q '^charge' search.toml && ! grep -q "^charge = $CHRG\$" search.toml; then
  echo "WARNING: $OUT/search.toml exists with another charge; it is kept (edit it to change settings)" >&2
fi
# A requeued or resubmitted job resumes from CREST's own restart file plus the CREST_CKPT checkpoints.
if [[ -f crest.restart ]] && ! grep -q '^restart *= *true' search.toml; then
  sed -i '1i restart = true' search.toml
fi

# The first submission with WORKER_SBATCH_ARGS writes dist_oloop.conf; a resubmission uses the existing file.
if [[ -n ${WORKER_SBATCH_ARGS:-} && ! -f dist_oloop.conf ]]; then
  cat > dist_oloop.conf <<EOF
workdir = $OUT/dist_passes
worker_script = $REPO/slurm/dist_worker.sbatch
worker_env = $WORKER_ENV
sbatch_args = $WORKER_SBATCH_ARGS
worker_threads = 16
chunk = 64
throttle = 60
poll = 60
min_structures = 128
verify = 32
verify_other = 32
same_cpu = 0
local = 1
lost_wait = 90
submit = 1
max_rounds = 10
EOF
fi
[[ -f dist_oloop.conf ]] && export CREST_DIST_OLOOP=$OUT/dist_oloop.conf

source "$REPO/scripts/crest-env.sh" gpu
export OMP_NUM_THREADS=$T CREST_CKPT=40 CREST_RESUME_POST=fallback
requeue_self() { echo "segment $SEG: time limit near, requeue $(date -u +%FT%TZ)" | tee -a "segment-$SEG.txt"; scontrol requeue "$SLURM_JOB_ID"; }
trap requeue_self USR1
solvflag=(); [[ $SOLVENT != none ]] && solvflag=(--alpb "$SOLVENT")
"$CREST_BIN" --input search.toml --newversion --gfn2 --chrg "$CHRG" --uhf "$UHF" "${solvflag[@]}" -T "$T" "$@" \
  >> crest.out 2>> crest.err &
pid=$!
wait "$pid"; rc=$?
if kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi
echo "segment $SEG end $(date -u +%FT%TZ) rc $rc" | tee -a "segment-$SEG.txt"
grep -h 'CREST_ATM_GPU summary\|CREST_DIST> pass' crest.err crest.out 2>/dev/null | tail -n 5 || true
exit "$rc"
