#!/bin/bash -l
# dist_worker.sbatch: one Slurm array task = one chunk of one distributed optimisation pass (CREST_DIST_OLOOP).
# The CREST driver submits it itself:
#   sbatch --parsable --export=NONE --job-name=cdo_<tag> --array=<chunks>%<throttle> --cpus-per-task=<TW> \
#          --output=<pass>/logs/w_%A_%a.out <sbatch_args of the config> dist_worker.sbatch <pass dir>
# The task runs the driver's own crest binary (path and sha256 from <pass>/worker.env) as
# runtype ancopt_ensemble on <pass>/worker.toml in a job-private /tmp folder; crest reads the exact pass
# coordinates and the pass's optimizer settings from chunks/chunk_<c>.bin (CREST_DIST_OLOOP_WORKER), refuses on
# any settings mismatch, and writes results/chunk_<c>.bin atomically (CREST_DIST_OLOOP_RESULT).
# A task whose result exists exits at once (checked at the start and again just before crest starts), so requeued
# (preempted) or resubmitted tasks do not redo a chunk. The driver cancels pending tasks of chunks it computes itself.
# Nothing is deleted by this script; it works in $TMPDIR (or /tmp), which most Slurm sites clean after the job.
#SBATCH --job-name=crest-dist-w
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -Eo pipefail
P=${1:?usage: dist_worker.sbatch <pass dir>}
c=${SLURM_ARRAY_TASK_ID:?not an array task}
C=$(printf '%04d' "$c")
R=$P/results/chunk_$C.bin
echo "== crest-dist worker: chunk $c job ${SLURM_JOB_ID:-} array ${SLURM_ARRAY_JOB_ID:-}_${c} restart ${SLURM_RESTART_COUNT:-0} host $(hostname -s) $(date -u +%FT%TZ)"
if [[ -e $R ]]; then echo "chunk $c already has a result; nothing to do"; exit 0; fi
for f in PASS.txt worker.env worker.toml ref.xyz chunks.tsv "chunks/chunk_$C.bin"; do
  [[ -f $P/$f ]] || { echo "pass folder incomplete: $P/$f missing" >&2; exit 3; }
done
# PASS_ID TW NCHUNK WORKER_ENV CREST_EXE CREST_SHA (written by the driver)
source "$P/worker.env"
n=$(awk -v c="$c" '$1 == c {print $2}' "$P/chunks.tsv")
[[ -n $n ]] || { echo "chunk $c is not in chunks.tsv" >&2; exit 3; }
W=${TMPDIR:-/tmp}/crest-dist-${SLURM_JOB_ID:-$$}-$c
mkdir -p "$W"
cd "$W"
echo "$CREST_SHA  $CREST_EXE" | sha256sum -c --quiet || { echo "crest binary differs from the driver's" >&2; exit 3; }
cp "$P/worker.toml" worker.toml
cp "$P/ref.xyz" ref.xyz
# ancopt_ensemble needs an n-frame ensemble to size its arrays; crest then replaces every frame by the exact
# (Bohr, binary) coordinates of the chunk file.
for ((i = 0; i < n; i++)); do cat ref.xyz; done > ensemble.xyz
if [[ -n ${WORKER_ENV:-} ]]; then source "$WORKER_ENV" "$TW"; fi
unset CREST_DIST_OLOOP CREST_CKPT CREST_DIST_OLOOP_DUMP CREST_DIST_OLOOP_TEST_EXIT CREST_RESUME_POST
export CREST_DIST_OLOOP_WORKER=$P/chunks/chunk_$C.bin CREST_DIST_OLOOP_RESULT=$R
export OMP_NUM_THREADS=$TW
echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)  threads $TW  structures $n"
# the driver may have computed this chunk meanwhile (it computes chunks whose tasks are still pending)
if [[ -e $R ]]; then echo "chunk $c got a result while this task started; nothing to do"; exit 0; fi
t0=$(date +%s.%N)
"$CREST_EXE" --input worker.toml --newversion -T "$TW" > crest.out 2> crest.err
rc=$?
t1=$(date +%s.%N)
cp crest.out "$P/logs/chunk_${C}_${SLURM_JOB_ID:-x}.crest.out" || true
grep -h 'CREST_DIST>' crest.out crest.err || true
tail -n 3 crest.err || true
echo "crest rc $rc wall $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}') s"
if [[ -e $R ]]; then echo "chunk $c DONE $(date -u +%FT%TZ)"; exit 0; fi
echo "chunk $c: no result written" >&2
exit 1
