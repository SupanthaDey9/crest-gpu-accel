#!/usr/bin/env bash
# crest-places.sh: print an OMP_PLACES list that keeps each trajectory's threads on one CPU socket.
#
#   eval "$(scripts/crest-places.sh 6 5)"   # 6 trajectories x 5 threads: exports OMP_PLACES and CREST_THREADS=30
#   crest ... -T "$CREST_THREADS"
#
# CREST runs the trajectories of an MTD batch in one OpenMP team and the threads of each trajectory in a nested team;
# with OMP_PROC_BIND=spread,close, trajectory k takes the k-th block of places. With OMP_PLACES=cores, a block can span
# two sockets when the job's cores are split unevenly between them (24 + 8 is common on shared nodes), and that
# trajectory then runs 5-16% slower in our tests; every MTD iteration waits for it. This script lists only whole
# groups of <threads> cores that lie on one socket, so each block is one group. It reads the cores this process may use
# (Linux sysfs) and exits with status 1, printing nothing, if fewer than <trajectories> such groups exist (with
# 6 x 5 threads, 34 cores in any split are always enough).
set -euo pipefail
NTRAJ=${1:?usage: crest-places.sh <trajectories> <threads per trajectory>}
NTHR=${2:?usage: crest-places.sh <trajectories> <threads per trajectory>}
python3 - "$NTRAJ" "$NTHR" <<'PY'
import os, sys
ntraj, nthr = int(sys.argv[1]), int(sys.argv[2])
cores = {}
for c in sorted(os.sched_getaffinity(0)):
    t = f'/sys/devices/system/cpu/cpu{c}/topology/'
    key = (int(open(t + 'physical_package_id').read()), int(open(t + 'core_id').read()))
    cores.setdefault(key, []).append(c)
sockets = {}
for (s, _), cpus in sorted(cores.items(), key=lambda kv: min(kv[1])):
    sockets.setdefault(s, []).append(sorted(cpus))
groups = [g[i:i + nthr] for s in sorted(sockets) for g in [sockets[s]] for i in range(0, len(g) - nthr + 1, nthr)]
if len(groups) < ntraj:
    sys.exit(1)
places = ','.join('{' + ','.join(map(str, c)) + '}' for g in groups[:ntraj] for c in g)
print(f"export OMP_PLACES='{places}' OMP_PROC_BIND=spread,close CREST_THREADS={ntraj * nthr}")
PY
