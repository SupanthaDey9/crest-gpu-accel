# crest-env.sh: runtime settings for the accelerated CREST build. Source it before running crest:
#   source scripts/crest-env.sh            # tuned threads only (no GPU switches)
#   source scripts/crest-env.sh gpu        # also switch on the GPU three-body term and the CPU solver for optimizations
#
# The thread settings do not change the arithmetic of a single-threaded run; with more threads, results differ from run
# to run only through reduction order, as for any multithreaded CREST run. In gpu mode, CREST_ATM_GPU=1 swaps in the GPU
# three-body kernel, which agrees with the CPU routine to 4.4e-18 Eh/bohr but is not bitwise equal (docs/VALIDATION.md).

# Thread layout. The OpenMP default (dynamic,1 schedule, unbound threads) hid almost all thread scaling of the SCC in
# our tests; a static schedule with bound threads made MTD 2.18x faster per energy+gradient on the same node.
export OMP_SCHEDULE=STATIC
export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export OMP_STACKSIZE=256M
export OMP_MAX_ACTIVE_LEVELS=${OMP_MAX_ACTIVE_LEVELS:-2}
# glibc keeps thread stacks cached between parallel regions (cut a nested region from 961 to 304 us in our tests)
export GLIBC_TUNABLES=glibc.pthread.stack_cache_size=17179869184
# BLAS must stay single-threaded inside CREST's own OpenMP regions. With a pthreads OpenBLAS and this unset, OpenBLAS
# prints "Detect OpenMP Loop" warnings without end and can fill a disk.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited 2>/dev/null || true

if [[ ${1:-} == gpu ]]; then
  # D4 three-body (ATM) term on the GPU during MTD and MD; optimization passes always use the CPU routine.
  export CREST_ATM_GPU=1
  # The CUDA eigensolver is selected per calculation level with tblite_solver = "cuda" in the TOML input. This keeps
  # every optimization pass on the CPU solver: faster there (32 concurrent optimizations ran at 0.87-0.89x on one GPU)
  # and checkable bit for bit against distributed workers.
  export CREST_OPT_SOLVER=cpu
fi
