# Worker environment for the ISU Nova cluster (example of a site file; the build used these modules).
# Use it as worker_env in dist_oloop.conf, or copy the module lines into slurm/worker_env.sh.
module purge
module load openblas/0.3.27-openmp-xhezwie cuda/12.4.1-cz3ljd3
unset LD_PRELOAD LD_AUDIT CREST_OPT_SOLVER CREST_ATM_GPU CREST_ATM_GPU_AUDIT
export OMP_NUM_THREADS=${1:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_SCHEDULE=STATIC OMP_PLACES=cores OMP_PROC_BIND=spread,close OMP_STACKSIZE=256M \
       GLIBC_TUNABLES=glibc.pthread.stack_cache_size=17179869184
ulimit -s unlimited 2>/dev/null || true
