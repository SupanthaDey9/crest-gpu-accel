# worker_env.sh: environment of one distributed-pass worker. dist_worker.sbatch sources it with $1 = threads.
# Workers are submitted with --export=NONE, so everything a worker needs must be set here.
#
# 1. Site setup: load the same compiler runtime, OpenBLAS and CUDA libraries the crest binary was built with
#    (the binary links libcudart/libcusolver even though workers never use the GPU). Edit these lines for your
#    cluster, or point worker_env in dist_oloop.conf at a complete site file such as slurm/site/nova_worker_env.sh.
# module load openblas cuda

# 2. Workers never touch the GPU and never write checkpoints.
unset LD_PRELOAD LD_AUDIT CREST_OPT_SOLVER CREST_ATM_GPU CREST_ATM_GPU_AUDIT

# 3. Thread layout (same as scripts/crest-env.sh).
export OMP_NUM_THREADS=${1:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_SCHEDULE=STATIC OMP_PLACES=cores OMP_PROC_BIND=spread,close OMP_STACKSIZE=256M \
       GLIBC_TUNABLES=glibc.pthread.stack_cache_size=17179869184
ulimit -s unlimited 2>/dev/null || true
