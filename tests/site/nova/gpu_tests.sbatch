#!/bin/bash -l
# Nova: run tests/run_tests.sh on a GPU node, including the distributed-pass test and a comparison with the benchmarked binary.
# Usage: sbatch [--dependency=afterok:<build job>] tests/site/nova/gpu_tests.sbatch <release dir> <work dir> <benchmarked binary>
#SBATCH --job-name=crest-rel-gputest
#SBATCH --partition=scavenger
#SBATCH --account=<scavenger account>
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h200|l40s
#SBATCH --exclude=nova24-gpu-1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=<your work folder>/gputest-%j.log
set -uo pipefail
REL=$1; W=$2; REF=$3
module purge
module load openblas/0.3.27-openmp-xhezwie cuda/12.4.1-cz3ljd3
echo "node $(hostname -s)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
bash "$REL/tests/run_tests.sh" "$W/bin/crest" --reference "$REF" --workdir "$W/tests" -T 8 \
  --worker-env "$REL/slurm/site/nova_worker_env.sh" \
  --dist "--partition=nova --account=<account> --qos=normal --mem=8G --time=00:30:00"
echo "run_tests rc $?"
