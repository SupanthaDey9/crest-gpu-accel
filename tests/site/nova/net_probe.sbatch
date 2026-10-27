#!/bin/bash
#SBATCH --job-name=crest-rel-netprobe
#SBATCH --partition=nova
#SBATCH --account=<account>
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=<your work folder>/netprobe-%j.log
# Does a compute node reach GitHub? (decides whether the release build can fetch sources itself)
hostname
timeout 60 git ls-remote https://github.com/crest-lab/crest.git HEAD && echo NET_OK || echo NET_FAIL
module avail cmake ninja openblas cuda gcc 2>&1 | grep -E "cmake|ninja|openblas|cuda/12" | head -20
