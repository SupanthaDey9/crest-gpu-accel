#!/bin/bash -l
# Nova: clean build from GitHub + the identity gate against the validated v2 binary (ec9ec385), on one CPU node.
# Usage: sbatch tests/site/nova/build_and_identity.sbatch <release dir> <work dir> <v2 binary> <Ga input pdb>
#SBATCH --job-name=crest-rel-build
#SBATCH --partition=nova
#SBATCH --account=<account>
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=<your work folder>/build-%j.log
set -uo pipefail
REL=$1; W=$2; V2=$3; PDB=$4
mkdir -p "$W"
module purge
module load cmake/3.30.2-baujmzy ninja/1.12.1-py311-l2hkirj openblas/0.3.27-openmp-xhezwie cuda/12.4.1-cz3ljd3
OPENBLAS_LIB=/opt/rit/el9/20240815/app/linux-rhel9-x86_64_v3/gcc-11.4.1/openblas-0.3.27-xhezwiemqjrojal52eerifzaabnzadyq/lib/libopenblasp-r0.3.27.so
CUDA_ROOT=/opt/rit/el9/20240815/app/linux-rhel9-x86_64_v3/gcc-11.4.1/cuda-12.4.1-cz3ljd3jkg6gn5apjvurp6o3hdxfr4gj
echo "node $(hostname -s); cpu $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
gfortran --version | head -1; cmake --version | head -1; nvcc --version | tail -1
t0=$(date +%s)
CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran \
  bash "$REL/build.sh" --jobs 8 --build-dir "$W/build" --prefix "$W" --cuda "$CUDA_ROOT" --openblas "$OPENBLAS_LIB"
rc=$?
echo "BUILD rc $rc in $(( $(date +%s) - t0 )) s"
[[ $rc == 0 ]] || exit $rc
# Identity gate: -T 1 single point of the Ga peptide, CPU solver, every CREST_* unset, both binaries on this node.
unset $(env | sed -n 's/^\(CREST_[A-Z_]*\)=.*/\1/p') 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for tag in new v2; do
  b=$W/bin/crest; [[ $tag == v2 ]] && b=$V2
  d=$W/identity/$tag; rm -rf "$d"; mkdir -p "$d"; cp "$PDB" "$d/production.pdb"
  printf 'input = "production.pdb"\nruntype = "singlepoint"\nthreads = 1\nomp_nested = false\n\n[[calculation.level]]\nmethod = "gfn2"\ncharge = 4\nuhf = 0\nalpb = "water"\netemp = 300.0\naccuracy = 1.0\nmaxscc = 500\ntblite_solver = "cpu"\ntblite_gpu_audit = false\n' > "$d/sp.toml"
  ( cd "$d" && "$b" --input sp.toml --newversion --gfn2 --chrg 4 --uhf 0 --alpb water -T 1 > crest.out 2> crest.err )
  echo "$tag: rc $? energy $(sed -n 8p "$d/crest.engrad" | tr -d ' ') sha $(sha256sum "$d/crest.engrad" | cut -c1-16)"
done
if cmp -s "$W/identity/new/crest.engrad" "$W/identity/v2/crest.engrad"; then
  echo "IDENTITY PASS: crest.engrad of the new build is byte-identical to v2 (ec9ec385)"
else
  echo "IDENTITY FAIL"; python3 "$REL/tests/compare_engrad.py" "$W/identity/new/crest.engrad" "$W/identity/v2/crest.engrad"
fi
sha256sum "$W/bin/crest" "$V2"
