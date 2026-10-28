#!/usr/bin/env bash
# build.sh: build the accelerated CREST from upstream sources plus the patches in patches/.
#
#   ./build.sh                       # fetch sources from GitHub, build into ./build, install ./bin/crest
#   ./build.sh --sources DIR         # use pre-fetched sources (DIR/crest with submodules, DIR/deps/<name>)
#   ./build.sh --fetch-only DIR      # only fetch the pinned sources into DIR (for clusters without internet on nodes)
#
# Options: --jobs N (default 8), --build-dir DIR (default <repo>/build), --prefix DIR (default <repo>; binary goes to PREFIX/bin),
#          --cuda DIR (CUDA toolkit root; default: from nvcc on PATH), --openblas LIB (path to libopenblas*.so; default:
#          found by CMake with BLA_VENDOR=OpenBLAS), --gencode "ARGS" (nvcc -gencode list for the GPU ATM kernel).
#
# Requirements (tested versions in brackets): gfortran/gcc/g++ >= 11 [11.5.0], CMake >= 3.17 [3.30.2], Ninja [1.12.1],
# OpenBLAS with OpenMP threading [0.3.27], CUDA toolkit with cuSOLVER and cuBLAS [12.4.1], git, patch.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOBS=8; BUILD=$HERE/build; PREFIX=$HERE; SOURCES=''; FETCH_ONLY=''; CUDA_ROOT=''; OPENBLAS_LIB=''; GENCODE=''
while [[ $# -gt 0 ]]; do
  case $1 in
    --jobs) JOBS=$2; shift 2 ;;
    --build-dir) BUILD=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --sources) SOURCES=$2; shift 2 ;;
    --fetch-only) FETCH_ONLY=$2; shift 2 ;;
    --cuda) CUDA_ROOT=$2; shift 2 ;;
    --openblas) OPENBLAS_LIB=$2; shift 2 ;;
    --gencode) GENCODE=$2; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
done

# Pinned upstream revisions. CREST commit cfdc301f (master) is v3.0.2 plus 80 commits; its submodules are pinned by the
# CREST commit itself. The five FetchContent dependencies are pinned to the commits our validated build used.
CREST_URL=https://github.com/crest-lab/crest.git
CREST_REV=cfdc301f759686b0fd66ced63b5ddbd6c693fa4f
declare -A DEP_URL=(
  [dftd4]=https://github.com/dftd4/dftd4.git
  [mctc-lib]=https://github.com/grimme-lab/mctc-lib.git
  [mstore]=https://github.com/grimme-lab/mstore.git
  [multicharge]=https://github.com/grimme-lab/multicharge.git
  [s-dftd3]=https://github.com/dftd3/simple-dftd3.git
)
declare -A DEP_REV=(
  [dftd4]=7b2ff85a71a3630808fd8a2d972933f75b743f3c
  [mctc-lib]=c86c80002e74e7cfc48b7cc9878dd9caa85d4e70
  [mstore]=663245d739be0123da61c917e55116b0c3db4c74
  [multicharge]=282626e690aa7db2aec448d9032636a5cd75f25c
  [s-dftd3]=3425201769e7c56e98e3831f9bdb7be71cfaec0d
)

fetch() {
  local dst=$1
  mkdir -p "$dst/deps"
  if [[ ! -d $dst/crest/.git ]]; then
    git clone --quiet "$CREST_URL" "$dst/crest"
  fi
  git -C "$dst/crest" checkout --quiet "$CREST_REV"
  git -C "$dst/crest" submodule update --init --recursive --quiet
  for d in "${!DEP_URL[@]}"; do
    if [[ ! -d $dst/deps/$d/.git ]]; then
      git clone --quiet "${DEP_URL[$d]}" "$dst/deps/$d"
    fi
    git -C "$dst/deps/$d" checkout --quiet "${DEP_REV[$d]}"
  done
  echo "sources ready in $dst"
}

check_rev() {  # verify a git checkout is at the pinned commit (skipped for plain copies without .git)
  local dir=$1 want=$2
  if [[ -d $dir/.git || -f $dir/.git ]]; then
    local got; got=$(git -C "$dir" rev-parse HEAD)
    [[ $got == "$want" ]] || { echo "$dir is at $got, expected $want" >&2; exit 65; }
  fi
}

if [[ -n $FETCH_ONLY ]]; then fetch "$FETCH_ONLY"; exit 0; fi
if [[ -z $SOURCES ]]; then SOURCES=$BUILD/sources; fetch "$SOURCES"; fi
check_rev "$SOURCES/crest" "$CREST_REV"
check_rev "$SOURCES/crest/subprojects/tblite" 6f6cd7d20d97b22ef00d420904343c7bb8e2afdf
for d in "${!DEP_REV[@]}"; do check_rev "$SOURCES/deps/$d" "${DEP_REV[$d]}"; done

# Work on a copy so that the fetched sources stay pristine.
SRC=$BUILD/src
rm -rf "$SRC"; mkdir -p "$SRC/deps"
cp -a "$SOURCES/crest" "$SRC/crest"
for d in "${!DEP_REV[@]}"; do cp -a "$SOURCES/deps/$d" "$SRC/deps/$d"; done
( cd "$SRC/crest" && patch -p1 --forward --quiet < "$HERE/patches/0001-crest-cfdc301f.patch" )
( cd "$SRC/crest/subprojects/tblite" && patch -p1 --forward --quiet < "$HERE/patches/0002-tblite-6f6cd7d.patch" )
for d in dftd4 mctc-lib multicharge s-dftd3; do
  ( cd "$SRC/deps/$d" && patch -p1 --forward --quiet < "$HERE/patches/deps/$d.patch" )
done
echo "patches applied"

if [[ -z $CUDA_ROOT ]]; then
  command -v nvcc >/dev/null || { echo "nvcc not found: load CUDA or pass --cuda DIR" >&2; exit 66; }
  CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
fi
BLAS_ARGS=(-DBLA_VENDOR=OpenBLAS -DBLA_SIZEOF_INTEGER=4)
if [[ -n $OPENBLAS_LIB ]]; then
  BLAS_ARGS+=(-DBLAS_openblas_LIBRARY:FILEPATH="$OPENBLAS_LIB" -DBLAS_LIBRARIES:STRING="$OPENBLAS_LIB"
              -DLAPACK_LIBRARIES:STRING="$OPENBLAS_LIB" -DCMAKE_BUILD_RPATH="$(dirname "$OPENBLAS_LIB")")
fi
EXTRA=()
[[ -n $GENCODE ]] && EXTRA+=(-DCREST_ATM_CUDA_GENCODE="$GENCODE")

# OpenMP/BLAS threads must not leak into the configure step.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cmake -S "$SRC/crest" -B "$BUILD/cmake" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER="${CC:-gcc}" -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_Fortran_COMPILER="${FC:-gfortran}" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" "${BLAS_ARGS[@]}" \
  -DWITH_OpenMP=ON -DWITH_TESTS=ON \
  -DWITH_TBLITE_GPU=ON -DWITH_TBLITE_GPU_DENSITY_EXPERIMENT=ON -DWITH_TBLITE_GPU_PERSISTENCE_EXPERIMENT=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DGFN0_FIND_METHOD:STRING=subproject -DGFNFF_FIND_METHOD:STRING=subproject -DLWONIOM_FIND_METHOD:STRING=subproject \
  -DPVOL_FIND_METHOD:STRING=subproject -DTBLITE_FIND_METHOD:STRING=subproject \
  -DTEST-DRIVE_FIND_METHOD:STRING=subproject -DTOML-F_FIND_METHOD:STRING=subproject -DTOMLF_FIND_METHOD:STRING=subproject \
  -DDFTD4_FIND_METHOD:STRING=fetch -DFETCHCONTENT_SOURCE_DIR_DFTD4:PATH="$SRC/deps/dftd4" \
  -DMCTCLIB_FIND_METHOD:STRING=fetch -DFETCHCONTENT_SOURCE_DIR_MCTC-LIB:PATH="$SRC/deps/mctc-lib" \
  -DMSTORE_FIND_METHOD:STRING=fetch -DFETCHCONTENT_SOURCE_DIR_MSTORE:PATH="$SRC/deps/mstore" \
  -DMULTICHARGE_FIND_METHOD:STRING=fetch -DFETCHCONTENT_SOURCE_DIR_MULTICHARGE:PATH="$SRC/deps/multicharge" \
  -DSDFTD3_FIND_METHOD:STRING=fetch -DFETCHCONTENT_SOURCE_DIR_S-DFTD3:PATH="$SRC/deps/s-dftd3" \
  "${EXTRA[@]}"
# Only the crest executable: the dftd4 and CREST test programs are not linked against the GPU ATM objects.
cmake --build "$BUILD/cmake" --target crest-exe -j "$JOBS"

mkdir -p "$PREFIX/bin"
cp "$BUILD/cmake/crest" "$PREFIX/bin/crest"
echo "built $PREFIX/bin/crest"
sha256sum "$PREFIX/bin/crest" 2>/dev/null || shasum -a 256 "$PREFIX/bin/crest"
for m in 'CREST_ATM_GPU summary' 'CREST_OPT_SOLVER cpu:' 'CREST_DIST>' 'CREST_CKPT>'; do
  grep -aqF -- "$m" "$PREFIX/bin/crest" || { echo "marker '$m' missing from the binary" >&2; exit 67; }
done
echo "feature markers present; next: $HERE/tests/run_tests.sh $PREFIX/bin/crest"
