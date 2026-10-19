#!/usr/bin/env bash
# Configure and build the gfx906 (MI50 / MI60 / Radeon VII) engine inside the ROCm 7.14 image.
#
#   ./deploy/mi50/build.sh              # configure build-714 if needed, then build the strata target
#   JOBS=8 ./deploy/mi50/build.sh       # fewer compile jobs (the engine is running, RAM is tight)
#   BUILD_DIR=build-714 ./deploy/mi50/build.sh
#
# The layout is the one docs/MI50.md describes: this checkout is <root>/Strata, the GGUF models are in
# <root>/models and the runtime data in <root>/Strata-data. Override HOST_ROOT, MODELS_HOST or GGML_SRC
# when yours differs. The checkout directory must stay named "Strata": the scripts address it as
# /work/Strata inside the container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_ROOT="${HOST_ROOT:-$(dirname "$REPO_ROOT")}"
MODELS_HOST="${MODELS_HOST:-$HOST_ROOT/models}"
IMAGE="${IMAGE:-mixa3607/rocm-gfx906:7.14-complete}"
CTR="${CTR:-strata714}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-714}"
BUILD_NAME="$(basename "$BUILD_DIR")"
JOBS="${JOBS:-$(nproc)}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

if [ -z "${GGML_SRC:-}" ]; then
    if [ -d "$REPO_ROOT/third_party/llama.cpp" ]; then
        GGML_SRC="$REPO_ROOT/third_party/llama.cpp"
    else
        GGML_SRC="$REPO_ROOT/build-906/_deps/strata_llamacpp-src"
    fi
fi
[ -d "$GGML_SRC" ] || {
    echo "missing pinned llama.cpp source: $GGML_SRC" >&2
    echo "see docs/MI50.md: download setup.py's LLAMA_CPP_COMMIT into third_party/llama.cpp" >&2
    exit 1
}
case "$GGML_SRC" in
    "$HOST_ROOT"/*) GGML_SRC_C="/work${GGML_SRC#$HOST_ROOT}" ;;
    *) echo "GGML_SRC must live under HOST_ROOT ($HOST_ROOT) so the container can see it" >&2; exit 1 ;;
esac

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed"
[ -e /dev/kfd ] || die "/dev/kfd is missing (amdgpu driver not loaded?)"

# ---------------------------------------------------------------- container --
if ! docker inspect "$CTR" >/dev/null 2>&1; then
    say "creating container $CTR ($IMAGE) ..."
    docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
    render_gid="$(getent group render 2>/dev/null | cut -d: -f3 || true)"
    render_gid="${render_gid:-110}"
    mounts=(-v "$HOST_ROOT:/work")
    [ -d "$MODELS_HOST" ] && mounts+=(-v "$MODELS_HOST:/models:ro")
    docker run -d --name "$CTR" \
        --network host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add "$render_gid" \
        --ipc=host --ulimit memlock=-1 --shm-size=16g \
        "${mounts[@]}" \
        "$IMAGE" sleep infinity >/dev/null
else
    docker start "$CTR" >/dev/null 2>&1 || true
fi
docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true || die "container $CTR is not running"

# ------------------------------------------------------------------- tools ---
if ! docker exec "$CTR" bash -c '
        command -v cmake >/dev/null && command -v ninja >/dev/null &&
        printf "#include <cstdint>\nint main(){return 0;}\n" > /tmp/.strata_toolcheck.cpp &&
        /opt/rocm/llvm/bin/clang++ -std=c++20 -c /tmp/.strata_toolcheck.cpp -o /tmp/.strata_toolcheck.o' \
        >/dev/null 2>&1; then
    say "installing cmake / ninja / libstdc++-13-dev in the container ..."
    # the gfx906 third-party apt source is dead (403); the main sources still work, so ignore its exit code
    docker exec "$CTR" bash -c "apt-get update -qq" >/dev/null 2>&1 || true
    docker exec "$CTR" bash -c "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake ninja-build libstdc++-13-dev" \
        || die "could not install the build tools in the container"
fi

# ---------------------------------------------------------------- configure --
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    say "configuring $BUILD_NAME (gfx906, ROCm 7.14) ..."
    docker exec -u "$HOST_UID:$HOST_GID" -e HOME=/tmp -w /work/Strata "$CTR" bash -c "
        set -e
        export ROCM_PATH=/opt/rocm
        cmake -S . -B '$BUILD_NAME' -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
          -DSTRATA_GGML_DIR='$GGML_SRC_C' \
          -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
    "
else
    say "reusing $BUILD_NAME/CMakeCache.txt (delete it to reconfigure)"
fi

# -------------------------------------------------------------------- build --
say "building $BUILD_NAME/strata with -j$JOBS ..."
docker exec -u "$HOST_UID:$HOST_GID" -e HOME=/tmp -w /work/Strata "$CTR" bash -c "
    set -e
    export ROCM_PATH=/opt/rocm
    exec cmake --build '$BUILD_NAME' --target strata -j '$JOBS'
"
say "done: $BUILD_DIR/strata"
