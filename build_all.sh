#!/bin/bash
set -e

export FLASHINFER_LOCAL_VERSION=rtpllm-`date +%Y%m%d_%H%M%S`-`git rev-parse --short HEAD`
export FLASHINFER_CUDA_ARCH_LIST="10.0a"
export MAX_JOBS=144

# Script to build all three FlashInfer wheel packages:
# 1. flashinfer-python (core package)
# 2. flashinfer-cubin (pre-compiled cubin binaries)
# 3. flashinfer-jit-cache (pre-compiled JIT cache)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_OUTPUT="${SCRIPT_DIR}/build_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
SKIP_JIT_CACHE=false
SKIP_CUBIN=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-jit-cache)
            SKIP_JIT_CACHE=true
            shift
            ;;
        --skip-cubin)
            SKIP_CUBIN=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-jit-cache  Skip building flashinfer-jit-cache wheel"
            echo "  --skip-cubin      Skip building flashinfer-cubin wheel"
            echo "  --clean           Clean previous builds before building"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  FLASHINFER_CUDA_ARCH_LIST  CUDA architectures for JIT cache (default: auto-detect)"
            echo "  FLASHINFER_LOCAL_VERSION   Local version suffix (e.g., cu129)"
            echo "  MAX_JOBS                   Maximum parallel compilation jobs"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  FlashInfer Build All Wheels Script"
echo "=========================================="
echo ""

# Display build environment
log_info "Build Environment:"
echo "  Working directory: ${SCRIPT_DIR}"
echo "  Output directory:  ${BUILD_OUTPUT}"
echo "  Python version:    $(python3 --version 2>&1)"
echo "  Git commit:        $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "  CUDA arch list:    ${FLASHINFER_CUDA_ARCH_LIST:-auto-detect}"
echo "  Local version:     ${FLASHINFER_LOCAL_VERSION:-none}"
echo ""

# Create output directory
mkdir -p "${BUILD_OUTPUT}"

# Clean previous builds if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning previous builds..."
    rm -rf "${BUILD_OUTPUT}"/*
    rm -rf "${SCRIPT_DIR}/dist"
    rm -rf "${SCRIPT_DIR}/build"
    rm -rf "${SCRIPT_DIR}/flashinfer-cubin/dist"
    rm -rf "${SCRIPT_DIR}/flashinfer-cubin/build"
    rm -rf "${SCRIPT_DIR}/flashinfer-jit-cache/dist"
    rm -rf "${SCRIPT_DIR}/flashinfer-jit-cache/build"
    log_success "Clean completed"
fi

# Ensure build tool is installed
log_info "Ensuring build tools are installed..."
pip install --quiet build wheel

# ============================================
# Step 1: Build flashinfer-python
# ============================================
echo ""
echo "=========================================="
echo "  Step 1/3: Building flashinfer-python"
echo "=========================================="

cd "${SCRIPT_DIR}"
rm -rf dist build *.egg-info

log_info "Building flashinfer-python wheel..."
python -m build --wheel --no-isolation

# Copy to output directory
cp dist/*.whl "${BUILD_OUTPUT}/"
PYTHON_WHL=$(ls dist/*.whl | head -1)
log_success "Built: $(basename ${PYTHON_WHL})"

# Install flashinfer-python for subsequent builds
log_info "Installing flashinfer-python for subsequent builds..."
pip install --quiet --force-reinstall dist/*.whl

# ============================================
# Step 2: Build flashinfer-cubin
# ============================================
if [ "$SKIP_CUBIN" = true ]; then
    log_warning "Skipping flashinfer-cubin build (--skip-cubin)"
else
    echo ""
    echo "=========================================="
    echo "  Step 2/3: Building flashinfer-cubin"
    echo "=========================================="

    cd "${SCRIPT_DIR}/flashinfer-cubin"
    rm -rf dist build *.egg-info flashinfer_cubin/cubins

    log_info "Building flashinfer-cubin wheel (this will download cubins)..."
    python -m build --wheel --no-isolation

    # Copy to output directory
    cp dist/*.whl "${BUILD_OUTPUT}/"
    CUBIN_WHL=$(ls dist/*.whl | head -1)
    log_success "Built: $(basename ${CUBIN_WHL})"
fi

# ============================================
# Step 3: Build flashinfer-jit-cache
# ============================================
if [ "$SKIP_JIT_CACHE" = true ]; then
    log_warning "Skipping flashinfer-jit-cache build (--skip-jit-cache)"
else
    echo ""
    echo "=========================================="
    echo "  Step 3/3: Building flashinfer-jit-cache"
    echo "=========================================="

    # Set CUDA arch list if not already set
    if [ -z "${FLASHINFER_CUDA_ARCH_LIST}" ]; then
        # Try to auto-detect from current GPU
        if command -v nvidia-smi &> /dev/null; then
            GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
            if [ -n "${GPU_ARCH}" ]; then
                export FLASHINFER_CUDA_ARCH_LIST="${GPU_ARCH:0:1}.${GPU_ARCH:1}"
                log_info "Auto-detected CUDA architecture: ${FLASHINFER_CUDA_ARCH_LIST}"
            fi
        fi

        # If still not set, use common defaults
        if [ -z "${FLASHINFER_CUDA_ARCH_LIST}" ]; then
            export FLASHINFER_CUDA_ARCH_LIST="8.0 8.9 9.0"
            log_warning "Using default CUDA architectures: ${FLASHINFER_CUDA_ARCH_LIST}"
        fi
    fi

    # Set MAX_JOBS if not set
    if [ -z "${MAX_JOBS}" ]; then
        MEM_AVAILABLE_GB=$(free -g 2>/dev/null | awk '/^Mem:/ {print $7}' || echo "16")
        NPROC=$(nproc 2>/dev/null || echo "8")
        MAX_JOBS=$(( MEM_AVAILABLE_GB / 8 ))
        if (( MAX_JOBS < 1 )); then
            MAX_JOBS=1
        elif (( NPROC < MAX_JOBS )); then
            MAX_JOBS=$NPROC
        fi
        export MAX_JOBS
    fi
    log_info "Using MAX_JOBS=${MAX_JOBS}"

    cd "${SCRIPT_DIR}/flashinfer-jit-cache"
    rm -rf dist build *.egg-info flashinfer_jit_cache/jit_cache

    log_info "Building flashinfer-jit-cache wheel (this will compile kernels)..."
    log_warning "This may take a while depending on CUDA architectures..."
    python -m build --wheel --no-isolation

    # Copy to output directory
    cp dist/*.whl "${BUILD_OUTPUT}/"
    JIT_CACHE_WHL=$(ls dist/*.whl | head -1)
    log_success "Built: $(basename ${JIT_CACHE_WHL})"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "  Build Summary"
echo "=========================================="
echo ""
log_success "All builds completed successfully!"
echo ""
echo "Output directory: ${BUILD_OUTPUT}"
echo ""
echo "Built wheels:"
ls -lh "${BUILD_OUTPUT}"/*.whl 2>/dev/null || echo "  (no wheels found)"
echo ""

# Verify wheels
log_info "Verifying wheel contents..."
for whl in "${BUILD_OUTPUT}"/*.whl; do
    if [ -f "$whl" ]; then
        echo "  $(basename $whl):"
        unzip -l "$whl" 2>/dev/null | tail -1 | awk '{print "    Files: " $2 ", Size: " $1 " bytes"}'
    fi
done

echo ""
log_success "Done! Wheels are available in: ${BUILD_OUTPUT}"

