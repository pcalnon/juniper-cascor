#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Script:        profile_training.bash
# Purpose:       Sampling profiler script using py-spy for training profiling
#
# P3-NEW-002: Sampling Profiling Infrastructure
#
# Usage:
#   ./util/profile_training.bash                    # Profile with default settings
#   ./util/profile_training.bash --output custom    # Custom output prefix
#   ./util/profile_training.bash --duration 60      # Profile for 60 seconds
#   ./util/profile_training.bash --svg              # Generate SVG flame graph
#   ./util/profile_training.bash --speedscope       # Generate speedscope format
#
# Prerequisites:
#   pip install py-spy
#   # OR for system-wide install with root privileges:
#   sudo pip install py-spy
#
#####################################################################################################################################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PROFILE_DIR="${PROJECT_ROOT}/profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default settings
OUTPUT_PREFIX="training_profile"
DURATION=""  # Empty means run to completion
FORMAT="speedscope"  # speedscope, svg, or raw
RATE=100  # Sampling rate in Hz

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Juniper Cascor Training Profiler (py-spy)
==========================================

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -o, --output PREFIX     Output file prefix (default: training_profile)
    -d, --duration SECS     Profile duration in seconds (default: run to completion)
    -r, --rate HZ           Sampling rate in Hz (default: 100)
    -f, --format FORMAT     Output format: speedscope, svg, raw (default: speedscope)
    --svg                   Shortcut for --format svg
    --speedscope            Shortcut for --format speedscope
    --native                Include native C/C++ frames
    --subprocesses          Profile subprocesses
    --idle                  Include idle time

Examples:
    $0                              # Profile with defaults
    $0 --svg                        # Generate SVG flame graph
    $0 --duration 30 --rate 200    # 30s at 200Hz
    $0 --output my_profile --svg   # Custom output as SVG

Output files are saved to: ${PROFILE_DIR}/
EOF
}

# Parse arguments
NATIVE=""
SUBPROCESSES=""
IDLE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -r|--rate)
            RATE="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        --svg)
            FORMAT="svg"
            shift
            ;;
        --speedscope)
            FORMAT="speedscope"
            shift
            ;;
        --native)
            NATIVE="--native"
            shift
            ;;
        --subprocesses)
            SUBPROCESSES="--subprocesses"
            shift
            ;;
        --idle)
            IDLE="--idle"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if py-spy is installed
if ! command -v py-spy &> /dev/null; then
    log_error "py-spy is not installed."
    log_info "Install with: pip install py-spy"
    log_info "Or for system-wide: sudo pip install py-spy"
    exit 1
fi

# Create profile directory
mkdir -p "${PROFILE_DIR}"

# Determine output filename based on format
case $FORMAT in
    svg)
        OUTPUT_FILE="${PROFILE_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.svg"
        FORMAT_FLAG="--format svg"
        ;;
    speedscope)
        OUTPUT_FILE="${PROFILE_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.json"
        FORMAT_FLAG="--format speedscope"
        ;;
    raw)
        OUTPUT_FILE="${PROFILE_DIR}/${OUTPUT_PREFIX}_${TIMESTAMP}.txt"
        FORMAT_FLAG="--format raw"
        ;;
    *)
        log_error "Unknown format: $FORMAT"
        exit 1
        ;;
esac

# Build py-spy command
PYSPY_CMD="py-spy record"
PYSPY_CMD+=" --output ${OUTPUT_FILE}"
PYSPY_CMD+=" --rate ${RATE}"
PYSPY_CMD+=" ${FORMAT_FLAG}"

if [[ -n "${DURATION}" ]]; then
    PYSPY_CMD+=" --duration ${DURATION}"
fi

if [[ -n "${NATIVE}" ]]; then
    PYSPY_CMD+=" ${NATIVE}"
fi

if [[ -n "${SUBPROCESSES}" ]]; then
    PYSPY_CMD+=" ${SUBPROCESSES}"
fi

if [[ -n "${IDLE}" ]]; then
    PYSPY_CMD+=" ${IDLE}"
fi

# Add the Python script to profile
PYSPY_CMD+=" -- python ${SRC_DIR}/main.py"

log_info "Starting py-spy profiler..."
log_info "Output: ${OUTPUT_FILE}"
log_info "Format: ${FORMAT}"
log_info "Rate: ${RATE} Hz"
if [[ -n "${DURATION}" ]]; then
    log_info "Duration: ${DURATION}s"
else
    log_info "Duration: Until training completes"
fi

echo ""
log_info "Command: ${PYSPY_CMD}"
echo ""

# Run py-spy
# Note: py-spy may require root on some systems
if [[ $EUID -ne 0 ]]; then
    log_warn "Running without root. If profiling fails, try: sudo $0 $*"
fi

eval "${PYSPY_CMD}"

log_success "Profile saved to: ${OUTPUT_FILE}"

# Post-processing hints
case $FORMAT in
    svg)
        log_info "View flame graph by opening ${OUTPUT_FILE} in a web browser"
        ;;
    speedscope)
        log_info "View profile at https://www.speedscope.app/"
        log_info "Upload ${OUTPUT_FILE} to visualize"
        ;;
    raw)
        log_info "Raw profile data saved. Use py-spy to convert:"
        log_info "  py-spy dump --pid <PID>"
        ;;
esac

echo ""
log_success "Profiling complete!"
