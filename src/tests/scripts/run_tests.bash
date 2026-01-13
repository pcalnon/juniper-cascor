#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     Cascade Correlation Neural Network
# File Name:     run_tests.sh
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2025-09-26
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    Script to run Cascade Correlation Network tests with various options.
#    Supports unit tests, integration tests, performance tests, and coverage reporting.
#####################################################################################################################################################################################################

set -e # Exit on any error

# Get script directory
# trunk-ignore(shellcheck/SC2312)
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
TESTS_DIR="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "$(dirname "${TESTS_DIR}")")"

# Default values
# TODO: move these to a config file
RUN_UNIT=true
RUN_INTEGRATION=false
RUN_PERFORMANCE=false
RUN_SLOW=false
RUN_GPU=false
VERBOSE=false
COVERAGE=true
PARALLEL=false
SPECIFIC_TEST=""
OUTPUT_DIR="${TESTS_DIR}/reports"
echo "Output Dir: ${OUTPUT_DIR}"
MARKERS=""
FAILED_ONLY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
	local color=$1
	local message=$2
	echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
	cat <<EOF
Usage: $0 [OPTIONS]

Run Cascade Correlation Network tests with various options.

OPTIONS:
    -u, --unit             Run unit tests (default: true)
    -i, --integration      Run integration tests (default: false)
    -p, --performance      Run performance tests (default: false)
    -s, --slow             Run slow tests (default: false)
    -g, --gpu              Run GPU tests (default: false)
    -v, --verbose          Verbose output (default: false)
    -c, --coverage         Run with coverage (default: true)
    --no-coverage          Disable coverage reporting
    -j, --parallel         Run tests in parallel (default: false)
    -t, --test TEST        Run specific test (can be file or pattern)
    -m, --markers MARKERS  Run tests with specific markers
    -f, --failed           Re-run only previously failed tests
    -o, --output-dir DIR   Output directory for reports (default: tests/reports)
    -h, --help             Show this help message

EXAMPLES:
    $0                          # Run unit tests with coverage
    $0 -i -s                    # Run integration and slow tests
    $0 -t test_forward_pass.py  # Run specific test file
    $0 -m "unit and accuracy"   # Run tests with both unit and accuracy markers
    $0 --no-coverage -j         # Run without coverage, in parallel
    $0 -f                       # Re-run only failed tests

TEST MARKERS:
    unit                   Unit tests for individual components
    integration            Integration tests for full workflows  
    performance            Performance and benchmarking tests
    slow                   Tests that take a long time to run
    gpu                    Tests that require GPU/CUDA
    multiprocessing        Tests that use multiprocessing
    spiral                 Tests specifically for spiral problem solving
    correlation            Tests for correlation coefficient calculations
    network_growth         Tests for network growth algorithms
    candidate_training     Tests for candidate unit training
    validation             Tests for input validation functions
    accuracy               Tests for accuracy calculation methods
    early_stopping         Tests for early stopping logic

EOF
}

# Parse command line arguments
while (($# > 0)); do
	case $1 in
	-u | --unit)
		RUN_UNIT=true
		shift
		;;
	-i | --integration)
		RUN_INTEGRATION=true
		shift
		;;
	-p | --performance)
		RUN_PERFORMANCE=true
		shift
		;;
	-s | --slow)
		RUN_SLOW=true
		shift
		;;
	-g | --gpu)
		RUN_GPU=true
		shift
		;;
	-v | --verbose)
		VERBOSE=true
		shift
		;;
	-c | --coverage)
		COVERAGE=true
		shift
		;;
	--no-coverage)
		COVERAGE=false
		shift
		;;
	-j | --parallel)
		PARALLEL=true
		shift
		;;
	-t | --test)
		SPECIFIC_TEST="$2"
		shift 2
		;;
	-m | --markers)
		MARKERS="$2"
		shift 2
		;;
	-f | --failed)
		FAILED_ONLY=true
		shift
		;;
	-o | --output-dir)
		OUTPUT_DIR="$2"
		shift 2
		;;
	-h | --help)
		show_usage
		exit 0
		;;
	*)
		print_color "${RED}" "Unknown option: $1"
		show_usage
		exit 1
		;;
	esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Change to tests directory
cd "${TESTS_DIR}" || {
    print_color "${RED}" "Failed to change directory to ${TESTS_DIR}"
    exit 1
}

print_color "${BLUE}" "=== Cascade Correlation Network Test Runner ==="
print_color "${BLUE}" "Tests Directory: ${TESTS_DIR}"
print_color "${BLUE}" "Output Directory: ${OUTPUT_DIR}"
print_color "${BLUE}" "Project Root: ${PROJECT_ROOT}"

# Build pytest command
PYTEST_CMD="python -m pytest"

# Add basic options
if [[ "${VERBOSE}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} -v"
else
	PYTEST_CMD="${PYTEST_CMD} -q"
fi

# Add parallel execution
if [[ "${PARALLEL}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} -n auto"
fi

# Add coverage options
if [[ "${COVERAGE}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} --cov=../cascade_correlation --cov=../candidate_unit"
	PYTEST_CMD="${PYTEST_CMD} --cov-report=html:${OUTPUT_DIR}/htmlcov"
	PYTEST_CMD="${PYTEST_CMD} --cov-report=xml:${OUTPUT_DIR}/coverage.xml"
	PYTEST_CMD="${PYTEST_CMD} --cov-report=term-missing"
fi

# Add JUnit XML report
PYTEST_CMD="${PYTEST_CMD} --junitxml=${OUTPUT_DIR}/junit.xml"

# Handle failed only
if [[ "${FAILED_ONLY}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} --lf"
fi

# Handle GPU tests
if [[ "${RUN_GPU}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} --gpu"
fi

# Handle slow tests
if [[ "${RUN_SLOW}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} --slow"
fi

# Handle integration tests
if [[ "${RUN_INTEGRATION}" == "true" ]]; then
	PYTEST_CMD="${PYTEST_CMD} --integration"
fi

# Build marker expression
MARKER_EXPR=""

if [[ "${MARKERS}" != "" ]]; then
	MARKER_EXPR="${MARKERS}"
else
	# Build marker expression based on flags
	MARKER_PARTS=()

	if [[ "${RUN_UNIT}" == "true" ]]; then
		MARKER_PARTS+=("unit")
	fi

	if [[ "${RUN_INTEGRATION}" == "true" ]]; then
		MARKER_PARTS+=("integration")
	fi

	if [[ "${RUN_PERFORMANCE}" == "true" ]]; then
		MARKER_PARTS+=("performance")
	fi

	# Join with "or" if multiple markers
	if [[ ${#MARKER_PARTS[@]} -gt 0 ]]; then
		MARKER_EXPR=$(
			IFS=" or "
			echo "${MARKER_PARTS[*]}"
		)
	fi
fi

if [[ "${MARKER_EXPR}" != "" ]]; then
	PYTEST_CMD="${PYTEST_CMD} -m \"$MARKER_EXPR\""
fi

# Add specific test if provided
if [[ "${SPECIFIC_TEST}" != "" ]]; then
	PYTEST_CMD="${PYTEST_CMD} ${SPECIFIC_TEST}"
fi

# Print configuration
print_color "${YELLOW}" "=== Test Configuration ==="
print_color "${YELLOW}" "Unit Tests: ${RUN_UNIT}"
print_color "${YELLOW}" "Integration Tests: ${RUN_INTEGRATION}"
print_color "${YELLOW}" "Performance Tests: ${RUN_PERFORMANCE}"
print_color "${YELLOW}" "Slow Tests: ${RUN_SLOW}"
print_color "${YELLOW}" "GPU Tests: ${RUN_GPU}"
print_color "${YELLOW}" "Coverage: ${COVERAGE}"
print_color "${YELLOW}" "Parallel: ${PARALLEL}"
print_color "${YELLOW}" "Verbose: ${VERBOSE}"
if [[ ${SPECIFIC_TEST} != "" ]]; then
	print_color "${YELLOW}" "Specific Test: ${SPECIFIC_TEST}"
fi
if [[ ${MARKER_EXPR} != "" ]]; then
	print_color "${YELLOW}" "Markers: ${MARKER_EXPR}"
fi
print_color "${YELLOW}" "=========================="

# Run pytest
print_color "${BLUE}" "Running: ${PYTEST_CMD}"
print_color "${BLUE}" "=========================="

# Execute the command
eval "${PYTEST_CMD}"
TEST_EXIT_CODE="$?"

# Report results
print_color "${BLUE}" "=========================="

if [[ ${TEST_EXIT_CODE} == "0" ]]; then
	print_color "${GREEN}" "✓ All tests passed!"
else
	print_color "${RED}" "✗ Some tests failed: exit code: ${TEST_EXIT_CODE}"
fi

# Show report locations
if [ "${COVERAGE}" = true ]; then
	print_color "${BLUE}" "Coverage HTML report: ${OUTPUT_DIR}/htmlcov/index.html"
	print_color "${BLUE}" "Coverage XML report: ${OUTPUT_DIR}/coverage.xml"
fi
print_color "${BLUE}" "JUnit XML report: ${OUTPUT_DIR}/junit.xml"

exit "${TEST_EXIT_CODE}"
