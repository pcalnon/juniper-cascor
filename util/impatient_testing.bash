#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     impatient_testing.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date Created:  2025-11-05
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This files runs tests with 5 sec status updates using kill -0
#
#####################################################################################################################################################################################################
# Notes:
#
#     python -m pytest integration/test_spiral_problem.py -v --no-cov --slow --integration -x 2>&1 &
#     source /opt/miniforge3/etc/profile.d/conda.sh
#     conda activate JuniperCascor
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define Top level constants
#####################################################################################################################################################################################################
export TRUE=0
export FALSE=1


#####################################################################################################################################################################################################
# Define Script Constants
#####################################################################################################################################################################################################
export SLEEPY_TIME=5
NAP_COUNT=1

export SECONDS=60
export MINUTES=60
export HOURS=24
export DAYS=365

export SECOND_SECONDS="1"
export MINUTE_SECONDS="${SECONDS}"
export HOUR_SECONDS=$(( MINUTE_SECONDS * MINUTES ))
export DAY_SECONDS=$(( HOUR_SECONDS * HOURS ))
export YEAR_SECONDS=$(( DAY_SECONDS * DAYS ))

export YEARS_LABEL="Y"
export WEEKS_LABEL="W"
export DAYS_LABEL="D"
export HOURS_LABEL="h"
export MINUTES_LABEL="m"
export SECONDS_LABEL="s"

export ELAPSED_TIME=0
export ELAPSED_INFO=""


#####################################################################################################################################################################################################
# Define Script Environment Constants
#####################################################################################################################################################################################################
JUNIPER_CASCOR_PATH="${HOME}/Development/python/Juniper/JuniperCascor/juniper_cascor"
TESTS_REL_PATH="src/tests"
TESTS_PATH="${JUNIPER_CASCOR_PATH}/${TESTS_REL_PATH}"

NOHUP_FILENAME="nohup.out"
NOHUP_FILE="${TESTS_PATH}/${NOHUP_FILENAME}"


#####################################################################################################################################################################################################
# Define Pytest Constants
#####################################################################################################################################################################################################
TEST_NAME="integration/test_spiral_problem.py"
PYTEST_PARAMS="-v --no-cov --slow --integration -x"


#####################################################################################################################################################################################################
# Define Script Functions
#####################################################################################################################################################################################################
function scale_to_current_interval(){
    ELAPSED_SECONDS="$1"
    INTERVAL_SECONDS="$2"
    export INTERVAL_ELAPSED=$(( ELAPSED_SECONDS / INTERVAL_SECONDS ))
    # export ELAPSED_TIME=$(( ELAPSED_SECONDS % INTERVAL_SECONDS ))
    if (( INTERVAL_ELAPSED > 0 )); then
        export ELAPSED_TIME=$(( ELAPSED_SECONDS - ( INTERVAL_ELAPSED * INTERVAL_SECONDS ) ))
    fi
    echo "${INTERVAL_ELAPSED}"
}

function build_output_string(){
    INTERVAL_ELAPSED="$1"
    INTERVAL_LABEL="$2"
    if (( INTERVAL_ELAPSED > 0 )) || [[ "${SHOW_ZEROS}" == "${TRUE}" ]]; then
        export ELAPSED_INFO+="${INTERVAL_ELAPSED} ${INTERVAL_LABEL}, "
        export SHOW_ZEROS="${TRUE}"
    fi
}


function prep_time_info(){
    local NAP_COUNT="$1"
    export ELAPSED_TIME=$(( NAP_COUNT * SLEEPY_TIME ))
    YEARS_ELAPSED="$(scale_to_current_interval "${ELAPSED_TIME}" "${YEAR_SECONDS}")"
    DAYS_ELAPSED="$(scale_to_current_interval "${ELAPSED_TIME}" "${DAY_SECONDS}")"
    HOURS_ELAPSED="$(scale_to_current_interval "${ELAPSED_TIME}" "${HOUR_SECONDS}")"
    MINUTES_ELAPSED="$(scale_to_current_interval "${ELAPSED_TIME}" "${MINUTE_SECONDS}")"
    SECONDS_ELAPSED="$(scale_to_current_interval "${ELAPSED_TIME}" "${SECOND_SECONDS}")"
    export ELAPSED_INFO=""
    export SHOW_ZEROS="${FALSE}"
    build_output_string "${YEARS_ELAPSED}" "${YEARS_LABEL}"
    build_output_string "${DAYS_ELAPSED}" "${DAYS_LABEL}"
    build_output_string "${HOURS_ELAPSED}" "${HOURS_LABEL}"
    build_output_string "${MINUTES_ELAPSED}" "${MINUTES_LABEL}"
    build_output_string "${SECONDS_ELAPSED}" "${SECONDS_LABEL}"
}


#####################################################################################################################################################################################################
# Perform pre-test tasks
#####################################################################################################################################################################################################
cd "${TESTS_PATH}" || exit 1
pwd
if [[ -f "${NOHUP_FILE}" ]]; then
    echo "Truncating Nohup.out file"
    cat /dev/null > "${NOHUP_FILE}"
else
    echo "NOHUP.out file not found"
fi
python --version


#####################################################################################################################################################################################################
# Launch Tests as background task
#####################################################################################################################################################################################################
echo -ne "\nnohup time python -m pytest \"${TEST_NAME}\" \"${PYTEST_PARAMS}\" 2>&1 &"
nohup time python -m pytest "${TEST_NAME}" ${PYTEST_PARAMS} 2>&1 &
pid=$!
echo -ne "\n\nHello Pytest Pid: \"${pid}\"\n"


#####################################################################################################################################################################################################
# Perform status check at interval defined by SLEEPY_TIME
#####################################################################################################################################################################################################
echo "Sleepy Time: ${SLEEPY_TIME}"
sleep "${SLEEPY_TIME}"
# echo "while kill -0 $pid 2>/dev/null; do"
echo "while kill -0 $pid; do"
# while kill -0 $pid 2>/dev/null; do
while kill -0 $pid; do
    sleep "${SLEEPY_TIME}"
    NAP_COUNT=$(( NAP_COUNT + 1 ))
    prep_time_info "${NAP_COUNT}"
    echo -ne "Nap Count: ${NAP_COUNT}\tElapsed Time: ~${ELAPSED_INFO}\n"
done
