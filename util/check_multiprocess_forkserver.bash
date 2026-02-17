#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       1.0.0
# File Name:     check_multiprocess_forkserver.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date Created:  2025-02-17
# Last Modified: 2026-01-28
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This file is the sourced conf file for the check_multiprocess_forkserver.bash script. The conf file defines all script constants.
#
#####################################################################################################################################################################################################
# Notes:
#
#
#   source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c "
#   import sys
#   sys.path.insert(0, 'src')
#
#   from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig
#
#   config = CascadeCorrelationConfig()
#   print(f'Address: {config.candidate_training_queue_address}')
#   print(f'Type:    {type(config.candidate_training_queue_address)}')
#   print(f'Context: {config.candidate_training_context_type}')
#   "
#
#####################################################################################################################################################################################################
# References:
#
#     SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
#     CONF_PATH="$(dirname "$(dirname "${SCRIPT_PATH}")")/conf"
#     CONF_FILENAME="$(basename -s ".bash" "${SCRIPT_PATH}").conf"
#     CONF_FILE="${SCRIPT_PATH}/${SCRIPT_FILENAME}"
#
#     CONF_PATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf"
#     CONF_FILENAME="$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"
#     CONF_FILE="${CONF_PATH}/${CONF_FILENAME}"
#
#     CONF_FILE="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf/$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"
#
#     source "${CONF_FILE}";  SUCCESS="$?"
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Define Project Environment Constants
#####################################################################################################################################################################################################
export PROJECT_NAME="Juniper"
export SUBPROJECT_NAME="JuniperCascor"
export APPLICATION_NAME="juniper_cascor"


#####################################################################################################################################################################################################
# Define Environment file system constants
#####################################################################################################################################################################################################
export APP_DIR="${HOME}/Development/python/${PROJECT_NAME}/${SUBPROJECT_NAME}/${APPLICATION_NAME}"

UTIL_DIR_NAME="util"
SRC_DIR_NAME="src"

export UTIL_DIR="${APP_DIR}/${UTIL_DIR_NAME}"
export SRC_DIR="${APP_DIR}/${SRC_DIR_NAME}"

MAIN_FILENAME="main.py"
export MAIN_FILE="${SRC_DIR}/${MAIN_FILENAME}"


#####################################################################################################################################################################################################
# Define Script Constants
#####################################################################################################################################################################################################
export CONDA_INIT="/opt/miniforge3/etc/profile.d/conda.sh"

export TIMEOUT="120"

export GREP_SEARCH_STRING="Manager started|Failed to start manager|Address already|correlation=|hidden|grow"


#####################################################################################################################################################################################################
# Run Juniper Cascor main
#####################################################################################################################################################################################################
cd "${SRC_DIR}" || exit 1
pwd

echo "source \"${CONDA_INIT}\" && conda activate \"${SUBPROJECT_NAME}\""
# shellcheck disable=SC1090
source "${CONDA_INIT}" && conda activate "${SUBPROJECT_NAME}"

# check if candidate_training_queue_address is being properly set to ('127.0.0.1', 0):
python -c "
import sys
sys.path.insert(0, 'src')

from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig()
print(f'Address: {config.candidate_training_queue_address}')
print(f'Type:    {type(config.candidate_training_queue_address)}')
print(f'Context: {config.candidate_training_context_type}')
"

exit 0
