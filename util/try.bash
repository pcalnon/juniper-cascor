#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Prototype:     juniper_cascor
# File Name:     try.bash
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
#
# Date Created:  2025-11-05
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2025 Paul Calnon
#
# Description:
#    This try.bash file--and its symlinks--launch the juniper_cascor prototype.
#
#####################################################################################################################################################################################################
# Notes:
#
#     This script performs the following actions for the current Project:
#         1.  Applies the cargo linter to the source files
#         2.  Builds the current project with the debug target
#         3.  Sets the expected Environment Variables for the Application
#         4.  Adds the expected command line arguments
#         5.  Executes the project's binary
#
#####################################################################################################################################################################################################
# References:
#
#     SCRIPT_PATH = /home/pcalnon/Development/python/Juniper/src/prototypes/cascor/util/try.bash
#     SCRIPT_PATH = /home/pcalnon/Development/python/Juniper/src/prototypes/cascor/util
#     SCRIPT_PATH = /home/pcalnon/Development/python/Juniper/src/prototypes/cascor
#     SCRIPT_PATH = /home/pcalnon/Development/python/Juniper/src/prototypes
#     SCRIPT_PATH = /home/pcalnon/Development/python/Juniper/src/prototypes/util
#
#####################################################################################################################################################################################################
# TODO :
#     Integrate config infrastructure from juniper cascor into this sub-project
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################

#####################################################################################################
# Define Debug Constants
#####################################################################################################
# export TRUE="true"
export TRUE="0"
# export FALSE="false" # trunk-ignore(shellcheck/SC2034)
export FALSE="1" # trunk-ignore(shellcheck/SC2034)

DEBUG="${TRUE}"
# DEBUG="${FALSE}"

# CURRENT_SCRIPT="prototypes/cascor/util/try.bash:"

#####################################################################################################
# Define Global Functions
####################################################################################################
# Define local Functions
function get_script_path() {
	# local source="${BASH_SOURCE[0]}"
	# shellcheck disable=SC2155
	local source="$(realpath "${BASH_SOURCE[0]}")"
	while [[ -L ${source} ]]; do
		# local dir="$(cd -P "$(dirname "${source}")" && pwd)"
		local dir
		dir="$(cd -P "$(dirname "${source}")" && pwd)"
		source="$(readlink "${source}")"
		[[ ${source} != /* ]] && source="${dir}/${source}"
	done
	# trunk-ignore(shellcheck/SC2312)
	echo "$(cd -P "$(dirname "${source}")" && pwd)/$(basename "${source}")"
}

export -f get_script_path

####################################################################################################
# export CURRENT_SCRIPT="prototypes/cascor/util/try.bash:"
# TODO: Extract this from env
export CURRENT_SCRIPT="juniper_cascor/util/try.bash:"


####################################################################################################
# Get Python path
####################################################################################################
# PYTHON_PATH="$(which python)"
# trunk-ignore(shellcheck/SC2034)
# TODO: Extract this from env
export PYTHON_PATH="/opt/miniforge3/envs/JuniperCascor/bin/python"


####################################################################################################
# Define GLobal Constants for Prototype Util Shell Script code
####################################################################################################
# shellcheck disable=SC2155
export SCRIPT_FULL_PATH="$(get_script_path)"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} SCRIPT_FULL_PATH: ${SCRIPT_FULL_PATH}"
# shellcheck disable=SC2155
export SCRIPT_NAME="$(basename "${SCRIPT_FULL_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} SCRIPT_NAME: ${SCRIPT_NAME}"
# shellcheck disable=SC2155
export SCRIPT_PATH="$(dirname "${SCRIPT_FULL_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} SCRIPT_PATH: ${SCRIPT_PATH}"

# shellcheck disable=SC2155
export PYTHON_UTIL_PATH="${SCRIPT_PATH}"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_UTIL_PATH: ${PYTHON_UTIL_PATH}"
# shellcheck disable=SC2155
export PYTHON_UTIL_NAME="$(basename "${PYTHON_UTIL_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_UTIL_NAME: ${PYTHON_UTIL_NAME}"
# shellcheck disable=SC2155

# shellcheck disable=SC2155
export PYTHON_PROTO_PATH="$(dirname "${PYTHON_UTIL_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PROTO_PATH: ${PYTHON_PROTO_PATH}"
# shellcheck disable=SC2155
export PYTHON_PROTO_NAME="$(basename "${PYTHON_PROTO_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PROTO_NAME: ${PYTHON_PROTO_NAME}"

BASH_EXT="bash"
export PYTHON_PROTO_SCRIPT_NAME="${PYTHON_PROTO_NAME}.${BASH_EXT}"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PROTO_SCRIPT_NAME: ${PYTHON_PROTO_SCRIPT_NAME}"

PYTHON_PROTO_SCRIPT_DIR_NAME="util"
export PYTHON_PROTO_SCRIPT="${PYTHON_PROTO_PATH}/${PYTHON_PROTO_SCRIPT_DIR_NAME}/${PYTHON_PROTO_SCRIPT_NAME}"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PROTO_SCRIPT: ${PYTHON_PROTO_SCRIPT}"

# shellcheck disable=SC2155
export PYTHON_PARENT_PROTO_PATH="$(dirname "${PYTHON_PROTO_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_PROTO_PATH: ${PYTHON_PARENT_PROTO_PATH}"
# shellcheck disable=SC2155
export PYTHON_PARENT_PROTO_NAME="$(basename "${PYTHON_PARENT_PROTO_PATH}")"
[[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_PROTO_NAME: ${PYTHON_PARENT_PROTO_NAME}"

# # shellcheck disable=SC2155
# export PYTHON_PARENT_UTIL_NAME="util"
# [[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_UTIL_NAME: ${PYTHON_PARENT_UTIL_NAME}"
# # shellcheck disable=SC2155
# export PYTHON_PARENT_UTIL_PATH="${PYTHON_PARENT_PROTO_PATH}/${PYTHON_PARENT_UTIL_NAME}"
# [[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_UTIL_PATH: ${PYTHON_PARENT_UTIL_PATH}"

# # shellcheck disable=SC2155
# export PYTHON_PARENT_SCRIPT_NAME="${SCRIPT_NAME}"
# [[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_SCRIPT_NAME: ${PYTHON_PARENT_SCRIPT_NAME}"
# # shellcheck disable=SC2155
# export PYTHON_PARENT_SCRIPT="${PYTHON_PARENT_UTIL_PATH}/${PYTHON_PARENT_SCRIPT_NAME}"
# [[ ${DEBUG} == "${TRUE}" ]] && echo "${CURRENT_SCRIPT} PYTHON_PARENT_SCRIPT: ${PYTHON_PARENT_SCRIPT}"

####################################################################################################
# Define Cascor Prototype Logging files
####################################################################################################
PYTHON_PROTO_LOGGING_DIR_NAME="logs"
export PYTHON_PROTO_LOGGING_DIR="${PYTHON_PROTO_PATH}/${PYTHON_PROTO_LOGGING_DIR_NAME}"

PYTHON_PROTO_LOG_FILE_NAME_ROOT="${PYTHON_PROTO_NAME}"
PYTHON_PROTO_LOGGING_FILE_EXT=".log"
export PYTHON_PROTO_LOG_FILE_NAME="${PYTHON_PROTO_LOG_FILE_NAME_ROOT}${PYTHON_PROTO_LOGGING_FILE_EXT}"

export PYTHON_PROTO_LOG_FILE="${PYTHON_PROTO_LOGGING_DIR}/${PYTHON_PROTO_LOG_FILE_NAME}"
export PYTHON_PROTO_LOG_FILE_BORKED="${PYTHON_PROTO_PATH}/${PYTHON_PROTO_LOG_FILE_NAME}"

####################################################################################################
# Clear log files before running
####################################################################################################
truncate -s 0 "${PYTHON_PROTO_LOG_FILE}"
truncate -s 0 "${PYTHON_PROTO_LOG_FILE_BORKED}"

####################################################################################################
# Call Parent Util script passing in Prototype Name
####################################################################################################
# echo "${CURRENT_SCRIPT} ${PYTHON_PARENT_SCRIPT} \"${PYTHON_PROTO_NAME}\""
echo "${CURRENT_SCRIPT} ${PYTHON_PROTO_SCRIPT} \"${PYTHON_PROTO_NAME}\""

# ${PYTHON_PARENT_SCRIPT} "${PYTHON_PROTO_NAME}"
${PYTHON_PROTO_SCRIPT} "${PYTHON_PROTO_NAME}"
