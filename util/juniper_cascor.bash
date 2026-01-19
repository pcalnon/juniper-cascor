#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     juniper_cascor.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date Created:  2025-10-11
# Last Modified: 2026-01-19
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#    This script performs initial environment setup and launches the JuniperCascor Application to train and evaluate the current Cascade Correlation Neural Network prototype
#
#####################################################################################################################################################################################################
# Notes:
#     This script is assumed to be located in a **/<Project Name>/utils/ dir for the Current Project
#     Languages are all assumed to be installed in and accessible from conda
#
#     Key Constants Defined in the juniper_cascor.conf file
#         PROJECT_NAME
#         PROTOTYPE_PROJECT == TRUE|FALSE
#         CURRENT_PROJECT
#         PROJECT_PATH
#         HOME_DIR
#         MAIN_FILE
#         LANGUAGE_NAME
#         LANGUAGE_PATH
#         PYTHON, JAVASCRIPT, RUST, JAVA, RUBY, NODE, GO, CPP, C, R
#         CASCOR_NAME
#         CASCOR_PATH
#         CASCOR
#
########################################################################################################)#############################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO:
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################



####################################################################################################################################################################################################
# Source script config file
#####################################################################################################################################################################################################
set -o functrace
# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")" && INIT_CONF="$(dirname "$(dirname "${PARENT_PATH_PARAM}")")/conf/init.conf"
# shellcheck disable=SC2015,SC1090
[[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found. Unable to Continue."; exit 1; }


#####################################################################################################################################################################################################
# Validate Environment
#    NOTE: This check was already carried out in the juniper_cascor.conf config file
#####################################################################################################################################################################################################
log_info "Validating Environment: Conda Active Env: \"${CONDA_ACTIVE_ENV}\", Expected Conda Env: \"${CONDA_ENV_NAME}\""
log_info "Validating Environment: Python Version: ${PYTHON_VERSION}, Expected Python Version: ${LANGUAGE_VERS}"
if [[ "${CONDA_ACTIVE_ENV}" != "${CONDA_ENV_NAME}" ]]; then
    log_error "Active Conda Environment is Wrong: Found: \"${CONDA_ACTIVE_ENV}\", Expected: \"${CONDA_ENV_NAME}\""
elif [[ "${PYTHON_VERSION}" != "${LANGUAGE_VERS}" ]]; then
    log_error "Python Version is Wrong: Found: ${PYTHON_VERSION}, Expected: ${LANGUAGE_VERS}"
else
    log_info "Successfully Validated Env: Python Version: ${PYTHON_VERSION}, Conda Environment: ${CONDA_ACTIVE_ENV}"
fi


#####################################################################################################################################################################################################
# Initialize Project Main File
#####################################################################################################################################################################################################
log_debug "CASCOR_MAIN_FILE: ${CASCOR_MAIN_FILE}"
PYTHON_SCRIPT="${CASCOR_MAIN_FILE}"
log_debug "Python Script: ${PYTHON_SCRIPT}"


#####################################################################################################################################################################################################
# Display Environment Values
#####################################################################################################################################################################################################
log_debug "Base Dir: ${BASE_DIR}"
log_debug "Current OS: ${CURRENT_OS}"
log_debug "Python: ${PYTHON} (ver: $("${PYTHON}" "--version"))"


#####################################################################################################################################################################################################
# Launch the Main function of the Juniper CasCor Application
#####################################################################################################################################################################################################
log_debug "time \"${PYTHON}\" \"${PYTHON_SCRIPT}\""
time "${PYTHON}" "${PYTHON_SCRIPT}"

exit $(( TRUE ))
