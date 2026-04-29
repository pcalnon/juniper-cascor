#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCascor
# Application:   juniper_cascor
# Purpose:       Juniper Project Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     proto.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date Created:  2025-10-11
# Last Modified: 2026-01-12
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#
#####################################################################################################################################################################################################
# Notes:
#
########################################################################################################)#############################################################################################
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
# @author: <NAME>
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Initialize script by sourcing the init_conf.bash config file
#####################################################################################################################################################################################################
set -o functrace

# ─── Free-threading interpreter guard ───────────────────────────────────────
# Abort cleanly under a CPython 3.14 free-threading build to avoid the
# psutil PyInit__psutil_linux segfault triggered by ABI-mismatched native
# wheels in the JuniperCascor env. Override with
# CASCOR_ALLOW_FREE_THREADING=1.
if [[ "${CASCOR_ALLOW_FREE_THREADING:-}" != "1" ]]; then
    if python -c "import sys, sysconfig; sys.exit(0 if sysconfig.get_config_var('Py_GIL_DISABLED') else 1)" 2>/dev/null; then
        cat >&2 <<'EOF'

ERROR: pytest cannot run under a free-threading CPython build (Py_GIL_DISABLED=1).
       Recreate the JuniperCascor env on a regular (GIL) Python:
         conda env remove -n JuniperCascor -y
         conda create -n JuniperCascor python=3.13 -c conda-forge -y
         conda activate JuniperCascor && pip install -e .
       Override at your own risk: CASCOR_ALLOW_FREE_THREADING=1.

EOF
        exit 2
    fi
fi

# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")" && INIT_CONF="conf/init.conf"
# shellcheck disable=SC2015
# shellcheck source=conf/init.conf
# shellcheck disable=SC1091
[[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found. Unable to Continue."; exit 1; }


#####################################################################################################################################################################################################
# Script to run tests with proper PYTHONPATH
#####################################################################################################################################################################################################

# Get absolute path to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

# Export PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# Run pytest with all arguments passed through
cd "${SCRIPT_DIR}" || exit 1
/opt/miniforge3/envs/JuniperPython/bin/python -m pytest "$@"
