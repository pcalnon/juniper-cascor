#!/usr/bin/env bash


APP_DIR="${HOME}/Development/python/Juniper/JuniperCascor/juniper_cascor"

UTIL_DIR_NAME="util"
SRC_DIR_NAME="src"

UTIL_DIR="${APP_DIR}/${UTIL_DIR_NAME}"
SRC_DIR="${APP_DIR}/${SRC_DIR_NAME}"

TIMEOUT="120"


cd "${APP_DIR}"
pwd

source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && timeout 120 python src/main.py 2>&1 | grep -E "Manager started|Failed to start manager|Address already|correlation=|hidden|grow"


