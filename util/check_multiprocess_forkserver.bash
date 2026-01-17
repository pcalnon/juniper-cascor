#!/usr/bin/env bash


APP_DIR="${HOME}/Development/python/Juniper/JuniperCascor/juniper_cascor"

UTIL_DIR_NAME="util"
SRC_DIR_NAME="src"

UTIL_DIR="${APP_DIR}/${UTIL_DIR_NAME}"
SRC_DIR="${APP_DIR}/${SRC_DIR_NAME}"

cd "${SRC_DIR}"
pwd

# check if candidate_training_queue_address is being properly set to ('127.0.0.1', 0):
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate JuniperCascor && python -c "
import sys
sys.path.insert(0, 'src')

from cascade_correlation.cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

config = CascadeCorrelationConfig()
print(f'Address: {config.candidate_training_queue_address}')
print(f'Type:    {type(config.candidate_training_queue_address)}')
print(f'Context: {config.candidate_training_context_type}')
"

