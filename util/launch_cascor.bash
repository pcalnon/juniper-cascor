#!/usr/bin/env bash

# Cascor listens on:
#
# 127.0.0.1:8200
# 0.0.0.0:8200

APPLICATION_DIR="${HOME}/Development/python/Juniper/juniper-cascor"
SOURCE_DIR="${APPLICATION_DIR}/src"

export JUNIPER_CASCOR_PORT="8201"
export JUNIPER_CASCOR_AUTO_START="true"

cd "${SOURCE_DIR}"
# python main.py
python server.py
