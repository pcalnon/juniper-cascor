#!/bin/bash

PARAMS="$*"

# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/auto_grad_test.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-0.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-1.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-2.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-3.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-3a.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-4.py"
# SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/torchexplorer_test-4a.py"
SCRIPT="/home/pcalnon/Development/python/dynamic_nn/src/prototypes/claude_sonnet_3.7_0.py"

python --version

time python "${SCRIPT}" "${PARAMS}"
