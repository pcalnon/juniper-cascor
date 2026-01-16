#!/usr/bin/env bash


TEST_NAME="integration/test_spiral_problem.py"
PYTEST_PARAMS="-v --no-cov --slow --integration -x"

JUNIPER_CASCOR_PATH="${HOME}/Development/python/Juniper/JuniperCascor/juniper_cascor"
TESTS_REL_PATH="src/tests"
TESTS_PATH="${JUNIPER_CASCOR_PATH}/${TESTS_REL_PATH}"

# source /opt/miniforge3/etc/profile.d/conda.sh
# conda activate JuniperCascor

cd "${TESTS_PATH}"
pwd

# python -m pytest integration/test_spiral_problem.py -v --no-cov --slow --integration -x 2>&1 &
python --version
echo -ne "\nnohup python -m pytest \"${TEST_NAME}\" \"${PYTEST_PARAMS}\" 2>&1 &"
nohup python -m pytest "${TEST_NAME}" ${PYTEST_PARAMS} 2>&1 &

pid=$!
echo "Hello Pytest Pid: \"${pid}\""

sleep 5
while kill -0 $pid 2>/dev/null; do
    sleep 5
