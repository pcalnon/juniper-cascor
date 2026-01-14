#!/usr/bin/env bash
########################################################################################################################################################################################
# Application: Juniper
# Script Name: try.bash
# Script Path: <Project>/util/try.bash
#
# Description: This script performs the following actions for the current Project:
#
#                 1.  Applies the cargo linter to the source files
#                 2.  Builds the current project with the debug target
#                 3.  Sets the expected Environment Variables for the Application
#                 4.  Adds the expected command line arguments
#                 5.  Executes the project's binary
#
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Notes:
#
#
#
#######################################################################################################################################################################################

#####################################################################################################
# Specify the Python script to run:
####################################################################################################
#PYTHON_SCRIPT_NAME="binary_classifier.py"
#PYTHON_SCRIPT_NAME="mnist_classifier.py"
#PYTHON_SCRIPT_NAME="mnist_dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="full_dynamic_classifier.py"
#PYTHON_SCRIPT_NAME="static_nn.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-00.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-01.py"
#PYTHON_SCRIPT_NAME="dynamic_layer_nn-02.py"
#PYTHON_SCRIPT_NAME="dynamic_nodes_nn-00.py"

#export PYTHON_SCRIPT_NAME="dynamic_nodes_nn-02.py"
export PYTHON_SCRIPT_NAME="main.py"
#export PYTHON_SCRIPT_DIR="pytorch_static"
export PYTHON_SCRIPT_DIR="."
export PYTHON_SCRIPT_PATH="${PYTHON_SCRIPT_DIR}/${PYTHON_SCRIPT_NAME}"

#####################################################################################################
# Define Global Configuration File Constants
####################################################################################################
# export ROOT_PROJ_NAME="dynamic_nn"
# export ROOT_PROJ_NAME="juniper"
export ROOT_PROJ_NAME="Juniper"
export ROOT_SUBPROJECT_NAME="JuniperCascor"
export ROOT_APPLICATION_NAME="juniper_cascor"
export ROOT_CONF_NAME="conf"
export ROOT_CONF_FILE_NAME="script_util.cfg"
export ROOT_PROJ_DIR="${HOME}/Development/python/${ROOT_PROJ_NAME}/${ROOT_SUBPROJECT_NAME}/${ROOT_APPLICATION_NAME}"
export ROOT_CONF_DIR="${ROOT_PROJ_DIR}/${ROOT_CONF_NAME}"
export ROOT_CONF_FILE="${ROOT_CONF_DIR}/${ROOT_CONF_FILE_NAME}"
# shellcheck disable=SC1090
source "${ROOT_CONF_FILE}"

####################################################################################################
# Configure Script Environment
####################################################################################################
GET_OS_SCRIPT="__get_os_name.bash"
GET_PROJECT_SCRIPT="__get_project_dir.bash"
DATE_FUNCTIONS_SCRIPT="__git_log_weeks.bash"

# Determine Project Dir
# shellcheck disable=SC1090,SC2155
export BASE_DIR=$(${GET_PROJECT_SCRIPT} "${BASH_SOURCE[0]}")
echo "Base Dir: ${BASE_DIR}"
# Determine Host OS
# shellcheck disable=SC1090,SC2155
export CURRENT_OS=$(${GET_OS_SCRIPT})
# Define Script Functions
# shellcheck disable=SC1090
source "${DATE_FUNCTIONS_SCRIPT}"

####################################################################################################
# Define Script Constants
####################################################################################################
export DATA_DIR="${BASE_DIR}/${DATA_DIR_NAME}"
export SOURCE_DIR="${BASE_DIR}/${SOURCE_DIR_NAME}"
export CONFIG_DIR="${BASE_DIR}/${CONFIG_DIR_NAME}"
export LOGGING_DIR="${BASE_DIR}/${LOGGING_DIR_NAME}"
export UTILITY_DIR="${BASE_DIR}/${UTILITY_DIR_NAME}"

# shellcheck disable=SC2155
export PYTHON="$(which python3)"

export PYTHON_SCRIPT="${SOURCE_DIR}/${PYTHON_SCRIPT_PATH}"

####################################################################################################
# Display Environment Values
####################################################################################################
echo "Base Dir: ${BASE_DIR}"
echo "Current OS: ${CURRENT_OS}"
echo "Python: ${PYTHON} (ver: $(${PYTHON} --version))"
echo "Python Script: ${PYTHON_SCRIPT}"
echo " "

####################################################################################################
# Execute Python script
####################################################################################################
#echo "time ${PYTHON} ${PYTHON_SCRIPT} >./output.log"
echo "time ${PYTHON} ${PYTHON_SCRIPT}"

#time ${PYTHON} ${PYTHON_SCRIPT} >./output.log
time "${PYTHON}" "${PYTHON_SCRIPT}"
