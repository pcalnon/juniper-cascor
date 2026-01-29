#!/usr/bin/env bash
###########################################################################################################################################################################
# Project: Juniper
# File: make.bash
# Author: Paul Calnon
#
# Created Date: 2025-08-15
# Last Modified: 2026-01-12
# Version: 1.0
#
# License: GNU General Public License v3.0
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Repository:
# Description: This script compiles a Python source file into bytecode.
# Usage: ./make.bash <source_file.py> [<parent_directory>]
###########################################################################################################################################################################
# Examples:
#    ./make.bash src/prototypes/cascor/src/cascade_correlation/candidate_unit/candidate_unit.py
#    ./make.bash candidate_unit.py ./src/cascade_correlation/candidate_unit
###########################################################################################################################################################################
# Notes:
#     This script takes a Python source file as the first argument and an optional parent directory as the second argument.
#     The script will compile the source file into bytecode and place it in the appropriate directory structure.
#     If the parent directory is not provided, it will use the current working directory as the base directory.
#     The script will also validate the input parameters and ensure that the source file exists before attempting to compile it.
###########################################################################################################################################################################
# References:
#    - Python documentation on bytecode compilation: https://docs.python.org/3/library/compile.html
#    - Bash scripting best practices: https://tldp.org/LDP/Bash-Beginners-Guide/html/
#    - GNU General Public License v3.0: https://www.gnu.org/licenses/gpl-3.0.html
#    - SPDX License List: https://spdx.org/licenses/
###########################################################################################################################################################################
# TODO :
#
#
###########################################################################################################################################################################


###########################################################################################################################################################################
# Define Script Global Constants
TRUE="true"
FALSE="false"

# DEBUG="${TRUE}"       # Set to "true" to enable debug output
DEBUG="${FALSE}"    # Set to "false" to disable debug output

EXT="py"


###########################################################################################################################################################################
# Define Script Global Environment Variables
CWD="/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src"
# SOURCE_DIR_NAME="src"


###########################################################################################################################################################################
# Define Script Variables
SOURCE_DIR=""
PARENT_DIR=""
SOURCE_FILE=""
SOURCE_FILE_NAME=""
SOURCE_FILE_NAME_BASE=""
SOURCE_FILE_NAME_RAW=""
SOURCE_FILE_PATH=""
# CLASS_SUBDIR_NAME=""
# CLASS_PARENT_DIR=""
# CLASS_PARENT_PATH=""


###########################################################################################################################################################################
# Define Function to display usage information
function blank_line() {
    echo -ne "\n"
}

function usage() {
    blank_line
    echo "Usage: $0   <-f|--file <source_file.py>> [-p|--path <parent_directory>] [-h|--help]"
    blank_line
    echo "Compiles a Python source file into bytecode."
    blank_line
    echo "Arguments:"
    echo "  -f|--file <source_file.py>      The Python source file to compile."
    echo "  -p|--path <parent_directory>    Optional parent directory for the source file."
    blank_line
    echo "Examples:"
    echo "  $0 --file src/prototypes/cascor/src/cascade_correlation/candidate_unit/candidate_unit.py"
    echo "  $0 -f candidate_unit.py --path ./src/cascade_correlation/candidate_unit"
    blank_line
    exit 1
}


###########################################################################################################################################################################
# Define Functions to provide output and error handling
function error() {
    blank_line
    echo "Error: $1"
    echo "Use '$0 --help' for usage information."
    blank_line
    exit 1
}

function warning() {
    echo "Warning: $1"
}

function info() {
    echo "Info: $1"
}

function debug() {
    if [[ "${DEBUG}" == "true" ]]; then
        echo "Debug: $1"
    fi
}


###########################################################################################################################################################################
# Parse Command Line Arguments -- results returned in global variables FILE_NAME and PARENT_DIR
# TODO: add arg parsing while loop to handle multiple arguments
function parse_args() {
    debug "Parsing command line arguments"

    # Initialize variables
    # PARAMS="$@"
    PARAMS="$*"
    debug "Input Parameters: ${PARAMS}"
    # if [[ "${PARAMS}" == "" ]]; then
    if [[ "${PARAMS}" == "" ]]; then
        usage
    fi

    # Parse the command line arguments
    debug "Parse the command line arguments"
    # while [[ "$#" > 0 ]]; do
    while (( $# > 0 )); do
        debug "Processing argument: $1"
        case "$1" in
            -h|--help)
                debug "Received help switch:  $1"
                usage
                ;;
            -f|--file)
                debug "Received file switch: $1"
                shift
                if [[ "$1" == "" ]]; then
                    error "No file name provided after -f/--file option."
                fi
                FILE_NAME="$1"
                debug "File Name: ${FILE_NAME}"
                ;;
            -p|--path)
                shift
                if [[ -z "$1" ]]; then
                    error "No parent directory provided after -p/--parent option."
                fi
                PARENT_DIR="$1"
                ;;
            *)
                error "Unexpected argument: $1"
                ;;
        esac
        shift
    done
    debug "Parsed arguments: FILE_NAME=${FILE_NAME}, PARENT_DIR=${PARENT_DIR}"
}


###########################################################################################################################################################################
# Define Function to Check for overlapping paths between the current directory and the source file directory
#     NOTE: This code block is only intended to handle reasonable Source File paths
#     TODO: To handle pathological Source File paths, we would need to use at least one stack
function check_overlapping_paths() {
    # Define local variables
    local CURRENT_DIR="$1"
    local SOURCE_FILE_DIR="$2"
    local PREFIX="${CURRENT_DIR}"
    local SUFFIX=""
    local INFIX=""
    local DELIMITER="/"

    # Get the Source File Directory
    [[ "${SOURCE_FILE_DIR%%\/*}" == "." ]] && SUFFIX="${SOURCE_FILE_DIR#*/}" || SUFFIX="${SOURCE_FILE_DIR}"
    debug "Source File Directory: \"${SUFFIX}\""
    if [[ "${SUFFIX}" != "" ]]; then
        # error "Source file directory is empty. Please provide a valid source file path."
        INFIX="${PREFIX}"

        # log the initial values
        debug "Initial Prefix: ${PREFIX}"
        debug "Initial Infix: ${INFIX}"
        debug "Initial Suffix: ${SUFFIX}"

        # Check for overlapping paths
        debug "Checking for overlapping paths"
        debug "Initial:  Infix: \"${INFIX}\", Prefix: \"${PREFIX}\", Suffix: \"${SUFFIX}\""
        # until [[ "${PREFIX}" == "" ]]; do
        until [[ "${INFIX}" == "" ]]; do
            # FOUND="$(echo "${SUFFIX}" | grep "${PREFIX}")"
            debug "FOUND=\"\$(echo \"${SUFFIX}\" | grep \"${INFIX}\")\""
            FOUND="$(echo "${SUFFIX}" | grep "${INFIX}")"
            debug "Match: \"${FOUND}\""
            if [[ "${FOUND}" != "" ]]; then
                debug "Found Match:  Infix: \"${INFIX}\", Prefix: \"${PREFIX}\", Suffix: \"${SUFFIX}\""

                # Remove the matched part from the prefix
                debug "Remove the matched/duplicate sub-path from the prefix"
                PREFIX="${PREFIX%"${INFIX}"*}"
                debug "Updated Prefix: ${PREFIX}"
                INFIX=""
                debug "Completed Final duplicate sub-path check: \"${INFIX}\""
                break
            else
                debug "No match found. Continuing to check for overlapping paths."
                INFIX="${INFIX#*/}"
                debug "Removed remaining top level directory from Infix: \"${INFIX}\""
                debug "Match Not Found:  Infix: \"${INFIX}\", Prefix: \"${PREFIX}\", Suffix: \"${SUFFIX}\""
            fi
            debug "Completed Current duplicate sub-path check"
            debug "Iteration Complete: Infix: \"${INFIX}\", Prefix: \"${PREFIX}\", Suffix: \"${SUFFIX}\""
        done
        debug "Remove Trailing Delimiter from Prefix"
        PREFIX="${PREFIX%/}"
        debug "Final check: Infix: \"${INFIX}\", Prefix: \"${PREFIX}\", Suffix: \"${SUFFIX}\""
        debug "Delimiter: \"${DELIMITER}\""
    else
        debug "Source file directory is empty. Using current directory as source file path."
        DELIMITER=""
        debug "Updated Delimiter: \"${DELIMITER}\""
    fi

    # Validate the potentially overlapping paths parsing.
    debug "Validate Parsing of the potentially overlapping paths."
    if [[ "${INFIX}" != "" ]]; then
        error "Failed to parse overlapping paths. Unable to validate the source file path. Infix is not empty: ${INFIX}"
    fi

    # Construct the Source Directory without overlapping paths
    debug "Construct the Source Directory without overlapping paths"
    SOURCE_DIR="${PREFIX}${DELIMITER}${SUFFIX}"
    debug "Source Dir without overlap: ${SOURCE_DIR}"
}

###########################################################################################################################################################################
# Define Function to compile the source file
function compile_source_file() {
    local SOURCE_FILE="$1"
    if [[ "${SOURCE_FILE}" == "" ]]; then
        error "No source file provided for compilation."
    elif [[ ! -f "${SOURCE_FILE}" ]]; then
        error "Source file does not exist: ${SOURCE_FILE}"
    fi
    ! python -m py_compile "${SOURCE_FILE}" && { error "Failed to compile source file: ${SOURCE_FILE}"; } || { info "Successfully compiled source file: ${SOURCE_FILE}"; }
}


###########################################################################################################################################################################
# Main Script Execution
###########################################################################################################################################################################

###########################################################################################################################################################################
# Parses Command Line args and returns the file name and parent directory
blank_line
FILE_NAME=""
debug "Raw Parsed Arguments: FILE_NAME=${FILE_NAME}, PARENT_DIR=${PARENT_DIR}"

PARENT_DIR=""
parse_args "$@"
debug "Validation 0: Raw Parsed Arguments: FILE_NAME=${FILE_NAME}, PARENT_DIR=${PARENT_DIR}"

###########################################################################################################################################################################
# Validate Parsed source file Arguments
if [[ "${FILE_NAME}" == "" ]]; then
    error "Compilation target not selected.  Please provide a Python source file as the first argument."
fi
debug "Validation 1: File Name: ${FILE_NAME}"

# Parse the source file name and source file path
SOURCE_FILE_PATH="${FILE_NAME%/*}"
debug "Raw Source File Path: \"${SOURCE_FILE_PATH}\""
SOURCE_FILE_NAME="${FILE_NAME##*/}"
debug "Raw Source File Name: \"${SOURCE_FILE_NAME}\""

# Validate the source file name
if [[ "${SOURCE_FILE_NAME}" == "" ]]; then
    error "Source file name is empty. Please provide a valid Python source file."
fi
debug "Validation 2: Source File Name: ${SOURCE_FILE_NAME}"

# Validate the source file path
if [[ "${SOURCE_FILE_PATH}" == "" ]]; then
    debug "No partial path provided with source file. Using current directory as source file path."
    SOURCE_FILE_PATH="./"
fi
debug "Validation 3: Source File Path: ${SOURCE_FILE_PATH}"

# Validate combination of parsed source file parent directory and Current Working Directory
if [[ "${PARENT_DIR}" == "" ]]; then
    warning "compilation target directory not provided.  Using current working directory as parent directory."
    PARENT_DIR="${CWD}"
else
    SOURCE_DIR=""
    check_overlapping_paths "${CWD}" "${PARENT_DIR}"
    PARENT_DIR="${SOURCE_DIR}"
fi
debug "Validation 4: Parent Dir: ${PARENT_DIR}"
# Validate integration of Parent directory and current working directory
if [[ "${PARENT_DIR}" == "" ]]; then
    error "Failed to parse or generate a valid Parent directory. Parent directory variable is empty."
fi
debug "Validation 5: Parent Directory: ${PARENT_DIR}"

# Validate combination of Parent directory and source file directory
SOURCE_DIR=""
check_overlapping_paths "${PARENT_DIR}" "${SOURCE_FILE_PATH}"
debug "Validation 6: Source Dir: ${SOURCE_DIR}"

# Validate the integration of Parent directory and source file path
if [[ "${SOURCE_DIR}" == "" ]]; then
    error "Failed to parse or generate a valid Source directory. Source directory variable is empty."
fi
debug "Validation 7: Parent Directory and Source file path integrated: ${SOURCE_DIR}"

# Validate the existence of the source directory
if [[ ! -d "${SOURCE_DIR}" ]]; then
    error "Source directory does not exist: ${SOURCE_DIR}"
fi
debug "Validation 8: Source Directory Exists: ${SOURCE_DIR}"

# Build and Verify Source File Name
SOURCE_FILE_NAME_RAW="${FILE_NAME##*/}"
if [[ "${SOURCE_FILE_NAME_RAW}" == "" ]]; then
    error "Source file name is empty after parsing. Please provide a valid Python source file."
fi
debug "Validation 9: Source File Name Raw: ${SOURCE_FILE_NAME_RAW}"

# Validate the source file name base
debug "Source File Name Raw: ${SOURCE_FILE_NAME_RAW}"
SOURCE_FILE_NAME_BASE="${SOURCE_FILE_NAME_RAW%.*}"
debug "Raw Source File Name Base: \"${SOURCE_FILE_NAME_BASE}\""
if [[ "${SOURCE_FILE_NAME_BASE}" == "" ]]; then
    error "Source file name base is empty after parsing. Please provide a valid Python source file."
fi
debug "Validation 10: Source File Name Base: \"${SOURCE_FILE_NAME_BASE}\""

# Validate the source file class directory structure
SOURCE_FILE_CLASS_DIR="${SOURCE_DIR##*/}"
if [[ "${SOURCE_FILE_CLASS_DIR}" != "${SOURCE_FILE_NAME_BASE}" ]]; then
    error "Source file class directory directory structure is not valid for the current class."
fi
debug "Validation 11: Source File Class Dir is valid: ${SOURCE_FILE_CLASS_DIR}"

# Validate the source file name and extension
SOURCE_FILE_NAME="${SOURCE_FILE_NAME_RAW}"
if [[ "${SOURCE_FILE_NAME_BASE}" == "${SOURCE_FILE_NAME_RAW}" ]]; then
    warning "Source file name provided does not include a python extension."
    SOURCE_FILE_NAME="${SOURCE_FILE_NAME}.${EXT}"
fi
debug "Validation 12: Source File name with extension: ${SOURCE_FILE_NAME}"

# Validate the source file path and name
SOURCE_FILE="${SOURCE_DIR}/${SOURCE_FILE_NAME}"
if [[ "${SOURCE_FILE}" == "" ]]; then
    error "Source file path is empty after combining source directory and source file name."
fi
debug "Validation 13: Final Source File Path: ${SOURCE_FILE}"

# Validate the final source file
if [[ ! -f "${SOURCE_FILE}" ]]; then
    error "Source file does not exist: ${SOURCE_FILE}"
fi
debug "Validation 14: Source File Exists: ${SOURCE_FILE}"

# Compile the source file
blank_line
info "$(echo -ne "Python Path: \t$(which python || ${TRUE})\n")"
info "$(echo -ne "Python Ver:  \t$(python --version || ${TRUE})\n")"
info "$(echo -ne "File Name:   \t${SOURCE_FILE_NAME}\n")"
info "$(echo -ne "File Path:   \t${SOURCE_FILE}\n\n")"
blank_line

compile_source_file "${SOURCE_FILE}"

if [[ $? != 0 ]]; then
    error "Source file failed to compile successfully: ${SOURCE_FILE}"
fi
info "Completed attempt to Compile Source file: ${SOURCE_FILE}"
blank_line
