#!/usr/bin/env bash
########################################################################################################################################################################
# Author: Paul Calnon
# Date: 2026-03-12
# Version: 1.1.0
# Status: Development
# License: MIT
# Copyright: 2026 Paul Calnon
# Repository: https://github.com/pcalnon/juniper-ml
#
# Description:
#    this script is a wrapper for the claude code agent.
#    this script is not intended to be used directly.
#    this script performs the actual call to launch the claude code agent.
#    it is used to launch the claude code agent with the appropriate parameters.
#    it is also used to validate the session id and resume the session.
#    it is also used to save the session id to a file.
#    it is also used to retrieve the session id from a file.
########################################################################################################################################################################
# Notes:
#
#   Old Index Assignment Syntax:
#     MODEL_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     EFFORT_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     SESSION_ID_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     WORKTREE_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     REMOTE_CONTROL_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     PERMISSIONS_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     RESUME_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     SPACER_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     PROMPT_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     PATH_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     FILE_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     HEADLESS_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     FORK_SESSION_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     MODEL_VERSION_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     SCRIPT_VERSION_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     HELP_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#     USAGE_FLAGS_INDEX="${INITIAL_INDEX}" && INITIAL_INDEX=$((INITIAL_INDEX + 1))
#
########################################################################################################################################################################
# References:
#
#    echo "0. Script Call"
#    echo "1. Initial Script Params: \"${*}\""
#
#    declare -a PARAMS=("${@}")
#    echo "2. Script Params: \"${PARAMS[*]}\""
#
#    declare -a PPARAMS=()
#    declare -a SPPARAMS=()
#    declare -a FPARAMS=()
#    declare -a SFPARAMS=()
#
#    echo "3. Function Definition"
#    function bla() {
#        FPARAMS=("${@}")
#        echo "5.0. Function Param 1: \"${1}\""
#        echo "5.1. Function Params: \"${FPARAMS[*]}\""
#        shift 2
#        echo "5.1.1. Function Param 1: \"${1}\""
#        echo "5.2. Shifted Function Params: \"${*}\""
#        SFPARAMS=("${@}")
#        echo "5.3. Shifted Function Params: \"${SFPARAMS[*]}\""
#        set "${PARAMS[@]}"
#        echo "5.3.1. Function Param 1: \"${1}\""
#        echo "5.4. Set Script Params: \"${*}\""
#        PPARAMS=("${@}")
#        echo "5.5. Parent Params: \"${PPARAMS[*]}\""
#        shift 2
#        SPPARAMS=("${@}")
#        echo "5.5.1. Shifted Parent Param 1: \"${1}\""
#        echo "5.6. Shifted Parent Params: \"${*}\""
#        echo "5.7. Shifted Parent Params: \"${SPPARAMS[*]}\""
#        set "${SFPARAMS[@]}"
#        echo "5.7. Shifted Function Params: \"${*}\""
#        shift 2
#        echo "5.7.1. Function Param 1: \"${1}\""
#        echo "5.8. Double Shifted Function Params: \"${*}\""
#    }
#
#    echo "4. Function Call"
#    echo "5. bla 0 1 2 3 4 5 6 7 8 9"
#    bla 0 1 2 3 4 5 6 7 8 9
#
#    echo "6. Final Script Params \"${*}\""
#    echo "6.1. Script Param 1: \"${1}\""
#    set "${SPPARAMS[@]}"
#    echo "7. Function Params: \"${SFPARAMS[*]}\""
#    echo "7.1. Final Script Params \"${*}\""
#    echo "7.2. Script Param 1: \"${1}\""
#    echo "8. Shifted Function Params: \"${SFPARAMS[*]}\""
#    echo "9. Shifted Parent Params: \"${SPPARAMS[*]}\""
#
#    elif [[ ( ( "${CLAUDE_RESUME_VALUE[*]}" == "" ) || ( "${#CLAUDE_RESUME_VALUE[@]}" == "0" ) ) && ( ( "${session_id}" != "${FALSE}" ) || ( "${session_id}" == "" ) ) ]]; then
#    elif [[ ( "${CLAUDE_RESUME_VALUE[*]}" == "" ) || ( "${#CLAUDE_RESUME_VALUE[@]}" == "0" ) ]]; then
#
########################################################################################################################################################################


########################################################################################################################################################################
# Define global Variables
########################################################################################################################################################################
TRUE="0"
FALSE="1"
UNSAFE_TARGET="2"  # Distinct save_session_id() status for a security refusal (e.g. a symlink target)

DEBUG="${TRUE}"

WTC_DEBUG="${WTC_DEBUG:-${DEBUG}}"
EXIT_AFTER_USAGE_DEFAULT="${TRUE}"

# Only require a successful session-ID save when NOT in wtc debug mode. Explicit comparison
# (robust to non-numeric/empty WTC_DEBUG, unlike the previous parity arithmetic).
export REQUIRE_SAVED_SESSION_ID="${TRUE}"
if [[ "${WTC_DEBUG}" == "${TRUE}" ]]; then
    export REQUIRE_SAVED_SESSION_ID="${FALSE}"
fi

SCRIPT_NAME="$(basename "$(realpath "${BASH_SOURCE[0]}")")"
# This script's own version, as its header states, for --script-version. Assigned unconditionally: claude_interactive.bash exports
# SCRIPT_VERSION as its own version, which this script would otherwise inherit and report as its own.
SCRIPT_VERSION="1.1.0"
# set -x
# PS4="+ ${SCRIPT_NAME}: ${FUNCNAME[1]}(${LINENO}): ${PS4}: "
# PS4="+ ${SCRIPT_NAME}: ${FUNCNAME[1]}(${LINENO}): "
# export PS4
# set +x

########################################################################################################################################################################
# Early function definitions (must precede top-level calls below)
########################################################################################################################################################################
function get_line_number() {
    [[ "${1:-""}" != "" ]] && { echo "${1%% *}" && return 0; } || { echo "-1" && return 1; }
}

function debug_log() {
    if [[ "${WTC_DEBUG}" == "${TRUE}" ]]; then
        local line && line="$( get_line_number "$( caller )" )"
        echo "${SCRIPT_NAME}: ${FUNCNAME[1]}(${line}): ${*}" 1>&2
    fi
}

function redact_uuid() {
    local value="$1"
    if (( ${#value} >= 12 )); then
        echo "${value:0:8}...${value: -4}"
    else
        echo "[redacted]"
    fi
}


########################################################################################################################################################################
# Define Script Directory Constants
########################################################################################################################################################################
SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSIONS_DIR="${WTC_SESSIONS_DIR:-${REPO_ROOT}/scripts/sessions}"
LOGS_DIR="${WTC_LOGS_DIR:-${REPO_ROOT}/logs}"

PROMPT_DIR_NAME_DEFAULT="prompts"
PROMPT_DIR_DEFAULT="${WTC_PROMPT_DIR_DEFAULT:-${REPO_ROOT}/${PROMPT_DIR_NAME_DEFAULT}}"
export PROMPT_DIR_DEFAULT

mkdir -p "${SESSIONS_DIR}" "${LOGS_DIR}"


########################################################################################################################################################################
# Define Claude Parameter Arrays index values
########################################################################################################################################################################
# Define array variable to save script input parameters as they are parsed
declare -a UPDATED_PARAMS=()

export PARAM_FLAG_INDEX="0"
export PARAM_VALUE_INDEX="1"

SESSION_ID_FILE=""


########################################################################################################################################################################
# Define Model Name Constants
#     NOTE: This is where you would add new model names and aliases
########################################################################################################################################################################

########################################################################################################################################################################
# Define Model Full Name Constants
debug_log "Define Model Name Constants"
# NOTE: Add new Model Full Name here
MODEL_FABLE_NAME="claude-fable-5"
MODEL_OPUS_NAME="claude-opus-5"
MODEL_SONNET_NAME="claude-sonnet-4-6"
MODEL_HAIKU_NAME="claude-haiku-4-5"


########################################################################################################################################################
# Define Model Name Array
debug_log "Define Model Name Array"
declare -a MODEL_NAME_ARRAY
MODEL_NAME_ARRAY+=("${MODEL_OPUS_NAME}")
MODEL_NAME_ARRAY+=("${MODEL_HAIKU_NAME}")
MODEL_NAME_ARRAY+=("${MODEL_SONNET_NAME}")
MODEL_NAME_ARRAY+=("${MODEL_FABLE_NAME}")


########################################################################################################################################################
# Define Model name Alias Constants
debug_log "Define Model Name Alias Constants"
# NOTE: Add new Model Name Alias here
MODEL_FABLE_ALIAS="fable"
MODEL_OPUS_ALIAS="opus"
MODEL_SONNET_ALIAS="sonnet"
MODEL_HAIKU_ALIAS="haiku"

MODEL_DEFAULT="${MODEL_OPUS_ALIAS}"
export MODEL_DEFAULT


########################################################################################################################################################
# Define Model Alias Array
debug_log "Define Model Alias Array"
declare -a MODEL_ALIAS_ARRAY
MODEL_ALIAS_ARRAY+=("${MODEL_OPUS_ALIAS}")
MODEL_ALIAS_ARRAY+=("${MODEL_HAIKU_ALIAS}")
MODEL_ALIAS_ARRAY+=("${MODEL_SONNET_ALIAS}")
MODEL_ALIAS_ARRAY+=("${MODEL_FABLE_ALIAS}")


########################################################################################################################################################
# Define Array for Model Name and Alias Validation Checks
debug_log "Define Model Name and Alias Validation Array"
declare -a MODEL_TEST_ARRAY
# NOTE: Add new Model Name and Alias to Validation Array
MODEL_TEST_ARRAY+=("${MODEL_OPUS_NAME}")
MODEL_TEST_ARRAY+=("${MODEL_HAIKU_NAME}")
MODEL_TEST_ARRAY+=("${MODEL_SONNET_NAME}")
MODEL_TEST_ARRAY+=("${MODEL_FABLE_NAME}")
MODEL_TEST_ARRAY+=("${MODEL_OPUS_ALIAS}")
MODEL_TEST_ARRAY+=("${MODEL_HAIKU_ALIAS}")
MODEL_TEST_ARRAY+=("${MODEL_SONNET_ALIAS}")
MODEL_TEST_ARRAY+=("${MODEL_FABLE_ALIAS}")


########################################################################################################################################################################
# Define Claude Code Level of Effort Values Constants
########################################################################################################################################################################

########################################################################################################################################################################
# Define Claude Code Effort Values Constants
debug_log "Define Effort values"
EFFORT_LOW="low"
EFFORT_MED="medium"
EFFORT_HIGH="high"
EFFORT_XHIGH="xhigh"
EFFORT_MAX="max"
EFFORT_AUTO="auto"

# max is the default again (Overtoad ruling, 2026-11-08). Claude Code has no "auto" effort: it answers --effort auto with
# "Unknown --effort value 'auto'" and ignores it, so an auto value sends no --effort flag at all (see CLAUDE_CODE_PARAMS).
# EFFORT_DEFAULT="${EFFORT_AUTO}"
EFFORT_DEFAULT="${EFFORT_MAX}"
export EFFORT_DEFAULT


########################################################################################################################################################################
# Define Claude Code Effort Array
debug_log "Define Effort Array"
declare -a EFFORT_ARRAY
EFFORT_ARRAY+=("${EFFORT_LOW}")
EFFORT_ARRAY+=("${EFFORT_MED}")
EFFORT_ARRAY+=("${EFFORT_HIGH}")
EFFORT_ARRAY+=("${EFFORT_XHIGH}")
EFFORT_ARRAY+=("${EFFORT_MAX}")
EFFORT_ARRAY+=("${EFFORT_AUTO}")


########################################################################################################################################################################
# Initialize Parsed Param values
########################################################################################################################################################################

########################################################################################################################################################################
# Define prompt parameter values
debug_log "Define Parsed Param values"
export PATH_NAME=""
export FILE_NAME=""
export PROMPT_FILE=""
export PROMPT_VALUE=""

PROMPT_VALUE_DEFAULT="Hello World, Claude!"
export PROMPT_VALUE_DEFAULT


########################################################################################################################################################################
# Define Parameter Flag and Value Arrays
debug_log "Define Parameter Flag and Value Arrays"
declare -a CLAUDE_MODEL_VALUE
declare -a CLAUDE_EFFORT_VALUE
declare -a CLAUDE_WORKTREE_VALUE
declare -a CLAUDE_REMOTE_CONTROL_VALUE
declare -a CLAUDE_SESSION_ID_VALUE
declare -a CLAUDE_PERMISSIONS_VALUE
declare -a CLAUDE_RESUME_VALUE
declare -a CLAUDE_HEADLESS_VALUE
declare -a CLAUDE_FORK_SESSION_VALUE


########################################################################################################################################################################
# Param values to be passed to Claude
debug_log "Define Param values to be passed to Claude"
CLAUDE_CODE_PROMPT=""
CLAUDE_CODE_PARAMS=()


########################################################################################################################################################################
# Valid Prompt Boolean Flags
debug_log "Define Valid Prompt Boolean Flags"
VALID_FILE_PARAM="${FALSE}"
VALID_PATH_PARAM="${FALSE}"
VALID_PROMPT_PARAM="${FALSE}"

PROMPT_DEFAULT="Hello World, Claude!"
export PROMPT_DEFAULT


########################################################################################################################################################################
# Define Input parameter flags
########################################################################################################################################################################

########################################################################################################################################################################
# Define Claude Code parameter flags
debug_log "Define Claude Code parameter flags"
CLAUDE_PERMISSIONS_FLAGS="--dangerously-skip-permissions"
CLAUDE_REMOTE_CONTROL_FLAGS="--remote-control"
CLAUDE_FORK_SESSION_FLAGS="--fork-session"
CLAUDE_SESSION_ID_FLAGS="--session-id"
CLAUDE_WORKTREE_FLAGS="--worktree"
CLAUDE_VERSION_FLAGS="--version"
CLAUDE_HEADLESS_FLAGS="--print"
CLAUDE_RESUME_FLAGS="--resume"
CLAUDE_EFFORT_FLAGS="--effort"
CLAUDE_MODEL_FLAGS="--model"
export CLAUDE_PROMPT_FLAGS="--prompt"
# Ends Claude Code's options: the prompt follows it, as the last argument (see "Build Claude Code Prompt")
CLAUDE_END_OF_OPTIONS_FLAG="--"

########################################################################################################################################################################
# Define all valid Script Input parameter flags:
debug_log "Define Input parameter flags"
SPACER_FLAGS="--"
USAGE_FLAGS="-u | --usage"
HELP_FLAGS="-h | --help | -?"
FILE_FLAGS="-f | --file | --file-name"
PATH_FLAGS="-l | --path | --file-path"
PROMPT_FLAGS="-p | --prompt | --prompt-string"
WORKTREE_FLAGS="-w | ${CLAUDE_WORKTREE_FLAGS} | --work-tree | --working-tree"
MODEL_VERSION_FLAGS="-v | -V | ${CLAUDE_VERSION_FLAGS} | --claude-version"
# --version is Claude Code's own version flag, so it asks for Claude Code's version alone; this script's version takes a flag of its
# own (Overtoad ruling, 2026-11-04). The two patterns once shared --version, and the model-version pattern, matched first, won.
# -v follows --version, as `claude -v` does (Overtoad ruling, 2026-11-06); until then it printed this script's version.
SCRIPT_VERSION_FLAGS="--script-version"
EFFORT_FLAGS="-e | ${CLAUDE_EFFORT_FLAGS} | --effort-type | --effort-level"
HEADLESS_FLAGS="-a | ${CLAUDE_HEADLESS_FLAGS} | --agent | --silent | --headless"
REMOTE_CONTROL_FLAGS="-c | --remote | --control | ${CLAUDE_REMOTE_CONTROL_FLAGS}"
PERMISSIONS_FLAGS="-s | --skip | --skip-permissions | ${CLAUDE_PERMISSIONS_FLAGS}"
SESSION_ID_FLAGS="-i | --id | --session | ${CLAUDE_SESSION_ID_FLAGS} | --session-name | -t | --thread | --thread-name | --thread-id"
MODEL_FLAGS="-m | ${CLAUDE_MODEL_FLAGS} | --llm-model | --ai-model | --model-type"
RESUME_FLAGS="-r | ${CLAUDE_RESUME_FLAGS} | --resume-thread | --resume-session"
FORK_SESSION_FLAGS="--fork | ${CLAUDE_FORK_SESSION_FLAGS} | --resume-fork | --resume-fork_session"


########################################################################################################################################################################
# Initialize Index values for Valid Patterns Array
debug_log "Define Index values for Valid Patterns Array"
INITIAL_INDEX=0
MODEL_FLAGS_INDEX="${INITIAL_INDEX}";          INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "MODEL_FLAGS_INDEX: ${MODEL_FLAGS_INDEX}"
EFFORT_FLAGS_INDEX="${INITIAL_INDEX}";         INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "EFFORT_FLAGS_INDEX: ${EFFORT_FLAGS_INDEX}"
WORKTREE_FLAGS_INDEX="${INITIAL_INDEX}";       INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "WORKTREE_FLAGS_INDEX: ${WORKTREE_FLAGS_INDEX}"
REMOTE_CONTROL_FLAGS_INDEX="${INITIAL_INDEX}"; INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "REMOTE_CONTROL_FLAGS_INDEX: ${REMOTE_CONTROL_FLAGS_INDEX}"
SESSION_ID_FLAGS_INDEX="${INITIAL_INDEX}";     INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "SESSION_ID_FLAGS_INDEX: ${SESSION_ID_FLAGS_INDEX}"
PERMISSIONS_FLAGS_INDEX="${INITIAL_INDEX}";    INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "PERMISSIONS_FLAGS_INDEX: ${PERMISSIONS_FLAGS_INDEX}"
RESUME_FLAGS_INDEX="${INITIAL_INDEX}";         INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "RESUME_FLAGS_INDEX: ${RESUME_FLAGS_INDEX}"
SPACER_FLAGS_INDEX="${INITIAL_INDEX}";         INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "SPACER_FLAGS_INDEX: ${SPACER_FLAGS_INDEX}"
PROMPT_FLAGS_INDEX="${INITIAL_INDEX}";         INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "PROMPT_FLAGS_INDEX: ${PROMPT_FLAGS_INDEX}"
PATH_FLAGS_INDEX="${INITIAL_INDEX}";           INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "PATH_FLAGS_INDEX: ${PATH_FLAGS_INDEX}"
FILE_FLAGS_INDEX="${INITIAL_INDEX}";           INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "FILE_FLAGS_INDEX: ${FILE_FLAGS_INDEX}"
HEADLESS_FLAGS_INDEX="${INITIAL_INDEX}";       INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "HEADLESS_FLAGS_INDEX: ${HEADLESS_FLAGS_INDEX}"
FORK_SESSION_FLAGS_INDEX="${INITIAL_INDEX}";   INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "FORK_SESSION_FLAGS_INDEX: ${FORK_SESSION_FLAGS_INDEX}"
MODEL_VERSION_FLAGS_INDEX="${INITIAL_INDEX}";  INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "MODEL_VERSION_FLAGS_INDEX: ${MODEL_VERSION_FLAGS_INDEX}"
SCRIPT_VERSION_FLAGS_INDEX="${INITIAL_INDEX}"; INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "SCRIPT_VERSION_FLAGS_INDEX: ${SCRIPT_VERSION_FLAGS_INDEX}"
HELP_FLAGS_INDEX="${INITIAL_INDEX}";           INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "HELP_FLAGS_INDEX: ${HELP_FLAGS_INDEX}"
USAGE_FLAGS_INDEX="${INITIAL_INDEX}";          INITIAL_INDEX=$((INITIAL_INDEX + 1)); debug_log "USAGE_FLAGS_INDEX: ${USAGE_FLAGS_INDEX}"


########################################################################################################################################################################
# Initialize Parsed Param values
debug_log "Define Valid Patterns Array"
declare -a VALID_PATTERNS

VALID_PATTERNS[MODEL_FLAGS_INDEX]="${MODEL_FLAGS}"
VALID_PATTERNS[EFFORT_FLAGS_INDEX]="${EFFORT_FLAGS}"
VALID_PATTERNS[WORKTREE_FLAGS_INDEX]="${WORKTREE_FLAGS}"
VALID_PATTERNS[REMOTE_CONTROL_FLAGS_INDEX]="${REMOTE_CONTROL_FLAGS}"
VALID_PATTERNS[SESSION_ID_FLAGS_INDEX]="${SESSION_ID_FLAGS}"
VALID_PATTERNS[PERMISSIONS_FLAGS_INDEX]="${PERMISSIONS_FLAGS}"
VALID_PATTERNS[RESUME_FLAGS_INDEX]="${RESUME_FLAGS}"
VALID_PATTERNS[SPACER_FLAGS_INDEX]="${SPACER_FLAGS}"
VALID_PATTERNS[PROMPT_FLAGS_INDEX]="${PROMPT_FLAGS}"
VALID_PATTERNS[PATH_FLAGS_INDEX]="${PATH_FLAGS}"
VALID_PATTERNS[FILE_FLAGS_INDEX]="${FILE_FLAGS}"
VALID_PATTERNS[HEADLESS_FLAGS_INDEX]="${HEADLESS_FLAGS}"
VALID_PATTERNS[FORK_SESSION_FLAGS_INDEX]="${FORK_SESSION_FLAGS}"
VALID_PATTERNS[MODEL_VERSION_FLAGS_INDEX]="${MODEL_VERSION_FLAGS}"
VALID_PATTERNS[SCRIPT_VERSION_FLAGS_INDEX]="${SCRIPT_VERSION_FLAGS}"
VALID_PATTERNS[HELP_FLAGS_INDEX]="${HELP_FLAGS}"
VALID_PATTERNS[USAGE_FLAGS_INDEX]="${USAGE_FLAGS}"


########################################################################################################################################################################
# Define functions for wake_the_claude.bash script
########################################################################################################################################################################

########################################################################################################################################################################
# matches_pattern(): Match a value against a pipe-delimited pattern string
function matches_pattern() {
    local ip_value="$1"
    local pattern="$2"
    local IFS='|'
    local -a candidates=()
    local candidate
    read -r -a candidates <<< "${pattern}"
    for candidate in "${candidates[@]}"; do
        candidate="${candidate#"${candidate%%[![:space:]]*}"}"
        candidate="${candidate%"${candidate##*[![:space:]]}"}"
        [[ "${ip_value}" == "${candidate}" ]] && return "${TRUE}"
    done
    return "${FALSE}"
}

########################################################################################################################################################################
# is_valid_uuid(): Validate UUID format (32 hex digits with optional hyphens)
function is_valid_uuid() {
    # example uuid="7632f5ab-4bac-11e6-bcb7-0cc47a6c4dbd"
    local uuid="$1"
    debug_log "Validating UUID format: $(redact_uuid "${uuid}")"
    if [[ ${uuid//-/} =~ ^[[:xdigit:]]{32}$ ]]; then
        debug_log "UUID is valid: $(redact_uuid "${uuid}")"
        return "${TRUE}"
    else
        debug_log "UUID is invalid"
        return "${FALSE}"
    fi
}

########################################################################################################################################################################
# Generate UUID with fallbacks for environments without uuidgen
function generate_uuid() {
    local generated_uuid=""
    if [[ ( "${generated_uuid}" == "" ) && ( -x "$(command -v uuidgen)" ) ]]; then
        generated_uuid="$(uuidgen 2>/dev/null)"
        if ! is_valid_uuid "${generated_uuid}"; then
            generated_uuid=""
        fi
    fi
    if [[ ( "${generated_uuid}" == "" ) && ( -r "/proc/sys/kernel/random/uuid" ) ]]; then
        generated_uuid="$(cat "/proc/sys/kernel/random/uuid" 2>/dev/null)"
        if ! is_valid_uuid "${generated_uuid}"; then
            generated_uuid=""
        fi
    fi
    if [[ ( "${generated_uuid}" == "" ) && ( -x "$(command -v python3)" ) ]]; then
        generated_uuid="$(python3 -c 'import uuid; print(uuid.uuid4())' 2>/dev/null)"
        if ! is_valid_uuid "${generated_uuid}"; then
            generated_uuid=""
        fi
    fi
    debug_log "Freshly Generated UUID: ${generated_uuid}"
    if [[ "${generated_uuid}" == "" ]]; then
        debug_log "Error: No valid UUID generated"
        return "${FALSE}"
    elif ! is_valid_uuid "${generated_uuid}"; then
        debug_log "Error: Generated UUID is not a valid UUID"
        return "${FALSE}"
    fi
    debug_log "Validated the Generated UUID: ${generated_uuid}"
    echo "${generated_uuid}"
    return "${TRUE}"
}

########################################################################################################################################################################
# is_valid_file(): Check if a file is valid
function is_valid_file() {
    debug_log "Checking if file is valid"
    local current_file="${1:-""}"
    debug_log "Current File: \"${current_file}\""
    local result_value="${TRUE}"

    if [[ ! -f "${current_file}" ]]; then
        debug_log "File is not valid: \"${current_file}\""
        result_value="${FALSE}"
    else
        debug_log "File is valid: \"${current_file}\""
    fi
    debug_log "Result Value: \"${result_value}\""

    return "${result_value}"
}

########################################################################################################################################################################
# is_valid_dir(): Check if a directory is valid
function is_valid_dir() {
    debug_log "Checking if directory is valid"
    local current_dir="${1:-""}"
    debug_log "Current Directory: \"${current_dir}\""
    local result_value="${TRUE}"

    if [[ ! -d "${current_dir}" ]]; then
        debug_log "Directory is not valid: \"${current_dir}\""
        result_value="${FALSE}"
    else
        debug_log "Directory is valid: \"${current_dir}\""
    fi
    debug_log "Result Value: \"${result_value}\""

    return "${result_value}"
}

########################################################################################################################################################################
# get_valid_session_id_file(): Get a valid session id file path to save current session id
function get_valid_session_id_file() {
    local session_id="$1" && shift
    debug_log "Session ID: ${session_id}"
    local safe_filename
    safe_filename="$(basename "${session_id}").txt"
    debug_log "Safe Filename: ${safe_filename}"
    local target_path="${SESSIONS_DIR}/${safe_filename}"
    debug_log "Target Path: ${target_path}"
    if [[ -L "${target_path}" ]]; then
        debug_log "Error: target file is a symlink — refusing to write"
        return "${UNSAFE_TARGET}"
    fi
    debug_log "Validated Target Path: ${target_path}"
    echo "${target_path}"
    return "${TRUE}"
}

########################################################################################################################################################################
# save_session_id(): Extract session ID from value string and persist to file
function save_session_id() {
    local session_id_value="$1"
    local session_id="$2"
    debug_log "Extracting Session ID and saving to file"
    debug_log "Session ID Value: \"${session_id_value}\""
    debug_log "Session ID: \"${session_id}\""
    if [[ "${session_id}" == "" ]]; then
        session_id="$(echo "${session_id_value}" | awk -F " " '{print $2;}')"
    fi
    debug_log "Parsed Session ID: \"${session_id}\""
    is_valid_uuid "${session_id}"
    RESULT="$?"
    if [[ "${RESULT}" != "${TRUE}" ]]; then
        debug_log "Error: Session ID is not a valid UUID — refusing to write file"
        return "${FALSE}"
    fi
    debug_log "Validated Parsed Session ID UUID: \"${session_id}\""
    local target_path=""
    local target_status="${TRUE}"
    # The file is named for the session id itself. Assign apart from the declaration: "local x=$(...)" reports the
    # status of "local", not of the function, and that status is how a symlink target is refused (UNSAFE_TARGET).
    target_path="$(get_valid_session_id_file "${session_id}")"
    target_status="$?"
    if [[ "${target_status}" != "${TRUE}" ]]; then
        debug_log "Error: Failed to get a valid session id file path."
        return "${target_status}"
    fi
    debug_log "Target Path: \"${target_path}\""
    if ! echo "${session_id}" > "${target_path}"; then
        debug_log "Error: Failed to write Session ID to file: \"${target_path}\""
        return "${FALSE}"
    fi
    debug_log "Saved Session ID $(redact_uuid "${session_id}") to file: \"${target_path}\""
    SESSION_ID_FILE="${target_path}"
    return "${TRUE}"
}

########################################################################################################################################################################
# retrieve_session_id(): Read session ID from a file in SESSIONS_DIR
function retrieve_session_id() {
    local session_id_file="$1"
    debug_log "Retrieving Session ID from file: ${session_id_file}"
    local session_id
    session_id="$(cat "${session_id_file}")"
    debug_log "Completed retrieving Session ID from file: \"${session_id_file}\""
    echo "${session_id}"
    return "${TRUE}"
}

########################################################################################################################################################################
# validate_session_id(): Validate session ID as UUID or resolve from session file
function validate_session_id() {
    local session_id="$1"
    local session_id_filename="${session_id}"
    local session_id_file="${SESSIONS_DIR}/${session_id_filename}"

    # 1st Pass: Check if the session id is empty or invalid
    debug_log "Validating Session ID, 1st Pass: \"$(redact_uuid "${session_id}")\""
    if [[ "${session_id}" == "" ]]; then
        debug_log "Session ID validation, 1st Pass: Failed for empty session id"
        return "${FALSE}"
    elif is_valid_uuid "${session_id}"; then
        debug_log "Session ID validation, 1st Pass: Input is a valid UUID"
        echo "${session_id}"
        return "${TRUE}"
    elif [[ "${session_id_filename}" == */* ]]; then
        debug_log "Session ID validation, 1st Pass: Failed for session id filename containing path separators"
        return "${FALSE}"
    elif [[ "${session_id_filename}" != *.txt ]]; then
        debug_log "Session ID validation, 1st Pass: Failed for session id filename not having .txt extension"
        return "${FALSE}"
    else
        debug_log "Session ID validation, 1st Pass: Succeeded for \"$(redact_uuid "${session_id}")\""
    fi

    # 2nd Pass: Step 1: Check if the session id is a file and retrieve the session id from the file
    if [[ -f "${session_id_file}" ]]; then
        debug_log "Session ID is a file: \"${session_id_file}\""
        session_id="$(retrieve_session_id "${session_id_file}")"
        debug_log "Completed retrieving Session ID from file: \"$(redact_uuid "${session_id}")\""
    else
        debug_log "Verified Session ID is not a file, Checking for valid UUID: \"$(redact_uuid "${session_id}")\""
    fi

    # 2nd Pass: Step 2: Check if the session id is a valid uuid
    if is_valid_uuid "${session_id}"; then
        debug_log "Session ID is valid: \"$(redact_uuid "${session_id}")\""
    else
        debug_log "Session ID file did not contain a valid UUID"
        return "${FALSE}"
    fi

    # Return the validated session id
    debug_log "Session ID validation, 2nd Pass: Succeeded for \"$(redact_uuid "${session_id}")\""
    echo "${session_id}"
    return "${TRUE}"
}

########################################################################################################################################################################
# Validate Model parameter value
function validate_model() {
    debug_log "Validating Model parameter value"
    local model_param="${1,,}"
    debug_log "Model parameter value: ${model_param}"
    # Start invalid: only a match against the allowlist below validates the model.
    local valid_model="${FALSE}"
    debug_log "1. Valid Model parameter value: ${valid_model}"
    # Local, so the loop cannot overwrite the caller's variable: bash scoping is dynamic, and parse_add_model_flags()
    # keeps the model it is validating in its own local named current_model.
    local current_model=""

    debug_log "Validate Claude Code Model Param Value: ${model_param}"
    for current_model in "${MODEL_TEST_ARRAY[@]}"; do
        if [[ "${model_param}" == "${current_model,,}" ]]; then
            debug_log "Model Validated: ${model_param}"
            valid_model="${TRUE}"
            break
        fi
    done
    debug_log "2. Valid Model parameter value: ${valid_model}"
    if [[ "${valid_model}" != "${TRUE}" ]]; then
        debug_log "Error: Specified Model is not Valid: ${model_param}"
    else
        debug_log "Successfully Validated Claude Code Model Param Value: ${model_param}"
    fi
    debug_log "3. Valid Model parameter value: ${valid_model}"
    return "${valid_model}"
}

########################################################################################################################################################################
# Validate Effort parameter value
function validate_effort() {
    local EFFORT_PARAM="${1,,}"
    local VALID_EFFORT="${FALSE}"
    debug_log "Validate Claude Code Effort Param Value: ${EFFORT_PARAM}"
    for CURRENT_EFFORT in "${EFFORT_ARRAY[@]}"; do
        if [[ "${EFFORT_PARAM}" == "${CURRENT_EFFORT,,}" ]]; then
            debug_log "Effort Validated: ${EFFORT_PARAM}"
            VALID_EFFORT="${TRUE}"
            break
        fi
    done
    if [[ "${VALID_EFFORT}" != "${TRUE}" ]]; then
        # The caller reports the failure, and the parse loop prints usage once and exits
        debug_log "Error: Specified Effort is not Valid: ${EFFORT_PARAM}"
    fi
    debug_log "Validated Claude Code Effort Param Value: ${EFFORT_PARAM}"
    return "${VALID_EFFORT}"
}


########################################################################################################################################################################
# Define functions to parse and process each input parameter flag type
########################################################################################################################################################################

########################################################################################################################################################################
# Add session id flags
function parse_session_id_flags() {
    debug_log "Parsing session id flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local session_id="${2:-""}"
    debug_log "Session ID: \"$(redact_uuid "${session_id}")\""
    local validated_session_id=""
    local status_value="${TRUE}"
    local result_value="${TRUE}"

    # Scenario 1: No Session ID Flag provided
    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Session ID Flag."
        result_value="${UNSAFE_TARGET}"
        return "${result_value}"
    fi
    shift  # shift to consume the session id flag

    # Scenario 2: a previous session is being resumed, or the session id was switched off with a boolean false value.
    #     Claude Code gets no session id, and the ignored value, when there is one, is consumed so it is not parsed as a flag.
    if (( ${#CLAUDE_RESUME_VALUE[@]} > 0 )) || [[ "${session_id}" == "${FALSE}" ]]; then
        debug_log "Warning: Ignoring Session ID Value Param since ID Flagged as False or Resume Previous Thread has been specified."
        CLAUDE_SESSION_ID_VALUE=()
        if [[ ( "${session_id}" != "" ) && ( "${session_id:0:1}" != "-" ) ]]; then
            shift  # shift to consume the ignored session id value
        fi
    else
        # Scenario 3: a boolean true value asks for a newly generated session id
        if [[ "${session_id}" == "${TRUE}" ]]; then
            debug_log "Warning: Received True Session ID Value. A new Session ID will be generated."
            shift  # shift to consume the session id boolean true value
            session_id=""
        fi

        # Scenario 4: a valid session id -- a UUID, or the name of a .txt file in the sessions directory that holds one
        validated_session_id="$(validate_session_id "${session_id}")"
        status_value="$?"
        if [[ ( "${status_value}" == "${TRUE}" ) && ( "${validated_session_id}" != "" ) ]]; then
            debug_log "Session ID validated: $(redact_uuid "${validated_session_id}")"
            shift  # shift to consume the valid session id value
            session_id="${validated_session_id}"
        # Scenario 5: no usable session id (no value, a flag follows, or a boolean true value): generate a new one.
        #     A value that is present but invalid is left in place, so it is parsed -- and rejected -- as the next flag.
        else
            debug_log "Warning: Received Session ID Flag but no Valid Session ID Name."
            debug_log "Session ID Value not Provided, Assigning a new UUID as Session ID."
            session_id="$(generate_uuid)"
            status_value="$?"
            if [[ ( "${status_value}" != "${TRUE}" ) || ( "${session_id}" == "" ) ]]; then
                debug_log "Error: Failed to generate a valid UUID for Session ID."
                result_value="${FALSE}"
            fi
        fi

        # Save the session id to its file: a symlink target is always refused, any other failure only when a save is required
        if [[ "${result_value}" == "${TRUE}" ]]; then
            CLAUDE_SESSION_ID_VALUE=("${CLAUDE_SESSION_ID_FLAGS}" "${session_id}")
            debug_log "Saving Session ID to File: $(redact_uuid "${session_id}")"
            # Called directly, not inside $(...): save_session_id() records SESSION_ID_FILE, which a subshell would discard
            save_session_id "${CLAUDE_SESSION_ID_VALUE[@]}"
            status_value="$?"
            if [[ "${status_value}" == "${UNSAFE_TARGET}" ]]; then
                debug_log "Error: Refusing to save Session ID to an unsafe (symlink) target. Exiting..."
                result_value="${FALSE}"
            elif [[ ( "${status_value}" != "${TRUE}" ) && ( "${REQUIRE_SAVED_SESSION_ID}" == "${TRUE}" ) ]]; then
                debug_log "Error: Failed to Save Session ID value. Exiting..."
                result_value="${FALSE}"
            elif [[ "${status_value}" != "${TRUE}" ]]; then
                debug_log "Warning: Failed to Save Session ID value."
            else
                debug_log "Session ID saved: $(redact_uuid "${session_id}") to file: \"${SESSION_ID_FILE}\""
            fi
        fi
    fi
    debug_log "Claude Session ID Value: [${CLAUDE_SESSION_ID_VALUE[0]:-""} $(redact_uuid "${CLAUDE_SESSION_ID_VALUE[1]:-""}")]"
    debug_log "Result Value: \"${result_value}\""

    # Update the current params array
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: $(redact_uuid "${CURRENT_PARAMS[@]}")"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: $(redact_uuid "${UPDATED_PARAMS[@]}")"

    return "${result_value}"
}

########################################################################################################################################################################
# parse_resume_flags(): Parse resume flags
function parse_resume_flags() {
    debug_log "Parsing resume flags"
    local resume_flag="${1:-""}"
    debug_log "Resume Flag: \"${resume_flag}\""
    local resume_value="${2:-""}"
    debug_log "Resume Value: \"${resume_value}\""
    local session_id=""
    local status_value="${TRUE}"
    local result_value="${TRUE}"

    # Scenario 1: this scenario means: we have received an empty resume flag
    if [[ "${resume_flag}" == "" ]]; then
        debug_log "Error: Received Empty Resume Flag. Exiting..."
        result_value="${FALSE}"
    # Scenario 2: this scenario means: we have received a false boolean resume value, so a previous session is NOT resumed
    elif [[ "${resume_value}" == "${FALSE}" ]]; then
        debug_log "Warning: Received False Resume Value."
        shift 2  # shift to consume the resume flag and resume boolean false value
        CLAUDE_RESUME_VALUE=()
    # Scenario 3: this scenario means: we have received a true boolean resume value: resume the session named by an earlier
    #     session id flag or, when there is none, let Claude Code offer its own list of sessions to resume
    elif [[ "${resume_value}" == "${TRUE}" ]]; then
        debug_log "Warning: Received Resume Flag with a true boolean value instead of a Resume Session ID."
        shift 2  # shift to consume the resume flag and resume boolean true value
        if (( ${#CLAUDE_SESSION_ID_VALUE[@]} > PARAM_VALUE_INDEX )); then
            session_id="${CLAUDE_SESSION_ID_VALUE[PARAM_VALUE_INDEX]}"
            debug_log "Resuming the Session ID already assigned: $(redact_uuid "${session_id}")"
            CLAUDE_RESUME_VALUE=("${CLAUDE_RESUME_FLAGS}" "${session_id}")
        else
            debug_log "Warning: No Session ID has been assigned. Letting Claude Code offer a session to resume."
            CLAUDE_RESUME_VALUE=("${CLAUDE_RESUME_FLAGS}")
        fi
        # Resuming replaces starting a new session, so claude code must not also receive a session id
        CLAUDE_SESSION_ID_VALUE=()
    # Scenario 4: this scenario means: no session id follows the resume flag -- the input ends, or the next input is a flag
    elif [[ ( "${resume_value}" == "" ) || ( "${resume_value:0:1}" == "-" ) ]]; then
        debug_log "Error: Received Resume Flag but no Valid Session ID to Resume. Exiting..."
        shift  # shift to consume the resume flag
        result_value="${FALSE}"
    # Scenario 5: this scenario means: we have received a resume value -- a UUID, or the name of a .txt file in the sessions directory that holds one
    else
        shift 2  # shift to consume the resume flag and resume value
        session_id="$(validate_session_id "${resume_value}")"
        status_value="$?"
        if [[ ( "${status_value}" == "${TRUE}" ) && ( "${session_id}" != "" ) ]]; then
            debug_log "Session ID validated: $(redact_uuid "${session_id}")"
            # Claude Code gets the resolved UUID, never the name of the file it was read from
            CLAUDE_RESUME_VALUE=("${CLAUDE_RESUME_FLAGS}" "${session_id}")
            CLAUDE_SESSION_ID_VALUE=()
        else
            debug_log "Error: Session ID is invalid. Exiting..."
            result_value="${FALSE}"
        fi
    fi
    debug_log "Claude Resume Value: [${CLAUDE_RESUME_VALUE[0]:-""} $(redact_uuid "${CLAUDE_RESUME_VALUE[1]:-""}")]"
    debug_log "Result Value: \"${result_value}\""

    # Update current params array
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################
# Add Model Flag
function parse_add_model_flags() {
    debug_log "Parsing add model flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local current_model="${2:-""}"
    debug_log "Current Model: \"${current_model}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Model Flag."
        CLAUDE_MODEL_VALUE=()
        result_value="${UNSAFE_TARGET}"
    elif [[ "${current_model}" == "${FALSE}" ]]; then
        debug_log "Warning: Received False Boolean as Model Name."
        debug_log "Model not included in claude code parameters array"
        CLAUDE_MODEL_VALUE=() && shift 2  # shift to consume the model flag and model boolean false value
    elif [[ "${current_model}" == "${TRUE}" ]]; then
        debug_log "Warning: Received True Boolean as Model Name. Setting Default Model Name: \"${MODEL_DEFAULT}\""
        CLAUDE_MODEL_VALUE=("${CLAUDE_MODEL_FLAGS}" "${MODEL_DEFAULT}") && shift 2  # shift to consume the model flag and model boolean true value
    # No model name follows the model flag: the input ends, or the next input is a flag
    elif [[ ( "${current_model}" == "" ) || ( "${current_model:0:1}" == "-" ) ]]; then
        debug_log "Error: Received Model Flag but no Model Name. Exiting..."
        CLAUDE_MODEL_VALUE=() && shift  # shift to consume the model flag
        result_value="${FALSE}"
    elif validate_model "${current_model}"; then
        debug_log "Selected Model Name validated: \"${current_model}\""
        CLAUDE_MODEL_VALUE=("${CLAUDE_MODEL_FLAGS}" "${current_model}") && shift 2 # shift to consume the model flag and model name
    else
        debug_log "Error: Received an Invalid Model Name: \"${current_model}\". Exiting..."
        CLAUDE_MODEL_VALUE=() && shift 2 # shift to consume the model flag and invalid model name
        result_value="${FALSE}"
    fi
    debug_log "Claude Model Value: \"${CLAUDE_MODEL_VALUE[*]}\""
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Add Effort Flag
function parse_add_effort_flags() {
    debug_log "Parsing effort flags"
    local current_flag=""
    current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local current_effort=""
    current_effort="${2:-""}"
    debug_log "Current Effort: \"${current_effort}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Effort Flag. Exiting..."
        result_value="${UNSAFE_TARGET}"
        CLAUDE_EFFORT_VALUE=()
    # Match every effort alias (-e, --effort, --effort-type, --effort-level), not only Claude Code's own --effort
    elif matches_pattern "${current_flag}" "${EFFORT_FLAGS}"; then
        debug_log "Received a Valid Effort Flag"
        shift  # shift to consume the effort flag

        if [[ "${current_effort}" == "${FALSE}" ]]; then
            debug_log "Warning: Received False Effort Value."
            CLAUDE_EFFORT_VALUE=() && shift  # shift to consume the effort boolean false value
        elif [[ "${current_effort}" == "${TRUE}" ]]; then
            debug_log "Warning: Received True Boolean as Effort Value. Setting Default Effort: \"${EFFORT_DEFAULT}\""
            CLAUDE_EFFORT_VALUE=("${CLAUDE_EFFORT_FLAGS}" "${EFFORT_DEFAULT}") && shift  # shift to consume the effort boolean true value
        # No effort value follows the effort flag: the input ends, or the next input is a flag
        elif [[ ( "${current_effort}" == "" ) || ( "${current_effort:0:1}" == "-" ) ]]; then
            debug_log "Error: Received Effort Flag but no valid Effort Value. Exiting..."
            CLAUDE_EFFORT_VALUE=()
            result_value="${FALSE}"
        elif validate_effort "${current_effort}"; then
            debug_log "Current Effort Name validated: \"${current_effort}\""
            CLAUDE_EFFORT_VALUE=("${CLAUDE_EFFORT_FLAGS}" "${current_effort}") && shift  # shift to consume the effort name
        else
            debug_log "Error: Received an Invalid Effort Value: \"${current_effort}\". Exiting..."
            CLAUDE_EFFORT_VALUE=() && shift  # shift to consume the invalid effort value
            result_value="${FALSE}"
        fi
        debug_log "Claude Effort Value: [${CLAUDE_EFFORT_VALUE[*]}]"
        debug_log "Result Value: \"${result_value}\""
    else
        debug_log "Warning: Received a non-empty but Invalid Effort flag: \"${current_flag}\"."
        CLAUDE_EFFORT_VALUE=() && shift  # shift to consume the invalid effort flag
        result_value="${FALSE}"
    fi
    debug_log "Effort Value: \"${CLAUDE_EFFORT_VALUE[*]}\""
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Add Permissions Flag
function parse_add_permissions_flags() {
    debug_log "Parsing add permissions flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Permissions Flag."
        result_value="${UNSAFE_TARGET}"
        CLAUDE_PERMISSIONS_VALUE=()
    elif matches_pattern "${current_flag}" "${PERMISSIONS_FLAGS}"; then
        debug_log "Received Valid Permissions Flag: \"${current_flag}\""
        # The flag takes no value: whatever follows it is parsed as the next flag
        shift  # shift to consume the permissions flag
        CLAUDE_PERMISSIONS_VALUE=("${CLAUDE_PERMISSIONS_FLAGS}")
    else
        debug_log "Warning: Received an Invalid Permissions Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid permissions flag
        CLAUDE_PERMISSIONS_VALUE=()
    fi
    debug_log "Result Value: \"${result_value}\""
    debug_log "Permissions Value: \"${CLAUDE_PERMISSIONS_VALUE[*]}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################
# Add worktree flag
function parse_add_worktree_flags() {
    debug_log "Parsing add worktree flags"
    local worktree_flag="${1:-""}"
    debug_log "Worktree Flag: \"${worktree_flag}\""
    local worktree_value="${2:-""}"
    debug_log "Worktree Value: \"${worktree_value}\""
    local result_value="${TRUE}"

    if [[ "${worktree_flag}" == "" ]]; then
        debug_log "Error: Received Empty Worktree Flag. Exiting..."
        result_value="${UNSAFE_TARGET}"
        CLAUDE_WORKTREE_VALUE=()
    elif [[ "${worktree_value}" == "${FALSE}" ]]; then
        debug_log "Warning: Received False Worktree Value."
        shift 2 # shift to consume the worktree flag and worktree boolean false value
        CLAUDE_WORKTREE_VALUE=()
    # No worktree name: a true boolean value, the end of the input, or a flag next. Claude Code assigns the name.
    elif [[ ( "${worktree_value}" == "${TRUE}" ) || ( "${worktree_value}" == "" ) || ( "${worktree_value:0:1}" == "-" ) ]]; then
        debug_log "Warning: Received Worktree Flag without a Worktree Name. Letting Claude Code assign one."
        shift  # shift to consume the worktree flag
        if [[ "${worktree_value}" == "${TRUE}" ]]; then
            shift  # shift to consume the worktree boolean true value
        fi
        CLAUDE_WORKTREE_VALUE=("${CLAUDE_WORKTREE_FLAGS}")
    else
        debug_log "Selected Worktree Name: \"${worktree_value}\""
        CLAUDE_WORKTREE_VALUE=("${CLAUDE_WORKTREE_FLAGS}" "${worktree_value}") && shift 2  # shift to consume the worktree flag and worktree name
    fi
    debug_log "Result Value: \"${result_value}\""
    debug_log "Worktree Value: \"${CLAUDE_WORKTREE_VALUE[*]}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Add Remote Control flag: sent bare, never with a value (Overtoad ruling, 2026-11-04), so Claude Code names the session itself.
#     A 0 (true) or 1 (false) after the flag turns remote control on or off, as for the other flags here. Any other word after it is
#     refused rather than forwarded: Claude Code would read a word after --remote-control as the session's name.
function parse_add_remote_control_flags() {
    debug_log "Parsing add remote control flags"
    local remote_control_flag="${1:-""}"
    debug_log "Remote Control Flag: \"${remote_control_flag}\""
    local remote_control_value="${2:-""}"
    debug_log "Remote Control Value: \"${remote_control_value}\""
    local result_value="${TRUE}"

    if [[ "${remote_control_flag}" == "" ]]; then
        debug_log "Error: Received Empty Remote Control Flag."
        result_value="${UNSAFE_TARGET}"
        CLAUDE_REMOTE_CONTROL_VALUE=()
    elif ! matches_pattern "${remote_control_flag}" "${REMOTE_CONTROL_FLAGS}"; then
        debug_log "Error: Received an Invalid Remote Control Flag: \"${remote_control_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid remote control flag
        CLAUDE_REMOTE_CONTROL_VALUE=()
    elif [[ "${remote_control_value}" == "${FALSE}" ]]; then
        debug_log "Warning: Received False Remote Control Value."
        shift 2  # shift to consume the remote control flag and remote control boolean false value
        CLAUDE_REMOTE_CONTROL_VALUE=()
    # Bare: a true boolean value, the end of the input, or a flag next
    elif [[ ( "${remote_control_value}" == "${TRUE}" ) || ( "${remote_control_value}" == "" ) || ( "${remote_control_value:0:1}" == "-" ) ]]; then
        debug_log "Received Remote Control Flag. Claude Code names the session."
        shift  # shift to consume the remote control flag
        if [[ "${remote_control_value}" == "${TRUE}" ]]; then
            shift  # shift to consume the remote control boolean true value
        fi
        CLAUDE_REMOTE_CONTROL_VALUE=("${CLAUDE_REMOTE_CONTROL_FLAGS}")
    else
        debug_log "Error: Remote Control takes no value, but received: \"${remote_control_value}\"."
        result_value="${FALSE}" && shift  # shift to consume the remote control flag
        CLAUDE_REMOTE_CONTROL_VALUE=()
    fi
    debug_log "Result Value: \"${result_value}\""
    debug_log "Remote Control Value: \"${CLAUDE_REMOTE_CONTROL_VALUE[*]}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Add headless flag
function parse_add_headless_flags() {
    debug_log "Parsing add headless flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Headless Flag. Exiting..."
        result_value="${UNSAFE_TARGET}"
    elif matches_pattern "${current_flag}" "${HEADLESS_FLAGS}"; then
        debug_log "Received Headless Flag: \"${current_flag}\""
        shift  # shift to consume the headless flag
    else
        debug_log "Error: Received an Invalid Headless Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid headless flag
    fi
    debug_log "Headless Value: \"${current_flag}\""
    if [[ "${result_value}" == "${TRUE}" ]]; then
        debug_log "Headless parameter Flag Value: \"${current_flag}\""
        CLAUDE_HEADLESS_VALUE=("${CLAUDE_HEADLESS_FLAGS}")
    else
        debug_log "Warning: Received an Invalid Headless Flag: \"${current_flag}\". Setting Headless parameter Flag Value to empty array."
        CLAUDE_HEADLESS_VALUE=()
    fi
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Add fork session flag
function parse_add_fork_session_flags() {
    debug_log "Parsing add fork session flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Fork Session Flag."
        result_value="${UNSAFE_TARGET}"
    elif matches_pattern "${current_flag}" "${FORK_SESSION_FLAGS}"; then
        debug_log "Received Fork Session Flag: \"${current_flag}\""
        shift  # shift to consume the fork session flag
    else
        debug_log "Error: Received an Invalid Fork Session Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid fork session flag
    fi
    debug_log "Result Value: \"${result_value}\""
    if [[ "${result_value}" == "${TRUE}" ]]; then
        debug_log "Fork Session parameter Flag Value: \"${current_flag}\""
        CLAUDE_FORK_SESSION_VALUE=("${CLAUDE_FORK_SESSION_FLAGS}")
    else
        debug_log "Warning: Received an Invalid Fork Session Flag: \"${current_flag}\". Setting Fork Session parameter Flag Value to empty array."
        CLAUDE_FORK_SESSION_VALUE=()
    fi
    debug_log "Fork Session Value: \"${CLAUDE_FORK_SESSION_VALUE[*]}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################################################################################
# Skip input param Spacer flag
function parse_spacer_flags() {
    debug_log "Received Spacer Flag"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Spacer Flag."
        result_value="${UNSAFE_TARGET}"
    elif matches_pattern "${current_flag}" "${SPACER_FLAGS}"; then
        debug_log "Received Spacer Flag: \"${current_flag}\""
        shift  # shift to consume the spacer flag
    else
        debug_log "Error: Received an Invalid Spacer Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid spacer flag
    fi
    debug_log "Result Value: \"${result_value}\""
    # Note: This flag is not being added to the updated Claude Code params because it is not a revelvant flag for claude code

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################
# Parse Command Line Provided Prompt
# NOTE: the "--prompt" flag is not added to updated Claude Code params because it's not a valid flag for claude code
function parse_prompt_value_flags() {
    debug_log "Parsing prompt value flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local current_prompt="${2:-""}"
    debug_log "Current Prompt: [${#current_prompt} chars]"
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Prompt Value Flag."
        result_value="${UNSAFE_TARGET}"
    elif matches_pattern "${current_flag}" "${PROMPT_FLAGS}"; then
        debug_log "Received Prompt Value Flag: \"${current_flag}\""
        shift  # shift to consume the prompt value flag

        # The word after a prompt flag is the prompt, whatever it starts with. It is consumed as opaque text and sent after "--",
        # so a prompt such as "--dangerously-skip-permissions" or "--print" is never read as a flag, here or by Claude Code. It
        # once was here: a value starting with "--" was left in the input to be parsed as a flag, and the default prompt sent.
        if (( $# == 0 )); then
            debug_log "Warning: Received a Prompt Flag at the end of the input. Setting Default Prompt Value: \"${PROMPT_VALUE_DEFAULT}\""
            PROMPT_VALUE="${PROMPT_VALUE_DEFAULT}" && VALID_PROMPT_PARAM="${TRUE}"
        elif [[ "${current_prompt}" == "${FALSE}" ]]; then
            debug_log "Warning: Received False Prompt Value. No prompt string will be sent."
            PROMPT_VALUE="" && VALID_PROMPT_PARAM="${FALSE}" && shift  # shift to consume the prompt boolean false value
        elif [[ ( "${current_prompt}" == "${TRUE}" ) || ( "${current_prompt}" == "" ) ]]; then
            debug_log "Warning: Received an Edge Case Prompt Value: \"${current_prompt}\". Setting Default Prompt Value: \"${PROMPT_VALUE_DEFAULT}\""
            PROMPT_VALUE="${PROMPT_VALUE_DEFAULT}" && VALID_PROMPT_PARAM="${TRUE}" && shift  # shift to consume the boolean true value or the empty prompt
        else
            debug_log "Received an apparent valid Prompt Value [${#current_prompt} chars]."
            PROMPT_VALUE="${current_prompt}" && VALID_PROMPT_PARAM="${TRUE}" && shift  # shift to consume the prompt value
        fi
    else
        debug_log "Error: Received an Invalid Prompt Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid prompt flag
    fi
    debug_log "Prompt Value: [${#PROMPT_VALUE} chars]"
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################
# Parse Prompt File Path Flag
#     The prompt file's contents are read once parsing is done, by load_prompt_file()
function parse_prompt_path_flags() {
    debug_log "Parsing prompt path flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local current_path="${2:-""}"
    debug_log "Current Path: \"${current_path}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Prompt Path Flag."
        result_value="${UNSAFE_TARGET}"
    elif ! matches_pattern "${current_flag}" "${PATH_FLAGS}"; then
        debug_log "Error: Received an Invalid Prompt Path Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid prompt path flag
    # No path follows the path flag: the input ends, or the next input is a flag
    elif [[ ( "${current_path}" == "" ) || ( "${current_path:0:1}" == "-" ) ]]; then
        debug_log "Error: Received Path Flag but no Path Value. Exiting..."
        result_value="${FALSE}" && shift  # shift to consume the prompt path flag
    else
        shift 2  # shift to consume the prompt path flag and prompt path value
        PATH_NAME="${current_path}"
        if [[ "${PATH_NAME: -1}" == "/" ]]; then
            PATH_NAME="${PATH_NAME%/}"
            debug_log "Removed trailing slash from path"
        fi
        # The path names the prompt file itself
        if is_valid_file "${PATH_NAME}"; then
            debug_log "Provided Pathname is a valid file"
            VALID_PATH_PARAM="${TRUE}"
            PROMPT_FILE="${PATH_NAME}"
        # The path names the directory holding the prompt file that the file flag names -- which may not be parsed yet
        elif is_valid_dir "${PATH_NAME}"; then
            debug_log "Provided Pathname is a valid directory"
            if [[ ( "${FILE_NAME}" != "" ) && ( -f "${PATH_NAME}/${FILE_NAME}" ) ]]; then
                debug_log "Combined Pathname and Filename is a valid file"
                VALID_PATH_PARAM="${TRUE}"
                PROMPT_FILE="${PATH_NAME}/${FILE_NAME}"
            elif [[ "${FILE_NAME}" == "" ]]; then
                debug_log "Filename has not yet been parsed"
                # A directory with no file name is not a prompt. Drop any prompt an earlier --path resolved:
                # --path good.md --path emptydir otherwise sent good.md and exited 0.
                PROMPT_FILE=""
                VALID_FILE_PARAM="${FALSE}"
                VALID_PATH_PARAM="${FALSE}"
            else
                debug_log "Error: Prompt file is invalid when Pathname is combined with Filename. Exiting..."
                result_value="${FALSE}"
            fi
        else
            debug_log "Error: received an invalid Prompt File Path. Exiting..."
            result_value="${FALSE}"
        fi
    fi
    debug_log "Prompt File: \"${PROMPT_FILE}\""
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}

########################################################################################################################################################################
# Parse prompt file flag
#     The prompt file's contents are read once parsing is done, by load_prompt_file()
function parse_prompt_file_flags() {
    debug_log "Parsing prompt file flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local current_file="${2:-""}"
    debug_log "Current File: \"${current_file}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Prompt File Flag."
        result_value="${UNSAFE_TARGET}"
    elif ! matches_pattern "${current_flag}" "${FILE_FLAGS}"; then
        debug_log "Error: Received an Invalid Prompt File Flag: \"${current_flag}\"."
        result_value="${FALSE}" && shift  # shift to consume the invalid prompt file flag
    # No file name follows the file flag: the input ends, or the next input is a flag
    elif [[ ( "${current_file}" == "" ) || ( "${current_file:0:1}" == "-" ) ]]; then
        debug_log "Error: Received File Flag but no File Value. Exiting..."
        result_value="${FALSE}" && shift  # shift to consume the prompt file flag
    else
        shift 2  # shift to consume the prompt file flag and prompt file name
        FILE_NAME="${current_file}"
        # The file name is a path to the prompt file on its own
        if is_valid_file "${FILE_NAME}"; then
            debug_log "Provided Filename is a valid file"
            VALID_FILE_PARAM="${TRUE}"
            PROMPT_FILE="${FILE_NAME}"
        # The file name is relative to the directory that the path flag named
        elif [[ ( "${PATH_NAME}" != "" ) && ( -f "${PATH_NAME}/${FILE_NAME}" ) ]]; then
            debug_log "Combined Pathname and Filename is a valid file"
            VALID_FILE_PARAM="${TRUE}"
            PROMPT_FILE="${PATH_NAME}/${FILE_NAME}"
        # The path flag may follow: resolution waits for it (and fails after parsing if it never arrives)
        elif [[ "${PATH_NAME}" == "" ]]; then
            debug_log "Filename not yet valid, Pathname has not yet been parsed"
            # This name did not resolve, and no directory is waiting to join it. Drop any prompt an earlier
            # --file or --path resolved: --file good.md --file missing.md otherwise sent good.md and exited 0.
            PROMPT_FILE=""
            VALID_FILE_PARAM="${FALSE}"
            VALID_PATH_PARAM="${FALSE}"
        else
            debug_log "Error: received an invalid Prompt File. Exiting..."
            result_value="${FALSE}"
        fi
    fi
    debug_log "Prompt File: \"${PROMPT_FILE}\""
    debug_log "Result Value: \"${result_value}\""

    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    return "${result_value}"
}


########################################################################################################################################################################
# Parse active Claude Code version flags
function parse_model_version_flags() {
    debug_log "Parsing Model version flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    debug_log "Searching for claude command in PATH"
    local version_bin=""
    version_bin="$(type -P claude 2>/dev/null || true)"
    debug_log "Version Bin: \"${version_bin}\""
    local result_value="${TRUE}"
    local current_version="0"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Model Version Flag. Exiting..."
        result_value="${UNSAFE_TARGET}"
    # Every model-version alias: -V and --claude-version were once matched against --version alone, and exited 1
    elif matches_pattern "${current_flag}" "${MODEL_VERSION_FLAGS}"; then
        debug_log "Received Model Version Flag: \"${current_flag}\""
        shift
        # The errors go to stderr as well: with debug output off (WTC_DEBUG=1) they would leave a bare exit 1
        if [[ ( "${version_bin}" == "" ) || ( ! -x "${version_bin}" ) ]]; then
            debug_log "Error: claude command not found in PATH"
            echo "Error: claude command not found in PATH" >&2
            result_value="${FALSE}"
        elif [[ -x "${version_bin}" ]]; then
            current_version="$("${version_bin}" ${CLAUDE_VERSION_FLAGS})"
            debug_log "Current Version: \"${current_version}\""
            if [[ "${current_version}" == "" ]]; then
                debug_log "Error: Unable to retrieve Model Version. Exiting..."
                echo "Error: \"${version_bin} ${CLAUDE_VERSION_FLAGS}\" printed no version" >&2
                result_value="${FALSE}"
            else
                debug_log "Model Version: \"${version_bin} ${CLAUDE_VERSION_FLAGS}\": \"${current_version}\""
                echo -ne "\n${version_bin} ${current_version}\n\n"
            fi
        else
            debug_log "Error: claude command found in PATH but is not executable"
            result_value="${FALSE}"
        fi
    else
        debug_log "Warning: Received a non-empty but Invalid Model Version Flag: \"${current_flag}\"."
        shift  # shift to consume the invalid model version flag
        result_value="${FALSE}"
    fi
    debug_log "Result Value: \"${result_value}\""

    # Update current params array
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    set -e
    exit "${result_value}"
}

########################################################################################################################################################################
# Parse this script's version flags
function parse_script_version_flags() {
    debug_log "Parsing Script version flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local version_value="${SCRIPT_VERSION}"
    debug_log "Script Version: \"${version_value}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Script Version Flag."
        result_value="${UNSAFE_TARGET}"
    elif [[ "${version_value}" == "" ]]; then
        debug_log "Error: Unable to retrieve Script Version."
        shift  # shift to consume the script version flag
        result_value="${FALSE}"
    elif matches_pattern "${current_flag}" "${SCRIPT_VERSION_FLAGS}"; then
        debug_log "Received Script Version Flag: \"${current_flag}\""
        shift  # shift to consume the script version flag
    else
        debug_log "Error: Received an Invalid Script Version Flag: \"${current_flag}\"."
        shift  # shift to consume the invalid script version flag
        result_value="${FALSE}"
    fi
    debug_log "Result Value: \"${result_value}\""
    if [[ "${result_value}" == "${TRUE}" ]]; then
        debug_log "Displaying Script Version"
        echo -ne "\n${SCRIPT_NAME} ${version_value}\n\n"
        debug_log "${SCRIPT_NAME} ${version_value}"
    fi
    debug_log "Result Value: \"${result_value}\""

    # Update current params array
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    set -e
    exit "${result_value}"
}

########################################################################################################################################################################################################################################
# Print script usage and exit with fail (1)
function parse_usage_flags() {
    debug_log "Received Usage Flag"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Usage Flag."
        result_value="${UNSAFE_TARGET}"
    elif matches_pattern "${current_flag}" "${USAGE_FLAGS}"; then
        debug_log "Received Usage Flag: \"${current_flag}\""
        shift
    else
        debug_log "Error: Received an Invalid Usage Flag: \"${current_flag}\"."
        result_value="${FALSE}"
        shift
    fi
    debug_log "Result Value: \"${result_value}\""

    # Doing this here to ensure that the current params are updated in case the usage function fails and the function returns a non-zero value
    # Update current params array
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    # -u asks for the usage text alone, so it exits with a failing status (Overtoad ruling, 2026-11-04); -h exits 0
    usage "${FALSE}"
}

########################################################################################################################################################################################################################################
# Print script usage (help) and exit with success
function parse_help_flags() {
    debug_log "Parsing help flags"
    local current_flag="${1:-""}"
    debug_log "Current Flag: \"${current_flag}\""
    local result_value="${TRUE}"

    # Validate current flag
    if [[ "${current_flag}" == "" ]]; then
        debug_log "Error: Received Empty Help Flag."
        result_value="${UNSAFE_TARGET}"
    # Verify that the current flag is the help flag
    elif matches_pattern "${current_flag}" "${HELP_FLAGS}"; then
        debug_log "Received Help Flag: \"${current_flag}\""
        shift
    # If the current flag is not the help flag, return an error
    else
        debug_log "Error: Received an Invalid Help Flag: \"${current_flag}\"."
        result_value="${FALSE}"
        shift
    fi
    debug_log "Result Value: \"${result_value}\""

    # Doing this here to ensure that the current params are updated in case the usage function fails and the function returns a non-zero value
    debug_log "Updating Current Params"
    CURRENT_PARAMS=("${@}")
    debug_log "Current Params: size: ${#CURRENT_PARAMS[*]}, values: [${CURRENT_PARAMS[*]}]"
    UPDATED_PARAMS=("${CURRENT_PARAMS[@]}")
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: [${UPDATED_PARAMS[*]}]"

    usage "${result_value}" || return "${result_value}"
}


########################################################################################################################################################################################################################################
# Define general purpose functions
########################################################################################################################################################################################################################################

########################################################################################################################################################################################################################################
# load_prompt_file(): Read the prompt file that --file / --path named, and parsing validated, into CLAUDE_CODE_PROMPT
#     A file that cannot be read, or that holds no prompt, fails: claude code must not start without the prompt it was asked to send.
function load_prompt_file() {
    local prompt_file="${1:-""}"
    debug_log "Loading Prompt File: \"${prompt_file}\""
    local prompt_text=""

    if [[ ( "${prompt_file}" == "" ) || ( ! -f "${prompt_file}" ) || ( ! -r "${prompt_file}" ) ]]; then
        debug_log "Error: Prompt File is missing or unreadable: \"${prompt_file}\""
        return "${FALSE}"
    elif ! prompt_text="$(cat -- "${prompt_file}")"; then
        debug_log "Error: Failed to read Prompt File: \"${prompt_file}\""
        return "${FALSE}"
    elif [[ ! "${prompt_text}" =~ [^[:space:]] ]]; then
        debug_log "Error: Prompt File holds no prompt: \"${prompt_file}\""
        return "${FALSE}"
    fi
    CLAUDE_CODE_PROMPT="${prompt_text}"
    debug_log "Prompt File loaded [${#CLAUDE_CODE_PROMPT} chars]"
    return "${TRUE}"
}

########################################################################################################################################################################################################################################
# format_command_line(): Render a command line as it would be typed: a plain word as-is, anything else in single quotes
#     So the prompt shows as the one quoted argument claude receives. A single quote inside an argument is written '\''.
function format_command_line() {
    local plain_word='^[A-Za-z0-9_./:=@%+,-]+$'
    local formatted=""
    local arg=""
    local quoted=""
    for arg in "${@}"; do
        if [[ "${arg}" =~ ${plain_word} ]]; then
            quoted="${arg}"
        else
            quoted="'${arg//\'/\'\\\'\'}'"
        fi
        formatted+="${formatted:+ }${quoted}"
    done
    printf '%s' "${formatted}"
}

########################################################################################################################################################################################################################################
# get_pattern_index(): Get the index into the valid patterns array of the pattern that matches the current element
# TODO: Verify this function works properly with debug log statements.
function get_pattern_index() {
    debug_log "Parsing Pattern Index"
    local current_element="$1"
    debug_log "Current Element: \"${current_element}\""
    local pattern_index=-1
    local result_value="${TRUE}"
    debug_log "Valid Patterns: size: ${#VALID_PATTERNS[@]}, values: [${VALID_PATTERNS[*]}]"

    debug_log "Getting Pattern Index for Current Element: \"${current_element}\""
    for i in "${!VALID_PATTERNS[@]}"; do
        if matches_pattern "${current_element}" "${VALID_PATTERNS[$i]}"; then
            pattern_index=$i
            break
        fi
    done
    debug_log "Pattern Index: \"${pattern_index}\""
    echo "${pattern_index}"
    return "${result_value}"
}


########################################################################################################################################################################################################################################
# usage(): Display usage statement and exit
#     TODO: this needs to be updated to add the new flags and their descriptions
function usage() {
    INPUT_PARAM="$1"
    # SCRIPT_NAME="$(basename "$(realpath "${BASH_SOURCE[0]}")")"
    EXIT_AFTER="${EXIT_AFTER_USAGE_DEFAULT}"
    if [[ ( "${INPUT_PARAM}" == "${TRUE}") || ( "${INPUT_PARAM}" == "${FALSE}" ) ]]; then
        EXIT_AFTER="${INPUT_PARAM}"
    fi
    echo -ne "usage: ${SCRIPT_NAME}\n"
    echo -ne "\t ${USAGE_FLAGS}:\n"
    echo -ne "\t\tFlags that display this Usage statement, then exit with a failing status (1).\n"
    echo -ne "\t ${HELP_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the Display of this Help statement.\n"
    echo -ne "\t ${MODEL_VERSION_FLAGS}:\n"
    echo -ne "\t\tFlags that display the version of Claude Code (claude --version).\n"
    echo -ne "\t ${SCRIPT_VERSION_FLAGS}:\n"
    echo -ne "\t\tFlags that display the version of this script.\n"
    echo -ne "\t ${FILE_FLAGS}:\n"
    echo -ne "\t\tFlags to allow a filename to be specified for this script.\n"
    echo -ne "\t\tNOTE: Prompt string can be provided either as a string of text or as a file name or pathname to a file containing the prompt string.\n"
    echo -ne "\t ${PATH_FLAGS}:\n"
    echo -ne "\t\tFlags that allow a path to be specified.  Path can be an absolute or relative path to a file, or path can be combined with a received filename.\n"
    echo -ne "\t\tNOTE: Prompt string can be provided either as a string of text or as a file name or pathname to a file containing the prompt string.\n"
    echo -ne "\t ${EFFORT_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the effort used by the Claude Code model to be specified.\n"
    echo -ne "\t\tNOTE: Effort value must be one of the following: ${EFFORT_LOW}, ${EFFORT_MED}, ${EFFORT_HIGH}, ${EFFORT_XHIGH}, ${EFFORT_MAX}, or ${EFFORT_AUTO}.\n"
    echo -ne "\t\tNOTE: ${EFFORT_AUTO} sends no effort flag, so Claude Code uses its own default; a true boolean (0) sends ${EFFORT_DEFAULT}.\n"
    echo -ne "\t ${WORKTREE_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the worktree to be specified for use by the Claude Code model.\n"
    echo -ne "\t\tNOTE: Worktree value must be a valid worktree name.  If no worktree name is provided, Claude Code will assign a new worktree name.\n"
    echo -ne "\t ${RESUME_FLAGS}:\n"
    echo -ne "\t\tFlags that allow a previously created session to be resumed for use by the Claude Code model.\n"
    echo -ne "\t\tNOTE: Session value must be a valid session id.  If no session id is provided, operation will not continue.\n"
    echo -ne "\t ${FORK_SESSION_FLAGS}:\n"
    echo -ne "\t\tFlags that fork the resumed session into a new session id (use together with --resume; the original session is left untouched).\n"
    echo -ne "\t ${PROMPT_FLAGS}:\n"
    echo -ne "\t\tFlags that allow a prompt string to be specified for use by the Claude Code model.\n"
    echo -ne "\t\tNOTE: The word after a prompt flag is the prompt, even one starting with -; ${TRUE} sends the default prompt, ${FALSE} none.\n"
    echo -ne "\t\tNOTE: The prompt, from whichever flag, is sent as Claude Code's last argument, after ${CLAUDE_END_OF_OPTIONS_FLAG}, with no prompt flag.\n"
    echo -ne "\t\tNOTE: Prompt string can be provided either as a string of text or as a file name or pathname to a file containing the prompt string.\n"
    echo -ne "\t ${MODEL_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the specific anthropic model, used by Claude Code, to be specified.\n"
    echo -ne "\t ${SESSION_ID_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the session id to be specified for use by the Claude Code model.\n"
    echo -ne "\t\tNOTE: Session ID value must be a valid session id (i.e., a UUID); if no session id is provided, this script will generate and assign a new UUID as the session id.\n"
    echo -ne "\t ${HEADLESS_FLAGS}:\n"
    echo -ne "\t\tFlags that allow Claude Code's operating mode to be specified--either headless if headless flag is provided or interactive if headless flag is omitted.\n"
    echo -ne "\t ${REMOTE_CONTROL_FLAGS}:\n"
    echo -ne "\t\tFlags that allow Claude Code's remote control to be specified--either remote if remote control flag is provided or local if remote control flag is omitted.\n"
    echo -ne "\t\tNOTE: Remote control takes no value: Claude Code names the session.  A following ${TRUE} (true) or ${FALSE} (false) turns it on or off.\n"
    echo -ne "\t ${PERMISSIONS_FLAGS}:\n"
    echo -ne "\t\tFlags that allow the level of supervision applied to Claude Code's actions to be specified.\n"
    echo -ne "\t ${SPACER_FLAGS}:\n"
    echo -ne "\t\tFlags that allow a spacer to be specified.  Spacer is used to separate flags and values in the input parameters.\n"

    [[ ( "${EXIT_AFTER}" == "${TRUE}" ) || ( "${EXIT_AFTER}" == "${FALSE}" ) ]] && exit "${EXIT_AFTER}" || exit "${FALSE}"
}


########################################################################################################################################################################
# Begin Main Script Execution
#
# Example:
#   ./scripts/wake_the_claude.bash --id 0 --worktree 0 --model opus --effort max --file prompts/thread-handoff_automated-prompts/HANDOFF_2026-06-21_precommit-testing-audit-remediation.md -- --dangerously-skip-permissions --remote-control
#
# Notes:
#   input parameters cannot include both --prompt and --file
#   input parameters cannot include both --prompt and --path
#   if input parameters include both --file and --path, the --path value and --file value are assumed to be the full path to the file
#
########################################################################################################################################################################

########################################################################################################################################################################
# Parse input parameters
########################################################################################################################################################################
debug_log "Verify that input parameters have been provided"
if [[ "${*}" != "" ]]; then
    debug_log "Input Params: [${#} args]"
    # echo "Input Params: \"${*}\""
else
    debug_log "No input params provided."
    usage "${FALSE}"
fi
debug_log "Completed verifying that input parameters have been provided"


debug_log "Parse input parameters"
while (( ${#*} > 0 )); do

    # Parse current input param flag and break when all params complete
    CURRENT_ELEMENT="${1}" && shift
    debug_log "Current Flag: \"${CURRENT_ELEMENT}\""
    debug_log "Remaining Params: size: ${#*}, values: [${*}]"

    # Validate Input Param Flag
    debug_log "Validating Input Param Flag: \"${CURRENT_ELEMENT}\""
    if [[ "${CURRENT_ELEMENT}" == "" ]]; then
        debug_log "Completed parsing input params"
        break || return "${FALSE}"
    fi
    debug_log "Validated Input Param Flag: \"${CURRENT_ELEMENT}\""

    # Save current state of input parameters
    debug_log "Saving current state of input parameters"
    UPDATED_PARAMS=("${@}")
    debug_log "Remaining Params: size: ${#*}, values: [${*}]"
    # debug_log "Updated Params: size: ${#UPDATED_PARAMS[@]}, values: $(redact_uuid "${UPDATED_PARAMS[@]}")"
    debug_log "Updated Params: size: ${#UPDATED_PARAMS[*]}, values: [${UPDATED_PARAMS[*]}]"

    # Add Resume previous session flag
    debug_log "Parsing Input Param Flag: \"${CURRENT_ELEMENT}\""
    PATTERN_INDEX="$(get_pattern_index "${CURRENT_ELEMENT}")"
    debug_log "Pattern Index: ${PATTERN_INDEX}"

    case "${PATTERN_INDEX}" in
        "${EFFORT_FLAGS_INDEX}")         parse_add_effort_flags         "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${FILE_FLAGS_INDEX}")           parse_prompt_file_flags        "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${FORK_SESSION_FLAGS_INDEX}")   parse_add_fork_session_flags   "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${HEADLESS_FLAGS_INDEX}")       parse_add_headless_flags       "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${HELP_FLAGS_INDEX}")           parse_help_flags               "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${MODEL_FLAGS_INDEX}")          parse_add_model_flags          "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${MODEL_VERSION_FLAGS_INDEX}")  parse_model_version_flags      "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${PERMISSIONS_FLAGS_INDEX}")    parse_add_permissions_flags    "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${PROMPT_FLAGS_INDEX}")         parse_prompt_value_flags       "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${PATH_FLAGS_INDEX}")           parse_prompt_path_flags        "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${REMOTE_CONTROL_FLAGS_INDEX}") parse_add_remote_control_flags "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${RESUME_FLAGS_INDEX}")         parse_resume_flags             "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${SCRIPT_VERSION_FLAGS_INDEX}") parse_script_version_flags     "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${SESSION_ID_FLAGS_INDEX}")     parse_session_id_flags         "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${SPACER_FLAGS_INDEX}")         parse_spacer_flags             "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${USAGE_FLAGS_INDEX}")          parse_usage_flags              "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        "${WORKTREE_FLAGS_INDEX}")       parse_add_worktree_flags       "${CURRENT_ELEMENT}" "${UPDATED_PARAMS[@]}" ;;
        *)
            debug_log "Error: Received Invalid Input Param: \"${CURRENT_ELEMENT}\" (pattern index: ${PATTERN_INDEX})"
            usage "${FALSE}"
        ;;
    esac
    PARSE_RESULT="$?"

    # A parser that could not use its flag stops the launch: claude code never starts on input this script rejected
    if [[ "${PARSE_RESULT}" != "${TRUE}" ]]; then
        debug_log "Error: Failed to parse Input Param: \"${CURRENT_ELEMENT}\" (result: ${PARSE_RESULT}). Exiting..."
        # On stderr too: with debug output off (WTC_DEBUG=1) the refusal was the usage text alone, with no reason
        echo "Error: refused \"${CURRENT_ELEMENT}\": its value is missing or invalid (WTC_DEBUG=0 logs why)" >&2
        usage "${FALSE}"
    fi

    # Restore Updated Input Parameter for Script
    debug_log "Completed Parsing: \"${CURRENT_ELEMENT}\""
    debug_log "Setting Updated Params"
    set -- "${UPDATED_PARAMS[@]}"
    debug_log "Remaining Params: size: ${#*}, values: [${*}]" || return "${FALSE}"
done

# A file or path flag whose prompt file never resolved -- a file name whose directory was never given, or a directory
# with no file name -- must not launch claude code without the prompt it was asked to send.
if [[ ( "${VALID_PROMPT_PARAM}" != "${TRUE}" ) && ( ( "${FILE_NAME}" != "" ) || ( "${PATH_NAME}" != "" ) ) && ( "${PROMPT_FILE}" == "" ) ]]; then
    debug_log "Error: received an invalid Prompt File. Exiting..."
    echo "Error: no prompt file found for --file \"${FILE_NAME}\" and --path \"${PATH_NAME}\"" >&2
    usage "${FALSE}"
fi

# Claude Code options, in the order main has always sent them; the prompt follows as the last, positional argument
CLAUDE_CODE_PARAMS+=("${CLAUDE_RESUME_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_SESSION_ID_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_WORKTREE_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_REMOTE_CONTROL_VALUE[@]}")
# auto, in any case, sends no --effort flag (Overtoad ruling, 2026-11-08): Claude Code has no such value and would ignore it with a warning
if [[ ( "${CLAUDE_EFFORT_VALUE[1]:-}" != "" ) && ( "${CLAUDE_EFFORT_VALUE[1],,}" == "${EFFORT_AUTO}" ) ]]; then
    debug_log "Effort ${CLAUDE_EFFORT_VALUE[1]}: sending no effort flag, so Claude Code uses its own default"
    CLAUDE_EFFORT_VALUE=()
fi
CLAUDE_CODE_PARAMS+=("${CLAUDE_EFFORT_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_MODEL_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_HEADLESS_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_FORK_SESSION_VALUE[@]}")
CLAUDE_CODE_PARAMS+=("${CLAUDE_PERMISSIONS_VALUE[@]}")

debug_log "Completed Appending Claude Code Params: ${#CLAUDE_CODE_PARAMS[@]}"
debug_log "Claude Code Params: ${CLAUDE_CODE_PARAMS[*]}"
debug_log "Completed Parsing input parameters"


########################################################################################################################################################################
# Build Claude Code Prompt
########################################################################################################################################################################
debug_log "Build Claude Code Prompt"

if [[ ( "${PROMPT_VALUE}" != "" ) && ( "${VALID_PROMPT_PARAM}" == "${TRUE}" ) ]]; then
    CLAUDE_CODE_PROMPT="${PROMPT_VALUE}"
elif [[ ( "${PROMPT_FILE}" != "" ) && ( ( "${VALID_FILE_PARAM}" == "${TRUE}" ) || ( "${VALID_PATH_PARAM}" == "${TRUE}" ) ) ]]; then
    if ! load_prompt_file "${PROMPT_FILE}"; then
        echo "Error: the prompt file \"${PROMPT_FILE}\" could not be read, or holds no prompt" >&2
        exit 1
    fi
fi

# The prompt is the last argument, after "--", which ends Claude Code's options. Claude Code parses its command line with commander, where an
# option with an optional value (--worktree [name], --remote-control [name], --resume [value]) takes the next word as that value: without the
# "--", a bare --worktree just before the prompt made the prompt the worktree's name and sent no prompt at all. The "--" also keeps a prompt
# that starts with "-", such as a prompt file's YAML front matter, from being refused as an unknown option.
if [[ "${CLAUDE_CODE_PROMPT}" != "" ]]; then
    CLAUDE_CODE_PARAMS+=("${CLAUDE_END_OF_OPTIONS_FLAG}" "${CLAUDE_CODE_PROMPT}")
    debug_log "Prompt loaded [${#CLAUDE_CODE_PROMPT} chars]"
fi
debug_log "Completed building prompt, ${#CLAUDE_CODE_PARAMS[@]} total args"


########################################################################################################################################################################
# Execute Claude Code
########################################################################################################################################################################
debug_log "Executing claude with ${#CLAUDE_CODE_PARAMS[@]} args"

CLAUDE_BIN="$(type -P claude 2>/dev/null || true)"
# The errors that end the launch below go to stderr, not debug_log: with debug output off (WTC_DEBUG=1) they would leave a bare exit 1.
if [[ "${CLAUDE_BIN}" == "" ]] || [[ ! -x "${CLAUDE_BIN}" ]]; then
    echo "Error: claude command not found in PATH" >&2
    exit 1
fi

# Temp root (2026-10-26 rule): the session writes temp files to a Juniper repo's gitignored tmp/,
# never to the system temp dir, which systemd-tmpfiles ages while files are in use. Exported here,
# after the binary check, so a missing claude still reports itself first.
#   CLAUDE_CODE_TMPDIR  Claude Code's own temp root (scratchpad, task output): juniper-ml's primary
#                       tmp/. It must equal env.CLAUDE_CODE_TMPDIR in ~/.claude/settings.json, which
#                       overrides the process environment anyway.
#   TMPDIR/TMP/TEMP     for programs (tempfile, mktemp, pytest): the PRIMARY checkout's tmp/ of the
#                       Juniper repo the session starts in (it outlives a worktree), else juniper-ml's.
#                       The same answer as the user-level SessionStart hook (marker juniper-repo-tmpdir),
#                       which re-exports TMPDIR for every Bash command; a repo outside the Juniper tree
#                       never gets a tmp/ created in it.
# If no tmp/ can be made, stop: an unset or missing TMPDIR silently falls back to the system temp dir.
# shellcheck source=../util/juniper_repo_tmp.bash
source "${REPO_ROOT}/util/juniper_repo_tmp.bash"
HARNESS_TMP_DIR="$(juniper_repo_tmp "${REPO_ROOT}")" || {
    echo "Error: cannot create juniper-ml's tmp/ (from ${REPO_ROOT}); refusing to launch on the system temp dir" >&2
    exit 1
}
SESSION_TMP_DIR="${HARNESS_TMP_DIR}"
ECOSYSTEM_DIR="$(dirname "$(dirname "${HARNESS_TMP_DIR}")")"
if PWD_PRIMARY_DIR="$(juniper_repo_primary "${PWD}" 2>/dev/null)"; then
    case "${PWD_PRIMARY_DIR}" in
        "${ECOSYSTEM_DIR}"/juniper-*)
            SESSION_TMP_DIR="$(juniper_repo_tmp "${PWD_PRIMARY_DIR}")" || SESSION_TMP_DIR="${HARNESS_TMP_DIR}"
            ;;
    esac
fi
# GIT_CEILING_DIRECTORIES: a TMPDIR inside a checkout makes git, run anywhere under it, find the ENCLOSING
# checkout, so a "not a git repository" fixture built under tempfile/tmp_path stops being one. The ceiling
# stops discovery at the tmp/ dirs: the session's primary tmp/, plus the launch directory's own top-level
# tmp/ when that is a Juniper worktree (its Write-tool scratch lives there), then any inherited value.
# The same list as the SessionStart hook writes.
SESSION_CEILING="${SESSION_TMP_DIR}"
PWD_TOPLEVEL_DIR="$(git -C "${PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
case "${PWD_TOPLEVEL_DIR}" in
    "${ECOSYSTEM_DIR}"/juniper-* | "${ECOSYSTEM_DIR}"/worktrees/*)
        [[ "${PWD_TOPLEVEL_DIR}/tmp" == "${SESSION_TMP_DIR}" ]] || SESSION_CEILING="${SESSION_CEILING}:${PWD_TOPLEVEL_DIR}/tmp"
        ;;
esac
export TMPDIR="${SESSION_TMP_DIR}" TMP="${SESSION_TMP_DIR}" TEMP="${SESSION_TMP_DIR}" CLAUDE_CODE_TMPDIR="${HARNESS_TMP_DIR}"
export GIT_CEILING_DIRECTORIES="${SESSION_CEILING}${GIT_CEILING_DIRECTORIES:+:${GIT_CEILING_DIRECTORIES}}"
debug_log "TMPDIR=${TMPDIR} CLAUDE_CODE_TMPDIR=${CLAUDE_CODE_TMPDIR} GIT_CEILING_DIRECTORIES=${GIT_CEILING_DIRECTORIES}"

if [[ "${CLAUDE_HEADLESS_VALUE[*]}" != "" ]]; then
    NOHUP_LOG_FILE=""
    NOHUP_LOG_CANDIDATE="${LOGS_DIR}/wake_the_claude.nohup.log"
    if touch "${NOHUP_LOG_CANDIDATE}" 2>/dev/null; then
        NOHUP_LOG_FILE="${NOHUP_LOG_CANDIDATE}"
    else
        NOHUP_LOG_CANDIDATE="${HOME}/wake_the_claude.nohup.log"
        if touch "${NOHUP_LOG_CANDIDATE}" 2>/dev/null; then
            NOHUP_LOG_FILE="${NOHUP_LOG_CANDIDATE}"
        else
            echo "Error: Failed to open nohup log file at ${LOGS_DIR} or ${HOME}" >&2
            exit 1
        fi
    fi
    debug_log "Nohup log file: ${NOHUP_LOG_FILE}"
    printf '%s\n' "nohup $(format_command_line "${CLAUDE_BIN}" "${CLAUDE_CODE_PARAMS[@]}") >> ${NOHUP_LOG_FILE} 2>&1 &"
    nohup "${CLAUDE_BIN}" "${CLAUDE_CODE_PARAMS[@]}" >> "${NOHUP_LOG_FILE}" 2>&1 &
    debug_log "Launched claude headless via nohup (pid $!)"
else
    debug_log "\"${CLAUDE_BIN}\" ${CLAUDE_CODE_PARAMS[*]}"
    printf '%s\n' "$(format_command_line "${CLAUDE_BIN}" "${CLAUDE_CODE_PARAMS[@]}")"
    "${CLAUDE_BIN}" "${CLAUDE_CODE_PARAMS[@]}"
    CLAUDE_STATUS=$?
    if [[ "${CLAUDE_STATUS}" != "0" ]]; then
        debug_log "Error: claude exited with non-zero status ${CLAUDE_STATUS}"
        exit "${CLAUDE_STATUS}"
    fi
fi

debug_log "Completed Executing Claude Code"
exit 0
