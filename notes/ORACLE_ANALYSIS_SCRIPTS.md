# Analysis and Recommended Fixes for juniper_cascor.bash Helper Script Resolution Issues

## Amp Thread

Refer to the detailed analysis and recommendations in this AMP forum thread:
<https://ampcode.com/threads/T-019bbb98-53a8-77ad-8916-0e295eb3a5be#message-21-block-1>

## Task Prompt

Task:
Analyze the Juniper Cascor shell script infrastructure and identify all issues preventing proper execution. Specifically:

1. In `util/juniper_cascor.bash` (lines 61-74), the helper scripts `__get_project_dir.bash`, `__get_os_name.bash`, and `__git_log_weeks.bash` are referenced without full paths. The config file `script_util.cfg` defines paths based on `ROOT_UTIL_DIR` but this may not be set in the juniper_cascor.bash PATH before trying to call these scripts.

2. The BASE_DIR variable ends up empty because `__get_project_dir.bash` command is not found.

3. The SOURCE_DIR incorrectly resolves to `/src` instead of the actual path.

Review these files and confirm all the issues in the script path resolution chain.
Context:
The user is seeing this error when running ./try:

```bash
/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/util/juniper_cascor.bash: line 67: __get_project_dir.bash: command not found
Base Dir: 
Python Script: /src/./main.py
/opt/miniforge3/envs/JuniperCascor/bin/python3: can't open file '/src/./main.py': [Errno 2] No such file or directory
```

The helper scripts exist in `util/` but are called without the full path in juniper_cascor.bash.
Files:

- util/juniper_cascor.bash
- conf/script_util.cfg
- util/__get_project_dir.bash

---

## Detailed Analysis and Recommended Fixes

### High Level Summary

TL;DR

The immediate failure is because juniper_cascor.bash discards the absolute helper-script paths from script_util.cfg and redefines them as bare filenames, which are not on $PATH.
As a result __get_project_dir.bash is not found, BASE_DIR is empty, and SOURCE_DIR becomes /src, so Python is invoked with /src/./main.py which does not exist.
Independently, script_util.cfg’s idea of the project root and util script names/paths does not match your actual directory layout or helper file names, so even if you used the config-derived paths, they are currently wrong.

### Recommended approach (simple path)

Effort: M (1–3h) to cleanly fix and test.

#### Step 1 – Fix helper script resolution in juniper_cascor.bash

Right after sourcing script_util.cfg, you currently have:

61: GET_OS_SCRIPT="__get_os_name.bash"
62: GET_PROJECT_SCRIPT="__get_project_dir.bash"
63: DATE_FUNCTIONS_SCRIPT="__git_log_weeks.bash"

These lines override any paths defined in the config and leave you with just bare filenames. That causes the immediate command not found error.

##### Minimal fix – option A

Option A is independent of config’s ROOT_* paths

```bash
# After 'source "${ROOT_CONF_FILE}"' and before using the helpers
# Resolve this util directory from the current script location

UTIL_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

GET_OS_SCRIPT="${UTIL_DIR}/__get_os_name.bash"
GET_PROJECT_SCRIPT="${UTIL_DIR}/__get_project_dir.bash"
DATE_FUNCTIONS_SCRIPT="${UTIL_DIR}/__git_log_weeks.bash"
```

And remove lines 61–63 as they are now.

This guarantees you call the helper scripts from the actual util/ directory where the current script lives, matching the path that shows up in the error:

/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor/util/juniper_cascor.bash

#### Step 2 – Verify the helper files and names

Based on your description, the util directory contains:

- util/__get_project_dir.bash (we have this file)
- util/__get_os_name.bash
- util/__git_log_weeks.bash

Make sure these filenames are exactly as above; otherwise adjust the three variables accordingly.

#### Step 3 – Confirm BASE_DIR and SOURCE_DIR behavior

__get_project_dir.bash does:

```bash
SCRIPT_PATH="${1}"
SCRIPT_PATH="$(readlink -f "${SCRIPT_PATH}")"
SCRIPT_DIR="$(cd -P "$(dirname -- "${SCRIPT_PATH}")" ... && pwd)"
PROJECT_DIR="$(dirname -- "${SCRIPT_DIR}")"
BASE_DIR="${PROJECT_DIR}"
echo "${BASE_DIR}"
```

Given the main script is in:

```bash
.../Juniper/JuniperCascor/juniper_cascor/util/juniper_cascor.bash
```

the computed values will be:

```bash
    SCRIPT_DIR → .../Juniper/JuniperCascor/juniper_cascor/util
    PROJECT_DIR/BASE_DIR → .../Juniper/JuniperCascor/juniper_cascor
```

Then, in juniper_cascor.bash:

```bash
export DATA_DIR="${BASE_DIR}/${DATA_DIR_NAME}"   # .../juniper_cascor/data
export SOURCE_DIR="${BASE_DIR}/${SOURCE_DIR_NAME}" # .../juniper_cascor/src
...
export PYTHON_SCRIPT="${SOURCE_DIR}/${PYTHON_SCRIPT_PATH}" # .../src/./main.py
```

Once BASE_DIR is correctly set, SOURCE_DIR stops being /src and becomes the expected project src directory.

#### Step 4 – Fix CURRENT_OS and date helper sourcing

Still in juniper_cascor.bash:

```bash
export CURRENT_OS=$(${GET_OS_SCRIPT})
source "${DATE_FUNCTIONS_SCRIPT}"
```

With Step 1 in place, these will run as:

```bash
    /absolute/path/to/util/__get_os_name.bash
    source /absolute/path/to/util/__git_log_weeks.bash
```

Verify both helper scripts are executable (or at least readable for source) and have correct shebangs. Fix file permissions if needed:

```bash
chmod +x util/__get_os_name.bash util/__get_project_dir.bash util/__git_log_weeks.bash
```

---

## Rationale and trade-offs

### What’s actually breaking today

After you source "${ROOT_CONF_FILE}", the config exports:

```bash
export ROOT_PROJECT_DIR="${LANGUAGE_DEV_DIR}/${ROOT_PROJECT_DIR_NAME}"
export ROOT_UTIL_DIR="${ROOT_PROJECT_DIR}/${UTILITY_DIR_NAME}"

export GET_OS_SCRIPT="${ROOT_UTIL_DIR}/${GET_OS_SCRIPT_NAME}"
export GET_PROJECT_SCRIPT="${ROOT_UTIL_DIR}/${GET_PROJECT_SCRIPT_NAME}"
export DATE_FUNCTIONS_SCRIPT="${ROOT_UTIL_DIR}/${DATE_FUNCTIONS_NAME}"
export GIT_LOG_SCRIPT="${ROOT_UTIL_DIR}/${GIT_LOG_SCRIPT_NAME}"
```

But in juniper_cascor.bash you immediately overwrite those with:

```bash
GET_OS_SCRIPT="__get_os_name.bash"
GET_PROJECT_SCRIPT="__get_project_dir.bash"
DATE_FUNCTIONS_SCRIPT="__git_log_weeks.bash"
```

When you then do:

```bash
export BASE_DIR=$(${GET_PROJECT_SCRIPT} "${BASH_SOURCE[0]}")
```

the shell tries to execute __get_project_dir.bash as a command, searches $PATH, does not find it, and prints:

```bash
__get_project_dir.bash: command not found
```

Because the command failed with no stdout, BASE_DIR becomes empty. All subsequent paths built from BASE_DIR become absolute paths from root:

```bash
SOURCE_DIR="${BASE_DIR}/${SOURCE_DIR_NAME}" → /src
PYTHON_SCRIPT="${SOURCE_DIR}/${PYTHON_SCRIPT_PATH}" → /src/./main.py
```

which is exactly the error you see:

Python Script: /src/./main.py
python3: can't open file '/src/./main.py': [Errno 2] No such file or directory

### How to fix it: Option Simple

#### Follow Simple Path

The simple fix is to stop overriding the script paths with bare filenames and instead construct absolute paths from the current script location (or correctly from config), which is what Step 1 does.

#### Risks and guardrails

Config and actual layout are inconsistent.

script_util.cfg computes:

ROOT_PROJECT_DIR="${LANGUAGE_DEV_DIR}/${ROOT_PROJECT_DIR_NAME}"
ROOT_PROJECT_DIR_NAME="${JUNIPER_APPLICATION_NAME}"  # "JuniperCascor"

which yields:

/home/pcalnon/Development/python/JuniperCascor

but your real project path (from the error) is:

/home/pcalnon/Development/python/Juniper/JuniperCascor/juniper_cascor

So ROOT_UTIL_DIR and the config’s GET_* paths do not actually point to the real util/. That’s why I recommend basing helper paths directly on BASH_SOURCE[0] in this script for now, instead of trying to lean on ROOT_PROJECT_DIR.

Naming inconsistencies in the config:

In script_util.cfg:

export DATE_FUNCTIONS_NAME="__date_functions.bash"
export GIT_LOG_SCRIPT_NAME="git_log_weeks.bash"

but in juniper_cascor.bash you set:

DATE_FUNCTIONS_SCRIPT="__git_log_weeks.bash"

So:
    Config thinks “date functions” live in __date_functions.bash.
    Config thinks git-log helper is git_log_weeks.bash.
    Main script thinks the date-related helper to source is "__git_log_weeks.bash".

This mismatch means you cannot safely rely on DATE_FUNCTIONS_SCRIPT / GIT_LOG_SCRIPT from the config as-is. The simplest guardrail is to treat the util directory that contains juniper_cascor.bash as the single source of truth for these helper script names and paths, and update the config later for consistency.

Future scripts may copy the same pattern.

If other util scripts also override GET_OS_SCRIPT/GET_PROJECT_SCRIPT like this, they will repeat the same bug. When you fix juniper_cascor.bash, consider quickly grepping for GET_PROJECT_SCRIPT="__get_project_dir.bash" and cleaning those up in the same way.

### How to fix it: Option Advanced

#### When to consider the advanced path

You should revisit the design (and use the config’s directory model properly) if:

You want all projects / languages to share a single script_util.cfg and be portable across different directory layouts.
You plan to execute these scripts from different working directories or via symlinks and need fully robust, config-driven path resolution.
You add more tooling that relies on ROOT_PROJECT_DIR, ROOT_UTIL_DIR and want a single canonical definition for the project root.

#### Optional advanced path (if you want to centralize everything)

If/when you want a more centralized solution:

**Make script_util.cfg describe your actual layout:**

- Introduce additional variables (e.g. ROOT_PROJ_NAME, ROOT_SUBPROJECT_NAME, ROOT_APPLICATION_NAME) in the config to mirror what you currently hard-code in juniper_cascor.bash.
- Compute ROOT_PROJECT_DIR from those three to match the real path:

```bash
ROOT_PROJECT_DIR="${LANGUAGE_DEV_DIR}/${ROOT_PROJ_NAME}/${ROOT_SUBPROJECT_NAME}/${ROOT_APPLICATION_NAME}"
ROOT_UTIL_DIR="${ROOT_PROJECT_DIR}/${UTILITY_DIR_NAME}"
```

**Align helper names:**

- Pick canonical names: e.g. "__get_os_name.bash", "__get_project_dir.bash", "__git_log_weeks.bash".
- Update GET_OS_SCRIPT_NAME, GET_PROJECT_SCRIPT_NAME, GIT_LOG_SCRIPT_NAME, DATE_FUNCTIONS_NAME in script_util.cfg to match the real filenames.
- Use only the config-derived GET_OS_SCRIPT, GET_PROJECT_SCRIPT, DATE_FUNCTIONS_SCRIPT, GIT_LOG_SCRIPT and do not reassign them in individual scripts.

**Optionally extend PATH or always use absolute paths:**

- Either leave as-is and always call helpers via their absolute paths from config, or
- Add ${ROOT_UTIL_DIR} to your PATH in one central bootstrap script and then rely on bare names everywhere.

## Conclusion

For now, though, the minimal fix in juniper_cascor.bash (absolute helper paths derived from BASH_SOURCE[0]) is enough to resolve the error you’re seeing and get the Python script running.
