#!/usr/bin/env bash

########################################################################################################################################################################################################################################################
# References:
#     DIR="./conf/" && for i in $(ls "${DIR}"); do if [[ -f "${DIR}${i}" ]]; then echo "${i}"; sed -i "s/# Purpose:       Juniper Project Cascade Correlation Neural Network/# Purpose:       Juniper Project Cascade Correlation Neural Network/g" "${DIR}${i}"; else echo "Nope! ${i}"; fi; done
#     DIR="./util/" && for i in $(ls "${DIR}"); do if [[ -f "${DIR}${i}" ]]; then echo "${i}"; sed -i "s/CASCOR/CASCOR/g" "${DIR}${i}"; else echo "Nope! ${i}"; fi; done^C
#     DIR="./conf/" && for i in $(ls "${DIR}"); do if [[ -f "${DIR}${i}" ]]; then echo "${i}"; sed -i "s/# Last Modified: 2026-01-12
#
########################################################################################################################################################################################################################################################


########################################################################################################################################################################################################################################################
# Define Constants for source code update
########################################################################################################################################################################################################################################################
DIR_LIST="./conf/ ./util/ ./src/"

STRING_LIST_OLD=(\
"CASCOR" \
"Cascor" \
"cascor" \
"# Last Modified: 2026-01-12
"# Date Created:  " \
"# Purpose:       Juniper Project Cascade Correlation Neural Network" \
)

STRING_LIST_NEW=(\
"CASCOR" \
"Cascor" \
"cascor" \
"# Last Modified: 2026-01-12
"# Date Created:  " \
"# Purpose:       Juniper Project Cascade Correlation Neural Network" \
)



########################################################################################################################################################################################################################################################
# Perform string replacement for source files
########################################################################################################################################################################################################################################################

# List of directories to update
for j in ${DIR_LIST}; do
    # List of files found by recursive search of directory
    for k in $(find "${j}"); do
        echo "${k}"
        # Validate ojbect is valid file
        if [[ -f "${k}" ]];  then
            # Replace all old strings with new strings
            for (( i=0; i<"${#STRING_LIST_NEW[*]}"; i++ )); do
                OLD_VALUE="${STRING_LIST_OLD[${i}]}"
                NEW_VALUE="${STRING_LIST_NEW[${i}]}"
                echo "sed -i \"s/${OLD_VALUE}/${NEW_VALUE}/g\" \"${k}\""
                sed -i "s/${OLD_VALUE}/${NEW_VALUE}/g" "${k}"
            done
        else
            echo -ne "\tNope! \"${k}\"\n\n"
        fi
    done
done
