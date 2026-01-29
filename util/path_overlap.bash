#!/usr/bin/env bash

A="/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src"
echo "Current: ${A}"
B="./src/prototypes/cascor/src/cascade_correlation/candidate_unit/candidate_unit.py"
echo "Source: ${B}"

[[ "${B%%\/*}" == "." ]] && SUFFIX="${B#*/}" || SUFFIX="${B}"
PREFIX="${A}"
INFIX=""
echo "Prefix: ${PREFIX}"
echo "Suffix: ${SUFFIX}"

until [[ "${PREFIX}" == "" ]]; do
    FOUND="$(echo "${SUFFIX}" | grep "${PREFIX}")"
    echo "Match: \"${FOUND}\""
    if [[ "${FOUND}" != "" ]]; then
        INFIX="${PREFIX}"
	echo "Infix: ${INFIX}"
        PREFIX="${A%"${INFIX}"*}"
	echo "Prefix: ${PREFIX}"
        break
    else
        echo "Match Not Found"
    fi
    PREFIX="${PREFIX#*/}"
    echo "Prefix: ${PREFIX}"
done
echo "PRE: ${PREFIX}, SUF: ${SUFFIX}"
