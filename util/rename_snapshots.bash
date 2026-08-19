#!/usr/bin/env bash
########################################################################################################################################################################################################################
#
#
#
########################################################################################################################################################################################################################
# Notes:
#     DIR="/home/pcalnon/Development/python/Juniper/juniper-cascor/src/snapshots"; STUB="snapshot_2026"; ls "${DIR}/${STUB}"*
#
#     FILE_STUB="cascor_snapshot_20260813"; FILE_DIR="$(pwd)"; echo "File: ${FILE_DIR}/${FILE_STUB}"; for f in "${FILE_DIR}/${FILE_STUB}"*; do echo "f: ${f}"; done
#
#     for i in $(ls "${ORIGIN_DIR}/${FILE_STUB}"*); do
#
########################################################################################################################################################################################################################


########################################################################################################################################################################################################################
# Define Environment Constants
ROOT_DIR="${HOME}/Development"
LANG_DIR="python"
PROJ_DIR="Juniper"
REPO_DIR="juniper-cascor"
SRCE_DIR="src/snapshots"
DEST_DIR="src/cascor_snapshots"


########################################################################################################################################################################################################################
# Define Operational Constants
APPL_DIR="${ROOT_DIR}/${LANG_DIR}/${PROJ_DIR}/${REPO_DIR}"  #; echo "APPL_DIR: ${APPL_DIR}"
ORIGIN_DIR="${APPL_DIR}/${SRCE_DIR}"                        #; echo "ORIGIN_DIR: ${ORIGIN_DIR}"
TARGET_DIR="${APPL_DIR}/${DEST_DIR}"                        #; echo "TARGET_DIR: ${TARGET_DIR}"
FILE_STUB="snapshot_2026"                                   #; echo "FILE_STUB: ${FILE_STUB}"
FILE_EXT=".h5"                                              #; echo "FILE_EXT: ${FILE_EXT}"


########################################################################################################################################################################################################################
# Do the things
cd "${ORIGIN_DIR}" || exit

for i in "${ORIGIN_DIR}/${FILE_STUB}"*; do   # ; echo "i: ${i}"
    [[ -f "${i}" ]] || continue              # Verify file exists
    UUID="$(uuidgen)"                        # ; echo "uuid: ${UUID}"
    m="$(basename "${i}")"                   # ; echo "m: ${m}"
    j="$(echo "${m}" | tr -d "Z")"           # ; echo "j: ${j}"
    k="$(echo "${j}" | tr "T" "_")"          # ; echo "k: ${k}"
    l="$(basename -s "${FILE_EXT}" "${k}")"  # ; echo "l: ${l}"
    n="${l}_${UUID}${FILE_EXT}"              # ; echo "n: ${n}"
    p="${TARGET_DIR}/${n}"                   # ; echo "p: ${p}"
    echo -ne "Moving File: ${m}\n"
    echo -ne "Pre-Move check: \"$(ls "${p}" 2>/dev/null)\"\n"
    echo -ne "mv ${i} ${p}\n"
    mv "${i}" "${p}"
    echo -ne "POST-Move check: \"$(ls "${p}" 2>/dev/null)\"\n\n"
done

echo -ne "Verify Moved: \n$(ls "${ORIGIN_DIR}/${FILE_STUB}"* 2>/dev/null)\n\n"
