#!/usr/bin/env bash
# This script is intended to test the use of the result value of running the py_compile python command


SOURCE_FILE="/home/pcalnon/Development/python/Juniper/src/prototypes/cascor/src/spiral_problem/spiral_problem.py"



echo "python -m py_compile \"${SOURCE_FILE}\""
python -m py_compile "${SOURCE_FILE}"
echo "Result: $?"


python -m py_compile "${SOURCE_FILE}"
[[ $? ]] && echo "Good" || echo "Nope"



[[ $( python -m py_compile "${SOURCE_FILE}" ) ]] && echo "success" || echo "suck"


#BLA="$( python -m py_compile "${SOURCE_FILE}" )"
# echo "Bla: \"${BLA}\""

# [[ python "-mpy_compile" "${SOURCE_FILE}" ]] && echo "Frog" || echo "Toad"

# [[ python -m py_compile "${SOURCE_FILE}" ]] && echo "Yap" || echo "bla"

python -m py_compile "${SOURCE_FILE}" && echo "Yaz" || echo "Meh"
