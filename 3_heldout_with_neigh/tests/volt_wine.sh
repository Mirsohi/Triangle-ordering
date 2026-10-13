#!/bin/bash
args=()
for a in "$@"; do if [[ "$a" == /* ]]; then args+=("$(WINEDEBUG=-all winepath -w "$a")"); else args+=("$a"); fi; done
WINEDEBUG=-all exec wine /home/claude/review2/neigh_supp/volt/volt.exe "${args[@]}"
