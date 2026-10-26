#!/bin/bash
# The modernized experiment, unattended, by the rules of the experiments, real2:
#   1. extract the code of ARCHIVE over modern_methodology and shared;
#   2. pilot every model on the runs 94 to 125, run_modern_pilot.sh, in results_pilot;
#   3. check the search ranges on the pilot and keep the verdict in range_check_modern.txt;
#      a failed check does not stop the sequence, and is verified afterwards;
#   4. run the experiment, run_modern_main.sh;
#   5. pack the results in modern_results.tar.gz, in the project folder.
# Every step continues from what is already written, so after an error or a stop the same
# command resumes the sequence. This script is not in ARCHIVE, so the extraction cannot
# change it while it runs. Results of the earlier modernized experiment, by other rules or
# on other runs, must be moved aside first: the sequence stops at once if it finds any in
# the folders of this one.
#
# Usage, from the project folder, which holds experimentos/:
#   setsid nohup experimentos/run_modern_sequence.sh ARCHIVE [OPTIONS] \
#       > modern_sequence.log 2>&1 < /dev/null &
# OPTIONS go to run_modern_pilot.sh and to run_modern_main.sh, e.g. --workers 16.
# Stop the whole sequence: kill -TERM -- -$(cat experimentos/modern_sequence.pgid), and after
# 30 seconds kill -KILL -- -$(cat experimentos/modern_sequence.pgid), because the memory pass,
# which starts a process per task, does not end on TERM; then ps -g of that group shows
# nothing.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$here")"
work="$here/modern_methodology"

log() { echo "$(date '+%F %T') $*"; }
fail() {
    log "ERROR: $*"
    exit 1
}
# Whether the tasks file $1, with the run in its third column, holds a run below $2.
runs_below() {
    [ -f "$1" ] && awk -F, -v first="$2" 'NR > 1 && $3 < first { found = 1 } END { exit !found }' "$1"
}
# Whether the folder $1 records a configuration other than the protocol real2, tune-only
# if $2 is "pilot".
other_configuration() {
    local stored="$1/configuration.json"
    [ -f "$stored" ] || return 1
    grep -q '"protocol": "real2"' "$stored" || return 0
    if [ "$2" = pilot ]; then
        grep -q '"tune_only": true' "$stored" || return 0
    else
        grep -q '"tune_only"' "$stored" && return 0
    fi
    return 1
}

if [ $# -lt 1 ] || [[ "$1" == --* ]]; then
    echo "usage: $0 ARCHIVE [OPTIONS]" >&2
    exit 2
fi
archive="$(realpath -- "$1")"
shift
options=("$@")

exec 9> "$here/modern_sequence.lock"
flock -n 9 || fail "another sequence is running"
ps -o pgid= -p $$ | tr -d ' ' > "$here/modern_sequence.pgid"
log "The sequence starts, process group $(cat "$here/modern_sequence.pgid")"

[ -f "$archive" ] || fail "there is no archive at $archive"
tar -tzf "$archive" | grep -qx "experimentos/modern_methodology/run_modern_main.sh" ||
    fail "$archive does not hold experimentos/modern_methodology/run_modern_main.sh"

# Results of the earlier experiment in the folders of this one
other_configuration "$work/results_pilot" pilot &&
    fail "$work/results_pilot holds results of other rules: move it aside first"
other_configuration "$work/results_main" main &&
    fail "$work/results_main holds results of other rules: move it aside first"
for name in results_previous_protocol results_memory; do
    runs_below "$work/$name/tasks.csv" 126 &&
        fail "$work/$name holds runs below 126, of the earlier experiment: move it aside first"
done
if [ -d "$work/results_timing/batches" ] &&
    find "$work/results_timing/batches" -name '*.csv' | awk -F/ '{ sub(/\.csv$/, "", $NF); if ($NF + 0 < 126) found = 1 } END { exit !found }'; then
    fail "$work/results_timing holds runs below 126, of the earlier experiment: move it aside first"
fi

# 1. The code
log "Extracting $archive"
tar -xzf "$archive" -C "$project" || fail "the extraction failed"
grep -q -- "--protocol real2" "$work/run_modern_main.sh" ||
    fail "the extracted run_modern_main.sh is not that of this sequence"
cd "$work" || fail "there is no $work"

# 2. The pilot
log "Running the pilot"
./run_modern_pilot.sh ${options[@]+"${options[@]}"} >> run_modern_pilot.log 2>&1 ||
    fail "the pilot failed; see run_modern_pilot.log"
log "The pilot is complete"

# 3. The range check, kept for verification
log "Checking the search ranges"
if .venv/bin/python -m benchmark.range_check results_pilot > range_check_modern.txt 2>&1; then
    log "RANGE CHECK PASSED; see range_check_modern.txt"
else
    log "RANGE CHECK FAILED; see range_check_modern.txt. The experiment runs anyway, to be verified"
fi

# 4. The experiment
log "Running the experiment"
./run_modern_main.sh ${options[@]+"${options[@]}"} >> run_modern_main.log 2>&1 ||
    fail "the experiment failed; see run_modern_main.log"
log "The experiment is complete"

# 5. The results
log "Packing the results"
packed=()
for name in results_pilot results_main results_previous_protocol results_timing \
    results_memory run_modern_pilot.log range_check_modern.txt run_modern_main.log; do
    [ -e "$name" ] && packed+=("$name")
done
tar -czf "$project/modern_results.tar.gz" "${packed[@]}" || fail "the packing failed"
log "The sequence is complete; the results are in $project/modern_results.tar.gz"
