#!/bin/bash
# The rest of the real-data experiment, unattended, after the pilot that is already running:
#   1. wait for that pilot, the process PILOT of run_real_pilot.sh, to end, for at most a day;
#   2. extract the code of ARCHIVE over modern_methodology and shared;
#   3. add MicroTEDAClus+ without its noise label to the pilot, in results_real_pilot;
#   4. check the search ranges on the pilot and keep the verdict in range_check.txt; a failed
#      check does not stop the sequence, and is verified afterwards;
#   5. run the experiment, run_real_main.sh;
#   6. pack the results in real_results.tar.gz, in the project folder.
# Every step continues from what is already written, so after an error or a stop the same
# command with "none" as PILOT resumes the sequence. This script is not in ARCHIVE, so the
# extraction cannot change it while it runs.
#
# Usage, from the project folder, which holds experimentos/:
#   setsid nohup experimentos/run_real_sequence.sh PILOT ARCHIVE [OPTIONS] \
#       > real_sequence.log 2>&1 < /dev/null &
# PILOT is the process id of the running run_real_pilot.sh, or "none" if no pilot runs.
# OPTIONS go to run_real_pilot.sh and to run_real_main.sh, e.g. --workers 16 --memory-limit 5.
# Stop the whole sequence: kill -TERM -- -$(cat experimentos/real_sequence.pgid)
set -u

here="$(cd "$(dirname "$0")" && pwd)"
project="$(dirname "$here")"
work="$here/modern_methodology"
models=(MicroTEDAClusPlusNoNoise DenStream DBSTREAM BIRCH MiniBatchKMeans BICO)
wait_limit=$((24 * 3600))

log() { echo "$(date '+%F %T') $*"; }
fail() {
    log "ERROR: $*"
    exit 1
}

if [ $# -lt 2 ]; then
    echo "usage: $0 PILOT ARCHIVE [OPTIONS]" >&2
    exit 2
fi
pilot=$1
archive="$(realpath "$2")"
shift 2
options=("$@")

exec 9> "$here/real_sequence.lock"
flock -n 9 || fail "another sequence is running"
ps -o pgid= -p $$ | tr -d ' ' > "$here/real_sequence.pgid"
log "The sequence starts, process group $(cat "$here/real_sequence.pgid")"

[ -f "$archive" ] || fail "there is no archive at $archive"
tar -tzf "$archive" | grep -qx "experimentos/modern_methodology/run_real_main.sh" ||
    fail "$archive does not hold experimentos/modern_methodology/run_real_main.sh"

# 1. The pilot that is already running
if [ "$pilot" = none ]; then
    if pgrep -f "run_real_pilot.sh" > /dev/null; then
        fail "a pilot is running: give its process id instead of none"
    fi
else
    tr '\0' ' ' < "/proc/$pilot/cmdline" 2> /dev/null | grep -q "run_real_pilot.sh" ||
        fail "process $pilot is not run_real_pilot.sh"
    log "Waiting for the pilot, process $pilot, to end"
    waited=0
    while kill -0 "$pilot" 2> /dev/null; do
        [ "$waited" -lt "$wait_limit" ] || fail "the pilot still runs after a day"
        sleep 60
        waited=$((waited + 60))
    done
    log "The pilot ended"
fi
if [ "$(tail -n 1 "$work/run_real_pilot.log" 2> /dev/null)" != "The pilot is complete." ]; then
    log "WARNING: run_real_pilot.log does not end with 'The pilot is complete.'; step 3 resumes it"
fi

# 2. The code
log "Extracting $archive"
tar -xzf "$archive" -C "$project" || fail "the extraction failed"
grep -q "runs=(126 219)" "$work/run_real_main.sh" ||
    fail "the extracted run_real_main.sh is not that of this sequence"
cd "$work" || fail "there is no $work"

# 3. MicroTEDAClus+ without its noise label in the pilot
log "Adding MicroTEDAClus+ without its noise label to the pilot"
./run_real_pilot.sh ${options[@]+"${options[@]}"} > run_real_pilot_no_noise.log 2>&1 ||
    fail "the pilot failed; see run_real_pilot_no_noise.log"
log "The pilot is complete"

# 4. The range check, kept for verification
log "Checking the search ranges"
if .venv/bin/python -m benchmark.range_check results_real_pilot --models "${models[@]}" \
    > range_check.txt 2>&1; then
    log "RANGE CHECK PASSED; see range_check.txt"
else
    log "RANGE CHECK FAILED; see range_check.txt. The experiment runs anyway, to be verified"
fi

# 5. The experiment
log "Running the experiment"
./run_real_main.sh ${options[@]+"${options[@]}"} > run_real_main.log 2>&1 ||
    fail "the experiment failed; see run_real_main.log"
log "The experiment is complete"

# 6. The results
log "Packing the results"
packed=()
for name in results_real_pilot results_real_main results_timing_real_main \
    results_memory_real_main run_real_pilot.log run_real_pilot_no_noise.log range_check.txt \
    run_real_main.log; do
    [ -e "$name" ] && packed+=("$name")
done
tar -czf "$project/real_results.tar.gz" "${packed[@]}" || fail "the packing failed"
log "The sequence is complete; the results are in $project/real_results.tar.gz"
