"""Tasks in parallel processes, each written when it finishes.

A run that stops advancing ends with an error instead of waiting forever, and a
run that ends, for any reason, leaves no worker process behind.
"""

import csv
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

STALL_LIMIT = (
    3600  # seconds without a finished task after which a run is declared stalled
)
TASK_KEY = ("dataset", "model", "run")


def finished_tasks(tasks_file: Path) -> set:
    """(dataset, model, run) of the tasks that a file of finished tasks holds."""
    if not tasks_file.exists():
        return set()
    with tasks_file.open(newline="") as stored:
        return {
            (r["dataset"], r["model"], int(r["run"])) for r in csv.DictReader(stored)
        }


def _call(call):
    function, arguments = call
    return function(*arguments)


def _terminate(signal_number, frame):
    sys.exit(f"ERROR: the run was asked to end (signal {signal_number})")


def run_tasks(
    function,
    tasks,
    tasks_file: Path,
    columns,
    workers,
    stall_limit=STALL_LIMIT,
    tasks_per_process=None,
):
    """Run `function(*task)` for every task and append the row it returns to `tasks_file`.

    The tasks start in the given order, in `workers` processes. A row has the
    key of its task and a status, "OK" or the error. The program ends with an
    error if no task finishes within `stall_limit` seconds. With
    `tasks_per_process`, a process is replaced after that many tasks, so that
    with one every task starts in a new process.
    """
    signal.signal(signal.SIGTERM, _terminate)  # so that the workers end with the run
    is_new = not tasks_file.exists()
    started = time.perf_counter()
    with (
        tasks_file.open("a", newline="") as stored,
        multiprocessing.get_context("spawn").Pool(
            workers, maxtasksperchild=tasks_per_process
        ) as pool,
    ):
        writer = csv.DictWriter(stored, fieldnames=columns)
        if is_new:
            writer.writeheader()
        results = pool.imap_unordered(_call, [(function, task) for task in tasks])
        for done in range(1, len(tasks) + 1):
            try:
                row = results.next(timeout=stall_limit)
            except multiprocessing.TimeoutError:
                sys.exit(
                    f"ERROR: no task finished in {stall_limit} seconds; the run is stopped"
                )
            writer.writerow(row)
            stored.flush()
            os.fsync(stored.fileno())
            if row["status"] != "OK":
                print(
                    " ".join(str(row[k]) for k in TASK_KEY) + f": {row['status']}",
                    flush=True,
                )
            if done % 50 == 0 or done == len(tasks):
                hours = (time.perf_counter() - started) / 3600
                remaining = hours / done * (len(tasks) - done)
                print(
                    f"{done}/{len(tasks)} tasks, {hours:.2f} h elapsed, about {remaining:.1f} h left",
                    flush=True,
                )
