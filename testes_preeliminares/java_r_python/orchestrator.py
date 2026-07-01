import subprocess
import re
import csv
import sys


def parse_time_output(stderr_text: str) -> dict[str, float]:
    """Extracts wall clock time and peak RSS memory from /usr/bin/time -v output."""
    metrics = {}

    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", stderr_text)
    if rss_match:
        metrics["os_peak_rss_mb"] = float(rss_match.group(1)) / 1024.0

    time_match = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+([\d:.]+)", stderr_text
    )
    if time_match:
        time_str = time_match.group(1)
        parts = time_str.split(":")
        if len(parts) == 2:
            metrics["os_wall_clock_sec"] = float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            metrics["os_wall_clock_sec"] = (
                float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            )

    return metrics


def parse_internal_metrics(stdout_text: str, environment: str) -> dict[str, float]:
    """Extracts internally logged execution time and memory from the benchmark scripts."""
    metrics = {"internal_time_sec": float, "internal_peak_rss_mb": float}

    if environment == "Java":
        time_match = re.search(r"Execution Time \(ms\) \[.*?\]:\s+(\d+)", stdout_text)
        if time_match:
            metrics["internal_time_sec"] = float(time_match.group(1)) / 1000.0

    elif environment == "Bridge":
        time_match = re.search(
            r"Bridge Execution Time \[.*?\]:\s+([\d.]+)", stdout_text
        )
        if time_match:
            metrics["internal_time_sec"] = float(time_match.group(1))

        mem_match = re.search(
            r"Peak Memory Usage \(MB\) \[.*?\]:\s+([\d.]+)", stdout_text
        )
        if mem_match:
            metrics["internal_peak_rss_mb"] = float(mem_match.group(1))

    return metrics


def run_experiment(command: list) -> tuple:
    """Wraps command in /usr/bin/time and executes it."""
    time_cmd = ["/usr/bin/time", "-v"] + command
    result = subprocess.run(time_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Error executing {' '.join(command)}\nSTDERR:\n{result.stderr}",
            file=sys.stderr,
        )

    return result.stdout, result.stderr


def main():
    algorithms = ["CluStream", "DenStream", "StreamKM"]
    csv_input = "stream_data.csv"
    results_file = "benchmark_results.csv"

    results = []

    commands = {
        "Java": [
            "java",
            "-Xmx2g",
            "-cp",
            "target/classes:target/dependency/*",
            "benchmark.MOABenchmark",
            csv_input,
            "10",
        ],
        "Bridge": ["python3", "bridge_benchmark.py", csv_input],
    }

    for alg in algorithms:
        for env, base_cmd in commands.items():
            print(f"Executing {env} : {alg}...")
            cmd = base_cmd + [alg]

            stdout, stderr = run_experiment(cmd)

            os_metrics = parse_time_output(stderr)
            internal_metrics = parse_internal_metrics(stdout, env)

            row = {
                "Algorithm": alg,
                "Environment": env,
                "Internal_Exec_Time_sec": internal_metrics.get("internal_time_sec"),
                "Internal_Peak_RSS_MB": internal_metrics.get("internal_peak_rss_mb"),
                "OS_Wall_Clock_sec": os_metrics.get("os_wall_clock_sec"),
                "OS_Peak_RSS_MB": os_metrics.get("os_peak_rss_mb"),
            }
            results.append(row)

    with open(results_file, mode="w", newline="") as csvfile:
        fieldnames = [
            "Algorithm",
            "Environment",
            "Internal_Exec_Time_sec",
            "Internal_Peak_RSS_MB",
            "OS_Wall_Clock_sec",
            "OS_Peak_RSS_MB",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results successfully written to {results_file}")


if __name__ == "__main__":
    main()
