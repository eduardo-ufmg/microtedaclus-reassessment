import csv
import math
import re
import subprocess
import sys


def parse_time_output(stderr_text: str) -> dict[str, float | None]:
    metrics = {"os_peak_rss_mb": None, "os_wall_clock_sec": None}

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


def parse_internal_metrics(
    stdout_text: str, environment: str
) -> dict[str, float | None]:
    metrics = {"internal_time_sec": None, "internal_peak_rss_mb": None}

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
    elif environment == "River":
        time_match = re.search(r"River Execution Time \[.*?\]:\s+([\d.]+)", stdout_text)
        if time_match:
            metrics["internal_time_sec"] = float(time_match.group(1))

        mem_match = re.search(
            r"Peak Memory Usage \(MB\) \[.*?\]:\s+([\d.]+)", stdout_text
        )
        if mem_match:
            metrics["internal_peak_rss_mb"] = float(mem_match.group(1))

    return metrics


def run_experiment(command: list[str]) -> tuple[str, str]:
    time_cmd = ["/usr/bin/time", "-v"] + command
    result = subprocess.run(time_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Error executing {' '.join(command)}\nSTDERR:\n{result.stderr}",
            file=sys.stderr,
        )

    return result.stdout, result.stderr


def compute_statistics(data: list[float | None]) -> tuple[float | None, float | None]:
    valid_data = [x for x in data if x is not None]
    n = len(valid_data)

    if n == 0:
        return None, None

    mean = sum(valid_data) / n

    if n > 1:
        variance = sum((x - mean) ** 2 for x in valid_data) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return mean, std


def main():
    algorithms = ["CluStream", "DenStream", "StreamKM"]
    csv_input = "stream_data.csv"
    results_file = "benchmark_results.csv"
    runs = 8

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
        "River": ["python3", "river_benchmark.py", csv_input],
    }

    for alg in algorithms:
        for env, base_cmd in commands.items():
            print(f"Executing {env} : {alg} (Iterations: {runs})...")

            raw_metrics = {
                "Internal_Exec_Time_sec": [],
                "Internal_Peak_RSS_MB": [],
                "OS_Wall_Clock_sec": [],
                "OS_Peak_RSS_MB": [],
            }

            for i in range(runs):
                cmd = base_cmd + [alg]
                stdout, stderr = run_experiment(cmd)

                os_m = parse_time_output(stderr)
                int_m = parse_internal_metrics(stdout, env)

                raw_metrics["OS_Wall_Clock_sec"].append(os_m["os_wall_clock_sec"])
                raw_metrics["OS_Peak_RSS_MB"].append(os_m["os_peak_rss_mb"])
                raw_metrics["Internal_Exec_Time_sec"].append(int_m["internal_time_sec"])
                raw_metrics["Internal_Peak_RSS_MB"].append(
                    int_m["internal_peak_rss_mb"]
                )

            row = {"Algorithm": alg, "Environment": env}

            for metric, values in raw_metrics.items():
                mean, std = compute_statistics(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std

            results.append(row)

    with open(results_file, mode="w", newline="") as csvfile:
        fieldnames = ["Algorithm", "Environment"]
        for metric in raw_metrics.keys():
            fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results successfully written to {results_file}")


if __name__ == "__main__":
    main()
