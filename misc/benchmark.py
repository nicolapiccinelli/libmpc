import subprocess
import psutil
import time
import os
import sys
import statistics

# Check if we are sudo user otherwise exit
if os.geteuid() != 0:
    print("Please run this script as root.")
    exit()

# get the path to the executable directory from the command line argument
if len(sys.argv) != 2:
    print("Usage: python benchmark.py <path to executable directory>")
    exit()

executable_dir = sys.argv[1]

# Configurable list of executables and arguments
executables = [
    {"path": "quadrotor_ex", "args": []},
    {"path": "vanderpol_ex", "args": []},
    {"path": "networked_oscillators_ex", "args": []},
    {"path": "ugv_ex", "args": []},
]

# Update the paths to the executables
for exe in executables:
    exe["path"] = os.path.join(executable_dir, exe["path"])

# Number of runs for each executable
num_runs = 5

# Results log file
results_file = "results.csv"

# Time interval for monitoring memory usage
poll_stats_time_sec = 0.001

def measure_executable(exec_info):
    path = exec_info["path"]
    args = exec_info["args"]
    command = [path] + args

    print(f"Starting benchmark for: {path}")
    start_time = time.time()
    
    # try:
    # Start the process
    process = psutil.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Process started: PID={process.pid}")

    peak_memory = 0
    total_read_bytes = 0
    total_write_bytes = 0
    thread_count = 0
    ctx_switches_voluntary = 0
    ctx_switches_involuntary = 0

    # Monitor the process and check if it has completed
    while process.poll() is None:
        try:
            # Monitor peak memory in bytes
            mem_info = process.memory_info()
            peak_memory = max(peak_memory, mem_info.rss)

            # Monitor IO counters
            io_counters = process.io_counters()
            total_read_bytes = io_counters.read_bytes
            total_write_bytes = io_counters.write_bytes

            # Monitor thread count
            thread_count = max(thread_count, process.num_threads())

            # Monitor context switches
            ctx_switches = process.num_ctx_switches()
            ctx_switches_voluntary = ctx_switches.voluntary
            ctx_switches_involuntary = ctx_switches.involuntary

        except psutil.NoSuchProcess:
            print("Process terminated before monitoring could complete.")
            break
        time.sleep(poll_stats_time_sec)

    # Wait for process to finish
    stdout, stderr = process.communicate()
    print(f"Process finished with return code: {process.returncode}")

    if stderr:
        print(f"Error output:\n{stderr.decode()}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Benchmark completed for: {path}")
    print(f"Execution Time: {execution_time:.2f}s, Peak Memory: {peak_memory / (1024 ** 2):.2f} MB")

    # Convert to MB
    return {
        "command": " ".join(command),
        "execution_time": execution_time,
        "peak_memory": peak_memory / (1024 ** 2),
        "total_read_bytes": total_read_bytes / (1024 ** 2),
        "total_write_bytes": total_write_bytes / (1024 ** 2),
        "max_threads": thread_count,
        "ctx_switches_voluntary": ctx_switches_voluntary,
        "ctx_switches_involuntary": ctx_switches_involuntary,
    }

# Run tests and collect results
results = {}
for exe in executables:
    exe_path = exe["path"]
    results[exe_path] = {
        "execution_times": [],
        "peak_memories": [],
        "read_bytes": [],
        "write_bytes": [],
        "thread_counts": [],
        "voluntary_ctx_switches": [],
        "involuntary_ctx_switches": [],
    }

    print(f"Starting test for executable: {exe_path}")
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        result = measure_executable(exe)
        if "error" in result:
            print(f"Error during benchmarking of {exe_path}: {result['error']}")
            break
        else:
            results[exe_path]["execution_times"].append(result["execution_time"])
            results[exe_path]["peak_memories"].append(result["peak_memory"])
            results[exe_path]["read_bytes"].append(result["total_read_bytes"])
            results[exe_path]["write_bytes"].append(result["total_write_bytes"])
            results[exe_path]["thread_counts"].append(result["max_threads"])
            results[exe_path]["voluntary_ctx_switches"].append(result["ctx_switches_voluntary"])
            results[exe_path]["involuntary_ctx_switches"].append(result["ctx_switches_involuntary"])

    print(f"Test completed for: {exe_path}\n")

# Calculate stats and log results
with open(results_file, "w") as f:
    f.write(
        "Command,Mean Execution Time (s),Stddev Execution Time (s),"
        "Mean Peak Memory (MB),Stddev Peak Memory (MB),"
        "Mean Read Bytes (MB),Mean Write Bytes (MB),"
        "Max Threads,Mean Voluntary Context Switches,"
        "Mean Involuntary Context Switches\n"
    )
    for exe_path, stats in results.items():
        if stats["execution_times"]:
            mean_exec_time = statistics.mean(stats["execution_times"])
            stddev_exec_time = statistics.stdev(stats["execution_times"])
            mean_peak_memory = statistics.mean(stats["peak_memories"])
            stddev_peak_memory = statistics.stdev(stats["peak_memories"])
            mean_read_bytes = statistics.mean(stats["read_bytes"])
            mean_write_bytes = statistics.mean(stats["write_bytes"])
            max_threads = max(stats["thread_counts"])
            mean_voluntary_ctx_switches = statistics.mean(stats["voluntary_ctx_switches"])
            mean_involuntary_ctx_switches = statistics.mean(stats["involuntary_ctx_switches"])
            
            f.write(
                f"{exe_path},{mean_exec_time},{stddev_exec_time},"
                f"{mean_peak_memory},{stddev_peak_memory},"
                f"{mean_read_bytes},{mean_write_bytes},"
                f"{max_threads},{mean_voluntary_ctx_switches},"
                f"{mean_involuntary_ctx_switches}\n"
            )
            
            print(f"Stats for {exe_path}:")
            print(f"\tMean Execution Time: {mean_exec_time:.2f}s")
            print(f"\tStddev Execution Time: {stddev_exec_time:.2f}s")
            print(f"\tMean Peak Memory: {mean_peak_memory:.2f} MB")
            print(f"\tStddev Peak Memory: {stddev_peak_memory:.2f} MB")
            print(f"\tMean Read Bytes: {mean_read_bytes:.2f} MB")
            print(f"\tMean Write Bytes: {mean_write_bytes:.2f} MB")
            print(f"\tMax Threads: {max_threads}")
            print(f"\tMean Voluntary Context Switches: {mean_voluntary_ctx_switches:.2f}")
            print(f"\tMean Involuntary Context Switches: {mean_involuntary_ctx_switches:.2f}")
        else:
            f.write(f"{exe_path},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR\n")
            print(f"Could not collect stats for {exe_path} due to errors.")

print(f"All tests completed. Results saved to {results_file}")
