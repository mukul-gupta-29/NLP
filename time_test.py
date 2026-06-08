# time_test.py

import time
import subprocess
import sys

print("=" * 60)
print("Running IR System with Time Measurement")
print("=" * 60)

# --- Total system time ---
total_start = time.time()

result = subprocess.run(
    [sys.executable, "main.py",
     "-dataset", "cranfield/",
     "-out_folder", "output/"],
    capture_output=False   # shows output live in terminal
)

total_end = time.time()

print("\n" + "=" * 60)
print(f"  Total Run Time : {total_end - total_start:.4f} seconds")
print("=" * 60)