"""Wait for specific workflow execution IDs to complete."""
import time
from pathlib import Path
import sys

execution_ids = [
    "6985ee39-118e-4ca7-a983-2cf451f5e05d",
    "d0ca1c04-ff73-4363-908c-e5c17874069c",
    "7a81ac63-93f7-4a29-b98b-31501a6cb29f"
]

results_dir = Path("data/distributed_results")
timeout_minutes = 15
start_time = time.time()

print("⏳ Waiting for 3 workflows to complete...")
print(f"   Execution IDs:")
for eid in execution_ids:
    print(f"     - {eid}")

found_results = {}
while len(found_results) < len(execution_ids):
    elapsed = time.time() - start_time

    # Check for result files
    for eid in execution_ids:
        if eid not in found_results:
            result_files = list(results_dir.glob(f"*{eid}*.pkl"))
            result_files = [f for f in result_files if f.stat().st_size > 1000]

            if result_files:
                found_results[eid] = result_files[0]
                print(f"\n✅ Results received for {eid[:8]}...")
                print(f"   File: {result_files[0].name}")

    if len(found_results) == len(execution_ids):
        break

    if elapsed > timeout_minutes * 60:
        print(f"\n⏰ Timeout after {timeout_minutes} minutes")
        print(f"   Found {len(found_results)}/{len(execution_ids)} results")
        sys.exit(1)

    # Check every 10 seconds
    time.sleep(10)
    remaining = len(execution_ids) - len(found_results)
    print(f"   Waiting for {remaining} results... ({elapsed/60:.1f} min elapsed)", end='\r')

print(f"\n\n✅ All 3 workflows completed after {elapsed/60:.1f} minutes")
print(f"\nResult files:")
for eid, result_file in found_results.items():
    print(f"  - {result_file.name}")
