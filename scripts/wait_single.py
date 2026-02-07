"""Wait for a specific workflow execution ID."""
import time
from pathlib import Path
import sys

execution_id = "0ee098aa-bebe-4700-83b2-9fbd66cdd61d"
results_dir = Path("data/distributed_results")
timeout_minutes = 25
start_time = time.time()

print(f"⏳ Waiting for workflow: {execution_id[:8]}...")

while True:
    elapsed = time.time() - start_time

    # Check for result file
    result_files = list(results_dir.glob(f"*{execution_id}*.pkl"))
    result_files = [f for f in result_files if f.stat().st_size > 1000]

    if result_files:
        print(f"\n✅ Results received after {elapsed/60:.1f} minutes")
        print(f"   File: {result_files[0].name}")
        sys.exit(0)

    # Check for error
    error_files = list(results_dir.glob(f"error_{execution_id}*.json"))
    if error_files:
        print(f"\n❌ Workflow failed after {elapsed/60:.1f} minutes")
        print(f"   Error file: {error_files[0].name}")
        sys.exit(1)

    if elapsed > timeout_minutes * 60:
        print(f"\n⏰ Timeout after {timeout_minutes} minutes")
        sys.exit(1)

    time.sleep(10)
    mins = int(elapsed / 60)
    print(f"   Waiting... {mins}/{timeout_minutes} min", end='\r')
