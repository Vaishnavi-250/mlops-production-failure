import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("==========================================")
print("MLOPS END-TO-END PIPELINE")
print("==========================================")

steps = [
    ("Model Training", [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py")
    ]),
    ("Data Drift Detection", [
        sys.executable,
        str(PROJECT_ROOT / "monitoring" / "drift_detector.py")
    ]),
    ("System Monitoring", [
        sys.executable,
        str(PROJECT_ROOT / "monitoring" / "system_monitor.py")
    ])
]

for step_name, command in steps:

    print("\n------------------------------------------")
    print(f"STEP: {step_name}")
    print("------------------------------------------")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"PIPELINE FAILED: {step_name}")
        sys.exit(1)

    print(f"{step_name} completed successfully.")

print("\n==========================================")
print("MLOPS PIPELINE COMPLETED SUCCESSFULLY")
print("==========================================")