"""
FactoryBench / ur5 / pick_and_place / run_shared.py
Thin wrapper: loads UR5 config and runs via the shared base_runner.
The original run.py is left completely untouched.

Usage:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur5/pick_and_place/run_shared.py [--headless] [--seed 0] [--episodes N] [--events]
"""
import sys
from pathlib import Path

_FACTORYBENCH = str(Path(__file__).parent.parent.parent.resolve())
if _FACTORYBENCH not in sys.path:
    sys.path.insert(0, _FACTORYBENCH)

from pick_and_place.base_runner import main

if __name__ == "__main__":
    config_path = str(Path(__file__).parent / "config" / "task_shared.yaml")
    main(config_path)
