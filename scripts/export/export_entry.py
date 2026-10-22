#!/usr/bin/env python3
"""Overlap calibration only; serialize memory-intensive conversion/compilation."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

gate = os.environ.get("QNN_SNAPSHOT_EXPORT_GATE")
if gate:
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import llm_wrappers
    original = llm_wrappers.convert_pt2e
    waited = False
    def convert_after_gate(*args, **kwargs):
        global waited
        if not waited:
            print(f"[ExportGate] calibration complete; waiting for {gate}", flush=True)
            while True:
                status = json.loads(Path(gate).read_text())
                if status.get("status") in ("exported", "failed"):
                    break
                time.sleep(10)
            waited = True
            print("[ExportGate] previous export ended; conversion may proceed", flush=True)
        return original(*args, **kwargs)
    llm_wrappers.convert_pt2e = convert_after_gate
entry = "examples/qualcomm/oss_scripts/llama/llama.py"
sys.argv[0] = entry
runpy.run_path(entry, run_name="__main__")
