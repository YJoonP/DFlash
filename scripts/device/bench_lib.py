#!/usr/bin/env python3
"""Device-side helpers of the measurement scripts (run_*.py): adb shell, thermal / USB checks, rest between runs.

The run scripts set OUT (results folder, holding inputs/manifest.json) and ADB before creating Experiment().
cool() waits until at least INTER_RUN_REST_SECONDS have passed since the last run and then until two consecutive
readings show SoC < 38 C, battery < 33 C and no other model process on the phone. A USB disconnect event stops the
session (no reconnect). ANDROID_ADB_SERVER_PORT is taken from the environment if set.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

OUT = None                                     # set by the run script
ADB = [os.environ.get('ADB', 'adb')]           # the run script appends '-s', <serial>
ENV = dict(os.environ)
CONTINUOUS = os.environ.get('QNN_BENCH_CONTINUOUS') == '1'
WAIT_FOR_CHARGE = not CONTINUOUS and os.environ.get('QNN_BENCH_NO_CHARGE_WAIT') != '1'
INTER_RUN_REST_SECONDS = 0 if CONTINUOUS else 60
SPEC = importlib.util.spec_from_file_location('dflash_parser', Path(__file__).resolve().parent / 'dflash_parse.py')
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)


def shell(command, timeout=45):
    return subprocess.check_output(ADB + ['shell', command], env=ENV, timeout=timeout,
                                   stderr=subprocess.STDOUT).decode(errors='replace').strip()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Experiment:
    def __init__(self):
        self.status = {'status': 'starting', 'pid': os.getpid(), 'thinking': True,
                       'started': time.strftime('%Y-%m-%d %H:%M:%S')}
        self.last_end = 0.0
        previous_status = OUT / 'measurement_status.json'
        previous = json.loads(previous_status.read_text()) if previous_status.exists() else {}
        self.charge_hold = WAIT_FOR_CHARGE and previous.get('status') == 'charging'
        if previous.get('status') == 'resting_between_runs':
            recent_result = OUT / f"{previous.get('current')}.json"
            if recent_result.exists():
                elapsed = max(0, time.time() - recent_result.stat().st_mtime)
                self.last_end = time.monotonic() - elapsed
            if WAIT_FOR_CHARGE and previous.get('thermal', {}).get('battery_pct', 100) <= 10:
                self.charge_hold = True
        self.usb_command = "dumpsys usb | grep 'USB_STATE=DISCONNECTED' | tail -1"
        self.usb = shell(self.usb_command)
        self.manifest = json.loads((OUT / 'inputs/manifest.json').read_text())
        self.prompts = {r['id']: r for r in self.manifest}

    def update(self, **kwargs):
        self.status.update(kwargs)
        save(OUT / 'measurement_status.json', self.status)

    def check_usb(self):
        if shell(self.usb_command) != self.usb:
            raise RuntimeError('USB disconnect detected; stopped without reconnecting')

    def thermal(self):
        self.check_usb()
        text = shell("cat /sys/class/thermal/thermal_zone0/temp; dumpsys battery | grep -E '^  (level|temperature|Charge counter):'; ps -A -o NAME | grep -E '^(llama-|qnn_|decode_runner|simpleperf)' || true")
        lines = text.splitlines()
        level = int(re.search(r'^  level: (\d+)', text, re.M)[1])
        battery = int(re.search(r'^  temperature: (\d+)', text, re.M)[1])
        busy = [s for s in lines if re.match(r'^(llama-|qnn_|decode_runner|simpleperf)', s)]
        charge = re.search(r'^  Charge counter: (\d+)', text, re.M)
        return {'soc_mC': int(lines[0]), 'battery_dC': battery, 'battery_pct': level,
                'charge_counter_uAh': int(charge[1]) if charge else None, 'busy': busy}

    def cool(self):
        if CONTINUOUS:
            self.charge_hold = False
            while True:
                t = self.thermal()
                if not t['busy']:
                    return t
                self.update(status='waiting_for_other_device_process', thermal=t, continuous=True)
                print('DEVICE_BUSY', t, flush=True)
                time.sleep(5)
        remaining = INTER_RUN_REST_SECONDS - (time.monotonic() - self.last_end)
        while self.last_end and remaining > 0:
            t = self.thermal()
            if WAIT_FOR_CHARGE and t['battery_pct'] <= 10:
                self.charge_hold = True
            self.update(status='resting_between_runs', thermal=t,
                        rest_remaining_seconds=round(remaining),
                        minimum_inter_run_rest_seconds=INTER_RUN_REST_SECONDS)
            print('REST_AND_CHARGE', round(remaining), t, flush=True)
            time.sleep(min(30, remaining))
            remaining = INTER_RUN_REST_SECONDS - (time.monotonic() - self.last_end)
        self.status.pop('rest_remaining_seconds', None)
        stable = 0
        while stable < 2:
            t = self.thermal()
            if WAIT_FOR_CHARGE and t['battery_pct'] <= 10:
                self.charge_hold = True
            if WAIT_FOR_CHARGE and self.charge_hold and t['battery_pct'] < 15:
                stable = 0
                self.update(status='charging', thermal=t, pause_battery_pct=10, resume_battery_pct=15)
                print('CHARGE_WAIT', t, flush=True)
                time.sleep(60)
                continue
            self.charge_hold = False
            good = t['soc_mC'] < 38000 and t['battery_dC'] < 330 and not t['busy']
            stable = stable + 1 if good else 0
            if not good:
                self.update(status='cooling_or_device_busy', thermal=t)
                print('WAIT', t, flush=True)
            if stable < 2:
                time.sleep(5 if good else 20)
        return t
