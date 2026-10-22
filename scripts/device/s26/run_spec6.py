# S26 Ultra DFlash run on six Spec-Bench prompts, first turn only (results/s26_spec6_20261016): 81 writing, 101 reasoning,
# 121 coding, 141 stem, 161 translation, 401 math_reasoning. Final config only: K16on = ckpt10 v5 export with the row-10
# checkpoint (R16v5 / K16off give identical tokens). Same protocol as run_v5.py; prompts are read from {PH}/inputs_spec6
# (make_spec6_inputs.py) and logs written to {PH}/logs_spec6.
# usage: S26_SERIAL=<adb serial> [ADB=<adb binary>] [RESULTS_DIR=<folder>] python3 run_spec6.py TAG K16on [IDS]
import os, json, re, sys, time, importlib.util
from pathlib import Path
os.environ['QNN_BENCH_NO_CHARGE_WAIT'] = '1'
os.environ['QNN_BENCH_CONTINUOUS'] = '0'
REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('RESULTS_DIR', REPO / 'results/s26_spec6_20261016'))
OUT.mkdir(exist_ok=True)
PH = '/data/local/tmp/whdgns/q35_s26_20261012'
LIBV1P = f'{PH}/lib_v1p'
LIBSNAP = f'{PH}/lib_snap'
QNNLIB = f'{PH}/qnnlib'
PIN = 'QNN_SNAPSHOT_PIN_SHARDS=0,1,2,3,4,5,6,7,8 QNN_SNAPSHOT_PIN_BUDGET_MIB=512 '
CFG = {
    'K16on': (16, f'{PH}/ctx_ckpt10v5_w16', LIBV1P, 'QNN_DEFER_CHECKPOINT_ROW=10 QNN_ZERO_COPY=1 ' + PIN),
}
RUNNER = f'{PH}/runner_ckpt10'
PIN_IN = f'{PH}/inputs_spec6'
LOGS = f'{PH}/logs_spec6'
tag, cfgs = sys.argv[1], sys.argv[2].split(',')
spec = importlib.util.spec_from_file_location('paired', Path(__file__).resolve().parents[1] / 'bench_lib.py')
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
lib.OUT = OUT
lib.ADB = [os.environ.get('ADB', 'adb'), '-s', os.environ['S26_SERIAL']]   # S26 Ultra (SM8850)
x = lib.Experiment()
ids = [int(i) for i in sys.argv[3].split(',')] if len(sys.argv) > 3 else [81, 101, 121, 141, 161, 401]
for c in cfgs:
    ctx = CFG[c][1]
    if lib.shell(f"[ -f {ctx}/params.json ] && [ -f {ctx}/forward_8.bin ] && [ -x {RUNNER} ] && echo ok || echo missing") != 'ok':
        raise SystemExit(f'{c}: model {ctx} or runner not on the phone')
lib.shell(f'mkdir -p {LOGS}')


def command(cfg, qid):
    w, ctx, opp, extra = CFG[cfg]
    draft = f'{PH}/ctx_draft_b{w}'
    return ("timeout 900 /system/bin/time -v env QNN_PROF_BASIC=1 QNN_ARGMAX_U16=1 "
            f"'ADSP_LIBRARY_PATH={opp};{QNNLIB}' QNN_OP_PACKAGES={opp}/libQnnDeltaRuleOpPackage_HTP.so:DeltaRuleOpPackageInterfaceProvider:HTP "
            + extra +
            f"LD_LIBRARY_PATH={PH}/lib:{QNNLIB} "
            f"DFLASH_CHAT=1 DFLASH_THINK=1 DFLASH_BLOCK={w} DFLASH_DRAFT_CTX={draft} DFLASH_DRAFT_PARAMS={draft}/params.json "
            f"DFLASH_DRAFT_EMBD={PH}/draft_q35_embd_f16.bin DFLASH_NINJECT=16 DFLASH_NHIDDEN=8 DFLASH_MASKID=248077 "
            f"{RUNNER} --qnn --multi-context --ctx-dir {ctx} --params {ctx}/params.json "
            f"--tokenizer {PH}/Qwen3.5-4B-q4_0-outq4.gguf --log-level 1 -n 128 -c 1024 --temp 0.0 "
            f"-f {PIN_IN}/prompt_{qid}.txt")


for k, qid in enumerate(ids):
    order = cfgs[k % len(cfgs):] + cfgs[:k % len(cfgs)]
    for cfg in order:
        name = f'{tag}_{cfg}_{qid}'
        dest = OUT / f'{name}_run.json'
        if dest.exists():
            continue
        before = x.cool()
        print('START', name, before, flush=True)
        x.update(status='running', current=f'{name}_run', thermal=before)
        remote = f'{LOGS}/{name}.log'
        cmd = command(cfg, qid) + ' > ' + remote + ' 2>&1; echo RUN_EXIT=$?'
        (OUT / f'{name}.command.txt').write_text(cmd + '\n')
        t0 = time.time()
        result = lib.shell(cmd, timeout=930)
        x.last_end = time.monotonic()
        x.check_usb()
        log = lib.shell('cat ' + remote, timeout=90)
        (OUT / f'{name}.log').write_text(log)
        ec = int(re.search(r'RUN_EXIT=(\d+)', result)[1])
        ids_out, complete = lib.PARSER.tokdbg(log)
        m = lib.PARSER.row(f'exit={ec}\n' + log, 'sd', x.prompts[qid]['expected_n_past'])
        hp = re.search(r'\[HOSTPATH\] mode=(\S+) per step: target logits ([\d.]+) ms, draft logits ([\d.]+) ms, accept argmax ([\d.]+) ms', log)
        m['hostpath'] = {'mode': hp[1], 'target_ms': float(hp[2]), 'draft_ms': float(hp[3]), 'accept_ms': float(hp[4])} if hp else None
        r = {'config': cfg, 'id': qid, 'exit_code': ec, 'metrics': m, 'token_ids': ids_out, 'tokdbg_complete': complete,
             'thermal_before': before, 'thermal_after': x.thermal(), 'wall_seconds': time.time() - t0,
             'started_kst': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}
        dest.write_text(json.dumps(r, indent=2))
        x.update(status='resting_between_runs', current=f'{name}_run')
        print('DONE', name, json.dumps(m), flush=True)
        if ec:
            raise RuntimeError(name + ' failed: ' + log[-2000:])
x.update(status='complete', tag=tag)
print('ALL_DONE', tag, flush=True)
