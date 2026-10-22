# S25 version of s26/run_v5.py (results/s25_vs_old_adopted_20261014): same runner and protocol, S25 paths (v79 op package,
# the S25 b = 16 draft). Configs: O16 old adopted export (10-02), C16 exact conv export (10-11), R16v5 plain recompute v5,
# K16on / K16off ckpt10 v5 with / without the checkpoint.
# Configs alternate in order per prompt (the first config rotates); before each run at least 60 s rest and two readings
# with SoC < 38 C and battery < 33 C (bench_lib.py). Each run: thinking chat template, greedy, 128 new tokens, context 1024;
# writes <TAG>_<CFG>_<id>_run.json (metrics, token ids, temperatures), .command.txt (the exact phone command) and .log.
# Existing *_run.json are skipped, so a new session needs its own RESULTS_DIR holding inputs/ (prompt_<id>.txt, manifest.json).
# Phone paths (PH ...) are the folders of our runs: copy ctx_target/, ctx_draft/, the op package, the QNN libraries, the
# runner and the prompts there, or edit them.
# usage: S25_SERIAL=<adb serial> [ADB=<adb binary>] [RESULTS_DIR=<folder>] python3 run_s25v5.py TAG CFG[,CFG...] [IDS]
import os, json, re, sys, time, importlib.util
from pathlib import Path
os.environ['QNN_BENCH_NO_CHARGE_WAIT'] = '1'
os.environ['QNN_BENCH_CONTINUOUS'] = '0'
REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('RESULTS_DIR', REPO / 'results/s25_vs_old_adopted_20261014'))
OUT.mkdir(exist_ok=True)
PH = '/data/local/tmp/whdgns'
OG = f'{PH}/q35_onegrid_20261013'
OPP = f'{PH}/q35_ckpt5_w8_20261010/lib'          # adopted v1p op package (v79), as in every S25 run since 10-10
QNNLIB = f'{PH}/q35_dflash16/qnnlib'
DEV8 = f'{PH}/q35_snapshot_w8_20261010'          # prompts, draft embedding, runner libs
DEV16 = f'{PH}/q35_snapshot_w16pin_20261010'     # b16 draft
PIN = 'QNN_SNAPSHOT_PIN_SHARDS=0,1,2,3,4,5,6,7,8 QNN_SNAPSHOT_PIN_BUDGET_MIB=512 '
CFG = {
    'O16': (16, f'{PH}/q35_npuab_20261011/ctx_target_old_w16', OPP, 'QNN_ZERO_COPY=1 ' + PIN),
    'C16': (16, f'{PH}/q35_exactconv_20261011/ctx_target_w16', OPP, 'QNN_ZERO_COPY=1 ' + PIN),
    'R16v5': (16, f'{OG}/ctx_target_v5_w16', OPP, 'QNN_ZERO_COPY=1 ' + PIN),
    'K16on': (16, f'{OG}/ctx_target_ckpt10v5_w16', OPP, 'QNN_DEFER_CHECKPOINT_ROW=10 QNN_ZERO_COPY=1 ' + PIN),
    'K16off': (16, f'{OG}/ctx_target_ckpt10v5_w16', OPP, 'QNN_ZERO_COPY=1 ' + PIN),
}
RUNNER = f'{OG}/runner_ckpt10'
tag, cfgs = sys.argv[1], sys.argv[2].split(',')
spec = importlib.util.spec_from_file_location('paired', Path(__file__).resolve().parents[1] / 'bench_lib.py')
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
lib.OUT = OUT
lib.ADB = [os.environ.get('ADB', 'adb'), '-s', os.environ['S25_SERIAL']]   # S25 (SM8750)
x = lib.Experiment()
ids = [int(i) for i in sys.argv[3].split(',')] if len(sys.argv) > 3 else [81, 82, 83, 161, 162, 163, 241, 242, 243, 321, 322, 323, 401, 402, 403, 481, 482, 483]
for c in cfgs:
    ctx = CFG[c][1]
    if lib.shell(f"[ -f {ctx}/params.json ] && [ -f {ctx}/forward_8.bin ] && [ -x {RUNNER} ] && echo ok || echo missing") != 'ok':
        raise SystemExit(f'{c}: model {ctx} or runner not on the phone')
lib.shell(f'mkdir -p {OG}/logs')


def command(cfg, qid):
    w, ctx, opp, extra = CFG[cfg]
    draft = f'{DEV16}/ctx_draft'
    return ("timeout 900 /system/bin/time -v env QNN_PROF_BASIC=1 QNN_ARGMAX_U16=1 "
            f"'ADSP_LIBRARY_PATH={opp};{QNNLIB}' QNN_OP_PACKAGES={opp}/libQnnDeltaRuleOpPackage_HTP.so:DeltaRuleOpPackageInterfaceProvider:HTP "
            + extra +
            f"LD_LIBRARY_PATH={OG}/lib_ckpt10:{DEV8}/lib:{QNNLIB}:{PH}/q35_dflash16 "
            f"DFLASH_CHAT=1 DFLASH_THINK=1 DFLASH_BLOCK={w} DFLASH_DRAFT_CTX={draft} DFLASH_DRAFT_PARAMS={draft}/params.json "
            f"DFLASH_DRAFT_EMBD={DEV8}/draft_q35_embd_f16.bin DFLASH_NINJECT=16 DFLASH_NHIDDEN=8 DFLASH_MASKID=248077 "
            f"{RUNNER} --qnn --multi-context --ctx-dir {ctx} --params {ctx}/params.json "
            f"--tokenizer /data/local/tmp/dflash/Qwen3.5-4B-q4_0-outq4.gguf --log-level 1 -n 128 -c 1024 --temp 0.0 "
            f"-f {DEV8}/inputs/prompt_{qid}.txt")


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
        remote = f'{OG}/logs/{name}.log'
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
