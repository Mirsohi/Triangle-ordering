"""Failure reporting of the post-hoc package (audit M1), on a COPY of the mock pre-registered results.
S1  a pre-registered held-out graph recorded 'missing' (download failure, as run_eval writes it)  -> exit 0,
    \\PHHeldCountText "4 of 5 (Facebook-page-page: missing, download failed)", Check passed.
S2  additionally, one analysed graph raises an exception (its saved T order is deleted)            -> exit 2,
    Check reports the failure, CountText names the graph.
S3  the order file is restored and the run repeated                                               -> exit 0, the
    graph is retried and the stale posthoc_error.json is renamed.
Runs with --no-ils (colouring, checks, verification and timing only) for speed.
python3 tests/test_failures.py -> tests/test_failures.json
"""
import json, shutil, subprocess, sys
from pathlib import Path
M = Path('/home/claude/review1/mock'); T = M / 'failure_test'
PH = Path('/home/claude/review1/posthoc')
if T.exists(): shutil.rmtree(T)
shutil.copytree(M / 'pc_eval_results' / 'local', T / 'results')
plan = json.loads((T / 'results' / 'plan_partB.json').read_text())
plan['graphs'].insert(4, dict(name='Facebook-page-page', label='Facebook page--page', file=str(M / 'npz' / 'Facebook-page-page.npz')))
(T / 'results' / 'plan_partB.json').write_text(json.dumps(plan))
fb = T / 'results' / 'partB' / 'Facebook-page-page'; fb.mkdir(parents=True)
(fb / 'result.json').write_text(json.dumps(dict(name='Facebook-page-page', label='Facebook page--page', part='B',
    status='missing', reason='GraphMissing: download failed after 5 attempts: https://snap.stanford.edu/data/facebook_large.zip',
    retry_allowed=True)))


def run(out):
    cmd = [sys.executable, str(PH / 'posthoc_eval.py'), '--pc-eval', '/home/claude/pc_eval',
           '--artifact', '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data', '--results', str(T / 'results'),
           '--local-npz', str(M / 'npz'), '--out', str(out), '--no-ils']
    cp = subprocess.run(cmd, capture_output=True, text=True)
    mac = {}
    mf = out / 'posthoc_macros.tex'
    if mf.exists():
        for line in mf.read_text().splitlines():
            if line.startswith('\\providecommand'):
                k = line.split('{\\')[1].split('}')[0]; v = line.split('\\renewcommand{\\' + k + '}{', 1)[1][:-1]
                mac[k] = v
    return cp.returncode, {k: mac.get(k) for k in ('PHHeldCount', 'PHHeldPlanned', 'PHHeldCountText', 'PHHeldNotAnalysed',
                                                   'PHHeldCheck', 'PHSuiteCountText', 'PHSuiteCheck', 'PHAllCountText')}, cp.stdout[-600:]


res = {}
rc, mac, _ = run(T / 'out1'); res['S1'] = dict(rc=rc, macros=mac)
ok1 = rc == 0 and mac['PHHeldCountText'] == '4 of 5 (Facebook-page-page: missing, download failed)' and mac['PHHeldCheck'].startswith('passed')
tfile = T / 'results' / 'partB' / 'GitHub' / 'orders' / 'T.npy'; shutil.move(tfile, T / 'T.npy.bak')
rc, mac, _ = run(T / 'out2'); res['S2'] = dict(rc=rc, macros=mac)
ok2 = rc == 2 and mac['PHHeldCheck'].startswith('failed or not analysed on 1 graph') and "GitHub: post-hoc error, FileNotFoundError" in mac['PHHeldCountText'] \
    and mac['PHHeldCount'] == '3' and mac['PHSuiteCheck'].startswith('passed')
shutil.move(T / 'T.npy.bak', tfile)
rc, mac, _ = run(T / 'out2'); res['S3'] = dict(rc=rc, macros=mac,
    superseded=[p.name for p in (T / 'out2' / 'partB' / 'GitHub').glob('posthoc_error*.json')])
ok3 = rc == 0 and mac['PHHeldCount'] == '4' and not (T / 'out2' / 'partB' / 'GitHub' / 'posthoc_error.json').exists() \
    and len(res['S3']['superseded']) == 1
res['pass'] = dict(S1=ok1, S2=ok2, S3=ok3)
print(json.dumps(res, indent=1))
(PH / 'tests' / 'test_failures.json').write_text(json.dumps(res, indent=1))
