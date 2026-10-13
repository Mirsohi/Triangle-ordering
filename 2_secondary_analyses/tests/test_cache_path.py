"""Exercise the non-local loading path of posthoc_eval.py (npz cache in <data>/part<P>/cache/, as written by
run_eval.make_file_loader via graphs.save_cache) on a COPY of the mock results, with the dense-edge SHA-256
injected into provenance as the main run records it. One graph gets a wrong SHA to check the refusal."""
import sys, json, shutil, hashlib, subprocess
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/claude/pc_eval'); sys.path.insert(0, '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data/src')
import numpy as np, graphs
M = Path('/home/claude/review1/mock'); T = M / 'cache_test'
if T.exists(): shutil.rmtree(T)
shutil.copytree(M / 'pc_eval_results' / 'local', T / 'results')
for part in ('A', 'B'):
    for rp in sorted((T / 'results' / f'part{part}').glob('*/result.json')):
        name = rp.parent.name
        z = np.load(M / 'npz' / f'{name}.npz'); e = z['edges'].astype(np.int64); n = int(z['n'])
        a, b = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
        edges = np.stack([a, b], 1).astype(np.int32)
        c = T / 'data' / f'part{part}' / 'cache' / f'{name}.npz'; c.parent.mkdir(parents=True, exist_ok=True)
        graphs.save_cache(c, edges, np.arange(n, dtype=np.int64), dict(source='test'))
        r = json.loads(rp.read_text())
        sha = hashlib.sha256(np.ascontiguousarray(edges, dtype=np.int32).tobytes()).hexdigest()
        if part == 'B' and name == 'P2P-Gnutella05': sha = '0' * 64      # must be refused
        if part == 'A' and name == 'Bitcoin-Alpha': sha = None            # absent SHA must be refused in non-local mode
        if sha is not None:
            r.setdefault('provenance', {})['dense_edges_sha256'] = sha
        rp.write_text(json.dumps(r))
cmd = [sys.executable, '/home/claude/review1/posthoc/posthoc_eval.py', '--pc-eval', '/home/claude/pc_eval',
       '--artifact', '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data', '--results', str(T / 'results'),
       '--data', str(T / 'data'), '--out', str(T / 'posthoc'), '--no-ils']
cp = subprocess.run(cmd, capture_output=True, text=True)
print(cp.stdout[-1500:]); print('exit code', cp.returncode)
out = {}
for p in sorted((T / 'posthoc').glob('part*/*/posthoc_result.json')):
    r = json.loads(p.read_text()); out[f'{p.parent.parent.name}/{p.parent.name}'] = dict(status=r['status'], checks=r['checks'], loader=r.get('provenance', {}).get('loader'))
mac = {l.split('{\\')[1].split('}')[0]: l.split('}{', 2)[-1][:-1] for l in (T / 'posthoc' / 'posthoc_macros.tex').read_text().splitlines()
       if l.startswith('\\providecommand')}
out['_exit_code'] = cp.returncode
out['_macros'] = {k: mac[k] for k in ('PHSuiteCountText', 'PHSuiteCheck', 'PHHeldCountText', 'PHHeldCheck')}
out['_pass'] = (cp.returncode == 2 and out['partA/Bitcoin-Alpha']['status'] == 'reload_check_failed'
                and out['partB/P2P-Gnutella05']['status'] == 'reload_check_failed'
                and sum(v['status'] == 'done' for k, v in out.items() if not k.startswith('_')) == 6
                and mac['PHHeldCheck'].startswith('failed or not analysed on 1 graph'))
print(json.dumps(out, indent=1))
json.dump(out, open('/home/claude/review1/posthoc/tests/test_cache_path.json', 'w'), indent=1)
