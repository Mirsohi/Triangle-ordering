"""Plumbing test of --volt (native path and wsl:<path>) with a FAKE VOLT (writes a seeded random rank
permutation) and a fake wsl.exe on PATH. Checks that Neigh candidates enter R* (both directions), that a failing
seed is recorded, and that the command line matches the artifact's run_neigh().
Run: python tests/test_volt_plumbing.py <scratch dir>"""
import sys, os, json, subprocess, shutil, stat
from pathlib import Path
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FAKE = r'''#!/usr/bin/env python3
import sys, numpy as np
a = sys.argv[1:]
edge = a[0]; seed = int(a[a.index('--seed') + 1]); out = a[a.index('--rank-output') + 1]
assert a[1:3] == ['-o', 'neigh'] and a[3:5] == ['-n', '0.01'] and a[-3:] == ['--order-only', '-p', '1'], a
if seed == 4: sys.exit(3)            # simulate a failing seed
e = np.loadtxt(edge, dtype=np.int64).reshape(-1, 2); n = int(e.max()) + 1
np.savetxt(out, np.random.default_rng(seed).permutation(n), fmt='%d')
'''
WSL = r'''#!/usr/bin/env python3
import sys, os
a = sys.argv[1:]
if a[:2] == ['wslpath', '-a']: print('/fake' + a[2]); sys.exit(0)
assert a[0] == '-e'
args = [x[5:] if x.startswith('/fake/') else x for x in a[1:]]
os.execv(sys.executable, [sys.executable] + args)
'''
def main():
    sc = Path(sys.argv[1]).resolve()
    if sc.exists(): shutil.rmtree(sc)
    (sc / 'bin').mkdir(parents=True); (sc / 'npz').mkdir()
    for name, body in (('fakevolt', FAKE), ('wsl.exe', WSL)):
        p = sc / 'bin' / name; p.write_text(body); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    for g in ('ba-1', 'er-1'):
        os.symlink(f'/home/claude/ideas/ref/research_20260927/run_20260927/cases/{g}.npz', sc / 'npz' / f'{g}.npz')
    env = dict(os.environ, PATH=str(sc / 'bin') + os.pathsep + os.environ['PATH'])
    res = {}
    for mode, volt in (('native', str(sc / 'bin' / 'fakevolt')), ('wsl', 'wsl:' + str(sc / 'bin' / 'fakevolt'))):
        out = sc / f'out_{mode}'
        cmd = [sys.executable, str(PKG / 'run_eval.py'), '--artifact', '/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data',
               '--data', str(sc / 'data'), '--out', str(out), '--part', 'B', '--local-npz', str(sc / 'npz'), '--quick', '--volt', volt]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        for p in sorted((out / 'local_quick' / 'partB').glob('*/result.json')):
            j = json.loads(p.read_text())
            works = j['reference']['candidate_works']
            ok = [v for v in j['checks']['volt'] if v['ok']]
            assert len(ok) == 4 and not [v for v in j['checks']['volt'] if v['seed'] == 4][0]['ok']
            assert all(f'Neigh{s}' in works and f'Neigh{s}-rev' in works for s in (1, 2, 3, 5)) and 'Neigh4' not in works
            res[f'{mode}:{j["name"]}'] = dict(neigh=j['reference']['neigh_candidates'], R=j['reference']['label'])
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'volt_plumbing.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)); print('VOLT PLUMBING TEST PASSED')
main()
