"""End-to-end test of the FULL (non-local) code paths of run_eval.py and make_tables.py without SNAP access.

A mock artifact (real src/dove, mock manifests / per_graph.csv / stored Neigh ranks) and a local HTTP server that
serves mock files under the pre-registered SNAP paths are built from the case graphs. PC_EVAL_TEST_BASEURL
redirects the SNAP prefix to the local server; results are flagged and make_tables refuses them without
--allow-test. Exercised paths: gzip / zip-member / tar downloads, CSV headers, comments, 3-column files, self
loops, duplicates, a missing zip member, a 404 with retries, a corrupt archive, a bipartite graph (R* work 0),
a manifest-hash warning, an n/m check failure (excluded), stored-Neigh checks, resume/skip, and the 10-graph
held-out table with missing rows.
Run: python tests/test_pipeline_mock.py <scratch dir>
"""
import sys, os, json, gzip, io, zipfile, tarfile, hashlib, shutil, subprocess, threading, functools
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
ART = Path('/home/claude/ideas/ref/code_data/DOVE_AISTATS_2027_code_and_data')
sys.path.insert(0, str(ART / 'src'))
CASES = Path('/home/claude/ideas/ref/research_20260927/run_20260927/cases')
import graphs  # noqa: E402
from dove.orderings import candidate_work, check_order  # noqa: E402


def case_edges(name):
    z = np.load(CASES / f'{name}.npz')
    return z['edges'].astype(np.int64)


def text(edges, style, rng):
    out = io.StringIO()
    if style == 'comments':
        out.write('# Directed graph\n# FromNodeId\tToNodeId\n')
    if style == 'csv':
        out.write('node_1,node_2\n')
    if style == 'csvid':
        out.write('id_1,id_2\n')
    for u, v in edges.tolist():
        if style in ('csv', 'csvid'):
            out.write(f'{u},{v}\n')
        elif style == 'temporal':
            out.write(f'{u} {v} {int(rng.integers(1e9))}\n')
            if rng.random() < 0.2:
                out.write(f'{v} {u} {int(rng.integers(1e9))}\n')
        else:
            out.write(f'{u}\t{v}\n')
    if style == 'temporal':
        out.write('5 5 1\n')
    return out.getvalue().encode()


def gz(b):
    return gzip.compress(b)


def main():
    scratch = Path(sys.argv[1]).resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    srv = scratch / 'server'; art = scratch / 'artifact'; srv.mkdir(parents=True); art.mkdir()
    rng = np.random.default_rng(3)
    (art / 'src').mkdir()
    os.symlink(ART / 'src' / 'dove', art / 'src' / 'dove')

    # ---------------- Part A mock suite
    suite = {  # name: (split, case graph, style)
        'Ego-1': ('Development', 'sbm-0', 'plain'),
        'CA-GrQc': ('Development', 'ba-1', 'comments'),
        'P2P-Mock': ('Validation', 'P2P-Gnutella05', 'plain'),
        'Audit-Mock': ('Audit', 'er-0', 'comments'),
        'Audit-BadN': ('Audit', 'ws-0', 'plain'),
    }
    files = {}
    for name, (split, case, style) in suite.items():
        files[name] = text(case_edges(case), style, rng)
    # facebook.tar.gz with facebook/1.edges
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode='w:gz') as t:
        data = files['Ego-1']; ti = tarfile.TarInfo('facebook/1.edges'); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
        d2 = b'1 2\n'; ti = tarfile.TarInfo('facebook/1.circles'); ti.size = len(d2); t.addfile(ti, io.BytesIO(d2))
    (srv / 'facebook.tar.gz').write_bytes(tbuf.getvalue())
    (srv / 'ca-GrQc.txt.gz').write_bytes(gz(files['CA-GrQc']))
    for name, fn in (('P2P-Mock', 'p2p-Mock.txt'), ('Audit-Mock', 'audit-mock.txt'), ('Audit-BadN', 'audit-badn.txt')):
        (srv / f'{fn}.gz').write_bytes(gz(files[name]))
    man = {'development_benchmark': 'name,family,path\nEgo-1,ego-social,../facebook_ego/dataset/raw/snap/facebook/1.edges\n'
                                    'CA-GrQc,coauthorship,../raw_graphs/CA-GrQc.txt\n'}
    def mrow(name, fn, wrong_sha=False):
        sha = hashlib.sha256((srv / f'{fn}.gz').read_bytes()).hexdigest()
        if wrong_sha:
            sha = '0' * 64
        return f'{name},x,raw/{fn},https://snap.stanford.edu/data/{fn}.gz,{sha},{len(files[name])}\n'
    hdr = 'name,family,path,source_url,compressed_sha256,edge_file_bytes\n'
    man['real_benchmark'] = hdr + mrow('P2P-Mock', 'p2p-Mock.txt')
    man['heldout_benchmark'] = hdr + mrow('Audit-Mock', 'audit-mock.txt', wrong_sha=True) + mrow('Audit-BadN', 'audit-badn.txt')
    for d, content in man.items():
        (art / 'data' / d).mkdir(parents=True)
        (art / 'data' / d / 'manifest.csv').write_text(content)
    # per_graph.csv + stored best-seed Neigh ranks (a Check order of the dense graph, reversed, stands in for Neigh)
    res = {'Development': 'final_development', 'Validation': 'final_validation', 'Audit': 'final_heldout'}
    rows = {v: ['graph,n,m,neigh_best_seed,neigh_best_work,reference_work'] for v in res.values()}
    for name, (split, case, style) in suite.items():
        tmp = scratch / f'{name}.txt'; tmp.write_bytes(files[name])
        e, ids = graphs.load_graph_file(tmp, name)
        g = graphs.make_graphdata(name, e, ids)
        if name == 'P2P-Mock':  # real stored Neigh rank of the revision study (ids are dense already)
            assert g.n == 8846 and np.array_equal(ids, np.arange(g.n))
            rank = np.loadtxt(ART / 'results/revision20260914/P2P-Gnutella05.neigh.3.rank', dtype=np.int64)
        else:
            o = np.asarray(check_order(g), np.int64)[::-1]
            rank = np.empty(g.n, np.int64); rank[o] = np.arange(g.n)
        seed = 3
        w = candidate_work(g, np.argsort(rank, kind='stable'))
        rd = art / 'results' / res[split] / 'ranks'; rd.mkdir(parents=True, exist_ok=True)
        np.savetxt(rd / f'{name}.neigh.seed-{seed}.rank', rank, fmt='%d')
        n_rec = g.n + (1 if name == 'Audit-BadN' else 0)
        rows[res[split]].append(f'{name},{n_rec},{g.m},{seed},{w},{w + 5}')
    for d, rr in rows.items():
        (art / 'results' / d / 'per_graph.csv').write_text('\n'.join(rr) + '\n')

    # ---------------- Part B mock held-out files under the pre-registered paths
    def zipf(path, member, data):
        with zipfile.ZipFile(path, 'w') as z:
            if member:
                z.writestr(member, data)
            z.writestr('x/README.txt', 'readme')
    (srv / 'email-Eu-core.txt.gz').write_bytes(gz(text(case_edges('er-1'), 'comments', rng)))
    (srv / 'CollegeMsg.txt.gz').write_bytes(gz(text(case_edges('ws-1'), 'temporal', rng)))
    zipf(srv / 'lastfm_asia.zip', 'lastfm_asia/lastfm_asia_edges.csv', text(case_edges('Bitcoin-Alpha'), 'csv', rng))
    zipf(srv / 'deezer_europe.zip', None, b'')   # member absent -> missing
    zipf(srv / 'facebook_large.zip', 'facebook_large/musae_facebook_edges.csv', text(case_edges('P2P-Gnutella05'), 'csvid', rng))
    (srv / 'amazon0302.txt.gz').write_bytes(gz(text(case_edges('powerlaw_cluster-1'), 'comments', rng)))
    # wiki-Talk.txt.gz absent -> 404 after 5 attempts -> missing (download failure, retried on resume)
    (srv / 'bigdata/communities').mkdir(parents=True)
    (srv / 'bigdata/communities/com-youtube.ungraph.txt.gz').write_bytes(b'not a gzip file')
    (srv / 'web-Google.txt.gz').write_bytes(gz(text(case_edges('core_periphery-1'), 'comments', rng)))
    (srv / 'roadNet-CA.txt.gz').write_bytes(gz(text(case_edges('prospective_bipartite_0'), 'plain', rng)))

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(srv))
    handler.log_message = lambda *a, **k: None
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'
    env = dict(os.environ, PC_EVAL_TEST_BASEURL=base)
    out = scratch / 'out'; data = scratch / 'data'
    cmd = [sys.executable, str(PKG / 'run_eval.py'), '--artifact', str(art), '--data', str(data), '--out', str(out), '--part', 'both']
    r1 = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (scratch / 'run1.log').write_text(r1.stdout + r1.stderr)
    assert r1.returncode == 0, r1.stderr[-3000:]
    # resume: done graphs skipped, the download failure retried
    r2 = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (scratch / 'run2.log').write_text(r2.stdout + r2.stderr)
    assert r2.returncode == 0, r2.stderr[-3000:]
    st = {}
    for part in 'AB':
        for p in sorted((out / f'part{part}').glob('*/result.json')):
            j = json.loads(p.read_text()); st[j['name']] = (j['status'], j.get('reason', '')[:120])
    print(json.dumps(st, indent=1))
    expect = {'Ego-1': 'done', 'CA-GrQc': 'done', 'P2P-Mock': 'done', 'Audit-Mock': 'done', 'Audit-BadN': 'excluded_check_failed',
              'Email-Eu-core': 'done', 'CollegeMsg': 'done', 'LastFM-Asia': 'done', 'Deezer-Europe': 'missing',
              'Facebook-page-page': 'done', 'Amazon0302': 'done', 'Wiki-Talk': 'missing', 'com-Youtube': 'missing',
              'web-Google': 'done', 'roadNet-CA': 'done'}
    bad = {k: (st.get(k), v) for k, v in expect.items() if st.get(k, ('?',))[0] != v}
    assert not bad, bad
    assert 'skipping' in r2.stdout and 'attempting again' in r2.stdout
    # make_tables refuses test data, then writes with --allow-test
    paper = scratch / 'paper'
    shutil.copytree('/home/claude/paper_new', paper, ignore=shutil.ignore_patterns('*.png', '*.aux', '*.log', '*.out', '*.blg', '*.pdf'))
    shutil.copy('/home/claude/paper_new/figures/fig1.pdf', paper / 'figures' / 'fig1.pdf')
    r3 = subprocess.run([sys.executable, str(PKG / 'make_tables.py'), '--out', str(out), '--paper', str(paper)], capture_output=True, text=True)
    assert r3.returncode != 0 and 'refusing' in (r3.stdout + r3.stderr)
    r4 = subprocess.run([sys.executable, str(PKG / 'make_tables.py'), '--out', str(out), '--paper', str(paper), '--allow-test'], capture_output=True, text=True)
    print(r4.stdout, r4.stderr)
    assert r4.returncode == 0
    # --quick smoke mode (synthetic Part B; mock Part A has none of the quick development graphs)
    r5 = subprocess.run(cmd + ['--quick'], env=env, capture_output=True, text=True)
    (scratch / 'run_quick.log').write_text(r5.stdout + r5.stderr)
    assert r5.returncode == 0, r5.stderr[-3000:]
    q = sorted(json.loads(p.read_text())['status'] for p in (out / 'quick' / 'partB').glob('*/result.json'))
    assert q == ['done', 'done', 'done'], q
    r6 = subprocess.run([sys.executable, str(PKG / 'make_tables.py'), '--out', str(out / 'quick'), '--paper', str(scratch / 'paper_q')], capture_output=True, text=True)
    assert r6.returncode != 0 and 'quick' in (r6.stdout + r6.stderr)
    httpd.shutdown()
    summary = dict(statuses=st, make_tables_refused_without_flag=True, run1_tail=r1.stdout[-1500:])
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'pipeline_mock.json').write_text(json.dumps(summary, indent=1))
    print('PIPELINE MOCK TEST PASSED')


if __name__ == '__main__':
    main()
