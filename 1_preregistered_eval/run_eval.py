#!/usr/bin/env python3
"""Pre-registered evaluation runner (see PREREGISTRATION.md and README_RUN.md).

python run_eval.py --artifact <artifact dir> --data <download dir> --out <results dir>
                   [--part A|B|both] [--volt <exe>] [--quick] [--local-npz <dir>]

Part A: rebased 32-graph suite of the artifact.  Part B: the ten pre-registered held-out graphs.
Resumable: a graph whose result.json exists is skipped; results are written after each graph.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

EXPECTED_PREREG_SHA256 = 'bdda241b6d7d9f13fe0a0b643f56ed2c0c229dc596ea283edc23f28c50a7ed4b'
# artifact src/dove files imported by this package (dove/__init__ imports graph, orderings, relaxation)
EXPECTED_DOVE_SHA256 = {
    '__init__.py': 'bca61e0d0a9b689c3bf16910b10fcb6b6f1b5d2fdf7945e6a5b50de58b66136e',
    'graph.py': '665b789424600b50d8642db0d0b2c6587cb5a602af37d1abd167825b9a8615c9',
    'orderings.py': 'a27d2278cde6a2e84639977e0b9a47a66900b3479e2deea38a8a9d7420b13e32',
    'relaxation.py': 'c1930feb600560f96f781ed3e6e1c5350ed8b977f5f99e62a4c8489d989c129e',
}
SEEDS = (1, 2, 3)
QUICK_PART_A = ('Ego-3980', 'Ego-698', 'Ego-414')


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file_lf(path):
    """(raw sha, LF-normalised sha)."""
    b = Path(path).read_bytes()
    return sha_bytes(b), sha_bytes(b.replace(b'\r\n', b'\n'))


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f'[{utcnow()}] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding='utf-8')
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------ checks
def check_prereg(log):
    p = HERE / 'PREREGISTRATION.md'
    raw, lf = sha_file_lf(p)
    stored = (HERE / 'PREREGISTRATION.sha256').read_text(encoding='utf-8').split()[0].strip()
    if stored != EXPECTED_PREREG_SHA256:
        raise SystemExit(f'PREREGISTRATION.sha256 ({stored}) differs from the hash compiled into run_eval.py; refusing Part B')
    if raw == EXPECTED_PREREG_SHA256:
        log(f'pre-registration hash OK: {raw}')
        return dict(sha256=raw, normalised=False)
    if lf == EXPECTED_PREREG_SHA256:
        log(f'pre-registration hash OK after CRLF->LF normalisation (raw {raw}); content unchanged')
        return dict(sha256=lf, raw_sha256=raw, normalised=True)
    raise SystemExit(f'PREREGISTRATION.md hash {raw} != recorded {EXPECTED_PREREG_SHA256}: the protocol changed; refusing Part B')


def check_dove(artifact, allow, log):
    d = Path(artifact) / 'src' / 'dove'
    res = {}
    bad = []
    for f, h in EXPECTED_DOVE_SHA256.items():
        raw, lf = sha_file_lf(d / f)
        res[f] = raw
        if h not in (raw, lf):
            bad.append(f)
    if bad:
        msg = f'artifact dove sources differ from the expected versions: {bad}'
        if not allow:
            raise SystemExit(msg + ' (use --allow-dove-mismatch to override; this is logged)')
        log('WARNING ' + msg + ' (overridden)')
    return dict(files=res, mismatches=bad)


def environment(args, artifact_info, prereg):
    import numba, scipy
    try:
        import networkx
        nxv = networkx.__version__
    except Exception:  # noqa: BLE001
        nxv = None
    pkg = {p.name: sha_bytes(p.read_bytes()) for p in sorted(HERE.glob('*.py'))}
    pkg.update({p.name: sha_bytes(p.read_bytes()) for p in sorted(HERE.glob('PREREGISTRATION.*'))})
    return dict(utc=utcnow(), argv=sys.argv, python=sys.version, executable=sys.executable,
                platform=platform.platform(), processor=platform.processor(), machine=platform.machine(),
                cpu_count=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                networkx=nxv, threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                                         'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS')},
                package_sha256=pkg, artifact=artifact_info, prereg=prereg, args=vars(args))


# ------------------------------------------------------------------------------------------------ one graph
def process_graph(item, part, root, args, log, loader):
    """loader() -> (GraphData, provenance, extra_candidates_dict, checks_dict). Writes result.json."""
    import core
    import methods
    name = item['name']
    gdir = root / f'part{part}' / name
    gdir.mkdir(parents=True, exist_ok=True)
    rpath = gdir / 'result.json'
    if rpath.exists():
        prev = json.loads(rpath.read_text(encoding='utf-8'))
        if prev.get('status') in ('done', 'skipped_large', 'excluded_check_failed'):
            log(f'{name}: already done ({prev["status"]}), skipping')
            return prev
        if prev.get('status') == 'missing' and not prev.get('retry_allowed', False):
            log(f'{name}: recorded missing ({prev.get("reason")}), not retried')
            return prev
        log(f'{name}: previous status {prev.get("status")} ({prev.get("reason")}); attempting again')
    attempts_path = gdir / 'attempts.json'
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    crash_attempts = [a for a in attempts if a.get('kind') == 'started']
    if len(crash_attempts) >= 3 and not any(a.get('kind') == 'finished' for a in attempts):
        res = dict(name=name, label=item.get('label', name), split=item.get('split'), part=part, status='missing',
                   reason='failed three times inside the graph (see attempts.json and run_log)', retry_allowed=False)
        write_json(rpath, res)
        log(f'{name}: MISSING after three failed attempts')
        return res
    attempts.append(dict(kind='started', utc=utcnow()))
    write_json(attempts_path, attempts)

    base = dict(name=name, label=item.get('label', name), split=item.get('split'), part=part,
                quick=bool(args.quick), local=bool(args.local_npz), started=utcnow())
    t_graph = perf_counter()
    try:
        g, prov, extra, checks = loader()
    except Exception as ex:  # noqa: BLE001
        import graphs
        is_dl = isinstance(ex, graphs.GraphMissing) and 'download failed' in str(ex)
        res = dict(base, status='missing', reason=f'{type(ex).__name__}: {ex}', retry_allowed=is_dl,
                   traceback=traceback.format_exc())
        write_json(rpath, res)
        attempts.append(dict(kind='finished', utc=utcnow(), status='missing'))
        write_json(attempts_path, attempts)
        log(f'{name}: MISSING ({res["reason"]})')
        return res
    base.update(n=g.n, m=g.m, provenance=prov, checks=checks)
    log(f'{name}: n={g.n:,} m={g.m:,}')
    if checks.get('fatal'):
        res = dict(base, status='excluded_check_failed', reason=checks['fatal'])
        write_json(rpath, res)
        log(f'{name}: EXCLUDED ({checks["fatal"]})')
        return res
    if g.m > 20_000_000:
        res = dict(base, status='skipped_large', reason='more than 20M edges (pre-registered skip)')
        write_json(rpath, res)
        log(f'{name}: skipped for optimization (m > 20M)')
        return res

    prep = methods.Prepared(g)
    odir = gdir / 'orders'
    odir.mkdir(exist_ok=True)
    saved = {}

    def save(label, order, work):
        p = odir / f'{label}.npy'
        dtype = np.int32 if g.n < 2 ** 31 else np.int64
        np.save(p, np.asarray(order).astype(dtype))
        saved[label] = (p, int(work))

    # reference R*
    t = perf_counter()
    ref = methods.build_reference(g, prep, extra)
    ref_seconds = perf_counter() - t
    save('Rstar', ref['order'], ref['work'])
    R, RW = ref['order'], ref['work']
    log(f'{name}: R* = {ref["label"]} work {RW:,} (forward-only min {ref["forward_min"]:,}; {ref_seconds:.1f}s)')

    # T
    To, Tw, Tinfo = methods.run_T(prep, R)
    save('T', To, Tw)
    log(f'{name}: T work {Tw:,} ({Tinfo["rounds"]} rounds, {Tinfo["seconds"]:.2f}s)')

    # DOVE
    if RW == 0:
        Do, Dw, Dinfo = R, 0, dict(seconds=0.0, status='known_zero_optimum')
    else:
        Do, Dw, Dinfo = methods.run_dove(g, R)
    save('DOVE', Do, Dw)
    B = Dinfo['seconds']
    log(f'{name}: DOVE work {Dw:,} in {B:.2f}s (= T+ILS budget)')

    # T+ILS
    tils = {}
    for s in (SEEDS if not args.quick else (1,)):
        o, w, info = methods.run_tils(g, R, B, s)
        save(f'TILS_seed{s}', o, w)
        tils[s] = dict(work=w, info=info)
        log(f'{name}: T+ILS seed {s} work {w:,} ({info.get("total_seconds", 0):.2f}s of {B:.2f}s; '
            f'{info.get("ils", {}).get("proposals", 0):,} proposals)')

    # verification of every saved order (reloaded from disk)
    ver = {}
    for label, (p, w) in saved.items():
        ver[label] = methods.verify_order_file(g, prep, p, w)
    tris = {v['triangles'] for v in ver.values()}
    all_ok = all(v['ok'] for v in ver.values()) and len(tris) == 1
    log(f'{name}: verification {"PASSED" if all_ok else "FAILED"} ({len(ver)} orders, triangles {sorted(tris)})')

    res = dict(base, status='done', finished=utcnow(), graph_seconds=perf_counter() - t_graph,
               reference=dict(label=ref['label'], work=RW, forward_min=ref['forward_min'],
                              forward_min_label=ref['forward_min_label'], candidate_works=ref['works'],
                              builtin_seconds=ref['builtin_seconds'], seconds=ref_seconds,
                              neigh_candidates=sorted(k for k in extra)),
               T=dict(work=Tw, **Tinfo), DOVE=dict(work=Dw, **Dinfo), budget=B,
               TILS={str(k): v for k, v in tils.items()},
               verification=dict(orders=ver, triangles=sorted(tris), all_ok=all_ok, n_orders=len(ver)))
    write_json(rpath, res)
    attempts.append(dict(kind='finished', utc=utcnow(), status='done'))
    write_json(attempts_path, attempts)
    return res


# ------------------------------------------------------------------------------------------------ loaders
def make_file_loader(item, part, args, log):
    import graphs
    import methods

    def loader():
        cache = Path(args.data) / f'part{part}' / 'cache' / f'{item["name"]}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if part == 'A':
            path, prov = graphs.fetch_suite_graph(item, args.data, log)
        else:
            path, prov = graphs.fetch_heldout_graph(item, args.data, log)
        if cache.exists():
            edges, ids = graphs.load_cache(cache)
            prov['cache'] = str(cache)
        else:
            t = perf_counter()
            try:
                edges, ids = graphs.load_graph_file(path, item['name'])
            except Exception as ex:  # noqa: BLE001
                raise graphs.GraphMissing(f'parse failed: {ex!r}')
            prov['parse_seconds'] = perf_counter() - t
            graphs.save_cache(cache, edges, ids, dict(source=str(path)))
        g = graphs.make_graphdata(item['name'], edges, ids, item.get('family', 'unknown'))
        prov['dense_edges_sha256'] = hashlib.sha256(np.ascontiguousarray(edges, dtype=np.int32).tobytes()).hexdigest()
        extra, checks = {}, {}
        stored = methods.stored_neigh_ranks(args.artifact, item['name'])
        if part == 'B':
            if stored:
                raise RuntimeError(f'held-out graph {item["name"]} has stored Neigh ranks in the artifact: not held out')
        else:
            checks['n_matches_artifact'] = g.n == item['expected_n']
            checks['m_matches_artifact'] = g.m == item['expected_m']
            if not (checks['n_matches_artifact'] and checks['m_matches_artifact']):
                checks['fatal'] = f'n/m ({g.n},{g.m}) differ from artifact per_graph.csv ({item["expected_n"]},{item["expected_m"]})'
                return g, prov, extra, checks
            for lab, p in stored.items():
                extra[lab] = methods.order_from_rank_file(p, g.n)
            from dove.orderings import candidate_work
            key = f'NeighStored{item["expected_neigh_best_seed"]}'
            checks['stored_neigh_labels'] = list(stored)
            checks['stored_neigh_work'] = int(candidate_work(g, extra[key])) if key in extra else None
            checks['stored_neigh_work_matches'] = checks['stored_neigh_work'] == item['expected_neigh_best_work']
            if not checks['stored_neigh_work_matches']:
                checks['fatal'] = (f'stored Neigh work {checks["stored_neigh_work"]} != artifact neigh_best_work '
                                   f'{item["expected_neigh_best_work"]} (vertex ids do not match)')
                return g, prov, extra, checks
            checks['artifact_reference_work'] = item['artifact_reference_work']
        if args.volt:
            vo, vrec = methods.run_volt(args.volt, g, Path(args.out) / 'volt' / f'part{part}', log=log)
            extra.update(vo)
            checks['volt'] = vrec
        return g, prov, extra, checks
    return loader


def make_npz_loader(path, args, log):
    import graphs
    import methods

    def loader():
        with np.load(path) as z:
            edges = z['edges'].astype(np.int64)
            n = int(z['n'])
        a, b = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
        if np.any(a == b) or len(np.unique(a * n + b)) != len(a):
            raise ValueError('local npz graph is not simple')
        g = graphs.make_graphdata(Path(path).stem, np.stack([a, b], 1).astype(np.int32), np.arange(n, dtype=np.int64))
        extra = {lab: methods.order_from_rank_file(p, n) for lab, p in methods.stored_neigh_ranks(args.artifact, g.name).items()}
        checks = dict(stored_neigh_labels=list(extra))
        if args.volt:
            vo, vrec = methods.run_volt(args.volt, g, Path(args.out) / 'volt' / 'local', log=log)
            extra.update(vo)
            checks['volt'] = vrec
        return g, dict(source=str(path)), extra, checks
    return loader


def make_synthetic_loader(name, args, log):
    import graphs
    import networkx as nx

    def loader():
        gens = {'quick-plc': lambda: nx.powerlaw_cluster_graph(3000, 4, 0.2, seed=1),
                'quick-gnm': lambda: nx.gnm_random_graph(4000, 16000, seed=2),
                'quick-ba': lambda: nx.barabasi_albert_graph(3000, 3, seed=3)}
        G = gens[name]()
        e = np.array(G.edges(), dtype=np.int64)
        edges, ids = graphs.preprocess(e[:, 0], e[:, 1])
        return graphs.make_graphdata(name, edges, ids), dict(source='networkx'), {}, {}
    return loader


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact', required=True, type=Path)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--part', choices=['A', 'B', 'both'], default='both')
    ap.add_argument('--volt', default=None, help='VOLT executable, or wsl:<linux path> for a WSL build (optional)')
    ap.add_argument('--quick', action='store_true', help='smoke test: no held-out data, 1 seed, separate output dir')
    ap.add_argument('--local-npz', type=Path, default=None, help='run on local .npz graphs (testing)')
    ap.add_argument('--allow-dove-mismatch', action='store_true')
    args = ap.parse_args()
    args.artifact = args.artifact.resolve()
    args.data = args.data.resolve()
    args.out = args.out.resolve()
    if args.volt is not None and not args.volt.startswith('wsl:'):
        args.volt = str(Path(args.volt).resolve())
        if not Path(args.volt).is_file():
            ap.error(f'VOLT executable not found: {args.volt}')
    if args.quick and args.local_npz is None:
        root = args.out / 'quick'
    elif args.local_npz is not None:
        root = args.out / ('local_quick' if args.quick else 'local')
    else:
        root = args.out
    root.mkdir(parents=True, exist_ok=True)
    log = Logger(root / 'run_log.txt')
    log('=' * 100)
    log(f'run_eval start: {" ".join(sys.argv)}')

    sys.path.insert(0, str(args.artifact / 'src'))
    art = check_dove(args.artifact, args.allow_dove_mismatch, log)
    prereg = None
    if args.part in ('B', 'both'):
        prereg = check_prereg(log)   # before any held-out download
    import core
    import graphs
    env = environment(args, art, prereg)
    (root / 'env').mkdir(exist_ok=True)
    write_json(root / 'env' / f'env_{utcnow().replace(":", "")}.json', env)
    log(f'environment: {env["platform"]} | {env["processor"]} | python {sys.version.split()[0]} | numpy {env["numpy"]} '
        f'| scipy {env["scipy"]} | numba {env["numba"]}')
    t = perf_counter()
    core.warmup()
    log(f'numba warm-up/compilation {perf_counter() - t:.1f}s (excluded from all timings)')

    mode = 'quick' if args.quick else ('local' if args.local_npz else 'full')
    # ---------------- Part A
    if args.part in ('A', 'both'):
        if args.local_npz:
            files = sorted(args.local_npz.glob('*.npz'))
            groups = {}
            gcsv = args.local_npz / 'groups.csv'
            if gcsv.exists():
                import csv
                groups = {r['name']: r['split'] for r in csv.DictReader(open(gcsv, encoding='utf-8'))}
            plan = [dict(name=f.stem, label=f.stem, split=groups.get(f.stem, 'Development'), file=str(f)) for f in files]
            if args.quick:
                plan = plan[:3]
        else:
            plan = graphs.suite_plan(args.artifact)
            if args.quick:
                plan = [p for p in plan if p['name'] in QUICK_PART_A]
        write_json(root / 'plan_partA.json', dict(mode=mode, graphs=plan, seeds=list(SEEDS if not args.quick else (1,))))
        log(f'Part A: {len(plan)} graphs')
        for item in plan:
            loader = (make_npz_loader(item['file'], args, log) if args.local_npz else make_file_loader(item, 'A', args, log))
            process_graph(item, 'A', root, args, log, loader)
    # ---------------- Part B
    if args.part in ('B', 'both'):
        if args.local_npz:
            files = sorted(args.local_npz.glob('*.npz'))
            planned = [f.stem for f in files]
            ptxt = args.local_npz / 'planned.txt'
            if ptxt.exists():
                planned = [x.strip() for x in ptxt.read_text(encoding='utf-8').splitlines() if x.strip()]
            plan = [dict(name=nm, label=nm, file=str(args.local_npz / f'{nm}.npz')) for nm in planned]
            if args.quick:
                plan = plan[:2]
        elif args.quick:
            plan = [dict(name=nm, label=nm) for nm in ('quick-plc', 'quick-gnm', 'quick-ba')]
        else:
            plan = [dict(h) for h in graphs.HELDOUT]
        write_json(root / 'plan_partB.json', dict(mode=mode, graphs=plan, seeds=list(SEEDS if not args.quick else (1,)),
                                                  volt=str(args.volt) if args.volt else None, prereg=prereg))
        log(f'Part B: {len(plan)} graphs ({mode})')
        for item in plan:
            if args.local_npz:
                if not Path(item['file']).exists():
                    loader = (lambda nm=item['name']: (_ for _ in ()).throw(graphs.GraphMissing(f'download failed: local file for {nm} absent (test of the missing path)')))
                else:
                    loader = make_npz_loader(item['file'], args, log)
            elif args.quick:
                loader = make_synthetic_loader(item['name'], args, log)
            else:
                loader = make_file_loader(item, 'B', args, log)
            process_graph(item, 'B', root, args, log, loader)
    log('run_eval finished')


if __name__ == '__main__':
    main()
