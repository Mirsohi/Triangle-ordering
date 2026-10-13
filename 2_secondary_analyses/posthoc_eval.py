#!/usr/bin/env python3
"""POST-HOC, SECONDARY analysis of the finished pre-registered pc_eval run.  NOT PRE-REGISTERED.

python posthoc_eval.py --pc-eval <pc_eval dir> --artifact <artifact dir> --results <pc_eval results dir>
                       --data <pc_eval data dir> --out <posthoc output dir>
                       [--local-npz <dir>] [--parts A B] [--seeds 1 2 3] [--only NAME ...] [--no-ils]

For every graph whose pre-registered result.json has status 'done' (Part A suite, Part B held-out):
  1. reload the graph exactly as run_eval.py did (the npz cache in <data>/part<P>/cache/, or --local-npz), and
     check n, m, the dense-edge SHA-256 and the exact work of the saved R*, T, DOVE and T+ILS orders against
     result.json;
  2. build the 8 greedy-colouring orders (ph_kernels.py), time them, compute exact work;
  3. ILS-T (ILS from R*, no tangent prefix, final tangent pass), budget B = DOVE time from result.json, seeds;
  4. C+ILS-T from the pre-declared fixed variant DEG-desc and from the best of the 8 colouring orders (by exact
     work; all constructions charged), same budget and seeds;
  5. verify every produced order (reload from disk, permutation, work by pc_eval core, dove.candidate_work, pc_eval
     verification lister probes, timing-lister probes; triangle counts identical to result.json);
  6. kernel timing with the numba marker lister (ph_kernels.Lister) for R*, T, DOVE, T+ILS seed 1 and ILS-T seed 1
     (one warm-up call each, then 7 repetitions, orders alternated with a rotating start; median).
Writes <out>/part<P>/<graph>/posthoc_result.json (+ orders/*.npy) and, at the end, runs make_posthoc_tables.py.
Resumable: a graph with a posthoc_result.json is skipped. Nothing is written into the pc_eval results folder.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.dont_write_bytecode = True     # never write __pycache__ into the pc_eval folder

import numpy as np  # noqa: E402

TIMING_REPS = 7
MIN_SAMPLE_S = 0.02   # each timing sample averages k inner listings, k fixed per graph so that R* takes >= 20 ms
TIMED_LABELS = ('Rstar', 'T', 'DOVE', 'TILS_seed1', 'ILST_seed1')


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_graph(name, part, args, res):
    """Exactly as run_eval.py: make_file_loader (npz cache written by the main run) or make_npz_loader."""
    import graphs
    if args.local_npz is not None:
        path = args.local_npz / f'{name}.npz'
        with np.load(path) as z:
            edges = z['edges'].astype(np.int64)
            n = int(z['n'])
        a, b = np.minimum(edges[:, 0], edges[:, 1]), np.maximum(edges[:, 0], edges[:, 1])
        g = graphs.make_graphdata(Path(path).stem, np.stack([a, b], 1).astype(np.int32), np.arange(n, dtype=np.int64))
        return g, dict(source=str(path), loader='local npz (as run_eval.make_npz_loader)')
    cache = Path(args.data) / f'part{part}' / 'cache' / f'{name}.npz'
    if not cache.exists():
        raise FileNotFoundError(f'npz cache of the main run not found: {cache}')
    edges, ids = graphs.load_cache(cache)
    g = graphs.make_graphdata(name, edges, ids, 'unknown')
    return g, dict(source=str(cache), loader='npz cache (as run_eval.make_file_loader)',
                   dense_edges_sha256=hashlib.sha256(np.ascontiguousarray(edges, dtype=np.int32).tobytes()).hexdigest())


def verify(g, prep, order, claimed, tri_expected):
    """Full check of one order (in memory or reloaded)."""
    import core
    import ph_kernels as K
    from dove.orderings import candidate_work
    o = np.ascontiguousarray(order, dtype=np.int64)
    ok_perm = bool(core.is_permutation(o)) and o.shape == (g.n,)
    if not ok_perm:
        return dict(ok=False, permutation=False)
    w_core = prep.work(o)
    w_dove = int(candidate_work(g, o))
    probes, tri = core.adaptive_forward_lister(prep.n, prep.E, o)
    L = K.Lister(prep.n, prep.E, o)
    _, p2, t2 = L.run()
    ok = (w_core == w_dove == int(probes) == p2 == int(claimed)) and int(tri) == t2 == int(tri_expected)
    return dict(ok=bool(ok), permutation=True, claimed=int(claimed), core_work=int(w_core), dove_work=w_dove,
                verif_lister_probes=int(probes), timing_lister_probes=int(p2), triangles=int(tri),
                timing_lister_triangles=int(t2))


def process(name, part, res, args, log, out_root):
    import core
    import methods
    import ph_kernels as K
    import ph_methods as M
    from run_eval import write_json, utcnow
    gdir = out_root / f'part{part}' / name
    rp = gdir / 'posthoc_result.json'
    if rp.exists() and not args.force:
        log(f'{name}: posthoc result exists, skipping')
        return json.loads(rp.read_text(encoding='utf-8'))
    gdir.mkdir(parents=True, exist_ok=True)
    odir = gdir / 'orders'
    odir.mkdir(exist_ok=True)
    rdir = Path(args.results) / f'part{part}' / name
    t_graph = perf_counter()
    out = dict(name=name, label=res.get('label', name), split=res.get('split'), part=part, started=utcnow(),
               posthoc=True, preregistered=False)

    # ---- 1. reload and check against the pre-registered run
    g, prov = load_graph(name, part, args, res)
    prep = methods.Prepared(g)
    chk = dict(n=g.n == res['n'], m=g.m == res['m'])
    rsha = res.get('provenance', {}).get('dense_edges_sha256')
    if args.local_npz is None:
        # mandatory in non-local mode: the main run always records it (run_eval.make_file_loader)
        chk['dense_edges_sha256'] = rsha is not None and prov.get('dense_edges_sha256') == rsha
    elif rsha is not None and 'dense_edges_sha256' in prov:
        chk['dense_edges_sha256'] = prov['dense_edges_sha256'] == rsha
    tri_main = res['verification']['triangles']
    chk['main_triangles_unique'] = len(tri_main) == 1
    tri = int(tri_main[0])
    saved = {'Rstar': res['reference']['work'], 'T': res['T']['work'], 'DOVE': res['DOVE']['work']}
    for s, v in res['TILS'].items():
        saved[f'TILS_seed{s}'] = v['work']
    main_orders = {}
    for lab, w in saved.items():
        o = np.load(rdir / 'orders' / f'{lab}.npy').astype(np.int64)
        main_orders[lab] = o
        chk[f'work_{lab}'] = bool(core.is_permutation(o)) and prep.work(o) == int(w)
    out.update(n=g.n, m=g.m, provenance=prov, checks=chk)
    if not all(chk.values()):
        out.update(status='reload_check_failed')
        write_json(rp, out)
        log(f'{name}: RELOAD CHECK FAILED {chk}')
        return out
    R = main_orders['Rstar']
    RW = int(saved['Rstar'])
    B = float(res['budget'])
    log(f'{name}: reloaded n={g.n:,} m={g.m:,}; R* work {RW:,} and all saved works match result.json; B={B:.2f}s')

    # ---- 2. colouring orders
    edges = np.ascontiguousarray(g.edges, dtype=np.int64)
    cols = K.all_colour_orders(g.n, edges)
    colour = {}
    ver = {}
    for lab, (o, sec, k) in cols.items():
        w = prep.work(o)
        colour[lab] = dict(work=int(w), seconds=sec, colours=int(k))
        ver[f'colour_{lab}'] = verify(g, prep, o, w, tri)
    best_lab = min(colour, key=lambda lab: (colour[lab]['work'], list(colour).index(lab)))
    pairs_equal = all(colour[f'{v}-desc']['work'] == colour[f'{v}-asc-rev']['work'] and
                      colour[f'{v}-asc']['work'] == colour[f'{v}-desc-rev']['work'] for v in ('SL', 'DEG'))
    out['colour'] = dict(orders=colour, best_label=best_lab, best_work=colour[best_lab]['work'],
                         fixed_label=K.FIXED_VARIANT, fixed_work=colour[K.FIXED_VARIANT]['work'],
                         reversal_pairs_equal=bool(pairs_equal))
    log(f'{name}: colouring best {best_lab} work {colour[best_lab]["work"]:,} '
        f'(fixed {K.FIXED_VARIANT}: {colour[K.FIXED_VARIANT]["work"]:,}; R* {RW:,})')

    # ---- 3/4. ILS-T and C+ILS-T
    runs = {}
    if not args.no_ils:
        for meth, mk in (('ILST', lambda: (lambda: (R, dict(kind='Rstar')))),
                         ('CILST_fixed', lambda: M.fixed_colour_start(g, K.FIXED_VARIANT)),
                         ('CILST_best', lambda: M.best_colour_start(g))):
            runs[meth] = {}
            for s in args.seeds:
                tinfo = res['TILS'].get(str(s), {}).get('info', {})
                reserve = tinfo.get('reserve')
                o, w, info = M.run_ils_t(g, mk(), B, s, reserve=reserve)
                p = odir / f'{meth}_seed{s}.npy'
                np.save(p, o.astype(np.int32 if g.n < 2 ** 31 else np.int64))
                back = np.load(p).astype(np.int64)
                ver[f'{meth}_seed{s}'] = verify(g, prep, back, w, tri)
                runs[meth][str(s)] = dict(work=int(w), info=info)
                log(f'{name}: {meth} seed {s} work {w:,} ({info.get("total_seconds", 0):.2f}s of {B:.2f}s; '
                    f'start {info["start"].get("label", "R*")} {info["start_work"]:,}; '
                    f'{info.get("ils", {}).get("proposals", 0):,} proposals)')
    out['runs'] = runs

    # ---- 6. kernel timing
    timing = {}
    orders_t = {lab: main_orders[lab] for lab in TIMED_LABELS if lab in main_orders}
    if 'ILST' in runs and '1' in runs['ILST']:
        orders_t['ILST_seed1'] = np.load(odir / 'ILST_seed1.npy').astype(np.int64)
    listers = {lab: K.Lister(g.n, prep.E, o) for lab, o in orders_t.items()}
    labs = list(listers)
    samples = {lab: [] for lab in labs}
    warm = {}
    for lab in labs:   # warm-up (untimed) + probe/triangle check
        sec, pr, tr = listers[lab].run()
        warm[lab] = dict(seconds=sec, probes=pr, triangles=tr, probes_equal_work=pr == prep.work(orders_t[lab]),
                         triangles_ok=tr == tri)
    w_r = listers['Rstar'].run()[0]          # second untimed call, used only to size k
    inner = int(min(1000, max(1, np.ceil(MIN_SAMPLE_S / max(w_r, 1e-7)))))
    for rep in range(TIMING_REPS):
        for j in range(len(labs)):
            lab = labs[(rep + j) % len(labs)]
            sec, pr, tr = listers[lab].run(inner)
            samples[lab].append(sec)
    del listers
    for lab in labs:
        med = float(np.median(samples[lab]))
        timing[lab] = dict(median_s=med, samples=samples[lab], work=int(prep.work(orders_t[lab])),
                           warmup=warm[lab])
    base = timing['Rstar']['median_s']
    for lab in labs:
        timing[lab]['ratio_vs_Rstar'] = timing[lab]['median_s'] / base if base > 0 else None
    out['kernel_timing'] = dict(lister='numba marker-array adaptive-forward lister (ph_kernels.Lister); NOT the '
                                       'C++ bitmap kernel of the artifact timing study', repetitions=TIMING_REPS, inner_listings_per_sample=inner,
                                per_order=timing)
    log(f'{name}: kernel median R* {base * 1e3:.3f} ms; ratios ' +
        ', '.join(f'{lab} {timing[lab]["ratio_vs_Rstar"]:.4f}' for lab in labs if lab != 'Rstar'))

    # ---- ordering times (from the pre-registered run, for reference) and verification summary
    out['ordering_seconds'] = dict(reference=res['reference']['seconds'], T=res['T']['seconds'],
                                   DOVE=res['DOVE']['seconds'],
                                   TILS={s: v['info'].get('total_seconds') for s, v in res['TILS'].items()})
    tris = {v.get('triangles') for v in ver.values()}
    all_ok = all(v['ok'] for v in ver.values()) and tris == {tri} and all(
        w['probes_equal_work'] and w['triangles_ok'] for w in warm.values())
    out['verification'] = dict(orders=ver, all_ok=bool(all_ok), n_orders=len(ver), triangles=tri)
    out.update(status='done', finished=utcnow(), graph_seconds=perf_counter() - t_graph)
    write_json(rp, out)
    stale = gdir / 'posthoc_error.json'      # an earlier failed attempt of this graph
    if stale.exists():
        stale.replace(gdir / f'posthoc_error_superseded_{utcnow().replace(":", "")}.json')
    log(f'{name}: posthoc verification {"PASSED" if all_ok else "FAILED"} ({len(ver)} orders, triangles {tri:,}); '
        f'{out["graph_seconds"]:.1f}s')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pc-eval', required=True, type=Path, help='the unchanged pc_eval package folder')
    ap.add_argument('--artifact', required=True, type=Path)
    ap.add_argument('--results', required=True, type=Path, help='finished pc_eval results root (contains partA/partB)')
    ap.add_argument('--data', type=Path, default=None, help='pc_eval data dir (npz caches of the main run)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--local-npz', type=Path, default=None, help='testing: graphs from local npz files')
    ap.add_argument('--parts', nargs='+', default=['A', 'B'], choices=['A', 'B'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--no-ils', action='store_true', help='skip ILS-T/C+ILS-T (colouring, checks and timing only)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow-dove-mismatch', action='store_true')
    ap.add_argument('--allow-incomplete', action='store_true',
                    help='run even if the pre-registered run has graphs without a final status (not recommended)')
    args = ap.parse_args()
    args.results = args.results.resolve()
    args.out = args.out.resolve()
    if args.data is None and args.local_npz is None:
        ap.error('--data is required (unless --local-npz)')
    if args.out == args.results or args.results in args.out.parents:
        ap.error('--out must not be inside the pre-registered results folder')
    sys.path.insert(0, str(args.pc_eval.resolve()))
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(args.artifact.resolve() / 'src'))
    import run_eval   # pc_eval helpers (Logger, write_json, check_dove, utcnow); its main() is not run
    import core
    import ph_kernels
    args.out.mkdir(parents=True, exist_ok=True)
    log = run_eval.Logger(args.out / 'posthoc_log.txt')
    log('=' * 100)
    log('posthoc_eval start (POST-HOC, NOT PRE-REGISTERED): ' + ' '.join(sys.argv))
    art = run_eval.check_dove(args.artifact.resolve(), args.allow_dove_mismatch, log)
    pc_files = {p.name: sha_file(p) for p in sorted(args.pc_eval.resolve().glob('*.py'))}
    pc_files.update({p.name: sha_file(p) for p in sorted(args.pc_eval.resolve().glob('PREREGISTRATION.*'))})
    ph_files = {p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))}
    decl = HERE / 'PREDECLARED_VARIANT.md'
    import numba, scipy, platform
    env = dict(utc=run_eval.utcnow(), argv=sys.argv, python=sys.version, platform=platform.platform(),
               processor=platform.processor(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
               pc_eval_sha256=pc_files, posthoc_sha256=ph_files, artifact=art,
               predeclared_variant=ph_kernels.FIXED_VARIANT,
               predeclared_file_sha256=sha_file(decl) if decl.exists() else None, args=vars(args))
    run_eval.write_json(args.out / f'posthoc_env_{run_eval.utcnow().replace(":", "")}.json', env)
    t = perf_counter()
    core.warmup()
    ph_kernels.warmup()
    log(f'numba warm-up {perf_counter() - t:.1f}s')

    todo = []
    for part in args.parts:
        plan_path = args.results / f'plan_part{part}.json'
        if not plan_path.exists():
            log(f'Part {part}: no plan in {args.results}; skipped')
            continue
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan.get('mode') == 'quick':
            raise SystemExit('refusing quick-mode results')
        pending = []
        for item in plan['graphs']:
            rp = args.results / f'part{part}' / item['name'] / 'result.json'
            st = json.loads(rp.read_text(encoding='utf-8')).get('status') if rp.exists() else 'not_run'
            if st not in ('done', 'missing', 'skipped_large', 'excluded_check_failed'):
                pending.append(f'{item["name"]} ({st})')
        if pending and not args.allow_incomplete:
            raise SystemExit(f'the pre-registered run has not finished Part {part}: {pending}. Run this post-hoc '
                             'analysis only after RUN_EVAL.cmd has printed DONE (use --allow-incomplete to override).')
        for item in plan['graphs']:
            rp = args.results / f'part{part}' / item['name'] / 'result.json'
            if args.only and item['name'] not in args.only:
                continue
            if not rp.exists():
                log(f'{item["name"]}: no result.json (not run); skipped')
                continue
            res = json.loads(rp.read_text(encoding='utf-8'))
            if res.get('status') != 'done':
                log(f'{item["name"]}: status {res.get("status")}; skipped')
                continue
            todo.append((part, item['name'], res))
    nils = 0 if args.no_ils else 3 * len(args.seeds)
    est = sum(nils * r['budget'] + 30 + 6 * r['m'] * 2e-6 for _, _, r in todo)
    log(f'{len(todo)} completed graphs; estimated time about {est / 3600:.1f} h '
        f'({nils} ILS runs x B per graph, B summed = {sum(r["budget"] for _, _, r in todo):.0f}s, plus checks)')
    for part, name, res in todo:
        try:
            process(name, part, res, args, log, args.out)
        except Exception as ex:  # noqa: BLE001
            log(f'{name}: ERROR {ex!r}\n{traceback.format_exc()}')
            run_eval.write_json(args.out / f'part{part}' / name / 'posthoc_error.json',
                                dict(name=name, error=repr(ex), traceback=traceback.format_exc()))
    log('posthoc_eval finished; writing tables')
    import make_posthoc_tables
    _, _, problems = make_posthoc_tables.write_all(args.results, args.out, log)
    log('posthoc tables written')
    if problems:
        log(f'POST-HOC FAILURES on {len(problems)} graph(s): ' + '; '.join(f'{k}: {v}' for k, v in problems.items()))
        log('exit code 2 (tables were written, but the failed graphs are excluded and reported in the macros)')
        sys.exit(2)
    log('all analysed graphs passed')


if __name__ == '__main__':
    main()
