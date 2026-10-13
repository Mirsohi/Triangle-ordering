#!/usr/bin/env python3
"""SUPPLEMENTARY held-out run with Neigh (review round 2). NOT PART OF THE FROZEN PRE-REGISTRATION.
The analyses are fixed in SUPP_PROTOCOL.md (declared before the run).

python neigh_supp.py --pc-eval <pc_eval> --posthoc <pc_eval_posthoc> --artifact <artifact> --results <pc_eval results>
                     --data <pc_eval data> --out <output> --volt <volt.exe>
                     [--local-npz <dir>] [--parts V N] [--only NAME ...] [--seeds 1 2 3] [--timing-reps 21] [--force]

Part V: rerun VOLT with the stored artifact seed on each suite graph and compare the rank with the stored rank file.
Part N: for each completed held-out graph: reload check; Neigh seeds 1-5; R*N; re-score the frozen orders (Q1a);
        rerun T, DOVE, T+ILS from R*N if it differs from R* (Q1b); verify every new order; kernel timing.
Writes <out>/partV/<graph>/supp_result.json and <out>/partN/<graph>/supp_result.json, then make_supp_tables.py.
Resumable: a graph with a supp_result.json is skipped. The frozen results folder is only read.
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
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

MIN_SAMPLE_S = 0.02
VOLT_SEEDS = (1, 2, 3, 4, 5)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_order(path, o, n):
    np.save(path, np.asarray(o).astype(np.int32 if n < 2 ** 31 else np.int64))
    return np.load(path).astype(np.int64)


def reload_and_check(name, part, res, args):
    """Reload exactly as run_eval.py did (via pc_eval_posthoc.load_graph) and check against result.json."""
    import core
    import methods
    import posthoc_eval as PE
    g, prov = PE.load_graph(name, part, args, res)
    prep = methods.Prepared(g)
    chk = dict(n=g.n == res['n'], m=g.m == res['m'])
    rsha = res.get('provenance', {}).get('dense_edges_sha256')
    if args.local_npz is None:
        chk['dense_edges_sha256'] = rsha is not None and prov.get('dense_edges_sha256') == rsha
    rdir = Path(args.results) / f'part{part}' / name
    saved = {'Rstar': res['reference']['work'], 'T': res['T']['work'], 'DOVE': res['DOVE']['work']}
    for s, v in res['TILS'].items():
        saved[f'TILS_seed{s}'] = v['work']
    orders = {}
    for lab, w in saved.items():
        o = np.load(rdir / 'orders' / f'{lab}.npy').astype(np.int64)
        orders[lab] = o
        chk[f'work_{lab}'] = bool(core.is_permutation(o)) and prep.work(o) == int(w)
    return g, prov, prep, chk, saved, orders


# ------------------------------------------------------------------------------------------------ Part V
def process_validation(name, res, args, log, out_root):
    import methods
    gdir = out_root / 'partV' / name
    rp = gdir / 'supp_result.json'
    if rp.exists() and not args.force:
        prev = json.loads(rp.read_text(encoding='utf-8'))
        if prev.get('status') == 'no_stored_rank' or (prev.get('status') == 'done' and
                                                      prev.get('verification', {}).get('all_ok', True)):
            log(f'V {name}: result exists, skipping')
            return prev
        rp.replace(rp.with_name(f'supp_result_superseded_{prev.get("status")}.json'))
    from run_eval import write_json, utcnow
    gdir.mkdir(parents=True, exist_ok=True)
    stored = methods.stored_neigh_ranks(args.artifact, name)
    out = dict(name=name, part='V', started=utcnow(), supplementary=True, preregistered=False)
    if not stored:
        out.update(status='no_stored_rank')
        write_json(rp, out)
        log(f'V {name}: no stored artifact Neigh rank; skipped')
        return out
    g, prov, prep, chk, _, _ = reload_and_check(name, 'A', res, args)
    out.update(n=g.n, m=g.m, checks=chk)
    if not all(chk.values()):
        out.update(status='reload_check_failed')
        write_json(rp, out)
        log(f'V {name}: RELOAD CHECK FAILED {chk}')
        return out
    seeds = [int(lab[len('NeighStored'):]) for lab in stored]
    vo, vrec = methods.run_volt(args.volt, g, out_root / 'volt' / 'partV', seeds=seeds, log=log)
    comp = {}
    for lab, path in stored.items():
        s = int(lab[len('NeighStored'):])
        so = methods.order_from_rank_file(path, g.n)
        rec = dict(seed=s, stored_file=Path(path).name, stored_work=int(prep.work(so)))
        if f'Neigh{s}' in vo:
            no = vo[f'Neigh{s}']
            rec.update(new_work=int(prep.work(no)), identical=bool(np.array_equal(no, so)))
        else:
            rec.update(new_work=None, identical=False, error='VOLT seed failed')
        comp[str(s)] = rec
    failed = [c['seed'] for c in comp.values() if c['new_work'] is None]
    out.update(volt=vrec, comparisons=comp, all_identical=all(c['identical'] for c in comp.values()),
               status='volt_failed' if failed else 'done', finished=utcnow())
    write_json(rp, out)
    log(f'V {name}: stored Neigh seeds {seeds}: ' + ', '.join(
        f's{c["seed"]} {"IDENTICAL" if c["identical"] else "differs"} (stored {c["stored_work"]:,}, new {c["new_work"]})'
        for c in comp.values()))
    return out


# ------------------------------------------------------------------------------------------------ Part N
def process_heldout(name, res, args, log, out_root):
    import methods
    import posthoc_eval as PE
    import ph_kernels as K
    from dove.orderings import candidate_work
    from run_eval import write_json, utcnow
    gdir = out_root / 'partN' / name
    rp = gdir / 'supp_result.json'
    if rp.exists() and not args.force:
        prev = json.loads(rp.read_text(encoding='utf-8'))
        if prev.get('status') == 'no_stored_rank' or (prev.get('status') == 'done' and
                                                      prev.get('verification', {}).get('all_ok', True)):
            log(f'N {name}: result exists, skipping')
            return prev
        rp.replace(rp.with_name(f'supp_result_superseded_{prev.get("status")}.json'))
    gdir.mkdir(parents=True, exist_ok=True)
    odir = gdir / 'orders'
    odir.mkdir(exist_ok=True)
    t_graph = perf_counter()
    out = dict(name=name, label=res.get('label', name), part='N', started=utcnow(), supplementary=True,
               preregistered=False)

    # 1. reload check
    g, prov, prep, chk, saved, frozen = reload_and_check(name, 'B', res, args)
    tri = int(res['verification']['triangles'][0])
    out.update(n=g.n, m=g.m, provenance=prov, checks=chk)
    if not all(chk.values()):
        out.update(status='reload_check_failed')
        write_json(rp, out)
        log(f'N {name}: RELOAD CHECK FAILED {chk}')
        return out
    RW = int(saved['Rstar'])
    log(f'N {name}: reloaded n={g.n:,} m={g.m:,}; frozen R* {res["reference"]["label"]} {RW:,}')

    # 2. Neigh
    vo, vrec = methods.run_volt(args.volt, g, out_root / 'volt' / 'partN', seeds=VOLT_SEEDS, log=log)
    ver = {}
    neigh = {}
    for lab, o in vo.items():
        for l2, oo in ((lab, o), (lab + '-rev', o[::-1].copy())):
            w = int(prep.work(oo))
            w2 = int(candidate_work(g, oo))
            if w != w2:
                raise AssertionError(f'{l2}: work mismatch {w} vs {w2}')
            back = save_order(odir / f'{l2}.npy', oo, g.n)
            ver[l2] = PE.verify(g, prep, back, w, tri)
            neigh[l2] = w
    out['neigh'] = dict(volt=vrec, works=neigh, ok_seeds=sum(r['ok'] for r in vrec),
                        seconds_total=sum(r['seconds'] for r in vrec))
    if 0 < out['neigh']['ok_seeds'] < len(VOLT_SEEDS):
        log(f'N {name}: WARNING only {out["neigh"]["ok_seeds"]} of {len(VOLT_SEEDS)} VOLT seeds succeeded '
            '(protocol: failed seeds are reported and left out)')
    if out['neigh']['ok_seeds'] == 0:
        out.update(status='neigh_failed', finished=utcnow())
        write_json(rp, out)
        log(f'N {name}: ALL VOLT SEEDS FAILED; see {out_root / "volt" / "partN"}')
        return out

    # 3. R*N (ties keep the frozen R*)
    cand = [('Rstar', RW)] + list(neigh.items())
    best_lab, RNW = min(cand, key=lambda t: (t[1], [c[0] for c in cand].index(t[0])))
    RN = frozen['Rstar'] if best_lab == 'Rstar' else np.load(odir / f'{best_lab}.npy').astype(np.int64)
    neigh_best = min(neigh.values()) if neigh else None
    out['reference'] = dict(frozen_label=res['reference']['label'], frozen_work=RW, new_label=best_lab, new_work=int(RNW),
                            changed=best_lab != 'Rstar', neigh_best_work=neigh_best,
                            gap_percent=0.0 if RW == 0 else 100.0 * (RW - RNW) / RW)
    log(f'N {name}: Neigh {out["neigh"]["ok_seeds"]}/5 seeds, best {neigh_best}; R*N = {best_lab} {RNW:,} '
        f'(gap {out["reference"]["gap_percent"]:.3f}%)')

    # 4. Q1a: frozen orders against R*N
    out['frozen_works'] = {k: int(v) for k, v in saved.items()}

    # 5. Q1b: rerun from R*N
    rerun = {}
    if best_lab != 'Rstar':
        o, w, info = methods.run_T(prep, RN)
        back = save_order(odir / 'T_fromRN.npy', o, g.n)
        ver['T_fromRN'] = PE.verify(g, prep, back, w, tri)
        rerun['T'] = dict(work=int(w), seconds=info['seconds'])
        log(f'N {name}: T from R*N {w:,} ({info["seconds"]:.2f}s)')
        o, w, dinfo = methods.run_dove(g, RN)
        back = save_order(odir / 'DOVE_fromRN.npy', o, g.n)
        ver['DOVE_fromRN'] = PE.verify(g, prep, back, w, tri)
        rerun['DOVE'] = dict(work=int(w), seconds=dinfo['seconds'], initial_work=dinfo['initial_work'])
        if dinfo['initial_work'] != RNW:
            raise AssertionError(f'DOVE initial work {dinfo["initial_work"]} != R*N {RNW}')
        B = float(dinfo['seconds'])
        log(f'N {name}: DOVE from R*N {w:,} ({B:.2f}s = T+ILS budget)')
        rerun['TILS'] = {}
        for s in args.seeds:
            o, w, info = methods.run_tils(g, RN, B, s)
            back = save_order(odir / f'TILS_fromRN_seed{s}.npy', o, g.n)
            ver[f'TILS_fromRN_seed{s}'] = PE.verify(g, prep, back, w, tri)
            rerun['TILS'][str(s)] = dict(work=int(w), info=info)
            log(f'N {name}: T+ILS from R*N seed {s} {w:,} ({info.get("total_seconds", 0):.2f}s of {B:.2f}s)')
        rerun['budget'] = B
        rerun['reused'] = False
    else:
        rerun = dict(reused=True, budget=float(res['budget']),
                     T=dict(work=int(saved['T'])), DOVE=dict(work=int(saved['DOVE'])),
                     TILS={s: dict(work=int(v['work']), info=v['info']) for s, v in res['TILS'].items()})
        log(f'N {name}: R*N = frozen R*; frozen runs reused')
    out['rerun'] = rerun

    # 7. kernel timing
    orders_t = {'Rstar': frozen['Rstar'], 'TILS_seed1': frozen['TILS_seed1']}
    if best_lab != 'Rstar':
        orders_t['RstarN'] = RN
        orders_t['TILSN_seed1'] = np.load(odir / 'TILS_fromRN_seed1.npy').astype(np.int64)
    listers = {lab: K.Lister(g.n, prep.E, o) for lab, o in orders_t.items()}
    labs = list(listers)
    warm = {}
    for lab in labs:
        sec, pr, tr = listers[lab].run()
        warm[lab] = dict(seconds=sec, probes=pr, triangles=tr, probes_equal_work=pr == prep.work(orders_t[lab]),
                         triangles_ok=tr == tri)
    w_r = listers['Rstar'].run()[0]
    inner = int(min(1000, max(1, np.ceil(MIN_SAMPLE_S / max(w_r, 1e-7)))))
    samples = {lab: [] for lab in labs}
    for rep in range(args.timing_reps):
        for j in range(len(labs)):
            lab = labs[(rep + j) % len(labs)]
            samples[lab].append(listers[lab].run(inner)[0])
    del listers
    out['kernel_timing'] = dict(lister='numba marker-array adaptive-forward lister (ph_kernels.Lister)',
                                repetitions=args.timing_reps, inner_listings_per_sample=inner,
                                per_order={lab: dict(samples=samples[lab], median_s=float(np.median(samples[lab])),
                                                     work=int(prep.work(orders_t[lab])), warmup=warm[lab])
                                           for lab in labs})

    tris = {v.get('triangles') for v in ver.values()}
    all_ok = all(v['ok'] for v in ver.values()) and tris <= {tri} and all(
        w['probes_equal_work'] and w['triangles_ok'] for w in warm.values())
    out['verification'] = dict(orders=ver, all_ok=bool(all_ok), n_orders=len(ver), triangles=tri)
    out.update(status='done', finished=utcnow(), graph_seconds=perf_counter() - t_graph)
    write_json(rp, out)
    stale = gdir / 'supp_error.json'
    if stale.exists():
        stale.replace(gdir / f'supp_error_superseded_{utcnow().replace(":", "")}.json')
    log(f'N {name}: supplementary verification {"PASSED" if all_ok else "FAILED"} ({len(ver)} orders); '
        f'{out["graph_seconds"]:.1f}s')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pc-eval', required=True, type=Path)
    ap.add_argument('--posthoc', required=True, type=Path, help='the unchanged pc_eval_posthoc package folder')
    ap.add_argument('--artifact', required=True, type=Path)
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--volt', required=True, help='VOLT executable')
    ap.add_argument('--local-npz', type=Path, default=None, help='testing: graphs from local npz files')
    ap.add_argument('--parts', nargs='+', default=['V', 'N'], choices=['V', 'N'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--timing-reps', type=int, default=21)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--allow-dove-mismatch', action='store_true')
    args = ap.parse_args()
    args.results = args.results.resolve()
    args.out = args.out.resolve()
    args.volt = str(Path(args.volt).resolve())
    if not Path(args.volt).is_file():
        ap.error(f'VOLT executable not found: {args.volt}')
    if args.data is None and args.local_npz is None:
        ap.error('--data is required (unless --local-npz)')
    if args.out == args.results or args.results in args.out.parents:
        ap.error('--out must not be inside the pre-registered results folder')
    sys.path.insert(0, str(args.pc_eval.resolve()))
    sys.path.insert(0, str(args.posthoc.resolve()))
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(args.artifact.resolve() / 'src'))
    import run_eval
    import core
    import ph_kernels
    args.out.mkdir(parents=True, exist_ok=True)
    log = run_eval.Logger(args.out / 'supp_log.txt')
    log('=' * 100)
    log('neigh_supp start (SUPPLEMENTARY, NOT PRE-REGISTERED; declared in SUPP_PROTOCOL.md): ' + ' '.join(sys.argv))
    art = run_eval.check_dove(args.artifact.resolve(), args.allow_dove_mismatch, log)
    import numba, scipy, platform
    env = dict(utc=run_eval.utcnow(), argv=sys.argv, python=sys.version, platform=platform.platform(),
               processor=platform.processor(), numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
               volt_sha256=sha_file(args.volt),
               protocol_sha256=sha_file(HERE / 'SUPP_PROTOCOL.md') if (HERE / 'SUPP_PROTOCOL.md').exists() else None,
               supp_sha256={p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))},
               pc_eval_sha256={p.name: sha_file(p) for p in sorted(args.pc_eval.resolve().glob('*.py'))},
               posthoc_sha256={p.name: sha_file(p) for p in sorted(args.posthoc.resolve().glob('*.py'))},
               artifact=art, args={k: str(v) for k, v in vars(args).items()})
    run_eval.write_json(args.out / f'supp_env_{run_eval.utcnow().replace(":", "")}.json', env)
    t = perf_counter()
    core.warmup()
    ph_kernels.warmup()
    log(f'numba warm-up {perf_counter() - t:.1f}s')

    def done_graphs(part):
        plan_path = args.results / f'plan_part{part}.json'
        if not plan_path.exists():
            return []
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan.get('mode') == 'quick':
            raise SystemExit('refusing quick-mode results')
        todo = []
        for item in plan['graphs']:
            if args.only and item['name'] not in args.only:
                continue
            rp = args.results / f'part{part}' / item['name'] / 'result.json'
            if rp.exists():
                res = json.loads(rp.read_text(encoding='utf-8'))
                if res.get('status') == 'done':
                    todo.append((item['name'], res))
        return todo

    for part, fn in (('V', process_validation), ('N', process_heldout)):
        if part not in args.parts:
            continue
        todo = done_graphs('A' if part == 'V' else 'B')
        if part == 'N':
            est = sum(4 * r['budget'] + r['reference']['seconds'] + 30 for _, r in todo)
            log(f'Part N: {len(todo)} completed held-out graphs; at most about {est / 60:.0f} min')
        else:
            log(f'Part V: {len(todo)} completed suite graphs')
        for name, res in todo:
            try:
                fn(name, res, args, log, args.out)
            except Exception as ex:  # noqa: BLE001
                log(f'{part} {name}: ERROR {ex!r}\n{traceback.format_exc()}')
                d = args.out / f'part{part}' / name
                d.mkdir(parents=True, exist_ok=True)
                run_eval.write_json(d / 'supp_error.json', dict(name=name, error=repr(ex),
                                                                traceback=traceback.format_exc()))
    log('neigh_supp finished; writing tables')
    import make_supp_tables
    problems = make_supp_tables.write_all(args.results, args.out, log, validation='V' in args.parts)
    if problems:
        log(f'SUPPLEMENTARY FAILURES on {len(problems)} graph(s): ' + '; '.join(f'{k}: {v}' for k, v in problems.items()))
        log('exit code 2 (tables were written; failed graphs are excluded and named in the macros)')
        sys.exit(2)
    log('all analysed graphs passed')


if __name__ == '__main__':
    main()
