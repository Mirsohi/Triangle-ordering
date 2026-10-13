#!/usr/bin/env python3
"""Tables and macros of the POST-HOC (secondary, NOT pre-registered) analysis.

python make_posthoc_tables.py --pc-eval <pc_eval dir> --results <pc_eval results> --out <posthoc output dir>

Writes into <out>:
  posthoc_summary.json          every number used (per graph, per seed, aggregates)
  posthoc_per_graph.csv         one row per graph
  posthoc_macros.tex            \\PH... macros (documented in README_POSTHOC.md)
  posthoc_{suite,heldout}_colour_rows.tex   rows: graph & colour variants (% vs R*) & best & fixed & time
  posthoc_{suite,heldout}_ils_rows.tex      rows: graph & T+ILS & ILS-T & C+ILS-T fixed & C+ILS-T best
  posthoc_{suite,heldout}_timing_rows.tex   rows: graph & R* kernel ms & time ratios & ordering seconds
The pre-registered numbers are only read (result.json); they are never changed.
"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
# never let numba (pc_eval core, cache=True) or Python write caches into the pc_eval folder
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.dont_write_bytecode = True
import argparse
import csv
import json
import math
import statistics
VARIANTS = ('SL-asc', 'SL-desc', 'DEG-asc', 'DEG-desc')
METHODS = ('ILST', 'CILST_fixed', 'CILST_best')


def _mt():
    import make_tables   # pc_eval (formatting helpers only)
    return make_tables


def imp(R, w):
    return 0.0 if R == 0 else 100.0 * (R - w) / R


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else 0.0


def geo(x):
    x = [v for v in x if v is not None and v > 0]
    return math.exp(sum(math.log(v) for v in x) / len(x)) if x else None


def wtl(a, b):
    """seed-mean work a vs b: (#a<b, #a==b, #a>b)."""
    w = sum(1 for x, y in zip(a, b) if x < y)
    t = sum(1 for x, y in zip(a, b) if x == y)
    return w, t, len(a) - w - t


def graph_row(res, ph):
    R = int(res['reference']['work'])
    tils = [int(v['work']) for _, v in sorted(res['TILS'].items(), key=lambda kv: int(kv[0]))]
    row = dict(name=res['name'], label=res.get('label', res['name']), split=res.get('split'), part=res['part'],
               n=res['n'], m=res['m'], R=R, budget=float(res['budget']),
               T=int(res['T']['work']), DOVE=int(res['DOVE']['work']), TILS_works=tils,
               TILS_mean=sum(tils) / len(tils))
    row['imp_T'] = imp(R, row['T'])
    row['imp_DOVE'] = imp(R, row['DOVE'])
    row['imp_TILS'] = statistics.mean(imp(R, w) for w in tils)
    row['TILS_overshoot_frac_max'] = max(float(v['info'].get('overshoot', 0.0)) / max(float(res['budget']), 1e-12)
                                         for v in res['TILS'].values())
    col = ph['colour']
    for v in VARIANTS:
        row[f'col_{v}'] = col['orders'][v]['work']
        row[f'imp_col_{v}'] = imp(R, col['orders'][v]['work'])
        row[f'sec_col_{v}'] = max(col['orders'][v]['seconds'], col['orders'][v + '-rev']['seconds'])
    row['col_best_label'] = col['best_label']
    row['col_best'] = col['best_work']
    row['imp_col_best'] = imp(R, col['best_work'])
    row['col_fixed'] = col['fixed_work']
    row['imp_col_fixed'] = imp(R, col['fixed_work'])
    row['col_reversal_pairs_equal'] = col['reversal_pairs_equal']
    row['sec_col_all8'] = sum(o['seconds'] for o in col['orders'].values())
    for meth in METHODS:
        runs = ph['runs'].get(meth, {})
        ws = [int(v['work']) for _, v in sorted(runs.items(), key=lambda kv: int(kv[0]))]
        if not ws:
            continue
        imps = [imp(R, w) for w in ws]
        row[f'{meth}_works'] = ws
        row[f'{meth}_mean'] = sum(ws) / len(ws)
        row[f'imp_{meth}'] = statistics.mean(imps)
        row[f'imp_{meth}_sd'] = sd(imps)
        row[f'{meth}_overshoot_max'] = max(v['info'].get('overshoot', 0.0) for v in runs.values())
        row[f'{meth}_overshoot_frac_max'] = max(v['info'].get('overshoot', 0.0) / max(float(res['budget']), 1e-12)
                                                for v in runs.values())
        row[f'{meth}_total_s_mean'] = statistics.mean(v['info'].get('total_seconds', 0.0) for v in runs.values())
        row[f'{meth}_start_s_mean'] = statistics.mean(v['info'].get('start_seconds', 0.0) for v in runs.values())
        if meth == 'CILST_best':
            row['CILST_best_start_label'] = sorted({v['info']['start'].get('label') for v in runs.values()})
    kt = ph['kernel_timing']['per_order']
    row['kernel_Rstar_ms'] = 1e3 * kt['Rstar']['median_s']
    for lab in ('T', 'DOVE', 'TILS_seed1', 'ILST_seed1'):
        if lab in kt:
            row[f'tratio_{lab}'] = kt[lab]['ratio_vs_Rstar']
            row[f'wratio_{lab}'] = (kt[lab]['work'] / kt['Rstar']['work']) if kt['Rstar']['work'] else None
            row[f'kernel_{lab}_ms'] = 1e3 * kt[lab]['median_s']
    os_ = ph['ordering_seconds']
    row['sec_T'] = os_['T']
    row['sec_DOVE'] = os_['DOVE']
    row['sec_TILS_mean'] = statistics.mean(v for v in os_['TILS'].values() if v is not None)
    row['sec_reference'] = os_['reference']
    row['verification_ok'] = ph['verification']['all_ok']
    row['verification_orders'] = ph['verification']['n_orders']
    return row


def aggregate(rows):
    a = dict(count=len(rows))
    if not rows:
        return a
    for key in ['imp_T', 'imp_DOVE', 'imp_TILS', 'imp_col_best', 'imp_col_fixed'] + [f'imp_col_{v}' for v in VARIANTS]:
        vals = [r[key] for r in rows]
        a[key + '_mean'] = statistics.mean(vals)
        a[key + '_median'] = statistics.median(vals)
    a['col_best_beats_R'] = sum(1 for r in rows if r['col_best'] < r['R'])
    a['col_fixed_beats_R'] = sum(1 for r in rows if r['col_fixed'] < r['R'])
    a['col_best_label_counts'] = {v: sum(1 for r in rows if r['col_best_label'] == v) for v in
                                  sorted({r['col_best_label'] for r in rows})}
    a['sec_col_max'] = max(max(r[f'sec_col_{v}'] for v in VARIANTS) for r in rows)
    a['sec_col_fixed_max'] = max(r['sec_col_DEG-desc'] for r in rows)
    for meth in METHODS:
        rr = [r for r in rows if f'{meth}_mean' in r]
        if len(rr) != len(rows):
            continue
        vals = [r[f'imp_{meth}'] for r in rows]
        a[f'imp_{meth}_mean'] = statistics.mean(vals)
        a[f'imp_{meth}_median'] = statistics.median(vals)
        a[f'{meth}_ITR'] = wtl([r[f'{meth}_mean'] for r in rows], [r['R'] for r in rows])
        a[f'{meth}_vs_TILS'] = wtl([r[f'{meth}_mean'] for r in rows], [r['TILS_mean'] for r in rows])
        a[f'{meth}_minus_TILS_pp'] = statistics.mean(r[f'imp_{meth}'] - r['imp_TILS'] for r in rows)
        a[f'{meth}_vs_DOVE'] = wtl([r[f'{meth}_mean'] for r in rows], [r['DOVE'] for r in rows])
        a[f'{meth}_overshoot_max'] = max(r[f'{meth}_overshoot_max'] for r in rows)
        a[f'{meth}_overshoot_pct_max'] = 100 * max(r[f'{meth}_overshoot_frac_max'] for r in rows)
        a[f'{meth}_geo_work_ratio'] = geo([r[f'{meth}_mean'] / r['R'] for r in rows if r['R'] > 0])
        if meth != 'ILST' and all('ILST_mean' in r for r in rows):
            a[f'{meth}_vs_ILST'] = wtl([r[f'{meth}_mean'] for r in rows], [r['ILST_mean'] for r in rows])
            a[f'{meth}_minus_ILST_pp'] = statistics.mean(r[f'imp_{meth}'] - r['imp_ILST'] for r in rows)
    a['TILS_overshoot_pct_max'] = 100 * max(r['TILS_overshoot_frac_max'] for r in rows)
    a['TILS_geo_work_ratio'] = geo([r['TILS_mean'] / r['R'] for r in rows if r['R'] > 0])
    for lab in ('T', 'DOVE', 'TILS_seed1', 'ILST_seed1'):
        k = f'tratio_{lab}'
        if all(k in r for r in rows):
            a[f'geo_{k}'] = geo([r[k] for r in rows])
            a[f'geo_wratio_{lab}'] = geo([r[f'wratio_{lab}'] for r in rows])
            a[f'faster_{lab}'] = sum(1 for r in rows if r[k] < 1.0)
    a['kernel_Rstar_ms_median'] = statistics.median(r['kernel_Rstar_ms'] for r in rows)
    a['sec_DOVE_median'] = statistics.median(r['sec_DOVE'] for r in rows)
    a['sec_T_median'] = statistics.median(r['sec_T'] for r in rows)
    a['verification_orders'] = sum(r['verification_orders'] for r in rows)
    a['verification_failed'] = sum(1 for r in rows if not r['verification_ok'])
    return a


def tex_safe(t):
    t = ''.join(ch for ch in str(t) if ch not in '\\{}$^~#')
    return t.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def coverage_macros(prefix, cov, a):
    """cov: dict(planned=int, not_analysed={name: reason}, failed={name: reason})."""
    P = f'PH{prefix}'
    n_ok = a.get('count', 0)
    na = {tex_safe(k): tex_safe(v) for k, v in cov['not_analysed'].items()}
    out = {P + 'Count': str(n_ok), P + 'Planned': str(cov['planned']),
           P + 'NotAnalysed': ('none' if not na else '; '.join(f'{k} ({v})' for k, v in na.items())),
           P + 'CountText': (f'{n_ok} of {cov["planned"]}' +
                             ('' if not na else ' (' + '; '.join(f'{k}: {v}' for k, v in na.items()) + ')'))}
    nfail = len(cov['failed']) + a.get('verification_failed', 0)
    if nfail:
        out[P + 'Check'] = f'failed or not analysed on {nfail} graph{"s" if nfail != 1 else ""} (see log)'
    elif n_ok:
        out[P + 'Check'] = f'passed for all {a["verification_orders"]} post-hoc orders'
    else:
        out[P + 'Check'] = 'no graph analysed'
    return out


def macros(prefix, a, mt):
    """prefix: 'Suite', 'Held', 'Audit' or 'All'."""
    pct = mt.pct
    out = {}
    if not a.get('count'):
        return out
    P = f'PH{prefix}'
    out[P + 'Count'] = str(a['count'])
    if 'imp_col_best_mean' in a:
        out[P + 'ColBestMean'] = pct(a['imp_col_best_mean'])
        out[P + 'ColFixedMean'] = pct(a['imp_col_fixed_mean'])
        for v, nm in zip(VARIANTS, ('SLasc', 'SLdesc', 'DEGasc', 'DEGdesc')):
            out[P + f'Col{nm}Mean'] = pct(a[f'imp_col_{v}_mean'])
        out[P + 'ColBestBeatsRef'] = f'{a["col_best_beats_R"]} of {a["count"]}'
        out[P + 'ColFixedBeatsRef'] = f'{a["col_fixed_beats_R"]} of {a["count"]}'
        out[P + 'ColTimeMax'] = f'{a["sec_col_max"]:.2f}'
    for meth, nm in (('ILST', 'ILST'), ('CILST_fixed', 'CILSTFixed'), ('CILST_best', 'CILSTBest')):
        if f'imp_{meth}_mean' not in a:
            continue
        out[P + nm + 'Mean'] = pct(a[f'imp_{meth}_mean'])
        out[P + nm + 'Median'] = pct(a[f'imp_{meth}_median'])
        out[P + nm + 'ITR'] = '/'.join(map(str, a[f'{meth}_ITR']))
        out[P + nm + 'vsTILS'] = '/'.join(map(str, a[f'{meth}_vs_TILS']))
        out[P + nm + 'MinusTILS'] = pct(a[f'{meth}_minus_TILS_pp'])
        out[P + nm + 'vsDove'] = '/'.join(map(str, a[f'{meth}_vs_DOVE']))
        out[P + nm + 'Overshoot'] = f'{a[f"{meth}_overshoot_max"]:.2f}'
        out[P + nm + 'OvershootPct'] = f'{a[f"{meth}_overshoot_pct_max"]:.2f}'
        if f'{meth}_vs_ILST' in a:
            out[P + nm + 'vsILST'] = '/'.join(map(str, a[f'{meth}_vs_ILST']))
            out[P + nm + 'MinusILST'] = pct(a[f'{meth}_minus_ILST_pp'])
    out[P + 'TILSMean'] = pct(a['imp_TILS_mean'])
    out[P + 'TILSOvershootPct'] = f'{a["TILS_overshoot_pct_max"]:.2f}'
    out[P + 'DoveMean'] = pct(a['imp_DOVE_mean'])
    for lab, nm in (('T', 'T'), ('DOVE', 'Dove'), ('TILS_seed1', 'TILS'), ('ILST_seed1', 'ILST')):
        if f'geo_tratio_{lab}' in a:
            out[P + 'TimeRatio' + nm] = f'{a[f"geo_tratio_{lab}"]:.4f}'
            out[P + 'WorkRatio' + nm] = f'{a[f"geo_wratio_{lab}"]:.4f}'
            out[P + 'Faster' + nm] = f'{a[f"faster_{lab}"]} of {a["count"]}'
    out[P + 'KernelMsMedian'] = f'{a["kernel_Rstar_ms_median"]:.2f}'
    out[P + 'DoveSecMedian'] = f'{a["sec_DOVE_median"]:.2f}'
    return out


def rows_tex(rows, mt):
    pct, esc = mt.pct, mt.row_label
    colour, ils, timing = [], [], []
    for r in rows:
        lab = esc(r['label'])
        colour.append(' & '.join([lab, mt.num(r['R'], work=True)] + [pct(r[f'imp_col_{v}']) for v in VARIANTS] +
                                 [r['col_best_label'].replace('-', '--'), f'{r["sec_col_DEG-desc"]:.2f}']) + r' \\')
        cells = [lab, pct(r['imp_TILS'])]
        for meth in METHODS:
            cells.append(f'{pct(r[f"imp_{meth}"])}$\\pm${r[f"imp_{meth}_sd"]:.2f}' if f'imp_{meth}' in r else '--')
        ils.append(' & '.join(cells) + r' \\')
        tcells = [lab, f'{r["kernel_Rstar_ms"]:.2f}']
        for k in ('T', 'DOVE', 'TILS_seed1'):
            tcells.append(f'{r[f"tratio_{k}"]:.3f}' if f'tratio_{k}' in r else '--')
        tcells += [f'{r["sec_T"]:.3f}', f'{r["sec_DOVE"]:.1f}', f'{r["sec_TILS_mean"]:.1f}']
        timing.append(' & '.join(tcells) + r' \\')
    return colour, ils, timing


def write_all(results, out, log=print):
    mt = _mt()
    results, out = Path(results), Path(out)
    rows = {'A': [], 'B': []}
    # coverage bookkeeping: (part, split, name, None if analysed else (kind, reason))
    cover = []
    problems = {}
    for part in ('A', 'B'):
        plan_path = results / f'plan_part{part}.json'
        if not plan_path.exists():
            continue
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        for item in plan['graphs']:
            name = item['name']
            gdir = out / f'part{part}' / name
            pp = gdir / 'posthoc_result.json'
            rp = results / f'part{part}' / name / 'result.json'
            res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else None
            split = (res or {}).get('split') or item.get('split')
            if res is None:
                cover.append((part, split, name, ('prereg', 'not run in the pre-registered run')))
                continue
            if res.get('status') != 'done':
                st = res.get('status')
                why = str(res.get('reason', ''))
                short = {'missing': 'missing' + (', download failed' if 'download failed' in why else
                                                  (', ' + why[:60] if why else '')),
                         'skipped_large': 'skipped, more than 20M edges',
                         'excluded_check_failed': 'excluded, artifact check failed'}.get(st, f'status {st}')
                cover.append((part, split, name, ('prereg', short)))
                continue
            ph = json.loads(pp.read_text(encoding='utf-8')) if pp.exists() else None
            if ph is None or ph.get('status') != 'done':
                if ph is not None:
                    why = f'post-hoc {str(ph.get("status")).replace("_", " ")}'
                elif (gdir / 'posthoc_error.json').exists():
                    err = json.loads((gdir / 'posthoc_error.json').read_text(encoding='utf-8')).get('error', '')
                    why = f'post-hoc error, {err.split("(")[0][:40] or "exception"}'
                else:
                    why = 'post-hoc not run'
                cover.append((part, split, name, ('posthoc', why)))
                problems[f'part{part}/{name}'] = why
                continue
            row = graph_row(res, ph)
            if not row['verification_ok']:
                problems[f'part{part}/{name}'] = 'post-hoc verification FAILED'
            rows[part].append(row)
            cover.append((part, split, name, None))
    audit = [r for r in rows['A'] if r['split'] == 'Audit']
    aggs = dict(Suite=aggregate(rows['A']), Held=aggregate(rows['B']), Audit=aggregate(audit),
                All=aggregate(rows['A'] + rows['B']))
    sel = dict(Suite=lambda c: c[0] == 'A', Held=lambda c: c[0] == 'B',
               Audit=lambda c: c[0] == 'A' and c[1] == 'Audit', All=lambda c: True)
    coverage = {}
    for k in aggs:
        cs = [c for c in cover if sel[k](c)]
        coverage[k] = dict(planned=len(cs),
                           not_analysed={c[2]: c[3][1] for c in cs if c[3] is not None},
                           failed={c[2]: c[3][1] for c in cs if c[3] is not None and c[3][0] == 'posthoc'})
    mac = {}
    for k, a in aggs.items():
        mac.update(macros(k, a, mt))
        mac.update(coverage_macros(k, coverage[k], a))
    import ph_kernels
    mac['PHFixedVariant'] = ph_kernels.FIXED_VARIANT.replace('-', '--')
    lines = ['% POST-HOC, SECONDARY analysis (NOT pre-registered). Generated by make_posthoc_tables.py.',
             '% Kept separate from the pre-registered macros.tex; see README_POSTHOC.md for definitions.']
    for k in sorted(mac):
        lines.append(f'\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{mac[k]}}}')
    (out / 'posthoc_macros.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for part, nm in (('A', 'suite'), ('B', 'heldout')):
        c, i, t = rows_tex(rows[part], mt)
        (out / f'posthoc_{nm}_colour_rows.tex').write_text('\n'.join(c) + '\n', encoding='utf-8')
        (out / f'posthoc_{nm}_ils_rows.tex').write_text('\n'.join(i) + '\n', encoding='utf-8')
        (out / f'posthoc_{nm}_timing_rows.tex').write_text('\n'.join(t) + '\n', encoding='utf-8')
    allrows = rows['A'] + rows['B']
    if allrows:
        keys = []
        for r in allrows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / 'posthoc_per_graph.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in allrows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    (out / 'posthoc_summary.json').write_text(json.dumps(dict(
        note='POST-HOC, SECONDARY, NOT PRE-REGISTERED', aggregates=aggs, coverage=coverage, problems=problems,
        macros=mac, per_graph=allrows),
        indent=1), encoding='utf-8')
    log(f'posthoc tables: suite {mac.get("PHSuiteCountText", "0")}, held-out {mac.get("PHHeldCountText", "0")} -> {out}')
    return aggs, mac, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pc-eval', required=True, type=Path)
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.pc_eval.resolve()))
    sys.path.insert(0, str(HERE))
    _, _, problems = write_all(args.results.resolve(), args.out.resolve())
    if problems:
        print(f'POST-HOC FAILURES on {len(problems)} graph(s): {problems}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
