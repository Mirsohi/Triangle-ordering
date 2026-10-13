#!/usr/bin/env python3
"""Writes <paper>/generated/{macros.tex, suite_rows.tex, heldout_rows.tex} from run_eval.py results.

python make_tables.py --out <results dir> --paper <paper dir> [--allow-test]

Metrics follow PREREGISTRATION.md Section 3.11. Also writes <out>/tables_summary.json with every number used
(including per-seed and T+ILS-vs-T comparisons). Quick/local/test results are refused unless --allow-test.
"""
import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ('Development', 'Validation', 'Audit')
FINAL = ('done', 'missing', 'skipped_large', 'excluded_check_failed')


def imp(R, w):
    return 0.0 if R == 0 else 100.0 * (R - w) / R


def pct(x):
    s = f'{x:.2f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    if s in ('$-$0.00',):
        s = '0.00'
    return s


def num(x, work=False):
    x = int(x)
    if x < 1000:
        return str(x)
    if x < 1_000_000:
        return f'{x / 1e3:.1f}k'
    return f'{x / 1e6:.3f}m' if work else f'{x / 1e6:.2f}m'


def tex_escape(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def row_label(s):
    """Long labels with a space (Facebook page--page) are set on two lines to keep Table 5 in the column."""
    s = tex_escape(s)
    if len(s) > 14 and ' ' in s:
        a, b = s.split(' ', 1)
        return r'\begin{tabular}[t]{@{}l@{}}' + a + r'\\' + b + r'\end{tabular}'
    return s


def load_part(root, part):
    plan_path = root / f'plan_part{part}.json'
    if not plan_path.exists():
        return None, []
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    rows = []
    for item in plan['graphs']:
        rp = root / f'part{part}' / item['name'] / 'result.json'
        r = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else dict(name=item['name'], status='not_run')
        r.setdefault('label', item.get('label', item['name']))
        r.setdefault('split', item.get('split'))
        rows.append(r)
    return plan, rows


def graph_metrics(r):
    R = int(r['reference']['work'])
    T = int(r['T']['work'])
    D = int(r['DOVE']['work'])
    ws = [int(v['work']) for k, v in sorted(r['TILS'].items(), key=lambda kv: int(kv[0]))]
    imps = [imp(R, w) for w in ws]
    Wbar = sum(ws) / len(ws)
    def cmp(a, b):
        return 'win' if a < b else ('tie' if a == b else 'loss')
    return dict(name=r['name'], label=r['label'], split=r.get('split'), n=r['n'], m=r['m'], R=R, T=T, D=D, tils=ws,
                imp_T=imp(R, T), imp_D=imp(R, D), imp_tils=sum(imps) / len(imps),
                sd_tils=statistics.stdev(imps) if len(imps) > 1 else 0.0, Wbar=Wbar,
                itr='I' if Wbar < R else ('T' if Wbar == R else 'R'), vs_dove=cmp(Wbar, D), vs_T=cmp(Wbar, T),
                vs_dove_seeds=[cmp(w, D) for w in ws], vs_T_seeds=[cmp(w, T) for w in ws],
                ratio=(Wbar / R) if R > 0 else None, fwd=int(r['reference']['forward_min']),
                ref_drop=imp(int(r['reference']['forward_min']), R), ref_label=r['reference']['label'],
                budget=r['budget'], tils_seconds=[v['info'].get('total_seconds') for k, v in sorted(r['TILS'].items())],
                ver_ok=r['verification']['all_ok'], n_orders=r['verification']['n_orders'],
                n_orders_ok=sum(1 for v in r['verification']['orders'].values() if v['ok']),
                n_orders_bad=(sum(1 for v in r['verification']['orders'].values() if not v['ok'])
                              or (0 if r['verification']['all_ok'] else r['verification']['n_orders'])),
                triangles=r['verification']['triangles'],
                neigh=r['reference'].get('neigh_candidates', []), volt=r.get('checks', {}).get('volt'))


def aggregate(ms):
    if not ms:
        return None
    wtl = lambda key: '/'.join(str(sum(1 for x in ms if x[key] == k)) for k in ('win', 'tie', 'loss'))
    ratios = [x['ratio'] for x in ms if x['ratio'] is not None and x['ratio'] > 0]
    return dict(count=len(ms), mean_T=statistics.mean(x['imp_T'] for x in ms), mean_D=statistics.mean(x['imp_D'] for x in ms),
                mean_tils=statistics.mean(x['imp_tils'] for x in ms), median_tils=statistics.median(x['imp_tils'] for x in ms),
                itr='/'.join(str(sum(1 for x in ms if x['itr'] == k)) for k in 'ITR'),
                geo=math.exp(statistics.mean(math.log(r) for r in ratios)) if ratios else float('nan'),
                vs_dove=wtl('vs_dove'), vs_T=wtl('vs_T'),
                vs_dove_seeds='/'.join(str(sum(s.count(k) for s in (x['vs_dove_seeds'] for x in ms))) for k in ('win', 'tie', 'loss')),
                vs_T_seeds='/'.join(str(sum(s.count(k) for s in (x['vs_T_seeds'] for x in ms))) for k in ('win', 'tie', 'loss')),
                ref_drop=statistics.mean(x['ref_drop'] for x in ms),
                orders=sum(x['n_orders'] for x in ms), orders_ok=sum(x['n_orders_ok'] for x in ms),
                orders_failed=sum(x['n_orders_bad'] for x in ms),
                graphs_ver_ok=sum(1 for x in ms if x['ver_ok']),
                ci=bootstrap_ci([x['imp_tils'] for x in ms]),
                sign_p=sign_test_p(sum(1 for x in ms if x['vs_dove'] == 'win'), sum(1 for x in ms if x['vs_dove'] == 'loss')))


def check_text(a):
    """Completes 'Exact-work and triangle-count checks: ...'. A graph whose orders all pass individually but whose
    triangle counts differ between orders counts as failed for all of its orders."""
    if a is None:
        return None
    total = a['orders']
    failed = a['orders_failed']
    if failed == 0:
        return f'passed for all {total} saved orders'
    return f'failed for {failed} of {total} (see log)'


def bootstrap_ci(values, resamples=10_000, seed=0):
    """95% percentile bootstrap interval over graphs for the mean (PREREGISTRATION.md Amendment 2)."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    G = v.size
    idx = np.random.default_rng(seed).integers(0, G, size=(resamples, G))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_test_p(wins, losses):
    """Two-sided exact sign test, ties excluded."""
    N = wins + losses
    if N == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(N, i) for i in range(k + 1)) / 2 ** N
    return min(1.0, 2 * tail)


def p_text(p):
    if p < 0.001:
        return '<0.001'
    return f'{p:#.2g}'.rstrip('.')


def ci_text(ci):
    return f'[{pct(ci[0])}, {pct(ci[1])}]\\%'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--paper', required=True, type=Path)
    ap.add_argument('--allow-test', action='store_true', help='accept quick/local/test results (never for the paper)')
    args = ap.parse_args()
    root = args.out.resolve()
    planA, rowsA = load_part(root, 'A')
    planB, rowsB = load_part(root, 'B')
    if planA is None and planB is None:
        sys.exit(f'no plan_partA.json / plan_partB.json in {root}')
    problems = []
    for plan, part in ((planA, 'A'), (planB, 'B')):
        if plan is not None and plan.get('mode') != 'full':
            problems.append(f'Part {part} results are from a {plan.get("mode")} run')
    if planB is not None and planB.get('mode') == 'full':
        sys.path.insert(0, str(HERE))
        from run_eval import EXPECTED_PREREG_SHA256
        if not planB.get('prereg') or planB['prereg'].get('sha256') != EXPECTED_PREREG_SHA256:
            problems.append('Part B plan does not carry the pre-registered hash')
        if len(planB['graphs']) != 10:
            problems.append('Part B plan does not list the ten pre-registered graphs')
    for r in rowsA + rowsB:
        if r.get('provenance', {}).get('test_url_override'):
            problems.append(f'{r["name"]}: downloaded through a test URL override')
    if problems and not args.allow_test:
        sys.exit('refusing to write paper tables:\n  ' + '\n  '.join(problems))
    test = bool(problems)

    complete = True
    for rows in (rowsA, rowsB):
        for r in rows:
            if r.get('status') not in FINAL:
                complete = False
    if planA is None or planB is None:
        complete = False

    mA = [graph_metrics(r) for r in rowsA if r.get('status') == 'done']
    mB = [graph_metrics(r) for r in rowsB if r.get('status') == 'done']
    aA, aB = aggregate(mA), aggregate(mB)
    per_split = {s: aggregate([x for x in mA if x['split'] == s]) for s in SPLITS}
    planned_split = {s: sum(1 for r in rowsA if r.get('split') == s) for s in SPLITS}

    P = r'\pend'
    macros = {}
    macros['SuiteRefDrop'] = pct(aA['ref_drop']) if aA else P
    macros['SuiteDoveMean'] = pct(aA['mean_D']) if aA else P
    macros['SuiteTMean'] = pct(aA['mean_T']) if aA else P
    macros['SuiteTIKMean'] = pct(aA['mean_tils']) if aA else P
    macros['SuiteTIKMedian'] = pct(aA['median_tils']) if aA else P
    macros['SuiteTIKITR'] = aA['itr'] if aA else P
    macros['SuiteTIKGeo'] = f'{aA["geo"]:.4f}' if aA else P
    macros['SuiteTIKvsDove'] = aA['vs_dove'] if aA else P
    macros['SuiteTriCheck'] = check_text(aA) if aA else P
    nB = len(rowsB)
    macros['HeldCount'] = (str(aB['count']) if aB['count'] == nB else f'{aB["count"]} of {nB}') if aB else (f'0 of {nB}' if planB else P)
    macros['HeldDoveMean'] = pct(aB['mean_D']) if aB else P
    macros['HeldTMean'] = pct(aB['mean_T']) if aB else P
    macros['HeldTIKMean'] = pct(aB['mean_tils']) if aB else P
    macros['HeldTIKMedian'] = pct(aB['median_tils']) if aB else P
    macros['HeldTIKITR'] = aB['itr'] if aB else P
    macros['HeldTIKGeo'] = f'{aB["geo"]:.4f}' if aB else P
    macros['HeldTIKvsDove'] = aB['vs_dove'] if aB else P
    if planB is None:
        macros['HeldNeigh'] = P
    elif not planB.get('volt'):
        macros['HeldNeigh'] = 'none (VOLT not available)'
    else:
        full = sum(1 for x in mB if x['volt'] and sum(1 for v in x['volt'] if v.get('ok')) == 5)
        if mB and full == len(mB):
            macros['HeldNeigh'] = 'five seeded runs, both directions'
        else:
            macros['HeldNeigh'] = (f'up to five seeded runs, both directions (all five succeeded on {full} of '
                                   f'{len(mB)} graphs; see log)')
    macros['HeldTriCheck'] = check_text(aB) if aB else P
    # extra macros (not in the placeholder; pre-registered comparison against T and the suite size)
    macros['SuiteTIKCI'] = ci_text(aA['ci']) if aA else P
    macros['SuiteTIKvsDoveP'] = p_text(aA['sign_p']) if aA else P
    macros['SuiteTIKvsT'] = aA['vs_T'] if aA else P
    macros['HeldTIKCI'] = ci_text(aB['ci']) if aB else P
    macros['HeldTIKvsDoveP'] = p_text(aB['sign_p']) if aB else P
    macros['HeldTIKvsT'] = aB['vs_T'] if aB else P
    macros['SuiteCount'] = str(aA['count']) if aA else P

    gen = args.paper.resolve() / 'generated'
    gen.mkdir(parents=True, exist_ok=True)
    # every macro defined by the paper's current macros.tex (placeholder or previous output) must be defined again
    import re
    old_macros = gen / 'macros.tex'
    if old_macros.exists():
        needed = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', old_macros.read_text(encoding='utf-8')))
        missing = sorted(needed - set(macros))
        if missing:
            sys.exit(f'the paper expects macros that make_tables.py does not define: {missing}')
    lines = [r'\resultspendingfalse' if complete else r'\resultspendingtrue',
             '% Generated by pc_eval/make_tables.py' + (' -- TEST DATA, NOT FOR THE PAPER' if test else '') + '.',
             f'% Source: {root}']
    placeholder_order = ['SuiteRefDrop', 'SuiteDoveMean', 'SuiteTMean', 'SuiteTIKMean', 'SuiteTIKMedian', 'SuiteTIKITR',
                         'SuiteTIKGeo', 'SuiteTIKvsDove', 'SuiteTriCheck', 'HeldCount', 'HeldDoveMean', 'HeldTMean',
                         'HeldTIKMean', 'HeldTIKMedian', 'HeldTIKITR', 'HeldTIKGeo', 'HeldTIKvsDove', 'HeldNeigh',
                         'HeldTriCheck', 'SuiteTIKCI', 'SuiteTIKvsDoveP', 'SuiteTIKvsT', 'HeldTIKCI',
                         'HeldTIKvsDoveP', 'HeldTIKvsT', 'SuiteCount']
    for k in placeholder_order:
        lines.append(f'\\newcommand{{\\{k}}}{{{macros[k]}}}')
    (gen / 'macros.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # suite rows: Split & Graphs & DOVE & T & T+ILS & I/T/R
    srows = ['% Generated by pc_eval/make_tables.py' + (' -- TEST DATA' if test else '') + '.']
    for s in SPLITS:
        a = per_split[s]
        cnt = str(a['count']) if a else '0'
        if a is None or a['count'] != planned_split[s]:
            cnt = f'{a["count"] if a else 0} of {planned_split[s]}'
        if a:
            srows.append(f'{s} & {cnt} & {pct(a["mean_D"])} & {pct(a["mean_T"])} & {pct(a["mean_tils"])} & {a["itr"]} \\\\')
        else:
            srows.append(f'{s} & {cnt} & -- & -- & -- & -- \\\\')
    srows.append(r'\midrule')
    if aA:
        cnt = str(aA['count']) if aA['count'] == len(rowsA) else f'{aA["count"]} of {len(rowsA)}'
        srows.append(f'All & {cnt} & {pct(aA["mean_D"])} & {pct(aA["mean_T"])} & {pct(aA["mean_tils"])} & {aA["itr"]} \\\\')
    else:
        srows.append(r'All & 0 & \pend & \pend & \pend & \pend \\')
    for r in rowsA:
        if r.get('status') != 'done':
            srows.append(f'% not completed: {r["name"]} ({r.get("status")}: {r.get("reason", "")})'.replace('\n', ' '))
    (gen / 'suite_rows.tex').write_text('\n'.join(srows) + '\n', encoding='utf-8')

    # held-out rows: Graph & n & m & Lambda_R* & T & DOVE & T+ILS
    hrows = ['% Generated by pc_eval/make_tables.py' + (' -- TEST DATA' if test else '') + '.']
    byname = {x['name']: x for x in mB}
    for r in rowsB:
        lab = row_label(r['label'])
        if r.get('status') == 'done':
            x = byname[r['name']]
            hrows.append(f'{lab} & {num(x["n"])} & {num(x["m"])} & {num(x["R"], work=True)} & {pct(x["imp_T"])} & '
                         f'{pct(x["imp_D"])} & {pct(x["imp_tils"])}$\\pm${x["sd_tils"]:.2f} \\\\')
        elif r.get('status') == 'skipped_large':
            hrows.append(f'{lab} & {num(r["n"])} & {num(r["m"])} & \\multicolumn{{4}}{{c}}{{skipped ($m>20$M)}} \\\\')
        else:
            hrows.append(f'{lab} & \\multicolumn{{6}}{{c}}{{missing}} \\\\')
            hrows.append(f'% {r["name"]}: {r.get("status")} {r.get("reason", "")}'.replace('\n', ' ')[:300])
    hrows.append(r'\midrule')
    if aB:
        hrows.append(f'Mean & & & & {pct(aB["mean_T"])} & {pct(aB["mean_D"])} & {pct(aB["mean_tils"])} \\\\')
    else:
        hrows.append(r'Mean & & & & -- & -- & -- \\')
    (gen / 'heldout_rows.tex').write_text('\n'.join(hrows) + '\n', encoding='utf-8')

    summary = dict(test=test, problems=problems, complete=complete, macros=macros, suite=aA, suite_splits=per_split,
                   heldout=aB, suite_graphs=mA, heldout_graphs=mB,
                   not_completed=[dict(name=r['name'], status=r.get('status'), reason=r.get('reason'))
                                  for r in rowsA + rowsB if r.get('status') != 'done'])
    (root / 'tables_summary.json').write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
    print(f'wrote {gen / "macros.tex"}, suite_rows.tex, heldout_rows.tex (complete={complete}, test={test})')
    for k in placeholder_order:
        print(f'  \\{k} = {macros[k]}')
    if aA:
        print(f'  suite T+ILS vs DOVE per seed {aA["vs_dove_seeds"]}, vs T {aA["vs_T"]} (per seed {aA["vs_T_seeds"]})')
    if aB:
        print(f'  held-out T+ILS vs DOVE per seed {aB["vs_dove_seeds"]}, vs T {aB["vs_T"]} (per seed {aB["vs_T_seeds"]})')


if __name__ == '__main__':
    main()
