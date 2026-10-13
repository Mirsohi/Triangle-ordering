#!/usr/bin/env python3
"""Tables and macros of the SUPPLEMENTARY held-out Neigh run (not pre-registered; see SUPP_PROTOCOL.md).

python make_supp_tables.py --results <pc_eval results> --out <supplementary output dir>

Writes into <out>: supp_summary.json, supp_per_graph.csv, supp_macros.tex (\\NS... macros), supp_heldout_rows.tex
(rows: graph & R*N & gap & frozen T+ILS vs R*N & rerun T & DOVE & T+ILS vs R*N).
Returns {graph: problem} for graphs that failed or were not analysed.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

B_RESAMPLES = 20000
B_SEED = 2027


def imp(R, w):
    return 0.0 if R == 0 else 100.0 * (R - w) / R


def boot_ci(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(B_SEED)
    m = v[rng.integers(0, len(v), (B_RESAMPLES, len(v)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def itr(vals):
    return sum(v > 0 for v in vals), sum(v == 0 for v in vals), sum(v < 0 for v in vals)


def wtl(a, b):
    w = sum(x < y for x, y in zip(a, b))
    t = sum(x == y for x, y in zip(a, b))
    return w, t, len(a) - w - t


def capped_work(run, B, strict=True):
    """Deadline-capped T+ILS (SUPP_PROTOCOL.md). Returns (work, branch).
    no-final-pass rule (strict=False): a run whose total time exceeds B is scored with its ILS incumbent, i.e. the
        final tangent pass is dropped (the ILS phase itself may end a few ms after B because of batch granularity).
    strict rule (strict=True): only an order that existed at B counts: the ILS incumbent if ILS ended by B, else the
        tangent-phase order if that phase ended by B, else the start order R*."""
    info = run.get('info', {})
    tot = info.get('total_seconds')
    if tot is None or tot <= B:
        return int(run['work']), 'final'
    if not strict or info.get('ils_end', math.inf) <= B:
        return int(info['ils_incumbent_work']), 'ils'
    if info['T_phase'].get('seconds_since_start', math.inf) <= B:
        return int(info['T_phase']['work']), 'tangent'
    return int(info['start_work']), 'start'


def timing_stats(kt, a, b, seed=B_SEED):
    """ratio b/a: ratio of medians, paired geometric mean over repetitions and paired bootstrap interval."""
    po = kt['per_order']
    x, y = np.asarray(po[a]['samples'], float), np.asarray(po[b]['samples'], float)
    lr = np.log(y / x)
    rng = np.random.default_rng(seed)
    bm = lr[rng.integers(0, len(lr), (B_RESAMPLES, len(lr)))].mean(1)
    return dict(ratio_medians=po[b]['median_s'] / po[a]['median_s'], paired_geo=float(np.exp(lr.mean())),
                ci=[float(np.exp(np.quantile(bm, .025))), float(np.exp(np.quantile(bm, .975)))],
                faster_reps=int((lr < 0).sum()), reps=int(len(lr)),
                work_ratio=po[b]['work'] / po[a]['work'] if po[a]['work'] else None, base_ms=po[a]['median_s'] * 1e3)


def geo_ci(ratios):
    L = np.log(np.asarray(ratios, float))
    if len(L) == 0:
        return None, None
    rng = np.random.default_rng(B_SEED)
    g = np.exp(L[rng.integers(0, len(L), (B_RESAMPLES, len(L)))].mean(1))
    return float(np.exp(L.mean())), [float(np.quantile(g, .025)), float(np.quantile(g, .975))]


def f2(x):
    return 'n/a' if x is None else (f'$-${abs(x):.2f}' if x < 0 else f'{x:.2f}')


def ci_txt(ci, pct=True):
    return 'n/a' if ci is None else f'[{f2(ci[0])}, {f2(ci[1])}]' + ('\\%' if pct else '')


def p_txt(p):
    return '$<$0.001' if p < 0.001 else f'{p:.2g}'


def tex_safe(t):
    return str(t).replace('\\', '/').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')


def graph_row(res, sr):
    ref = sr['reference']
    RN = int(ref['new_work'])
    fw = sr['frozen_works']
    ftils = [v for k, v in sorted(fw.items()) if k.startswith('TILS_seed')]
    rr = sr['rerun']
    rtils_runs = [rr['TILS'][s] for s in sorted(rr['TILS'], key=int)]
    rtils = [int(r['work']) for r in rtils_runs]
    B = float(rr['budget'])
    capped = [capped_work(r, B, strict=True) for r in rtils_runs]
    nofinal = [capped_work(r, B, strict=False) for r in rtils_runs]
    row = dict(name=sr['name'], label=sr.get('label', sr['name']), n=sr['n'], m=sr['m'],
               R=int(ref['frozen_work']), R_label=ref['frozen_label'], RN=RN, RN_label=ref['new_label'],
               changed=bool(ref['changed']), gap=ref['gap_percent'], neigh_best=ref['neigh_best_work'],
               neigh_ok=sr['neigh']['ok_seeds'], neigh_seconds=sr['neigh']['seconds_total'],
               frozen_imp_T=imp(RN, fw['T']), frozen_imp_DOVE=imp(RN, fw['DOVE']),
               frozen_imp_TILS=imp(RN, float(np.mean(ftils))),
               frozen_imp_TILS_vsR=imp(ref['frozen_work'], float(np.mean(ftils))),
               frozen_TILS_mean=float(np.mean(ftils)), frozen_DOVE=int(fw['DOVE']),
               rerun_reused=bool(rr['reused']), rerun_budget=B,
               rerun_T=int(rr['T']['work']), rerun_DOVE=int(rr['DOVE']['work']), rerun_TILS_works=rtils,
               rerun_TILS_mean=float(np.mean(rtils)),
               rerun_imp_T=imp(RN, rr['T']['work']), rerun_imp_DOVE=imp(RN, rr['DOVE']['work']),
               rerun_imp_TILS=imp(RN, float(np.mean(rtils))),
               rerun_imp_TILS_capped=imp(RN, float(np.mean([w for w, _ in capped]))),
               rerun_imp_TILS_nofinal=imp(RN, float(np.mean([w for w, _ in nofinal]))),
               capped_branches=[br for _, br in capped],
               rerun_imp_TILS_sd=float(np.std([imp(RN, w) for w in rtils], ddof=1)) if len(rtils) > 1 else 0.0,
               rerun_TILS_max_over_frac=max((r.get('info', {}).get('total_seconds', 0) - B) / B for r in rtils_runs),
               verification_ok=sr['verification']['all_ok'], verification_orders=sr['verification']['n_orders'])
    kt = sr['kernel_timing']
    row['timing_frozen'] = timing_stats(kt, 'Rstar', 'TILS_seed1')
    if 'RstarN' in kt['per_order']:
        row['timing_rerun'] = timing_stats(kt, 'RstarN', 'TILSN_seed1')
        row['timing_RN_vs_R'] = timing_stats(kt, 'Rstar', 'RstarN')
    else:
        row['timing_rerun'] = row['timing_frozen']
    return row


def aggregate(rows):
    a = dict(count=len(rows))
    if not rows:
        return a
    for key in ('frozen_imp_T', 'frozen_imp_DOVE', 'frozen_imp_TILS', 'rerun_imp_T', 'rerun_imp_DOVE',
                'rerun_imp_TILS', 'rerun_imp_TILS_capped', 'rerun_imp_TILS_nofinal', 'gap', 'frozen_imp_TILS_vsR'):
        v = [r[key] for r in rows]
        a[key + '_mean'] = float(np.mean(v))
        a[key + '_median'] = float(np.median(v))
        a[key + '_itr'] = itr(v)
    a['gap_max'] = max(r['gap'] for r in rows)
    a['gap_max_graph'] = max(rows, key=lambda r: r['gap'])['label']
    a['changed'] = sum(r['changed'] for r in rows)
    a['changed_list'] = [r['label'] for r in rows if r['changed']]
    a['neigh_ok'] = sum(r['neigh_ok'] for r in rows)
    a['frozen_imp_TILS_ci'] = boot_ci([r['frozen_imp_TILS'] for r in rows])
    a['rerun_imp_TILS_ci'] = boot_ci([r['rerun_imp_TILS'] for r in rows])
    a['rerun_imp_TILS_capped_ci'] = boot_ci([r['rerun_imp_TILS_capped'] for r in rows])
    br = [b for r in rows for b in r['capped_branches']]
    a['capped_branch_counts'] = {k: br.count(k) for k in ('final', 'ils', 'tangent', 'start')}
    a['reused'] = sum(r['rerun_reused'] for r in rows)
    for other, key in (('DOVE', 'rerun_DOVE'), ('T', 'rerun_T')):
        w, t, l = wtl([r['rerun_TILS_mean'] for r in rows], [r[key] for r in rows])
        a[f'rerun_TILS_vs_{other}'] = (w, t, l)
        a[f'rerun_TILS_vs_{other}_p'] = sign_p(w, l)
    w, t, l = wtl([r['frozen_TILS_mean'] for r in rows], [r['frozen_DOVE'] for r in rows])
    a['frozen_TILS_vs_DOVE'] = (w, t, l)
    a['rerun_TILS_max_over_frac'] = max(r['rerun_TILS_max_over_frac'] for r in rows)
    # timing ratios are T+ILS / reference (below 1 = T+ILS lists faster). Geo/CI: geometric mean over graphs of the
    # per-graph paired geometric-mean ratio, bootstrap over graphs. Below/Above: graphs whose paired bootstrap interval
    # over repetitions lies entirely below/above 1.
    for key in ('timing_frozen', 'timing_rerun'):
        g, ci = geo_ci([r[key]['paired_geo'] for r in rows])
        a[key + '_geo'] = g
        a[key + '_ci'] = ci
        a[key + '_graphs_ci_below1'] = sum(r[key]['ci'][1] < 1 for r in rows)
        a[key + '_graphs_ci_above1'] = sum(r[key]['ci'][0] > 1 for r in rows)
        a[key + '_work_geo'] = geo_ci([r[key]['work_ratio'] for r in rows])[0]
    a['verified_orders'] = sum(r['verification_orders'] for r in rows)
    return a


def write_all(results, out, log=print, validation=True):
    results, out = Path(results), Path(out)
    problems = {}
    rows, vrows = [], []
    planB = json.loads((results / 'plan_partB.json').read_text(encoding='utf-8'))
    planned = [it['name'] for it in planB['graphs']]
    not_analysed = {}
    for name in planned:
        rp = results / 'partB' / name / 'result.json'
        res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else dict(status='not_run')
        if res.get('status') != 'done':
            not_analysed[name] = 'missing, download failed' if res.get('status') == 'missing' else res.get('status')
            continue
        sp = out / 'partN' / name / 'supp_result.json'
        ep = out / 'partN' / name / 'supp_error.json'
        if ep.exists():
            problems[name] = 'error: ' + json.loads(ep.read_text(encoding='utf-8')).get('error', '?')[:80]
            continue
        if not sp.exists():
            problems[name] = 'not run'
            continue
        sr = json.loads(sp.read_text(encoding='utf-8'))
        if sr.get('status') != 'done':
            problems[name] = sr.get('status')
            continue
        if not sr['verification']['all_ok']:
            problems[name] = 'verification failed'
            continue
        rows.append(graph_row(res, sr))
    planA = results / 'plan_partA.json'
    if validation and planA.exists():
        for it in json.loads(planA.read_text(encoding='utf-8'))['graphs']:
            sp = out / 'partV' / it['name'] / 'supp_result.json'
            ep = out / 'partV' / it['name'] / 'supp_error.json'
            if ep.exists() and not sp.exists():
                problems['V:' + it['name']] = 'validation error'
            elif not sp.exists():
                vres = results / 'partA' / it['name'] / 'result.json'
                if vres.exists() and json.loads(vres.read_text(encoding='utf-8')).get('status') == 'done':
                    problems['V:' + it['name']] = 'not run'
            elif sp.exists():
                sr = json.loads(sp.read_text(encoding='utf-8'))
                if sr.get('status') == 'done':
                    vrows.append(dict(name=it['name'], comparisons=sr['comparisons'], all_identical=sr['all_identical']))
                elif sr.get('status') != 'no_stored_rank':
                    problems['V:' + it['name']] = sr.get('status')
    a = aggregate(rows)
    v_total = sum(len(v['comparisons']) for v in vrows)
    v_ident = sum(c['identical'] for v in vrows for c in v['comparisons'].values())
    a['validation'] = dict(graphs=len(vrows), ranks=v_total, identical=v_ident,
                           nonidentical=[f'{v["name"]} s{c["seed"]}' for v in vrows for c in v['comparisons'].values()
                                         if not c['identical']])

    M = {}
    M['NSNote'] = 'supplementary, not pre-registered'
    M['NSHeldCount'] = str(a['count'])
    M['NSHeldPlanned'] = str(len(planned))
    txt = f'{a["count"]} of {len(planned)}'
    extra = [f'{k}: {v}' for k, v in list(not_analysed.items()) + list((k, v) for k, v in problems.items() if not k.startswith('V:'))]
    M['NSHeldCountText'] = tex_safe(txt + (f' ({"; ".join(extra)})' if extra else ''))
    held_problems = {k: v for k, v in problems.items() if not k.startswith('V:')}
    M['NSCheck'] = (f'passed for all {a.get("verified_orders", 0)} supplementary orders' if not held_problems else
                    tex_safe(f'failed or not analysed on {len(held_problems)} graph(s): ' + '; '.join(held_problems)))
    vprob = {k[2:]: v for k, v in problems.items() if k.startswith('V:')}
    if not validation:
        M['NSValidation'] = 'Part V not run in this execution'
    else:
      M['NSValidation'] = (f'{v_ident} of {v_total} stored artifact Neigh ranks reproduced exactly'
                         + (f' (differences: {tex_safe(", ".join(a["validation"]["nonidentical"]))})' if v_ident < v_total else '')
                         + (f'; not validated: {tex_safe("; ".join(f"{k}: {v}" for k, v in vprob.items()))}' if vprob else ''))
    if rows:
        M['NSNeighSeeds'] = f'{a["neigh_ok"]} of {5 * a["count"]}'
        M['NSRefChanged'] = str(a['changed'])
        M['NSRefChangedList'] = tex_safe(', '.join(a['changed_list']) or 'none')
        M['NSGapMean'] = f2(a['gap_mean'])
        M['NSGapMax'] = f2(a['gap_max'])
        M['NSGapMaxGraph'] = tex_safe(a['gap_max_graph'])
        for pre, key in (('NSFrozen', 'frozen'), ('NSRe', 'rerun')):
            M[pre + 'TMean'] = f2(a[f'{key}_imp_T_mean'])
            M[pre + 'DoveMean'] = f2(a[f'{key}_imp_DOVE_mean'])
            M[pre + 'TILSMean'] = f2(a[f'{key}_imp_TILS_mean'])
            M[pre + 'TILSMedian'] = f2(a[f'{key}_imp_TILS_median'])
            M[pre + 'TILSITR'] = '/'.join(map(str, a[f'{key}_imp_TILS_itr']))
            M[pre + 'TILSCI'] = ci_txt(a[f'{key}_imp_TILS_ci'])
        M['NSFrozenTILSvsDove'] = '/'.join(map(str, a['frozen_TILS_vs_DOVE']))
        M['NSFrozenTILSvsRMean'] = f2(a['frozen_imp_TILS_vsR_mean'])
        M['NSReTILSCappedMean'] = f2(a['rerun_imp_TILS_capped_mean'])
        M['NSReTILSCappedCI'] = ci_txt(a['rerun_imp_TILS_capped_ci'])
        M['NSReTILSNoFinalMean'] = f2(a['rerun_imp_TILS_nofinal_mean'])
        cb = a['capped_branch_counts']
        M['NSReRuns'] = str(sum(cb.values()))
        M['NSReOverRuns'] = str(sum(cb.values()) - cb['final'])
        M['NSReCappedBranches'] = (f'{cb["ils"]} scored with the ILS incumbent, {cb["tangent"]} with the tangent-phase '
                                   f'order, {cb["start"]} with the start order')
        M['NSReReused'] = str(a['reused'])
        for o in ('DOVE', 'T'):
            M[f'NSReTILSvs{o.title() if o == "DOVE" else o}'] = '/'.join(map(str, a[f'rerun_TILS_vs_{o}']))
            M[f'NSReTILSvs{o.title() if o == "DOVE" else o}P'] = p_txt(a[f'rerun_TILS_vs_{o}_p'])
        M['NSReMaxOver'] = f'{100 * max(0.0, a["rerun_TILS_max_over_frac"]):.1f}\\%'
        for pre, key in (('NSTimeFrozen', 'timing_frozen'), ('NSTimeRe', 'timing_rerun')):
            M[pre + 'Geo'] = f'{a[key + "_geo"]:.4f}'
            M[pre + 'CI'] = f'[{a[key + "_ci"][0]:.4f}, {a[key + "_ci"][1]:.4f}]'
            M[pre + 'Below'] = str(a[key + '_graphs_ci_below1'])
            M[pre + 'Above'] = str(a[key + '_graphs_ci_above1'])
            M[pre + 'WorkGeo'] = f'{a[key + "_work_geo"]:.4f}'
    lines = ['% Generated by pc_eval_neigh_supp/make_supp_tables.py. SUPPLEMENTARY, NOT PRE-REGISTERED.']
    for k, v in M.items():
        lines.append(f'\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}')
    (out / 'supp_macros.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    rl = ['% Generated by pc_eval_neigh_supp/make_supp_tables.py. SUPPLEMENTARY, NOT PRE-REGISTERED.',
          '% dagger: R*N = frozen R*, so the rerun columns are the frozen runs (reused).']
    for r in rows:
        lab = tex_safe(r['RN_label'] if r['changed'] else r['R_label'])
        dag = '$^\\dagger$' if r['rerun_reused'] else ''
        rl.append(f'{tex_safe(r["label"])}{dag} & {lab} & {f2(r["gap"])} & {f2(r["frozen_imp_TILS"])} & '
                  f'{f2(r["rerun_imp_T"])} & {f2(r["rerun_imp_DOVE"])} & {f2(r["rerun_imp_TILS"])}$\\pm${r["rerun_imp_TILS_sd"]:.2f} \\\\')
    if rows:
        rl.append('\\midrule')
        rl.append(f'Mean & & {f2(a["gap_mean"])} & {f2(a["frozen_imp_TILS_mean"])} & {f2(a["rerun_imp_T_mean"])} & '
                  f'{f2(a["rerun_imp_DOVE_mean"])} & {f2(a["rerun_imp_TILS_mean"])} \\\\')
    (out / 'supp_heldout_rows.tex').write_text('\n'.join(rl) + '\n', encoding='utf-8')
    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in r.items()} for r in rows]
    if flat:
        with open(out / 'supp_per_graph.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
    (out / 'supp_summary.json').write_text(json.dumps(dict(note='SUPPLEMENTARY, NOT PRE-REGISTERED', aggregates=a,
                                                           not_analysed=not_analysed, problems=problems, macros=M,
                                                           per_graph=rows, validation=vrows), indent=1, default=str),
                                           encoding='utf-8')
    log(f'supplementary tables: {a["count"]} held-out graphs; validation {v_ident}/{v_total} identical')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--no-validation', action='store_true', help='Part V was not run')
    args = ap.parse_args()
    problems = write_all(args.results, args.out, validation=not args.no_validation)
    raise SystemExit(2 if problems else 0)


if __name__ == '__main__':
    main()
