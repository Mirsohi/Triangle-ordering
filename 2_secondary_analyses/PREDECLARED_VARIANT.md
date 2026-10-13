# Pre-declared fixed greedy-colouring variant (review round 1, concern C3)

- Declared (UTC): 2026-09-28T08:04:00Z
- Declared BEFORE any result of the pre-registered PC run (pc_eval) was seen. At the time of writing, no PC
  result exists in this sandbox; the PC run is still in progress. It was also declared before any new
  sandbox experiment of review round 1 (Table 2 extension, posthoc mock test) was run.
- Data used: only `/home/claude/ideas/work/randomgraph/posthoc_split.csv`
  (sha256 a0a750a72eecf6c4d27049fbf31f0fc51798687b7a2f7c51a7fbd69f128d6c1c): 24 sampled graphs
  (ER8, PL2.1, PL2.5, PL3.0; n = 1e3, 1e4, 1e5; graph seeds 0, 1).

## Rule (fixed before looking at the numbers)
For each of the four variants, compute on each of the 24 graphs the ratio Λ(variant) / min over the four variants,
and take the geometric mean over the 24 graphs. The variant with the smallest geometric mean is the fixed variant
(ties: smaller maximum ratio).

## Result

| variant (process order, colour-class direction) | geo-mean ratio (24 graphs) | max ratio | #graphs best | geo-mean, n = 1e5 only |
|---|---|---|---|---|
| reverse degeneracy, desc | 1.0706 | 1.226 | 3 | 1.0680 |
| reverse degeneracy, asc  | 1.0742 | 1.391 | 8 | 1.0568 |
| **descending degree, desc** | **1.0597** | 1.239 | **11** | 1.0571 |
| descending degree, asc   | 1.0736 | 1.262 | 2 | 1.0689 |

**Fixed variant: `DEG-desc`** = greedy colouring with vertices processed by descending degree (identifier ties),
colour classes placed as blocks in DESCENDING colour order (highest colour first, colour 0 last), identifier ties
inside a class. This is `kc.color_order(g, 'degree', 'desc')` of `work/kclique/kc.py`.

Remarks.
- The choice does not depend on how the "degeneracy" process order is defined (the historical code used the reverse
  of the artifact's `degeneracy_order`, which is a core ordering, while the review asks for reverse smallest-last):
  the chosen variant uses the degree process order.
- By construction a proper colouring has no edge inside a colour class, so colour-descending and the REVERSAL of
  colour-ascending induce the same orientation, hence the same Λ (see NOTES.md, claim P1). "Both directions" of the
  four variants therefore give eight orders but only four distinct Λ values.
- The same fixed variant is used for C+ILS-T in the sandbox Table 2 extension and in the post-hoc PC package.

## Addendum, 2026-09-28T09:25Z (the declaration above is unchanged; DEG-desc remains the fixed variant)

Added after the checker's audit (09:25Z), when the pre-registered PC run had finished (09:03Z) and its coordinator had seen partial pre-registered results, but before any post-hoc run. The variant itself was fixed at 08:04Z, before any PC result existed, and this addendum does not change it.

- **Candidate set.** DEG-desc was chosen among the four variants implemented at the time. In those variants the
  "degeneracy" process order was the reverse of the artifact's `degeneracy_order`, which is a core ordering, not true
  smallest-last. With the corrected construction (reverse Matula-Beck smallest-last, as used in the post-hoc package
  and the new Table 2), **SL-desc gives better starts than DEG-desc on three of the four random-graph laws**. On the
  four mock public graphs, DEG-desc is the worst of the four starts (mean -71% vs R*, against -25% for DEG-asc).
- **Why DEG-desc is kept.** To honour the pre-declaration, DEG-desc stays the fixed, deployable variant.
- **Per-graph best reported alongside.** The post-hoc package and Table 2 also report the per-graph best of the eight
  orders. It is selected by exact work before the search, and its selection cost is charged. Both analyses are post
  hoc and not part of the pre-registered protocol.
- **The choice was weakly determined.** On all 24 graphs DEG-desc has the lowest geometric mean (1.0597, against
  1.0706 / 1.0736 / 1.0742). At n = 1e5 alone it essentially ties the reverse-core-order ascending variant (1.0571
  vs 1.0568).
- **Wording correction to the remark above.** Descending and reversed-ascending (and vice versa) give identical work,
  so the eight orders have **at most** four distinct Λ values. Fewer occur when two variants coincide; one graph in
  the checker's test had three.

Suggested paper wording: "As a fixed, deployable rule we pre-declared, before any held-out or post-hoc PC result, the
variant with the lowest geometric-mean start work on the 24 sampled graphs of our earlier study among the four
variants then implemented: greedy coloring in descending-degree order with color classes in descending order
(DEG-desc). That comparison used a core-ordering process order instead of true reverse smallest-last; with the
corrected construction, SL-desc gives better starts on three of four random-graph laws. We keep DEG-desc to honour
the pre-declaration and also report the per-graph best of the eight orders, selected by exact work before the search
with its cost charged. Both analyses are post hoc and not part of the pre-registered protocol."
