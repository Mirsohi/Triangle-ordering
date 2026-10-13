# External baseline

The paper evaluates the **Neigh** ordering from Lécuyer et al.'s VOLT
implementation. `scripts/setup_volt.sh` clones the public GPL-3.0 repository at
commit `31ea49141cc75d9a547c16a5c954c881676ae109`, applies
`volt_reproducibility.patch`, and builds it.

The patch does not alter the Neigh objective or update rule. It (i) replaces
out-of-bounds `std::vector` writes after `reserve` with sized vectors, (ii)
adds a deterministic seed option, (iii) exports the vertex-rank permutation,
and (iv) permits stopping after ordering so baseline timing excludes triangle
listing. DOVE independently validates each exported permutation and evaluates
its exact candidate work; it does not trust VOLT's surrogate score.

VOLT itself is not redistributed. It remains subject to its upstream GPL-3.0
license and attribution requirements.
